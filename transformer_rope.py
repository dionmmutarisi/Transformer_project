"""
transformer_rope.py — Rotary Position Embeddings (RoPE)
Drop-in replacement for learned positional embeddings.

Usage:
    python transformer_rope.py --mode toy
    python transformer_rope.py --mode wp --emb 512 --heads 8
    python transformer_rope.py --mode tune --n-trials 30
"""

import argparse, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import optuna
from optuna_integration.wandb import WeightsAndBiasesCallback
import wandb

from raw import load_toy, load_wp

ENTITY = "dionmutarisi-vrije-universiteit-amsterdam"
PROJECT = "transformer-3b"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",         default="toy",  choices=["toy","wp","tune"])
    p.add_argument("--emb",          type=int,   default=128)
    p.add_argument("--heads",        type=int,   default=4)
    p.add_argument("--depth",        type=int,   default=4)
    p.add_argument("--context",      type=int,   default=256)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--max-batches",  type=int,   default=50_000)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup",       type=int,   default=1_000)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--eval-every",   type=int,   default=10_000)
    p.add_argument("--sample-every", type=int,   default=10_000)
    p.add_argument("--wandb-name",   type=str,   default=None)
    p.add_argument("--n-trials",     type=int,   default=30)
    return p.parse_args()


# ── Q10: batch sampler ────────────────────────────────────────────────────────

def sample_batch(data, b, l):
    N = data.size(0)
    starts  = torch.randint(0, N - l, (b,))
    offsets = torch.arange(l + 1, device=data.device)
    return data[starts[:, None] + offsets[None, :]].detach()


# ── Q11: model ────────────────────────────────────────────────────────────────

# ── RoPE helper functions ─────────────────────────────────────────────────────

def precompute_rope_freqs(dim, max_seq_len, base=10000.0, device='cpu'):
    """Precompute cos/sin rotation tables for all positions up to max_seq_len."""
    i       = torch.arange(0, dim, 2, device=device).float()
    thetas  = 1.0 / (base ** (i / dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs   = torch.outer(positions, thetas)   # (max_seq_len, dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """
    Rotate Q or K by position-dependent angles.
    x   : (B, h, T, dk)
    cos : (T, dk/2)
    sin : (T, dk/2)
    """
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    cos = cos[:x.size(2)].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.size(2)].unsqueeze(0).unsqueeze(0)
    out = torch.stack([x_even * cos - x_odd * sin,
                       x_even * sin + x_odd * cos], dim=-1)
    return out.flatten(-2)


# ── RoPE model classes ────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb, heads, mask=False, context=256):
        super().__init__()
        assert emb % heads == 0
        self.heads, self.dk, self.mask = heads, emb // heads, mask
        self.tokeys     = nn.Linear(emb, emb, bias=False)
        self.toqueries  = nn.Linear(emb, emb, bias=False)
        self.tovalues   = nn.Linear(emb, emb, bias=False)
        self.unifyheads = nn.Linear(emb, emb)
        cos, sin = precompute_rope_freqs(self.dk, context)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x):
        B, T, d = x.size()
        h, dk   = self.heads, self.dk
        Q = self.toqueries(x).view(B,T,h,dk).transpose(1,2)
        K = self.tokeys(x)   .view(B,T,h,dk).transpose(1,2)
        V = self.tovalues(x) .view(B,T,h,dk).transpose(1,2)
        Q = apply_rope(Q, self.cos, self.sin)   # rotate Q
        K = apply_rope(K, self.cos, self.sin)   # rotate K — NOT V
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(dk)
        if self.mask:
            i_, j_ = torch.triu_indices(T, T, offset=1, device=x.device)
            scores[..., i_, j_] = float('-inf')
        out = torch.matmul(F.softmax(scores, dim=-1), V)
        return self.unifyheads(out.transpose(1,2).contiguous().view(B,T,d))


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, context=256, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(emb, heads, mask, context)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff    = nn.Sequential(nn.Linear(emb, ff_mult*emb), nn.ReLU(),
                                   nn.Linear(ff_mult*emb, emb))
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab, emb=300, heads=6, depth=6, context=256, dropout=0.1):
        super().__init__()
        self.context = context
        self.tok_emb = nn.Embedding(vocab, emb)
        # NO position embedding table — RoPE handles it inside attention
        self.blocks  = nn.Sequential(
            *[TransformerBlock(emb, heads, mask=True, context=context,
                               dropout=dropout)
              for _ in range(depth)])
        self.norm = nn.LayerNorm(emb)
        self.head = nn.Linear(emb, vocab)
        self.head.weight = self.tok_emb.weight   # weight tying
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T = x.size()
        # token embeddings only — position encoded inside attention via RoPE
        h = self.tok_emb(x)
        return self.head(self.norm(self.blocks(h)))   # (B, T, vocab)


# ── Q12: validation loss in bits ─────────────────────────────────────────────

@torch.no_grad()
def validation_loss_bits(model, val_data, context, n_batches=500,
                         batch_size=32, device="cpu"):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        batch  = sample_batch(val_data, batch_size, context).to(device)
        nll    = F.cross_entropy(model(batch[:,:-1])[:,-1,:],
                                 batch[:,-1], reduction="mean")
        total += nll.item() / math.log(2)
    model.train()
    return total / n_batches


# ── Q13: sampling ─────────────────────────────────────────────────────────────

def sample_token(logits, temperature=1.0):
    if temperature == 0.0:
        return int(logits.argmax().item())
    return int(dist.Categorical(F.softmax(logits / temperature, dim=0)).sample())


@torch.no_grad()
def generate(model, seed, n_tokens, context, temperature=1.0, device="cpu"):
    model.eval()
    gen = seed.clone().to(device)
    for _ in range(n_tokens):
        inp = gen[-context:].unsqueeze(0)
        tok = sample_token(model(inp)[0, -1], temperature)
        gen = torch.cat([gen, torch.tensor([tok], device=device)])
    model.train()
    return gen


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(step, target_lr, warmup, max_steps):
    if step < warmup:
        return target_lr * step / warmup
    p = (step - warmup) / max(1, max_steps - warmup)
    return target_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


# ── Q13/Q14: training loop ────────────────────────────────────────────────────

def train(train_data, val_data, i2c, vocab_size,
          emb=128, heads=4, depth=4, context=256, dropout=0.1,
          batch_size=32, max_batches=50_000, target_lr=3e-4,
          warmup_steps=1_000, grad_clip=1.0,
          eval_every=10_000, sample_every=10_000,
          seed_len=16, n_gen=200, temperature=0.8,
          device="cpu", wandb_run=None):

    model = AutoregressiveTransformer(
        vocab_size, emb, heads, depth, context, dropout).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}  device: {device}")

    no_decay = {"bias", "norm"}
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": 0.1},
        {"params": [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ], lr=target_lr, betas=(0.9, 0.95))

    init_val = validation_loss_bits(model, val_data, context,
                                    n_batches=100, batch_size=batch_size,
                                    device=device)
    print(f"Step 0 | val: {init_val:.4f} bits")
    if wandb_run:
        wandb_run.log({"val_loss_bits": init_val, "step": 0})

    running = 0.0
    for step in range(1, max_batches + 1):
        lr = get_lr(step, target_lr, warmup_steps, max_batches)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch  = sample_batch(train_data, batch_size, context).to(device)
        logits = model(batch[:, :-1])
        loss   = F.cross_entropy(logits.transpose(1,2), batch[:,1:],
                                 reduction="sum") / batch[:,:-1].numel()
        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running += loss.item()

        if step % 100 == 0:
            avg  = running / 100
            running = 0.0
            bits = avg / math.log(2)
            print(f"Step {step:6d} | {avg:.4f} nats ({bits:.4f} bits) "
                  f"| gnorm {gnorm:.3f} | lr {lr:.2e}")
            if wandb_run:
                wandb_run.log({"train_loss_nats": avg, "train_loss_bits": bits,
                               "grad_norm": float(gnorm), "lr": lr, "step": step})

        if step % eval_every == 0:
            val_bits = validation_loss_bits(model, val_data, context,
                                            n_batches=500, batch_size=batch_size,
                                            device=device)
            print(f"\n── val: {val_bits:.4f} bits ──\n")
            if wandb_run:
                wandb_run.log({"val_loss_bits": val_bits, "step": step})
            ckpt = f'/var/scratch/dmu224/experiments/checkpoint_step{step}.pt'
            torch.save(model.state_dict(), ckpt)
            print(f"Saved {ckpt}")

        if step % sample_every == 0 and val_data.size(0) > seed_len:
            start = torch.randint(0, val_data.size(0) - seed_len, (1,)).item()
            seed  = val_data[start: start + seed_len]
            gen   = generate(model, seed, n_gen, context,
                             temperature=temperature, device=device)
            def decode(ids):
                out = []
                for t in ids:
                    c = i2c[t] if isinstance(i2c, (list, dict)) else chr(t)
                    out.append(c if isinstance(c, str) else chr(t))
                return "".join(out)
            s_str = decode(seed.tolist())
            g_str = decode(gen[seed_len:].tolist())
            print(f"Seed: '{s_str}'\nGen:  '{g_str}'\n")
            if wandb_run:
                wandb_run.log({"sample": f"[{step}] {s_str} → {g_str}", "step": step})

    return model


# ── Optuna sweep ──────────────────────────────────────────────────────────────

def run_study(train_tensor, val_tensor, vocab_size, n_trials=30):
    cb = WeightsAndBiasesCallback(
        metric_name="val_loss_bits",
        wandb_kwargs={"entity": ENTITY, "project": PROJECT, "group": "optuna-sweep"},
        as_multirun=True,
    )

    @cb.track_in_wandb()
    def objective(trial):
        lr      = trial.suggest_float("lr",      1e-4, 1e-2, log=True)
        emb     = trial.suggest_categorical("emb",    [128, 256, 512])
        depth   = trial.suggest_int("depth",     2, 8)
        heads   = trial.suggest_categorical("heads",  [4, 8])
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        bs      = trial.suggest_categorical("batch_size", [16, 32, 64])
        ctx     = trial.suggest_categorical("context",    [128, 256])
        if emb % heads != 0:
            raise optuna.exceptions.TrialPruned()
        wandb.config.update(dict(lr=lr, emb=emb, depth=depth, heads=heads,
                                 dropout=dropout, batch_size=bs, context=ctx))

        model = AutoregressiveTransformer(vocab_size, emb, heads,
                                          depth, ctx, dropout).to(DEVICE)
        no_decay = {"bias", "norm"}
        opt = torch.optim.AdamW([
            {"params": [p for n,p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)], "weight_decay": 0.1},
            {"params": [p for n,p in model.named_parameters()
                        if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.95))

        best_val = float("inf")
        for step in range(1, 3001):
            batch  = sample_batch(train_tensor, bs, ctx).to(DEVICE)
            logits = model(batch[:, :-1])
            loss   = F.cross_entropy(logits.transpose(1,2), batch[:,1:],
                                     reduction="sum") / batch[:,:-1].numel()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            wandb.log({"train_loss_bits": loss.item()/math.log(2), "step": step})

            if step % 500 == 0:
                vb = validation_loss_bits(model, val_tensor, ctx,
                                          n_batches=100, batch_size=bs, device=DEVICE)
                wandb.log({"val_loss_bits": vb, "step": step})
                if vb < best_val:
                    best_val = vb
                trial.report(best_val, step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        return best_val

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=500),
        study_name="transformer-3b-sweep",
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[cb], n_jobs=1)

    best = study.best_trial
    print(f"\nBest: {best.value:.4f} bits")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    with wandb.init(entity=ENTITY, project=PROJECT, group="optuna-sweep",
                    name="study-summary", config=best.params) as run:
        run.summary["best_val_loss_bits"] = best.value
        run.summary["n_trials"]  = len(study.trials)
        run.summary["n_pruned"]  = sum(1 for t in study.trials
                                       if t.state == optuna.trial.TrialState.PRUNED)
    return study


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    os.makedirs("logs", exist_ok=True)

    if args.mode in ("toy", "tune"):
        (td, vd), (i2c, c2i) = load_toy(final=False)
        train_t = torch.tensor(td, dtype=torch.long)
        val_t   = torch.tensor(vd, dtype=torch.long)
        vocab   = len(i2c)
        print(f"Toy: vocab={vocab}, train={len(train_t)}, val={len(val_t)}")
    else:
        # Wikipedia — 256-byte vocabulary
        (td, vd), _ = load_wp(final=False)
        train_t = torch.tensor(td, dtype=torch.long) if not isinstance(td, torch.Tensor) else td.long()
        val_t   = torch.tensor(vd, dtype=torch.long) if not isinstance(vd, torch.Tensor) else vd.long()
        vocab   = 256
        i2c     = {i: chr(i) for i in range(256)}
        print(f"Wikipedia: vocab={vocab}, train={len(train_t):,}, val={len(val_t):,}")

    if args.mode == "tune":
        run_study(train_t, val_t, vocab, n_trials=args.n_trials)
    else:
        run_name = args.wandb_name or f"rope-{args.mode}"
        with wandb.init(entity=ENTITY, project=PROJECT,
                        name=run_name, group="learned-vs-rope",
                        config=vars(args)) as run:
            train(train_t, val_t, i2c, vocab,
                  emb=args.emb, heads=args.heads, depth=args.depth,
                  context=args.context, dropout=args.dropout,
                  batch_size=args.batch_size, max_batches=args.max_batches,
                  target_lr=args.lr, warmup_steps=args.warmup,
                  grad_clip=args.grad_clip, eval_every=args.eval_every,
                  sample_every=args.sample_every,
                  device=DEVICE, wandb_run=run)
