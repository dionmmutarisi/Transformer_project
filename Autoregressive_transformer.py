
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import optuna
from optuna_integration.wandb import WeightsAndBiasesCallback
import wandb


from raw import load_wp

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ENTITY  = "dionmutarisi-vrije-universiteit-amsterdam"
PROJECT = "transformer-3b"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Q10 — Batch sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_batch(data: torch.Tensor, b: int, l: int) -> torch.Tensor:
    """
    Sample b random windows of length l+1 from the corpus tensor.
    Returns shape (b, l+1). Split [:, :-1] = input, [:, 1:] = target.
    """
    N       = data.size(0)
    starts  = torch.randint(low=0, high=N - l, size=(b,))
    offsets = torch.arange(l + 1, device=data.device)
    indices = starts[:, None] + offsets[None, :]
    return data[indices].detach()


# ─────────────────────────────────────────────────────────────────────────────
# Q11 — Model
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb: int, heads: int, mask: bool = False):
        super().__init__()
        assert emb % heads == 0, "emb must be divisible by heads"
        self.heads = heads
        self.dk    = emb // heads
        self.mask  = mask

        self.tokeys     = nn.Linear(emb, emb, bias=False)
        self.toqueries  = nn.Linear(emb, emb, bias=False)
        self.tovalues   = nn.Linear(emb, emb, bias=False)
        self.unifyheads = nn.Linear(emb, emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.size()
        h, dk   = self.heads, self.dk

        Q = self.toqueries(x).view(B, T, h, dk).transpose(1, 2)   # (B,h,T,dk)
        K = self.tokeys(x)   .view(B, T, h, dk).transpose(1, 2)
        V = self.tovalues(x) .view(B, T, h, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)  # (B,h,T,T)

        if self.mask:
            is_, js = torch.triu_indices(T, T, offset=1, device=x.device)
            scores[..., is_, js] = float('-inf')

        weights = F.softmax(scores, dim=-1)
        out     = torch.matmul(weights, V)                         # (B,h,T,dk)
        out     = out.transpose(1, 2).contiguous().view(B, T, d)   # (B,T,d)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb: int, heads: int, mask: bool,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(emb, heads, mask=mask)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff    = nn.Sequential(
            nn.Linear(emb, ff_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_mult * emb, emb),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attention(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab: int, emb: int = 300, heads: int = 6,
                 depth: int = 6, context: int = 256, dropout: float = 0.1):
        super().__init__()
        self.context = context

        self.token_embedding    = nn.Embedding(vocab, emb)
        self.position_embedding = nn.Embedding(context, emb)
        self.blocks = nn.Sequential(
            *[TransformerBlock(emb, heads, mask=True, dropout=dropout)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb)
        self.head = nn.Linear(emb, vocab)
        self.head.weight = self.token_embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.size()
        assert T <= self.context
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_embedding(x) + self.position_embedding(positions)
        h = self.blocks(h)
        h = self.norm(h)
        return self.head(h)   # (B, T, vocab) — logits, no softmax


# ─────────────────────────────────────────────────────────────────────────────
# Q12 — Validation loss in bits
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validation_loss_bits(model: nn.Module, val_data: torch.Tensor,
                         context: int, n_batches: int = 1000,
                         batch_size: int = 32, device: str = "cpu") -> float:
    """Average −log₂ p(last token | context) over n_batches random windows."""
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        batch       = sample_batch(val_data, batch_size, context).to(device)
        inp         = batch[:, :-1]       # (B, context)
        target      = batch[:, -1]        # (B,) — last token only
        logits      = model(inp)          # (B, context, vocab)
        last_logits = logits[:, -1, :]    # (B, vocab)
        nll_nats    = F.cross_entropy(last_logits, target, reduction="mean")
        total      += nll_nats.item() / math.log(2)
    model.train()
    return total / n_batches


# ─────────────────────────────────────────────────────────────────────────────
# Q13 — Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample(lnprobs: torch.Tensor, temperature: float = 1.0) -> int:
    if temperature == 0.0:
        return int(lnprobs.argmax().item())
    p  = F.softmax(lnprobs / temperature, dim=0)
    return int(dist.Categorical(p).sample().item())


@torch.no_grad()
def generate(model: nn.Module, seed: torch.Tensor, n_tokens: int,
             context: int, temperature: float = 1.0,
             device: str = "cpu") -> torch.Tensor:
    model.eval()
    generated = seed.clone().to(device)
    for _ in range(n_tokens):
        inp        = generated[-context:].unsqueeze(0)
        last_logit = model(inp)[0, -1]
        next_tok   = sample(last_logit, temperature)
        generated  = torch.cat([generated,
                                 torch.tensor([next_tok], device=device)])
    model.train()
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, target_lr: float,
           warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return target_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return target_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ─────────────────────────────────────────────────────────────────────────────
# Q13 & Q14 — Training loop (with W&B logging)
# ─────────────────────────────────────────────────────────────────────────────

def train(train_data: torch.Tensor,
          val_data:   torch.Tensor,
          i2c,
          vocab_size: int,
          # model HPs
          emb:      int   = 300,
          heads:    int   = 6,
          depth:    int   = 6,
          context:  int   = 256,
          dropout:  float = 0.1,
          # training HPs
          batch_size:   int   = 32,
          max_batches:  int   = 50_000,
          target_lr:    float = 3e-4,
          warmup_steps: int   = 1_000,
          grad_clip:    float = 1.0,
          # eval / logging
          eval_every:      int   = 10_000,
          sample_every:    int   = 10_000,
          sample_seed_len: int   = 16,
          n_sample_tokens: int   = 200,
          temperature:     float = 0.8,
          device:          str   = "cpu",
          # W&B — pass None to disable
          wandb_run=None) -> nn.Module:
    """
    Full training loop. Logs to W&B when wandb_run is provided.
    """
    model = AutoregressiveTransformer(
        vocab=vocab_size, emb=emb, heads=heads,
        depth=depth, context=context, dropout=dropout,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    no_decay = {"bias", "norm"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.1},
        {"params": [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=target_lr, betas=(0.9, 0.95))

    # initial val loss
    init_val = validation_loss_bits(model, val_data, context,
                                    n_batches=200, batch_size=batch_size,
                                    device=device)
    print(f"Step 0 | val_loss_bits: {init_val:.4f}")
    if wandb_run:
        wandb_run.log({"val_loss_bits": init_val, "step": 0})

    running_loss = 0.0
    model.train()

    for step in range(1, max_batches + 1):

        # LR schedule
        lr = get_lr(step, target_lr, warmup_steps, max_batches)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # forward + loss
        batch      = sample_batch(train_data, batch_size, context).to(device)
        inp        = batch[:, :-1]
        target     = batch[:, 1:]
        logits     = model(inp)
        loss       = F.cross_entropy(logits.transpose(1, 2), target,
                                     reduction="sum") / inp.numel()

        # backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()

        # logging every 100 steps
        if step % 100 == 0:
            avg = running_loss / 100
            running_loss = 0.0
            print(f"Step {step:6d} | {avg:.4f} nats "
                  f"({avg/math.log(2):.4f} bits) | "
                  f"gnorm {grad_norm:.3f} | lr {lr:.2e}")
            if wandb_run:
                wandb_run.log({
                    "train_loss_nats": avg,
                    "train_loss_bits": avg / math.log(2),
                    "grad_norm":       grad_norm.item(),
                    "lr":              lr,
                    "step":            step,
                })

        # validation
        if step % eval_every == 0:
            val_bits = validation_loss_bits(model, val_data, context,
                                            n_batches=1000, batch_size=batch_size,
                                            device=device)
            print(f"\n── Step {step} | val_loss_bits: {val_bits:.4f} ──\n")
            if wandb_run:
                wandb_run.log({"val_loss_bits": val_bits, "step": step})

        # sampling
        if step % sample_every == 0:
            start = torch.randint(0, val_data.size(0) - sample_seed_len, (1,)).item()
            seed  = val_data[start: start + sample_seed_len]
            gen   = generate(model, seed, n_sample_tokens, context,
                             temperature=temperature, device=device)
            seed_str = "".join(i2c[t] for t in seed.tolist())
            gen_str  = "".join(i2c[t] for t in gen[sample_seed_len:].tolist())
            print(f"Seed:      '{seed_str}'")
            print(f"Generated: '{gen_str}'\n")
            if wandb_run:
                wandb_run.log({"sample": f"SEED: {seed_str} | GEN: {gen_str}",
                               "step": step})

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Optuna + W&B integration
# ─────────────────────────────────────────────────────────────────────────────

# Each Optuna trial becomes its own W&B run (as_multirun=True).
# This gives you the parallel coordinates + parameter importance panels
# in the W&B project page.

wandb_callback = WeightsAndBiasesCallback(
    metric_name="val_loss_bits",
    wandb_kwargs={
        "entity":  ENTITY,
        "project": PROJECT,
        "group":   "optuna-sweep",
    },
    as_multirun=True,
)


@wandb_callback.track_in_wandb()
def objective(trial: optuna.Trial) -> float:
    """One Optuna trial = one short training run. Returns val_loss_bits."""

    # ── suggest hyperparameters ───────────────────────────────────────────────
    lr         = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
    emb        = trial.suggest_categorical("emb",  [128, 256, 300])
    depth      = trial.suggest_int("depth",        2, 6)
    heads      = trial.suggest_categorical("heads", [4, 6])
    dropout    = trial.suggest_float("dropout",    0.0, 0.3, step=0.05)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # heads must divide emb — prune invalid combos immediately
    if emb % heads != 0:
        raise optuna.exceptions.TrialPruned()

    wandb.config.update(dict(
        lr=lr, emb=emb, depth=depth,
        heads=heads, dropout=dropout, batch_size=batch_size,
    ))

    # ── load data ────────────────────────────────────────────────────────────
    (train_data, val_data), (i2c, c2i) = load_wp(final=False)
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    val_tensor   = torch.tensor(val_data,   dtype=torch.long)
    vocab_size   = len(i2c)

    # ── build model ──────────────────────────────────────────────────────────
    model = AutoregressiveTransformer(
        vocab=vocab_size, emb=emb, heads=heads,
        depth=depth, context=256, dropout=dropout,
    ).to(DEVICE)

    no_decay = {"bias", "norm"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.1},
        {"params": [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    # ── short training run for tuning ─────────────────────────────────────────
    # 3000 steps is enough to compare HP settings without full convergence.
    N_STEPS    = 3_000
    CONTEXT    = 256
    EVAL_EVERY = 500
    best_val   = float("inf")
    model.train()

    for step in range(1, N_STEPS + 1):
        batch     = sample_batch(train_tensor, batch_size, CONTEXT).to(DEVICE)
        inp       = batch[:, :-1]
        target    = batch[:, 1:]
        logits    = model(inp)
        loss      = F.cross_entropy(logits.transpose(1, 2), target,
                                    reduction="sum") / inp.numel()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        wandb.log({"train_loss_bits": loss.item() / math.log(2), "step": step})

        if step % EVAL_EVERY == 0:
            val_bits = validation_loss_bits(
                model, val_tensor, CONTEXT,
                n_batches=200, batch_size=batch_size, device=DEVICE,
            )
            wandb.log({"val_loss_bits": val_bits, "step": step})

            if val_bits < best_val:
                best_val = val_bits

            # Pruning: tell Optuna the current metric; it may cut this trial
            # early if it's clearly worse than past trials at the same step.
            trial.report(best_val, step=step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_val


def run_study(n_trials: int = 30) -> optuna.Study:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=500,
        ),
        study_name="transformer-3b-sweep",
    )

    # n_jobs=1 is required — parallel trials corrupt the W&B callback state
    study.optimize(objective, n_trials=n_trials,
                   callbacks=[wandb_callback], n_jobs=1)

    best = study.best_trial
    print(f"\n── Best trial ──")
    print(f"  val_loss_bits: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # one final W&B run summarising the whole study
    with wandb.init(entity=ENTITY, project=PROJECT,
                    group="optuna-sweep", name="study-summary",
                    config=best.params) as run:
        run.summary["best_val_loss_bits"] = best.value
        run.summary["n_trials"]  = len(study.trials)
        run.summary["n_pruned"]  = sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        )

    return study


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        # python transformer_train.py tune
        run_study(n_trials=30)

    else:
        # python transformer_train.py
        # — full training run on toy data, logged to W&B
        (train_data, val_data), (i2c, c2i) = load_wp(final=False)
        train_tensor = torch.tensor(train_data, dtype=torch.long)
        val_tensor   = torch.tensor(val_data,   dtype=torch.long)
        vocab_size   = len(i2c)

        with wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name="full-run",
            config=dict(emb=128, heads=4, depth=4, context=128,
                        dropout=0.1, batch_size=64, lr=3e-4),
        ) as run:
            train(
                train_data   = train_tensor,
                val_data     = val_tensor,
                i2c          = i2c,
                vocab_size   = vocab_size,
                emb          = 128,
                heads        = 4,
                depth        = 4,
                context      = 128,
                dropout      = 0.1,
                batch_size   = 64,
                max_batches  = 50_000,
                target_lr    = 3e-4,
                warmup_steps = 1_000,
                grad_clip    = 1.0,
                eval_every   = 10_000,
                sample_every = 10_000,
                device       = DEVICE,
                wandb_run    = run,
            )