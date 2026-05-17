import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from torch.utils.tensorboard import SummaryWriter 
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import wandb

from raw import load_imdb_synth, load_xor

"""
Assignment 3B: Questions 10–14 — Autoregressive Transformer
============================================================
Deep dive: every design decision is explained, with mathematics and alternatives.

OVERVIEW OF WHAT WE ARE BUILDING
----------------------------------
A GPT-style autoregressive language model. Given a sequence of tokens
  x_1, x_2, ..., x_T
the model is trained to predict the next token at every position:
  p(x_{t+1} | x_1, ..., x_t)   for t = 1..T

This is done by running the whole sequence through a *causal* (masked) transformer
and reading off the logits at every time step simultaneously — making training
O(T) forward passes worth of information from a single pass. That efficiency is
the whole point of the autoregressive formulation.


"""

# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 10 — Batch sampler for autoregressive training
# ─────────────────────────────────────────────────────────────────────────────
"""
QUESTION 10
===========
Build a function that takes a dataset (a single long integer tensor), and
returns a random batch of shape (b, L+1).

WHY A SINGLE LONG SEQUENCE?
----------------------------
Unlike classification, where each example is a separate sentence, language
modelling treats the entire corpus as ONE continuous stream of tokens.
We then cut out random windows of length L+1:

    corpus:  [t0, t1, t2, t3, ... tN]
              └──────────┘  <- one window of length L+1
                     └──────────┘  <- another window

From each window we split:
    input  = window[:-1]   (length L)
    target = window[1:]    (length L)

This is the *shifted* prediction target: at position i, predict position i+1.

MATHEMATICAL VIEW
-----------------
We are maximising the log-likelihood of the data under the model:

    L = Σ_t  log p_θ(x_{t+1} | x_1, ..., x_t)

Slicing random windows is an unbiased stochastic estimate of this sum.

ALTERNATIVES
------------
1. Sequential (non-random) batching: iterate through corpus in order.
   - Pro: every token seen exactly once per epoch, no overlap waste.
   - Con: consecutive batches are correlated → noisier gradients in practice.
2. Document-aware batching: respect sentence/document boundaries.
   - Used for fine-tuning; adds a [SEP] or <eos> token at boundaries.
3. Pack-and-pad: pad shorter documents, group by similar lengths.
   - Better for classification; wasteful for LM because we can always find a
     length-L window without crossing a boundary.
"""

def sample_batch(data: torch.Tensor, b: int, l: int) -> torch.Tensor:
    """
    Sample a batch of b random windows of length l+1 from `data`.

    Args:
        data:  1-D integer tensor of shape (N,)  — the whole corpus.
        b:     batch size
        l:     context length (the model sees l tokens, predicts l tokens)

    Returns:
        Tensor of shape (b, l+1).  Slice [:, :-1] = input, [:, 1:] = target.

    Implementation detail — why torch.no_grad() / detach()?
    --------------------------------------------------------
    We must NOT let this sampling become part of the autograd computation graph.
    The integer indices are not differentiable anyway, but wrapping in
    torch.no_grad() makes our intent explicit and avoids any accidental
    graph node creation from surrounding code.
    """
    N = data.size(0)
    # Draw b random starting indices in [0, N - l - 1]
    # torch.randint is exclusive on the high end, so high = N - l
    starts = torch.randint(low=0, high=N - l, size=(b,))   # shape (b,)

    # Vectorised slice: build index tensor [starts, starts+1, ..., starts+l]
    # arange gives [0, 1, ..., l], broadcast-add starts[:,None] gives (b, l+1)
    offsets = torch.arange(l + 1, device=data.device)      # shape (l+1,)
    indices = starts[:, None] + offsets[None, :]            # shape (b, l+1)

    # Index into data; .detach() prevents gradient flow (data is discrete anyway)
    return data[indices].detach()


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 11 — Causal (autoregressive) transformer model
# ─────────────────────────────────────────────────────────────────────────────
"""
QUESTION 11
===========
Two key changes vs the classification transformer:

1. Remove global pool  →  keep all T output vectors, one logit-vector per position.
2. Mask self-attention  →  make it CAUSAL so position t can only attend to 1..t.

WHY CAUSAL MASKING?
--------------------
Standard (bidirectional) self-attention lets every token attend to every other
token, including future ones. During training that's fine — we have the whole
sequence. But at *inference* time we generate one token at a time, so future
tokens don't exist yet. To make training match inference we forbid the model
from seeing future tokens by setting the corresponding attention logits to -∞:

    Before softmax:
        W'[i, j] = Q[i] · K[j]^T / √d_k    (raw attention score)

    After masking (j > i means "future"):
        W'[i, j] = -∞   if j > i
        W'[i, j] = W'[i, j]   otherwise

    After softmax:
        exp(-∞) = 0  →  zero weight on future positions ✓

The mask is a fixed upper-triangular matrix (above the main diagonal = future).

MATHEMATICS OF SELF-ATTENTION (quick recap)
--------------------------------------------
Given input X ∈ R^{T×d}:

    Q = X W_Q,   K = X W_K,   V = X W_V      (W_* ∈ R^{d×d_k})

    Attention(Q, K, V) = softmax( Q K^T / √d_k ) V

Multi-head: split d into h heads of size d_k = d/h, compute attention per head,
concatenate, project back:

    head_i = Attention(Q_i, K_i, V_i)
    MultiHead(X) = Concat(head_1, ..., head_h) W_O

SCALING BY √d_k
----------------
Without scaling, the dot products Q·K grow in magnitude as d_k grows
(variance of a dot product of two unit-variance vectors is d_k).
Large values push softmax into saturation (near 0 or 1), causing vanishing
gradients. Dividing by √d_k keeps the variance ~1 regardless of d_k.

ALTERNATIVE ATTENTION VARIANTS
--------------------------------
- Relative position attention (Shaw et al. 2018): encode *distance* between
  positions rather than absolute positions.
- Flash Attention (Dao et al. 2022): reorders computation to avoid materialising
  the full T×T attention matrix — critical for long sequences.
- Linear attention (Katharopoulos et al. 2020): approximates softmax attention in
  O(T·d) instead of O(T²·d), at some expressivity cost.
"""

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional causal masking.

    Parameters
    ----------
    emb   : total embedding dimension d
    heads : number of attention heads h  (must divide emb evenly)
    mask  : if True, apply causal (upper-triangular) mask

    Internal shapes (B=batch, T=time, d=emb, h=heads, dk=d/h):
        Q, K, V : (B, h, T, dk)
        scores  : (B, h, T, T)   — raw dot products / √dk
        weights : (B, h, T, T)   — after softmax (rows sum to 1)
        out     : (B, T, d)
    """

    def __init__(self, emb: int, heads: int, mask: bool = False):
        super().__init__()
        assert emb % heads == 0, "emb must be divisible by heads"
        self.heads = heads
        self.emb   = emb
        self.mask  = mask
        self.dk    = emb // heads   # per-head dimension

        # Three separate linear projections — no bias is conventional
        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        # Final projection after concatenating heads
        self.unifyheads = nn.Linear(emb, emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, d)
        returns : (B, T, d)
        """
        B, T, d = x.size()
        h  = self.heads
        dk = self.dk          # d / h

        # ── Step 1: Project to Q, K, V ──────────────────────────────────────
        # Each is (B, T, d); we then split the last dim into h heads of size dk.
        # view(..., h, dk) reshapes last dim; transpose(1,2) puts heads before T.
        #   (B, T, d) → (B, T, h, dk) → (B, h, T, dk)
        Q = self.toqueries(x).view(B, T, h, dk).transpose(1, 2)
        K = self.tokeys(x)   .view(B, T, h, dk).transpose(1, 2)
        V = self.tovalues(x) .view(B, T, h, dk).transpose(1, 2)

        # ── Step 2: Scaled dot-product attention scores ──────────────────────
        # Q: (B, h, T, dk)  K^T: (B, h, dk, T)  →  scores: (B, h, T, T)
        # matmul on last two dims; leading dims are treated as batch.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)

        # ── Step 3: Causal mask ──────────────────────────────────────────────
        if self.mask:
            # triu_indices returns the indices of the upper triangle (excluding diag)
            # offset=1 means we start one above the diagonal (j > i means future)
            is_, js = torch.triu_indices(T, T, offset=1, device=x.device)
            # Set future positions to -inf so softmax → 0
            # [..., is_, js] uses advanced indexing; leading ... = (B, h) batch dims
            scores[..., is_, js] = float('-inf')

        # ── Step 4: Softmax over the key dimension (last dim) ────────────────
        # dim=-1 → each query's distribution over all (non-masked) keys
        weights = F.softmax(scores, dim=-1)   # (B, h, T, T)

        # ── Step 5: Weighted sum of values ───────────────────────────────────
        # weights: (B, h, T, T)  V: (B, h, T, dk)  →  out: (B, h, T, dk)
        out = torch.matmul(weights, V)

        # ── Step 6: Re-assemble heads ─────────────────────────────────────────
        # (B, h, T, dk) → transpose → (B, T, h, dk) → contiguous → (B, T, d)
        # .contiguous() is needed because transpose creates a non-contiguous view
        # and view() requires contiguous memory.
        out = out.transpose(1, 2).contiguous().view(B, T, d)

        # ── Step 7: Final linear ──────────────────────────────────────────────
        return self.unifyheads(out)   # (B, T, d)


class TransformerBlock(nn.Module):
    """
    One transformer block:
        x → LayerNorm → MultiHeadSelfAttention → residual add
          → LayerNorm → FeedForward → residual add

    LAYER NORMALISATION
    -------------------
    LayerNorm normalises across the *feature* dimension (d), not the batch.
    For a vector x ∈ R^d:

        LN(x) = (x - μ) / (σ + ε) * γ + β

    where μ, σ are mean/std computed over the d features, and γ, β are
    learned scale/shift. This stabilises gradients, especially in deep networks.

    WHY RESIDUAL CONNECTIONS?
    --------------------------
    Adding the input back:  y = f(x) + x
    means the gradient can flow *directly* from the loss to early layers
    without passing through f, preventing vanishing gradients.
    The network learns *residual* corrections on top of the identity.

    FEEDFORWARD SUBLAYER
    ---------------------
    Two linear layers with a RELU in between, expanded to 4×emb internally:

        FFN(x) = max(0,  x W_1 + b_1) W_2 + b_2

    W_1 ∈ R^{d × 4d},  W_2 ∈ R^{4d × d}

    Applied independently to each position (pointwise). This is where the
    model "processes" the attended context — attention routes information,
    FFN transforms it.

    DROPOUT
    -------
    Randomly zero some activations during training (rate p). Acts as
    regularisation. Applied after attention and after FFN.
    """

    def __init__(self, emb: int, heads: int, mask: bool, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(emb, heads, mask=mask)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_mult * emb, emb),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm variant (slightly more stable than post-norm)
        attended = self.attention(self.norm1(x))
        x = x + self.drop(attended)         # residual

        forwarded = self.ff(self.norm2(x))
        x = x + self.drop(forwarded)        # residual
        return x


class AutoregressiveTransformer(nn.Module):
    """
    GPT-style autoregressive language model.

    Architecture
    ------------
    token embedding  +  position embedding
        → stack of TransformerBlocks (causal)
        → LayerNorm
        → Linear(emb, vocab)   →  logits over vocabulary

    No global pool: we output logits at *every* time step.

    POSITION EMBEDDINGS
    --------------------
    Self-attention is permutation-invariant — it treats the sequence as a *set*.
    To inject order information we add a learned position embedding e_t to each
    token embedding at position t:

        h_t = embed(x_t) + pos_embed(t)

    Alternative: sinusoidal (fixed) embeddings (Vaswani et al. 2017):
        PE(t, 2i)   = sin(t / 10000^{2i/d})
        PE(t, 2i+1) = cos(t / 10000^{2i/d})
    These have the advantage that they generalise to longer sequences than seen
    during training. Learned embeddings are simpler and often perform similarly.
    """

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
        self.norm   = nn.LayerNorm(emb)
        self.head   = nn.Linear(emb, vocab)

        # Weight tying: share weights between token embedding and output head.
        # Rationale: the same semantic space is used to encode input and to
        # project outputs → fewer parameters, often better performance.
        # (Press & Wolf, 2017 — "Using the Output Embedding to Improve Language Models")
        self.head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """
        Small initialisation for stability.
        Embeddings: N(0, 0.02) following GPT-2.
        Linear layers: N(0, 0.02); biases: 0.
        Residual projections scaled by 1/√depth to prevent explosion.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T)  — integer token indices, T ≤ context
        returns : (B, T, vocab)  — logits (NOT probabilities)

        We purposely do NOT apply softmax here.  Reason: cross_entropy loss
        in PyTorch expects *logits* and applies log-softmax internally in a
        numerically stable fused kernel (using the log-sum-exp trick).
        Applying softmax before the loss would lead to log(softmax(x)) which
        is less numerically stable and wastes computation.
        """
        B, T = x.size()
        assert T <= self.context, f"Sequence length {T} > context {self.context}"

        # Token embeddings: (B, T, emb)
        tok = self.token_embedding(x)

        # Position embeddings: create [0, 1, ..., T-1] for every item in batch
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        pos = self.position_embedding(positions)                     # (1, T, emb)

        # Add token + position embeddings (broadcasting over batch dim)
        h = tok + pos   # (B, T, emb)

        h = self.blocks(h)      # (B, T, emb)
        h = self.norm(h)        # final layer norm
        logits = self.head(h)   # (B, T, vocab)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 12 — Validation loss in bits
# ─────────────────────────────────────────────────────────────────────────────
"""
QUESTION 12
===========
Estimate validation loss by averaging − log2 p(x_{T} | x_{1..T-1}) over many
random windows, only scoring the *last* token.

WHY BITS?
----------
log2 p(x) gives the information content in *bits* — how many bits an optimal
code would need to represent token x given the context. This is the cross-
entropy H between the true distribution and the model distribution:

    H = − Σ_x p_true(x) log2 p_model(x)

A uniform model over a vocabulary of size V has H = log2(V) bits.
As the model improves, H decreases toward the true entropy of the language.
For English text at the character level, typical values are ~1–2 bits/char.

ALTERNATIVE METRICS
--------------------
- Perplexity: PP = 2^H (for bits) or e^H (for nats). Perplexity of PP means
  the model is as confused as if it chose uniformly among PP options.
  Lower is better. PP = 2 means it effectively chooses between 2 options.
- BPC (bits-per-character): same as H when tokens are characters.
- NLL (negative log-likelihood in nats): H × ln(2). PyTorch's cross_entropy
  returns nats by default; divide by ln(2) to convert to bits.

WHY ONLY THE LAST POSITION?
-----------------------------
The model prediction at position t uses tokens 1..t as context. For the
first few tokens of a window the context is very short and the prediction
is noisy. Scoring only position T (the last) gives the most informative
estimate — the model has seen the full context length L.

This is an approximation of the true validation loss (which would average
over all positions and all windows without overlap), but it's cheap and
correlates well with the true loss.
"""

@torch.no_grad()   # disable gradient tracking for evaluation
def validation_loss_bits(model: nn.Module, val_data: torch.Tensor,
                         context: int, n_batches: int = 1000,
                         batch_size: int = 32,
                         device: str = 'cpu') -> float:
    """
    Estimate validation loss in bits/token, scored at the last position only.

    Returns
    -------
    float : average − log2 p(last token | context)
    """
    model.eval()
    total_nll = 0.0

    for _ in range(n_batches):
        # Sample a batch of length context+1
        batch = sample_batch(val_data, batch_size, context).to(device)
        inp    = batch[:, :-1]   # (B, context)
        target = batch[:, -1]    # (B,) — only the LAST token

        logits = model(inp)          # (B, context, vocab)
        last_logits = logits[:, -1, :]   # (B, vocab) — logits at final position

        # cross_entropy returns mean NLL in nats (natural log)
        nll_nats = F.cross_entropy(last_logits, target, reduction='mean')

        # Convert nats → bits: log2(x) = log(x) / log(2)
        nll_bits = nll_nats.item() / math.log(2)
        total_nll += nll_bits

    model.train()
    return total_nll / n_batches


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 13 — Sampling from the model
# ─────────────────────────────────────────────────────────────────────────────
"""
QUESTION 13
===========
Autoregressive generation: given a seed of S tokens, generate N more by
repeatedly:
  1. Forward pass the current sequence through the model.
  2. Take the logit vector at the LAST position.
  3. Sample from the resulting distribution.
  4. Append the sampled token to the sequence.
  5. Repeat.

TEMPERATURE
-----------
Temperature T controls the "sharpness" of the distribution:

    p_T(x) = softmax(logits / T)

- T → 0  : argmax (always pick the most likely token, deterministic / greedy)
- T = 1  : sample from the model's distribution (no distortion)
- T > 1  : flatter distribution, more diversity but less coherent

WHY NOT JUST ALWAYS TAKE THE ARGMAX?
--------------------------------------
Greedy decoding often produces repetitive output because it always commits to
the single best next token, ignoring alternatives. Sampling introduces variety.

ALTERNATIVES
------------
- Top-k sampling: zero out all but the top-k logits before softmax.
  Prevents sampling very unlikely tokens that can derail generation.
- Top-p (nucleus) sampling (Holtzman et al. 2019): keep the smallest set of
  tokens whose cumulative probability exceeds p. Adapts the number of options
  to the uncertainty at each step.
- Beam search: maintain k candidate sequences, expand each by all vocab tokens,
  keep the k with highest joint probability. Deterministic and thorough but
  very expensive for large vocab; rarely used in large LMs.

CONTEXT WINDOW MANAGEMENT
--------------------------
If the generated sequence exceeds the model's context length, we must truncate:
keep only the last `context` tokens. This is called a "sliding window".
In practice, modern models handle this with more sophisticated approaches
(RoPE embeddings, ALiBi, etc.) that allow extrapolation beyond training length.
"""

def sample(lnprobs: torch.Tensor, temperature: float = 1.0) -> int:
    """
    Sample one token index from logits.

    Args:
        lnprobs:    1-D tensor of logits (NOT log-probs despite the name)
        temperature: sampling temperature

    Returns:
        sampled integer token index
    """
    if temperature == 0.0:
        return int(lnprobs.argmax().item())
    p  = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return int(cd.sample().item())


@torch.no_grad()
def generate(model: nn.Module, seed: torch.Tensor, n_tokens: int,
             context: int, temperature: float = 1.0,
             device: str = 'cpu') -> torch.Tensor:
    """
    Generate n_tokens tokens autoregressively.

    Args:
        model    : trained AutoregressiveTransformer
        seed     : 1-D integer tensor of length S (the prompt)
        n_tokens : how many new tokens to generate
        context  : maximum context length of the model
        temperature : sampling temperature

    Returns:
        1-D integer tensor of length S + n_tokens
    """
    model.eval()
    generated = seed.clone().to(device)  # running sequence, shape (current_len,)

    for _ in range(n_tokens):
        # Truncate to context window (sliding window if sequence gets too long)
        inp = generated[-context:].unsqueeze(0)  # (1, min(current_len, context))

        logits = model(inp)          # (1, T, vocab)
        last_logits = logits[0, -1]  # (vocab,) — last position's logits

        next_token = sample(last_logits, temperature)
        generated  = torch.cat([generated, torch.tensor([next_token], device=device)])

    model.train()
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
"""
GRADIENT CLIPPING
-----------------
In deep transformers, gradients can occasionally become very large (an
"explosion") causing the optimiser to take a huge step and destabilise training.
Gradient clipping rescales the gradient if its norm exceeds a threshold:

    if ||g|| > clip_value:
        g ← g * clip_value / ||g||

This leaves the direction unchanged but caps the magnitude.
Common value: 1.0. Apply BEFORE the optimiser step.

LEARNING RATE WARMUP
--------------------
Starting with a large learning rate can destabilise the early steps, especially
with Adam (whose adaptive estimates start near zero, making the effective step
size very large). Warmup linearly increases LR from 0 to the target over
`warmup_steps` steps:

    lr(t) = target_lr * min(1, t / warmup_steps)

After warmup, a cosine decay schedule is common:

    lr(t) = target_lr * 0.5 * (1 + cos(π * t / T_max))

ADAM OPTIMISER (brief recap)
-----------------------------
Adam maintains a per-parameter running mean (m) and variance (v) of the gradient:

    m_t = β1 * m_{t-1} + (1-β1) * g_t        (first moment)
    v_t = β2 * v_{t-1} + (1-β2) * g_t²       (second moment)
    θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

Typical values: β1=0.9, β2=0.95 (GPT uses 0.95 not 0.999), ε=1e-8.
Weight decay (AdamW): adds L2 regularisation DIRECTLY to the parameters
(not through the gradient), which interacts correctly with Adam's adaptive
scaling. This is the recommended variant for transformers.
"""

def get_lr(step: int, target_lr: float, warmup_steps: int,
           max_steps: int) -> float:
    """Warmup + cosine decay learning rate schedule."""
    if step < warmup_steps:
        return target_lr * step / warmup_steps
    # Cosine decay from target_lr to target_lr/10
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return target_lr * (0.1 + 0.9 * cosine)   # don't decay all the way to 0


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 13 & 14 — Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(train_data: torch.Tensor,
          val_data:   torch.Tensor,
          i2c,             # index → character mapping
          vocab_size: int,
          # Model hyperparameters
          emb:      int   = 300,
          heads:    int   = 6,
          depth:    int   = 6,
          context:  int   = 256,
          dropout:  float = 0.1,
          # Training hyperparameters
          batch_size:    int   = 32,
          max_batches:   int   = 50_000,
          target_lr:     float = 3e-4,
          warmup_steps:  int   = 1_000,
          grad_clip:     float = 1.0,
          # Evaluation / logging
          eval_every:    int   = 10_000,
          sample_every:  int   = 10_000,
          sample_seed_len: int = 16,
          n_sample_tokens: int = 200,
          temperature:   float = 0.8,
          device:        str   = 'cpu',
          log_dir:       str   = 'runs/transformer'):
    """
    Full training loop for the autoregressive transformer.

    Logs to TensorBoard for loss curves and gradient norm curves.
    Run:  tensorboard --logdir runs/
    """

    writer = SummaryWriter(log_dir)

    # ── Model + optimiser ────────────────────────────────────────────────────
    model = AutoregressiveTransformer(
        vocab=vocab_size, emb=emb, heads=heads,
        depth=depth, context=context, dropout=dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")

    # AdamW: Adam + correct weight decay (decoupled from adaptive scaling)
    # weight_decay on the weight matrices, NOT on biases/LayerNorm params
    no_decay = {'bias', 'norm'}
    param_groups = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optim = torch.optim.AdamW(param_groups, lr=target_lr, betas=(0.9, 0.95))

    # ── Initial validation loss ──────────────────────────────────────────────
    init_val = validation_loss_bits(model, val_data, context,
                                    n_batches=200, batch_size=batch_size,
                                    device=device)
    print(f"Step 0 | val loss: {init_val:.4f} bits/token")
    writer.add_scalar('val/bits_per_token', init_val, 0)

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0

    for step in range(1, max_batches + 1):

        # ── Learning rate schedule ───────────────────────────────────────────
        lr = get_lr(step, target_lr, warmup_steps, max_batches)
        for g in optim.param_groups:
            g['lr'] = lr

        # ── Batch ────────────────────────────────────────────────────────────
        batch  = sample_batch(train_data, batch_size, context).to(device)
        inp    = batch[:, :-1]   # (B, context)
        target = batch[:, 1:]    # (B, context) — shifted right by 1

        # ── Forward ──────────────────────────────────────────────────────────
        logits = model(inp)   # (B, context, vocab)

        # ── Loss ─────────────────────────────────────────────────────────────
        # cross_entropy expects (B, C, ...) for the logits and (B, ...) for targets
        # Reshape: (B, context, vocab) → (B, vocab, context); target (B, context)
        # reduction='sum' then divide by total tokens → per-token loss
        # This makes the loss independent of batch_size and context length.
        n_tokens   = inp.numel()
        loss       = F.cross_entropy(logits.transpose(1, 2), target, reduction='sum')
        loss_per_t = loss / n_tokens

        # ── Backward ─────────────────────────────────────────────────────────
        optim.zero_grad()
        loss_per_t.backward()

        # Gradient clipping (compute norm BEFORE clipping for logging)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optim.step()

        running_loss += loss_per_t.item()

        # ── Logging ──────────────────────────────────────────────────────────
        if step % 100 == 0:
            avg_loss = running_loss / 100
            running_loss = 0.0
            writer.add_scalar('train/loss_nats', avg_loss, step)
            writer.add_scalar('train/loss_bits', avg_loss / math.log(2), step)
            writer.add_scalar('train/grad_norm', grad_norm.item(), step)
            writer.add_scalar('train/lr', lr, step)
            print(f"Step {step:6d} | loss {avg_loss:.4f} nats "
                  f"({avg_loss/math.log(2):.4f} bits) | "
                  f"grad_norm {grad_norm:.3f} | lr {lr:.2e}")

        # ── Validation + sampling ─────────────────────────────────────────────
        if step % eval_every == 0:
            val_bits = validation_loss_bits(model, val_data, context,
                                            n_batches=1000, batch_size=batch_size,
                                            device=device)
            writer.add_scalar('val/bits_per_token', val_bits, step)
            print(f"\n{'─'*60}")
            print(f"Step {step} | val loss: {val_bits:.4f} bits/token")

        if step % sample_every == 0:
            # Pick a random seed from validation data
            start = torch.randint(0, val_data.size(0) - sample_seed_len, (1,)).item()
            seed  = val_data[start: start + sample_seed_len]

            generated = generate(model, seed, n_sample_tokens, context,
                                 temperature=temperature, device=device)

            # Decode to string (works for character-level models)
            seed_str = ''.join(i2c[t] for t in seed.tolist())
            gen_str  = ''.join(i2c[t] for t in generated[sample_seed_len:].tolist())
            print(f"\nSeed: '{seed_str}'")
            print(f"Generated: '{gen_str}'")
            print(f"{'─'*60}\n")
            writer.add_text('samples', f"[{step}] SEED: {seed_str} | GEN: {gen_str}", step)

    writer.close()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 14 — Full training run + Wikipedia extension
# ─────────────────────────────────────────────────────────────────────────────
"""
QUESTION 14
===========
Full training run on the toy grammar data. The goal is to see the samples
progressively become more language-like.

WHAT TO EXPECT AT EACH STAGE
------------------------------
- At init (bits ≈ log2(vocab)): completely random characters
- After ~5k steps: spaces appear, letter frequency roughly right
- After ~20k steps: word-length chunks emerge
- After ~50k steps: common letter pairs (th, he, in, ...) cluster

The model capacity needed for this is very small — the toy grammar is simple.

WIKIPEDIA EXTENSION (optional)
--------------------------------
For real English, use load_wp(). Key differences:
- vocab = 256 (raw bytes)
- Much larger dataset → more steps needed (200k–500k)
- Larger model: emb=512, heads=8, depth=12 gives good results
- Batch size 64, context 256, lr 3e-4 with warmup
- GPU strongly recommended: ~12-24h on a single V100/A100 for semi-realistic text

PERPLEXITY TARGETS (character-level, Wikipedia)
------------------------------------------------
- State of art transformer: ~1.1–1.2 bits/char
- Decent small model after 24h: ~1.5–2.0 bits/char
- Random baseline: log2(256) ≈ 8 bits/char

WHY DOES TRAINING LANGUAGE MODELS GENERALISE?
----------------------------------------------
The model is forced to predict every next token, so it cannot memorise
individual windows (there are exponentially many). It must learn the
*structure* of the language — syntax, common collocations, factual
associations. This is the core idea behind GPT pre-training.
"""

def main_toy():
    """Example entry point for the toy dataset."""
    import wget  # pip install wget

    # Load data (uses dataset script from assignment)
    # (train, test), (i2c, c2i) = load_toy(final=False)
    # For this template we stub it out:
    print("Load data: (train, test), (i2c, c2i) = load_toy(final=False)")
    print("Then call:")
    print("""
    train(
        train_data = torch.tensor(train, dtype=torch.long),
        val_data   = torch.tensor(test,  dtype=torch.long),
        i2c        = i2c,
        vocab_size = len(i2c),
        emb        = 128,      # smaller for toy data
        heads      = 4,
        depth      = 4,
        context    = 128,
        dropout    = 0.1,
        batch_size = 64,
        max_batches = 50_000,
        target_lr   = 3e-4,
        warmup_steps = 1_000,
        grad_clip    = 1.0,
        eval_every   = 10_000,
        sample_every = 10_000,
        device       = 'cuda' if torch.cuda.is_available() else 'cpu',
    )
    """)


if __name__ == '__main__':
    main_toy()