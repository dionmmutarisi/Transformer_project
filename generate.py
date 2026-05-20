import torch
import torch.nn.functional as F
import torch.distributions as dist
import sys

from transformer_rope import AutoregressiveTransformer

CHECKPOINT = "/var/scratch/dmu224/experiments/checkpoint_step200000.pt"
VOCAB      = 256
EMB, HEADS, DEPTH, CONTEXT = 512, 8, 6, 256
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoregressiveTransformer(VOCAB, EMB, HEADS, DEPTH, CONTEXT).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
print(f"Loaded checkpoint. Device: {DEVICE}")

seed_text = input("\nEnter your seed text: ")
temperature = float(input("Temperature (0.0=greedy, 0.8=default, 1.0=creative): ") or "0.8")
n_tokens    = int(input("How many characters to generate? (default 200): ") or "200")

seed_ids = torch.tensor([ord(c) for c in seed_text], dtype=torch.long).to(DEVICE)
generated = seed_ids.clone()

with torch.no_grad():
    for _ in range(n_tokens):
        inp  = generated[-CONTEXT:].unsqueeze(0)
        logits = model(inp)[0, -1]
        if temperature == 0.0:
            next_tok = logits.argmax().item()
        else:
            p = F.softmax(logits / temperature, dim=0)
            next_tok = dist.Categorical(p).sample().item()
        generated = torch.cat([generated, torch.tensor([next_tok], device=DEVICE)])

output = "".join(chr(t) for t in generated.tolist())
print(f"\n{'─'*60}")
print(output)
print(f"{'─'*60}")
