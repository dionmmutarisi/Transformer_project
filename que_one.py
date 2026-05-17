import torch
import torch.nn as nn
from raw import load_imdb_synth, load_imdb, load_xor

# ----------------------------
# DATA
# ----------------------------
(train_data, train_labels), (test_data, test_labels), (i2w, w2i), num_classes = load_xor()

# ----------------------------
# QUESTION 1: BATCHING + PADDING
# ----------------------------
def pad_and_tensorize(batch, pad_token):
    """
    Convert a list of variable-length token sequences into a rectangular LongTensor.

    Why this exists:
    PyTorch tensors require every row to have the same length, but NLP sequences
    naturally have variable lengths. Padding is how we make them batchable.
    """
    max_len = max(len(seq) for seq in batch)
    padded = [seq + [pad_token] * (max_len - len(seq)) for seq in batch]
    return torch.tensor(padded, dtype=torch.long)

def labels_to_tensor(labels):
    """
    Class labels for CrossEntropyLoss must be integer class indices (torch.long),
    not one-hot vectors.
    """
    return torch.tensor(labels, dtype=torch.long)

# ----------------------------
# QUESTION 2: BASELINE MODEL
# ----------------------------
class PooledTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, pad_idx, emb_dim=300,  pool_type="mean"):

        super().__init__()

        # padding_idx makes the pad embedding a fixed zero vector.
        # This is important because padding should not carry semantic information.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx
        )

        self.linear = nn.Linear(emb_dim, num_classes)
        self.pad_idx = pad_idx
        self.pool_type = pool_type

    def forward(self, x):
        """
        x: LongTensor of shape (batch, time)
        returns: logits of shape (batch, num_classes)
        """
        emb = self.embedding(x)  # (batch, time, emb)

        if self.pool_type == "mean":
            # Masked mean pooling: only average real tokens
            mask = (x != self.pad_idx).unsqueeze(-1).float()   # (batch, time, 1)
            emb_masked = emb * mask
            lengths = mask.sum(dim=1).clamp(min=1.0)           # (batch, 1)
            pooled = emb_masked.sum(dim=1) / lengths           # (batch, emb)

        elif self.pool_type == "max":
            # Masked max pooling: padded positions should not win the max
            mask = (x != self.pad_idx).unsqueeze(-1)           # (batch, time, 1)
            emb_masked = emb.masked_fill(~mask, float("-inf"))
            pooled = emb_masked.max(dim=1).values              # (batch, emb)

        elif self.pool_type == "select":
            # First token only
            pooled = emb[:, 0, :]                              # (batch, emb)

        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        logits = self.linear(pooled)
        return logits

# ----------------------------
# TRAINING / EVALUATION HELPERS
# ----------------------------
def train_epoch(model, x_data, y_data, batch_size, pad_token, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Shuffle each epoch so SGD does not always see the same order
    indices = torch.randperm(len(x_data)).tolist()

    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start:start + batch_size]
        x_batch = [x_data[i] for i in batch_ids]
        y_batch = [y_data[i] for i in batch_ids]

        x_batch = pad_and_tensorize(x_batch, pad_token).to(device)
        y_batch = labels_to_tensor(y_batch).to(device)

        optimizer.zero_grad()

        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        loss.backward()
        optimizer.step()

        # Multiply by batch size so final average is per sample, not per batch
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += y_batch.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def evaluate_model(model, x_data, y_data, batch_size, pad_token, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for start in range(0, len(x_data), batch_size):
            x_batch = x_data[start:start + batch_size]
            y_batch = y_data[start:start + batch_size]

            x_batch = pad_and_tensorize(x_batch, pad_token).to(device)
            y_batch = labels_to_tensor(y_batch).to(device)

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)

            total_loss += loss.item() * y_batch.size(0)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += y_batch.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# ----------------------------
# MAIN TRAINING SCRIPT
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pad_token = w2i[".pad"]
vocab_size = len(i2w)
batch_size = 32
epochs = 5

pool_types = ["mean", "max", "select"]
results = {}

for pool in pool_types:
    print(f"\nTraining model with pooling = {pool}")

    model = PooledTextClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        pad_idx=pad_token,
        emb_dim=300,
        pool_type=pool
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_data, train_labels, batch_size, pad_token, optimizer, loss_fn, device
        )

        val_loss, val_acc = evaluate_model(
            model, test_data, test_labels, batch_size, pad_token, device
        )

        print(
            f"[{pool}] Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    results[pool] = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

print("\nFinal comparison:")
for pool, metrics in results.items():
    print(
        f"{pool:>6} | "
        f"Train Acc: {metrics['train_acc']:.4f} | "
        f"Val Acc: {metrics['val_acc']:.4f}"
    )