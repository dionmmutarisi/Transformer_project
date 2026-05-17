import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

from raw import load_imdb, load_imdb_synth, load_xor 


# ----------------------------
# DATA
# ----------------------------
(train_data, train_labels), (test_data, test_labels), (i2w, w2i), num_classes = load_xor()


# ----------------------------
# QUESTION 1: PADDING
# ----------------------------
def pad_and_tensorize(batch, pad_token):
    """
    Convert a list of variable-length token sequences into a padded LongTensor.

    This is still the same core padding logic from Question 1.
    The difference now is that it will be called inside DataLoader's collate_fn.
    """
    max_len = max(len(seq) for seq in batch)
    padded = [seq + [pad_token] * (max_len - len(seq)) for seq in batch]
    return torch.tensor(padded, dtype=torch.long)


def labels_to_tensor(labels):
    return torch.tensor(labels, dtype=torch.long)


# ----------------------------
# DATASET
# ----------------------------
class TextDataset(Dataset):
    """
    Minimal PyTorch Dataset.

    Why this exists:
    - __len__ tells DataLoader how many examples exist
    - __getitem__ tells DataLoader how to fetch one example

    The dataset should return *raw examples*.
    Padding should NOT happen here, because padding depends on the batch.
    """
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# ----------------------------
# COLLATE FUNCTION
# ----------------------------
def make_collate_fn(pad_token):
    """
    Returns a collate function that DataLoader will use to combine
    individual samples into one padded batch.

    Why use a factory function?
    Because collate_fn must know the pad token, and this lets us
    "capture" that value cleanly.
    """
    def collate_fn(batch):
        # batch is a list of tuples: [(x1, y1), (x2, y2), ...]
        x_batch, y_batch = zip(*batch)

        x_batch = pad_and_tensorize(list(x_batch), pad_token)
        y_batch = labels_to_tensor(list(y_batch))

        return x_batch, y_batch

    return collate_fn


# ----------------------------
# QUESTION 4: SIMPLE SELF-ATTENTION MODEL


class SimpleSelfAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=300, pool_type="select"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.pool_type = pool_type
        self.last_attention = None

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x)                          # (B, T, E)
        scores = torch.bmm(emb, emb.transpose(1, 2))    # (B, T, T)
        weights = F.softmax(scores, dim=-1)             # (B, T, T)
        self.last_attention = weights.detach()

        attended = torch.bmm(weights, emb)              # (B, T, E)

        if self.pool_type == "select":
            # Q5: use only the first token representation
            pooled = attended[:, 0, :]                  # (B, E)
        elif self.pool_type == "max":
            pooled = attended.max(dim=1).values
        elif self.pool_type == "mean":
            pooled = attended.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        logits = self.classifier(pooled)                # (B, C)
        return logits

# ----------------------------
# TRAINING / EVALUATION HELPERS
# ----------------------------
def get_grad_norm(model):
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm_sq += param_norm ** 2
    return total_norm_sq ** 0.5


def attention_entropy(attn):
    return -(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean().item()


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += y_batch.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    grad_norm = get_grad_norm(model)
    attn_ent = attention_entropy(model.last_attention) if model.last_attention is not None else None

    return avg_loss, avg_acc, grad_norm, attn_ent


def evaluate_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

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
learning_rate = 1e-3

# Build datasets
train_dataset = TextDataset(train_data, train_labels)
test_dataset = TextDataset(test_data, test_labels)

# Build dataloaders
# Training loader: shuffle=True so the model does not always see the data in the same order
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=make_collate_fn(pad_token)
)

# Evaluation loader: shuffle=False because order should not matter and reproducibility is cleaner
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=make_collate_fn(pad_token)
)

model = SimpleSelfAttentionClassifier(
    vocab_size=vocab_size,
    num_classes=num_classes,
    emb_dim=300
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

run = wandb.init(
    entity="dionmutarisi-vrije-universiteit-amsterdam",
    project="transformer-assignment",
    config={
        "model": "simple_self_attention",
        "dataset": "XOR",
        "embedding_dim": 300,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "shuffle_train": True,
        "shuffle_eval": False,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
)

for epoch in range(epochs):
    train_loss, train_acc, grad_norm, attn_ent = train_epoch(
        model, train_loader, optimizer, loss_fn, device
    )

    val_loss, val_acc = evaluate_model(
        model, test_loader, device
    )

    log_dict = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "grad_norm": grad_norm,
    }

    if attn_ent is not None:
        log_dict["attention_entropy"] = attn_ent

    wandb.log(log_dict)

    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Grad Norm: {grad_norm:.4f}"
    )

wandb.finish()