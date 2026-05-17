import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import wandb

from raw import load_imdb_synth, load_xor



# DATASET + COLLATE


class TextDataset(Dataset):
    """
    Minimal Dataset wrapper.

    Each item is:
        (token_sequence, class_label)

    We keep examples in raw variable-length form here.
    Padding is deferred to the collate function because padding is a
    batch-level concern, not a single-example concern.
    """
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def pad_and_tensorize(batch, pad_token, max_length=None):
    """
    Pad a batch of token sequences to equal length and return a LongTensor.

    """
    if max_length is not None:
        batch = [seq[:max_length] for seq in batch]

    max_len = max(len(seq) for seq in batch)
    padded = [seq + [pad_token] * (max_len - len(seq)) for seq in batch]
    return torch.tensor(padded, dtype=torch.long)


def labels_to_tensor(labels):
    return torch.tensor(labels, dtype=torch.long)


def make_collate_fn(pad_token, max_length=None):
    """
    Returns a collate_fn that DataLoader can use to assemble variable-length batches.
    """
    def collate_fn(batch):
        x_batch, y_batch = zip(*batch)
        x_batch = pad_and_tensorize(list(x_batch), pad_token, max_length=max_length)
        y_batch = labels_to_tensor(list(y_batch))
        return x_batch, y_batch
    return collate_fn



# SIMPLE SELF-ATTENTION MODEL (Q4/Q5 baseline for comparison)


class SimpleSelfAttentionClassifier(nn.Module):
    """
    Simple self-attention classifier from earlier questions.

    This uses:
    - token embeddings directly as queries/keys/values
    - no learned Q/K/V projections
    - no scaling by sqrt(d)
    - no multi-head decomposition

    We keep it here for direct comparison against full self-attention.
    """
    def __init__(self, vocab_size, num_classes, emb_dim=300, pool_type="select"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.pool_type = pool_type
        self.last_attention = None

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x)                          # (B, T, E)

        # Simple attention score matrix
        scores = torch.bmm(emb, emb.transpose(1, 2))    # (B, T, T)
        weights = F.softmax(scores, dim=-1)             # (B, T, T)
        self.last_attention = weights.detach()

        # Contextualized sequence
        attended = torch.bmm(weights, emb)              # (B, T, E)

        if self.pool_type == "select":
            pooled = attended[:, 0, :]                  # (B, E)
        elif self.pool_type == "max":
            pooled = attended.max(dim=1).values
        elif self.pool_type == "mean":
            pooled = attended.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        logits = self.classifier(pooled)                # (B, C)
        return logits



# 3. FULL SELF-ATTENTION MODULE (Q6)


class FullSelfAttention(nn.Module):
    """
    Full self-attention as requested in.

    Upgrades over simple self-attention:
    1. Separate learned linear projections for Q, K, V
    2. Scale dot products by sqrt(per-head-dim)
    3. Multi-head parallel attention, now I am using 6 heads
    4. Final output projection

    Input:
        x of shape (B, T, E)

    Output:
        attended sequence of shape (B, T, E)
    """
    def __init__(self, emb_dim=300, num_heads=6):
        super().__init__()

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Learned linear maps to create separate Q, K, V spaces
        self.to_queries = nn.Linear(emb_dim, emb_dim)
        self.to_keys = nn.Linear(emb_dim, emb_dim)
        self.to_values = nn.Linear(emb_dim, emb_dim)

        # Final projection after concatenating heads
        self.unify_heads = nn.Linear(emb_dim, emb_dim)

        self.last_attention = None

    def forward(self, x):
        """
        x: (B, T, E)
        returns: (B, T, E)
        """
        B, T, E = x.shape
        H = self.num_heads
        D = self.head_dim

        # learned Q/K/V projections
        q = self.to_queries(x)
        k = self.to_keys(x)
        v = self.to_values(x)


        # From (B, T, E) -> (B, T, H, D)
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)


# MOVE ATTENTION TO IT'S OWN FUNCTION, 

        # (B, T, H, D) -> (B, H, T, D), YOU CAN ALSO USE .permute(0, 2, 1, 3) HERE, BUT .transpose(1, 2) IS MORE EFFICIENT FOR SWAPPING TWO DIMENSIONS BECAUSE IT DOES NOT REQUIRE SPECIFYING ALL DIMENSIONS.
        #  .permute() IS MORE GENERAL AND CAN REORDER ANY NUMBER OF DIMENSIONS IN ANY WAY, 
        # BUT IT MIGHT INVOLVE AN UNNECESSARY COPY IF THE NEW ORDER IS NOT ALREADY CONTIGUOUS IN MEMORY. 
        # SINCE WE ARE ONLY SWAPPING TWO ADJACENT DIMENSIONS, .transpose() IS MORE EFFICIENT.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scaled dot-product attention scores
        # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D) #CHECK USING X

        # Softmax over attended-to positions
        weights = F.softmax(scores, dim=-1)
        self.last_attention = weights.detach()

        #  weighted sum of values
        # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        z = torch.matmul(weights, v)

        #  reorder back to (B, T, H, D)
        z = z.transpose(1, 2).contiguous()

        #  concatenate heads        # (B, T, H, D) -> (B, T, E)
        z = z.view(B, T, E) 
        # YOU CAN ALSO USE .reshape() HERE, BUT .view() IS FASTER IF THE TENSOR IS CONTIGUOUS IN MEMORY,
        # WHICH IT IS BECAUSE OF THE .contiguous() CALL ABOVE. .reshape() WOULD WORK TOO, 
        # BUT IT MIGHT INVOLVE AN UNNECESSARY COPY IF THE TENSOR IS NOT ALREADY IN THE RIGHT SHAPE. 
        # SINCE WE KNOW IT IS, .view() IS MORE EFFICIENT.

        # final linear operation
        out = self.unify_heads(z)   # (B, T, E)
        return out


# FULL SELF-ATTENTION CLASSIFIER 

class FullSelfAttentionClassifier(nn.Module):
    """
    Classification model using full self-attention.

    Structure:
        token ids -> embedding -> full self-attention -> pooling -> linear classifier
    """
    def __init__(self, vocab_size, num_classes, emb_dim=300, num_heads=6, pool_type="select"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = FullSelfAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.pool_type = pool_type

    def forward(self, x):
        emb = self.embedding(x)            # (B, T, E)
        attended = self.attention(emb)     # (B, T, E)

        if self.pool_type == "select":
            pooled = attended[:, 0, :]     # (B, E)

        elif self.pool_type == "max":
            pooled = attended.max(dim=1).values
        elif self.pool_type == "mean":
            pooled = attended.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        logits = self.classifier(pooled)
        return logits

    @property
    def last_attention(self):
        return self.attention.last_attention


# TRAINING / EVALUATION

def get_grad_norm(model):
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.data.norm(2).item()
            total_norm_sq += n ** 2
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

    attn_ent = None
    if model.last_attention is not None:
        attn_ent = attention_entropy(model.last_attention)

    return avg_loss, avg_acc, grad_norm, attn_ent


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

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



#  DATASET HELPER

def get_dataset(dataset_name):
    if dataset_name == "imdb_synth":
        return load_imdb_synth()
    elif dataset_name == "xor":
        return load_xor()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# EXPERIMENT RUNNER

def run_experiment(
    dataset_name,
    model_type,
    emb_dim=300,
    num_heads=8,
    pool_type="select",
    batch_size=32,
    epochs=5,
    learning_rate=1e-3,
    max_length=256,
    project="transformer-assignment-q6-q7",
    entity="dionmutarisi-vrije-universiteit-amsterdam"
):
    """
    Runs one controlled experiment.

    model_type:
        - "simple"
        - "full"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (train_data, train_labels), (test_data, test_labels), (i2w, w2i), num_classes = get_dataset(dataset_name)
    pad_token = w2i[".pad"]

    train_dataset = TextDataset(train_data, train_labels)
    test_dataset = TextDataset(test_data, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(pad_token, max_length=max_length)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(pad_token, max_length=max_length)
    )

    if model_type == "simple":
        model = SimpleSelfAttentionClassifier(
            vocab_size=len(i2w),
            num_classes=num_classes,
            emb_dim=emb_dim,
            pool_type=pool_type
        ).to(device)
    elif model_type == "full":
        model = FullSelfAttentionClassifier(
            vocab_size=len(i2w),
            num_classes=num_classes,
            emb_dim=emb_dim,
            num_heads=num_heads,
            pool_type=pool_type
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    run = wandb.init(
        entity=entity,
        project=project,
        reinit=True,
        config={
            "dataset": dataset_name,
            "model_type": model_type,
            "pool_type": pool_type,
            "embedding_dim": emb_dim,
            "num_heads": num_heads if model_type == "full" else 1,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }
    )

    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc, grad_norm, attn_ent = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        val_loss, val_acc = evaluate_model(
            model, test_loader, device
        )

        best_val_acc = max(best_val_acc, val_acc)

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "grad_norm": grad_norm,
        }

        if attn_ent is not None:
            log_dict["attention_entropy"] = attn_ent

        wandb.log(log_dict)

        print(
            f"[{dataset_name} | {model_type}] "
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    wandb.finish()
    return best_val_acc






if __name__ == "__main__":


    common_kwargs = {
        "emb_dim": 300,
        "num_heads": 6,
        "pool_type": "select",
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 1e-3,
        "max_length": 256,
    }

    # IMDb-synth comparison
    synth_simple = run_experiment(
        dataset_name="imdb_synth",
        model_type="simple",
        **common_kwargs
    )
    synth_full = run_experiment(
        dataset_name="imdb_synth",
        model_type="full",
        **common_kwargs
    )

    # XOR comparison
    xor_simple = run_experiment(
        dataset_name="xor",
        model_type="simple",
        **common_kwargs
    )
    xor_full = run_experiment(
        dataset_name="xor",
        model_type="full",
        **common_kwargs
    )

    print("\nFinal comparison summary")
    print(f"IMDb-synth | simple: {synth_simple:.4f} | full: {synth_full:.4f}")
    print(f"XOR        | simple: {xor_simple:.4f} | full: {xor_full:.4f}")