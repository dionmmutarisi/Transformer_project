import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import optuna
import wandb

from raw import load_imdb_synth, load_xor



# 1. DATASET + COLLATE LOGIC

# Why this section exists:
# PyTorch separates:
# - Dataset: how to access one example
# - DataLoader: how to batch examples
# - collate_fn: how to combine variable-length examples into one batch
#
# For text, examples have different lengths, so the default collate function
# is not sufficient. We need our own padding-aware collate_fn.


class TextDataset(Dataset):
    """
    Minimal dataset wrapper.

    Each item returned is a tuple:
        (token_sequence, class_label)

    We do NOT pad here because padding is a batch-level operation.
    Padding inside Dataset would force all examples to the same global length,
    which wastes memory and compute.
    """
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def pad_and_tensorize(batch, pad_token):
    """
    Convert a list of variable-length token lists into a padded LongTensor.

    Input:
        batch = [[1, 5, 9], [2, 8], [4]]

    Output:
        tensor([[1, 5, 9],
                [2, 8, pad],
                [4, pad, pad]])

    Why torch.long:
    Embedding layers expect integer token IDs, not floats.
    """
    max_len = max(len(seq) for seq in batch)
    padded = [seq + [pad_token] * (max_len - len(seq)) for seq in batch]
    return torch.tensor(padded, dtype=torch.long)


def labels_to_tensor(labels):
    """
    Convert class labels to a LongTensor.

    CrossEntropyLoss expects integer class indices, not one-hot vectors.
    """
    return torch.tensor(labels, dtype=torch.long)


def make_collate_fn(pad_token):
    """
    Returns a collate function that DataLoader can use.

    Why a nested function:
    DataLoader expects collate_fn(batch), but the function also needs access
    to pad_token. This factory pattern lets us "bind" pad_token once.
    """
    def collate_fn(batch):
        # batch is a list like:
        # [(x1, y1), (x2, y2), ..., (xB, yB)]
        x_batch, y_batch = zip(*batch)

        x_batch = pad_and_tensorize(list(x_batch), pad_token)
        y_batch = labels_to_tensor(list(y_batch))

        return x_batch, y_batch

    return collate_fn



# 2. MODEL: SIMPLE SELF-ATTENTION + SELECT POOL

# Question 5 is specifically about the simple self-attention model
# and says to use only the select pool for Synth and XOR tuning. :contentReference[oaicite:1]{index=1}


class SimpleSelfAttentionClassifier(nn.Module):
    """
    Simple self-attention classifier.

    Architecture:
        token ids -> embedding -> simple self-attention -> select pool -> linear classifier

    Important:
    This is NOT full transformer attention yet.
    There are no learned Q/K/V projections here.
    There is no scaling by sqrt(d).
    There are no heads, no residuals, no FFN, no layer norm.
    Those come later in the assignment.
    """
    def __init__(self, vocab_size, num_classes, emb_dim=300, pool_type="select"):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.pool_type = pool_type

        # Useful for diagnostics; not required by the assignment.
        self.last_attention = None

    def forward(self, x):
        """
        x: LongTensor of shape (batch, time)
        returns: logits of shape (batch, num_classes)
        """

        # Step 1: token embeddings
        # Shape: (B, T, E)
        emb = self.embedding(x)

        # Step 2: simple self-attention scores
        # Compare each token embedding against every other token embedding.
        #
        # (B, T, E) @ (B, E, T) -> (B, T, T)
        scores = torch.bmm(emb, emb.transpose(1, 2))

        # Step 3: normalize attention weights across attended-to positions
        # dim=-1 means: for each token, produce a probability distribution over tokens.
        weights = F.softmax(scores, dim=-1)

        # Save for analysis/debugging
        self.last_attention = weights.detach()

        # Step 4: mix token embeddings using the attention weights
        # (B, T, T) @ (B, T, E) -> (B, T, E)
        attended = torch.bmm(weights, emb)

        # Step 5: select pool
        # Q5 says to use only select pooling for Synth and XOR tuning.
        if self.pool_type == "select":
            pooled = attended[:, 0, :]   # first token representation only
        else:
            raise ValueError("For Question 5 tuning, pool_type must be 'select'.")

        # Step 6: classification logits
        logits = self.classifier(pooled)

        return logits



# 3. TRAINING / EVALUATION UTILITIES


def get_grad_norm(model):
    """
    Compute the global L2 gradient norm.

    Why this is useful:
    - Very large norms can indicate unstable training.
    - Very small norms can indicate stalled learning.
    - For research work, this becomes a basic optimization diagnostic.
    """
    total_norm_sq = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm_sq += param_norm ** 2

    return total_norm_sq ** 0.5


def attention_entropy(attn):
    """
    Measure how diffuse attention is.

    High entropy  -> attention spread out across many tokens
    Low entropy   -> attention concentrated on fewer tokens

    Not required by the assignment, but useful when diagnosing whether
    the attention mechanism is learning anything structured.
    """
    return -(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean().item()


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Train for one epoch.

    Returns:
        avg_loss, avg_acc, grad_norm, attn_entropy
    """
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
    """
    Evaluate the model without gradient computation.
    """
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



# 4. DATASET LOADING HELPER

# Why this helper exists:
# We want the Optuna objective to work for both IMDb-synth and XOR
# without duplicating code.


def get_dataset(dataset_name):
    """
    Load one dataset by name.

    Supported:
        - "imdb_synth"
        - "xor"
    """
    if dataset_name == "imdb_synth":
        return load_imdb_synth()
    elif dataset_name == "xor":
        return load_xor()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")



# 5. OPTUNA OBJECTIVE

# Each Optuna trial = one full experiment with one set of hyperparameters.
# Each trial also gets its own W&B run.
#
# This is important experimentally:
# one run should correspond to one hypothesis/configuration,
# otherwise the logs become mixed and meaningless.


def make_objective(dataset_name, device):
    def objective(trial):
        # ----------------------------------------------------
        # Hyperparameter search space
        # ----------------------------------------------------
        # Keep the search space focused on variables that are likely to matter.
        # The assignment specifically notes sensitivity to learning rate and
        # batch size for these problems. :contentReference[oaicite:2]{index=2}

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        emb_dim = trial.suggest_categorical("emb_dim", [64, 128, 300])
        epochs = trial.suggest_int("epochs", 10, 100)

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------
        (train_data, train_labels), (test_data, test_labels), (i2w, w2i), num_classes = get_dataset(dataset_name)
        pad_token = w2i[".pad"]
        vocab_size = len(i2w)

        # ----------------------------------------------------
        # Build Dataset / DataLoader
        # ----------------------------------------------------
        train_dataset = TextDataset(train_data, train_labels)
        test_dataset = TextDataset(test_data, test_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,   # shuffle for SGD training
            collate_fn=make_collate_fn(pad_token)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # keep evaluation deterministic
            collate_fn=make_collate_fn(pad_token)
        )

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------
        model = SimpleSelfAttentionClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            emb_dim=emb_dim,
            pool_type="select"
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # ----------------------------------------------------
        # Start a separate W&B run for this Optuna trial
        # ----------------------------------------------------
        run = wandb.init(
            entity="dionmutarisi-vrije-universiteit-amsterdam",
            project="transformer-assignment-q5",
            reinit=True,
            config={
                "dataset": dataset_name,
                "model": "simple_self_attention",
                "pool_type": "select",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "emb_dim": emb_dim,
                "epochs": epochs,
                "optimizer": "Adam",
                "trial_number": trial.number,
                "num_parameters": sum(p.numel() for p in model.parameters()),
            }
        )

        best_val_acc = 0.0

        # ----------------------------------------------------
        # Training loop
        # ----------------------------------------------------
        for epoch in range(epochs):
            train_loss, train_acc, grad_norm, attn_ent = train_epoch(
                model, train_loader, optimizer, loss_fn, device
            )

            val_loss, val_acc = evaluate_model(
                model, test_loader, device
            )

            # Log to W&B
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

            # Report intermediate value to Optuna
            # This lets Optuna prune clearly bad trials early.
            trial.report(val_acc, step=epoch)

            if trial.should_prune():
                wandb.finish()
                raise optuna.TrialPruned()

            best_val_acc = max(best_val_acc, val_acc)

        wandb.finish()
        return best_val_acc

    return objective



# 6. STUDY RUNNER

# This runs two separate Optuna studies:
# one for IMDb-synth and one for XOR.
#
# That is cleaner than mixing them into one study because the datasets
# have different difficulty profiles and different attainable ceilings.


def run_study(dataset_name, n_trials, device):
    """
    Run one Optuna study for a single dataset.
    """
    print(f"\nStarting Optuna study for dataset: {dataset_name}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        make_objective(dataset_name, device),
        n_trials=n_trials
    )

    print(f"\nBest trial for {dataset_name}:")
    print("  Best validation accuracy:", study.best_value)
    print("  Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study



# 7. MAIN


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # You can change the number of trials depending on your compute budget.
    # Start small while debugging, then increase once the pipeline is stable.
    n_trials = 20

    # Run separate studies for the two Q5 tuning datasets
    synth_study = run_study("imdb_synth", n_trials=n_trials, device=device)
    xor_study = run_study("xor", n_trials=n_trials, device=device)

    print("\nAll studies finished.")