"""Fast training path: fine-tune only the classifier head on precomputed
wav2vec2 embeddings (e.g. eminkorkut/deepfakevoice-wac2vec-4datasets on
Kaggle) instead of raw audio. Minutes instead of hours, since there's no SSL
forward pass -- just an MLP over 768-dim vectors.

Handles two things a naive run misses on this dataset: it's ~89% spoof /
11% bonafide (so raw accuracy is a misleading metric -- always report
per-class recall), and raw wav2vec2 embeddings benefit from standardization
before an MLP head (baked into the model as buffers via src.model.Normalize).

Usage (inside the Kaggle notebook, after attaching the wac2vec dataset):
    python -m src.train_embeddings \
        --parquet /kaggle/input/datasets/eminkorkut/deepfakevoice-wac2vec-4datasets/ASVspoof2019_train_wav2vec.parquet \
        --epochs 30
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

MULTI_PARQUET_HELP = (
    "one or more training embedding parquets (space-separated). Mixing in "
    "In-the-wild_train_wav2vec.parquet alongside ASVspoof2019_train_wav2vec.parquet "
    "matters: ASVspoof's bonafide clips are all clean studio recordings, so a "
    "model trained on ASVspoof alone learns 'genuine = studio-clean' and "
    "flags real-world genuine speech (like a live demo mic) as suspicious."
)

from config import (
    LEARNING_RATE, CHECKPOINT_PATH, ARTIFACTS_DIR, SSL_MODEL_NAME, EMBEDDING_DIM,
)
from src.device import get_device
from src.dataset import load_embedding_parquet
from src.model import EmbeddingClassifier


def per_class_recall(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    recalls = {}
    for cls, name in ((0, "bonafide"), (1, "spoof")):
        mask = labels == cls
        recalls[name] = (preds[mask] == cls).float().mean().item() if mask.any() else float("nan")
    return recalls


def run_epoch(model, loader, device, criterion, optimizer=None, collect=False):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_labels = [], []

    for embs, labels in loader:
        embs, labels = embs.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(embs)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_n += labels.size(0)
        if collect:
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    metrics = {"loss": total_loss / max(total_n, 1), "acc": total_correct / max(total_n, 1)}
    if collect:
        metrics["recall"] = per_class_recall(torch.cat(all_preds), torch.cat(all_labels))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, nargs="+", help=MULTI_PARQUET_HELP)
    parser.add_argument("--val-split", type=float, default=0.1, help="fraction held out for validation")
    parser.add_argument("--in-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    device = get_device()
    print(f"Training on device: {device}")

    X_parts, y_parts = [], []
    for path in args.parquet:
        Xi, yi = load_embedding_parquet(path)
        print(f"  {path}: {Xi.shape[0]} rows, bonafide={int((yi == 0).sum())}, spoof={int((yi == 1).sum())}")
        X_parts.append(Xi)
        y_parts.append(yi)
    X, y = torch.cat(X_parts), torch.cat(y_parts)
    n_bonafide, n_spoof = int((y == 0).sum()), int((y == 1).sum())
    print(f"Loaded {X.shape[0]} rows total, dim={X.shape[1]}, bonafide={n_bonafide}, spoof={n_spoof}")

    n = X.shape[0]
    perm = torch.randperm(n)
    n_val = int(n * args.val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Standardize using train-split stats only (no val leakage); baked into
    # the model itself so inference never needs a separate stats file.
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp_min(1e-6)

    # Inverse-frequency class weights: this dataset is ~89% spoof, so an
    # unweighted loss lets the model coast on the majority class and barely
    # learn to recognise genuine speech -- exactly the case the demo needs.
    train_bonafide, train_spoof = int((y_train == 0).sum()), int((y_train == 1).sum())
    total = train_bonafide + train_spoof
    class_weights = torch.tensor(
        [total / (2 * max(train_bonafide, 1)), total / (2 * max(train_spoof, 1))],
        dtype=torch.float32, device=device,
    )
    print(f"Class weights [bonafide, spoof] = {class_weights.tolist()}")

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = EmbeddingClassifier(in_dim=args.in_dim).to(device)
    model.head[0].set_stats(mean.to(device), std.to(device))
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "head.0" not in n],  # skip the fixed Normalize buffers
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_dl, device, criterion, optimizer)
        last = epoch == args.epochs
        val_m = run_epoch(model, val_dl, device, criterion, collect=last or epoch % 5 == 0)
        msg = (f"epoch {epoch}/{args.epochs} train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f} "
               f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.4f}")
        if "recall" in val_m:
            msg += f" val_recall(bonafide)={val_m['recall']['bonafide']:.4f} val_recall(spoof)={val_m['recall']['spoof']:.4f}"
        print(msg)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    with open(os.path.join(ARTIFACTS_DIR, "ssl_name.txt"), "w") as f:
        f.write(SSL_MODEL_NAME)
    print(f"Saved checkpoint to {CHECKPOINT_PATH} (ssl_name={SSL_MODEL_NAME}, "
          f"normalization baked into head.0 buffers, "
          f"load into VoiceGuardNet with strict=False for inference)")


if __name__ == "__main__":
    main()
