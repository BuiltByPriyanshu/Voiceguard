"""Fast training path: fine-tune only the classifier head on precomputed
wav2vec2 embeddings (e.g. eminkorkut/deepfakevoice-wac2vec-4datasets on
Kaggle) instead of raw audio. Minutes instead of hours, since there's no SSL
forward pass -- just an MLP over 768-dim vectors.

Usage (inside the Kaggle notebook, after attaching the wac2vec dataset):
    python -m src.train_embeddings \
        --parquet /kaggle/input/datasets/eminkorkut/deepfakevoice-wac2vec-4datasets/ASVspoof2019_train_wav2vec.parquet \
        --epochs 10
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import (
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, CHECKPOINT_PATH, ARTIFACTS_DIR,
    SSL_MODEL_NAME, EMBEDDING_DIM,
)
from src.device import get_device
from src.dataset import load_embedding_parquet
from src.model import EmbeddingClassifier


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for embs, labels in loader:
        embs, labels = embs.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(embs)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_n += labels.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="training embeddings parquet")
    parser.add_argument("--val-split", type=float, default=0.1, help="fraction held out for validation")
    parser.add_argument("--in-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS * 3)  # embeddings train fast, afford more epochs
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    device = get_device()
    print(f"Training on device: {device}")

    X, y = load_embedding_parquet(args.parquet)
    print(f"Loaded {X.shape[0]} rows, dim={X.shape[1]}, "
          f"bonafide={int((y == 0).sum())}, spoof={int((y == 1).sum())}")

    full_ds = TensorDataset(X, y)
    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = EmbeddingClassifier(in_dim=args.in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_dl, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_dl, device)
        print(f"epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    with open(os.path.join(ARTIFACTS_DIR, "ssl_name.txt"), "w") as f:
        f.write(SSL_MODEL_NAME)
    print(f"Saved checkpoint to {CHECKPOINT_PATH} (ssl_name={SSL_MODEL_NAME}, "
          f"load into VoiceGuardNet with strict=False for inference)")


if __name__ == "__main__":
    main()
