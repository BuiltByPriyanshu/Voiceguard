"""Fine-tune the classifier head on ASVspoof 2019 LA. Run in the cloud (CUDA).

Usage (inside the Kaggle notebook, after attaching datasets):
    python -m src.train \
        --protocol /kaggle/input/asvspoof-2019-la/.../train.trn.txt \
        --audio-dir /kaggle/input/asvspoof-2019-la/.../flac \
        --val-protocol /kaggle/input/asvspoof-2019-la/.../dev.trl.txt \
        --val-audio-dir /kaggle/input/asvspoof-2019-la/.../flac
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, CHECKPOINT_PATH, ARTIFACTS_DIR,
    SSL_MODEL_NAME,
)
from src.device import get_device
from src.dataset import ASVspoofDataset
from src.model import VoiceGuardNet


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(waveforms)
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
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--val-protocol", default=None)
    parser.add_argument("--val-audio-dir", default=None)
    parser.add_argument("--ssl-name", default=SSL_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    device = get_device()
    print(f"Training on device: {device}")

    train_ds = ASVspoofDataset(args.protocol, args.audio_dir)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    val_dl = None
    if args.val_protocol and args.val_audio_dir:
        val_ds = ASVspoofDataset(args.val_protocol, args.val_audio_dir)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = VoiceGuardNet(ssl_name=args.ssl_name).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_dl, device, optimizer)
        msg = f"epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
        if val_dl is not None:
            val_loss, val_acc = run_epoch(model, val_dl, device)
            msg += f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        print(msg)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    with open(os.path.join(ARTIFACTS_DIR, "ssl_name.txt"), "w") as f:
        f.write(args.ssl_name)
    print(f"Saved checkpoint to {CHECKPOINT_PATH} (ssl_name={args.ssl_name})")


if __name__ == "__main__":
    main()
