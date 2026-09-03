"""Cross-dataset evaluation: score the model on attacks it never trained on,
compute Equal Error Rate (EER). Reporting EER on an *unseen* set (ASVspoof
2021 DF or In-the-Wild) is the credibility metric for the pitch.

Usage:
    python -m src.eval_eer --protocol <unseen_protocol> --audio-dir <unseen_audio> \
        --dataset-name "ASVspoof2021-DF"
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

from config import CHECKPOINT_PATH, METRICS_PATH, ARTIFACTS_DIR, SSL_MODEL_NAME
from src.device import get_device
from src.dataset import ASVspoofDataset
from src.model import VoiceGuardNet


def compute_eer(labels, scores):
    """labels: 0/1 (1=spoof), scores: P(spoof). Returns EER in [0, 1]."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--dataset-name", required=True, help="label for metrics.json, e.g. ASVspoof2021-DF")
    parser.add_argument("--ckpt", default=CHECKPOINT_PATH)
    parser.add_argument("--ssl-name", default=SSL_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    model = VoiceGuardNet(ssl_name=args.ssl_name).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    ds = ASVspoofDataset(args.protocol, args.audio_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_labels, all_scores = [], []
    with torch.no_grad():
        for waveforms, labels in dl:
            waveforms = waveforms.to(device)
            probs = F.softmax(model(waveforms), dim=-1)[:, 1]  # P(spoof)
            all_scores.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    eer = compute_eer(all_labels, all_scores)
    print(f"EER on {args.dataset_name}: {eer * 100:.2f}%")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    metrics[args.dataset_name] = {"eer": eer, "eer_pct": round(eer * 100, 2), "n": len(all_labels)}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
