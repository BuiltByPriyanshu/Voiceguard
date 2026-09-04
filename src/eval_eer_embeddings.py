"""Cross-dataset EER on precomputed embeddings from an UNSEEN dataset (e.g.
ASVspoof2021_test_wav2vec.parquet or In-the-wild_test_wav2vec.parquet) --
the credibility number for the pitch. Fast: no audio, no SSL forward pass.

Usage:
    python -m src.eval_eer_embeddings \
        --parquet /kaggle/input/datasets/eminkorkut/deepfakevoice-wac2vec-4datasets/ASVspoof2021_test_wav2vec.parquet \
        --dataset-name ASVspoof2021-DF
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from config import CHECKPOINT_PATH, METRICS_PATH, ARTIFACTS_DIR, EMBEDDING_DIM
from src.device import get_device
from src.dataset import load_embedding_parquet
from src.model import EmbeddingClassifier
from src.eval_eer import compute_eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--dataset-name", required=True, help="label for metrics.json, e.g. ASVspoof2021-DF")
    parser.add_argument("--ckpt", default=CHECKPOINT_PATH)
    parser.add_argument("--in-dim", type=int, default=EMBEDDING_DIM)
    args = parser.parse_args()

    device = get_device()
    model = EmbeddingClassifier(in_dim=args.in_dim).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    X, y = load_embedding_parquet(args.parquet)
    X = X.to(device)
    n_bonafide, n_spoof = int((y == 0).sum()), int((y == 1).sum())
    print(f"Loaded {X.shape[0]} rows: bonafide={n_bonafide}, spoof={n_spoof}")

    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=-1)[:, 1]  # P(spoof)
        preds = logits.argmax(dim=-1).cpu()

    for cls, name in ((0, "bonafide"), (1, "spoof")):
        mask = y == cls
        recall = (preds[mask] == cls).float().mean().item() if mask.any() else float("nan")
        print(f"  {name} recall: {recall:.4f}")

    eer = compute_eer(y.tolist(), probs.cpu().tolist())
    print(f"EER on {args.dataset_name}: {eer * 100:.2f}%")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    metrics[args.dataset_name] = {"eer": eer, "eer_pct": round(eer * 100, 2), "n": len(y)}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
