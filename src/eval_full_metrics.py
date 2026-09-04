"""Experiment A (voice authenticity accuracy): full classification metrics
on a precomputed-embeddings parquet -- accuracy, precision, recall, F1,
EER, ROC-AUC, confusion matrix. Extends eval_eer_embeddings.py, which only
reports EER + per-class recall, with the fuller picture test-suite-style
reporting needs. Same model, same data path -- no new training.

Usage:
    python -m src.eval_full_metrics \
        --parquet artifacts/asvspoof2019_eval_sample10k_wav2vec.parquet \
        --dataset-name ASVspoof2019-LA-eval-sample10k
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix,
)

from config import CHECKPOINT_PATH, ARTIFACTS_DIR, EMBEDDING_DIM
from src.device import get_device
from src.dataset import load_embedding_parquet
from src.model import EmbeddingClassifier
from src.eval_eer import compute_eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--ckpt", default=CHECKPOINT_PATH)
    parser.add_argument("--in-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--out", default=os.path.join(ARTIFACTS_DIR, "test_suite_experiment_a.json"))
    args = parser.parse_args()

    device = get_device()
    model = EmbeddingClassifier(in_dim=args.in_dim).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    X, y = load_embedding_parquet(args.parquet)
    X = X.to(device)
    y_true = y.tolist()

    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=-1)[:, 1]  # P(spoof)
        preds = logits.argmax(dim=-1).cpu().tolist()

    probs_list = probs.cpu().tolist()

    acc = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1], zero_division=0
    )
    eer = compute_eer(y_true, probs_list)
    auc = roc_auc_score(y_true, probs_list)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else float("nan")

    result = {
        "dataset": args.dataset_name,
        "n": len(y_true),
        "n_bonafide": int((y == 0).sum()),
        "n_spoof": int((y == 1).sum()),
        "accuracy": round(acc, 4),
        "precision": {"bonafide": round(precision[0], 4), "spoof": round(precision[1], 4)},
        "recall": {"bonafide": round(recall[0], 4), "spoof": round(recall[1], 4)},
        "f1": {"bonafide": round(f1[0], 4), "spoof": round(f1[1], 4)},
        "eer_pct": round(eer * 100, 2),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        # Experiment E (false-positive safety) reads directly off this run:
        # false_positive_rate is the fraction of GENUINE speech incorrectly
        # flagged as synthetic -- the number that matters for not blocking
        # legitimate calls/transactions.
        "false_positive_rate": round(false_positive_rate, 4),
    }

    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
