"""Experiment D (contextual risk): keep the voice identical, vary call/
transaction context via the live /context API, and show interaction_risk
change while voice_authenticity stays constant -- the direct
demonstration of the risk-fusion architecture (src/risk_fusion.py). Hits
the live backend, so `uvicorn backend.app:app` must already be running.

Usage:
    python -m src.eval_context_demo demo_clips/teammate_ref.wav
"""
import argparse
import json

import requests

from config import ARTIFACTS_DIR

BASE = "http://127.0.0.1:8000"

SCENARIOS = [
    {"label": "ordinary call, known contact", "known_contact": True, "transaction_value": "none"},
    {"label": "known contact, low-value txn", "known_contact": True, "transaction_value": "low"},
    {"label": "unknown caller, no txn", "known_contact": False, "transaction_value": "none"},
    {"label": "unknown caller, high-value txn (urgent transfer)", "known_contact": False, "transaction_value": "high"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("--out", default=f"{ARTIFACTS_DIR}/test_suite_experiment_d.json")
    args = parser.parse_args()

    results = []
    for s in SCENARIOS:
        requests.post(f"{BASE}/context", json={
            "known_contact": s["known_contact"],
            "transaction_value": s["transaction_value"],
        })
        with open(args.clip, "rb") as f:
            r = requests.post(f"{BASE}/analyze", files={"file": (args.clip, f, "audio/wav")}).json()
        row = {
            "scenario": s["label"],
            "voice_authenticity": r["voice_authenticity"],
            "context_risk": r["context_risk"],
            "interaction_risk": r["interaction_risk"],
            "decision_band": r["decision"]["band"],
        }
        results.append(row)
        print(f"{s['label']:50s} voice={row['voice_authenticity']:3d} "
              f"context={row['context_risk']:3d} interaction={row['interaction_risk']:3d} "
              f"({row['decision_band']})")

    # reset to safe defaults
    requests.post(f"{BASE}/context", json={"known_contact": True, "transaction_value": "none"})

    with open(args.out, "w") as f:
        json.dump({"clip": args.clip, "scenarios": results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
