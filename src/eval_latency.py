"""Experiment B (streaming behaviour): how quickly does the system
recognize an attack, and what's the per-window inference latency? Reuses
RiskEngine exactly as the live app does -- same sliding window/hop, same
EMA smoothing -- so these numbers reflect actual demo behavior, not a
synthetic benchmark.

Usage:
    python -m src.eval_latency demo_clips/fraud_en.wav demo_clips/fraud_hi.wav
"""
import argparse
import json
import time

import numpy as np
import soundfile as sf

from config import ARTIFACTS_DIR, DEFAULT_ALERT_THRESHOLD
from src.infer import RiskEngine


def run_clip(path: str, engine: RiskEngine, threshold: int) -> dict:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != engine.sr:
        import torch, torchaudio
        wav_t = torch.from_numpy(audio).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, engine.sr)
        audio = wav_t.squeeze(0).numpy()

    engine.reset()
    win, hop = engine.win, engine.hop
    pos, t_idx = 0, 0.0
    trace, latencies = [], []
    time_to_alert = None

    while pos < len(audio):
        chunk = audio[pos:pos + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        t0 = time.time()
        risk = engine.push(chunk)
        latency_ms = (time.time() - t0) * 1000
        alert = risk >= threshold
        trace.append({"t": round(t_idx, 1), "risk": risk, "alert": alert})
        latencies.append(latency_ms)
        if alert and time_to_alert is None:
            time_to_alert = round(t_idx, 1)
        pos += hop
        t_idx += hop / engine.sr

    return {
        "path": path,
        "duration_s": round(len(audio) / engine.sr, 1),
        "time_to_first_alert_s": time_to_alert,
        "mean_inference_latency_ms": round(sum(latencies[1:]) / max(len(latencies) - 1, 1), 1)
        if len(latencies) > 1 else round(latencies[0], 1) if latencies else None,
        "first_window_latency_ms": round(latencies[0], 1) if latencies else None,
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--threshold", type=int, default=DEFAULT_ALERT_THRESHOLD)
    parser.add_argument("--out", default=f"{ARTIFACTS_DIR}/test_suite_experiment_b.json")
    args = parser.parse_args()

    engine = RiskEngine()
    results = [run_clip(clip, engine, args.threshold) for clip in args.clips]

    for r in results:
        print(f"{r['path']}: duration={r['duration_s']}s "
              f"time_to_first_alert={r['time_to_first_alert_s']}s "
              f"mean_latency={r['mean_inference_latency_ms']}ms "
              f"first_window_latency={r['first_window_latency_ms']}ms")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
