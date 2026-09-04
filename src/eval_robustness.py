"""Experiment C (real-world robustness): run genuine and cloned clips
through increasing degradation -- clean, noisy, reverb, telephony-style
compression -- and observe how the risk score shifts. Reuses
src/augment.py (already built for training-data hardening) for
noise/reverb; compression uses macOS's afconvert (AAC low-bitrate
round-trip) since no ffmpeg is installed here -- a reasonable stand-in
for codec/telephony degradation.

Usage:
    python -m src.eval_robustness demo_clips/teammate_ref.wav demo_clips/fraud_en.wav
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from config import ARTIFACTS_DIR, SAMPLE_RATE, DEFAULT_ALERT_THRESHOLD
from src.augment import add_noise, add_reverb, normalize_peak
from src.infer import RiskEngine


def compress_roundtrip(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Encode to low-bitrate AAC and back, via macOS's afconvert -- simulates
    telephony/codec compression without needing ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp:
        wav_in = os.path.join(tmp, "in.wav")
        m4a_path = os.path.join(tmp, "compressed.m4a")
        wav_out = os.path.join(tmp, "out.wav")
        sf.write(wav_in, waveform, sr)
        subprocess.run(
            ["afconvert", "-f", "m4af", "-d", "aac", "-b", "12000", wav_in, m4a_path],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{sr}", "-c", "1", m4a_path, wav_out],
            check=True, capture_output=True,
        )
        out, _ = sf.read(wav_out, dtype="float32", always_2d=False)
        return out


def make_conditions(waveform: np.ndarray, sr: int) -> dict:
    conditions = {"clean": waveform}
    conditions["noise_light"] = normalize_peak(add_noise(waveform, snr_db=25, seed=1))
    conditions["noise_heavy"] = normalize_peak(add_noise(waveform, snr_db=10, seed=2))
    conditions["reverb"] = normalize_peak(add_reverb(waveform, sr, room_scale=1.2))
    try:
        conditions["compressed_telephony"] = compress_roundtrip(waveform, sr)
    except Exception as e:
        print(f"  [skip compressed_telephony: {e}]")
    return conditions


def score_clip(waveform: np.ndarray, engine: RiskEngine) -> dict:
    win, hop = engine.win, engine.hop
    engine.reset()
    pos = 0
    peak = 0
    final = 0
    while pos < len(waveform):
        chunk = waveform[pos:pos + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        risk = engine.push(chunk)
        peak = max(peak, risk)
        final = risk
        pos += hop
    return {"peak_risk": peak, "final_risk": final}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+", help="WAV files, expected to already be labeled by filename convention")
    parser.add_argument("--threshold", type=int, default=DEFAULT_ALERT_THRESHOLD)
    parser.add_argument("--out", default=f"{ARTIFACTS_DIR}/test_suite_experiment_c.json")
    args = parser.parse_args()

    engine = RiskEngine()
    results = {}

    for path in args.clips:
        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != SAMPLE_RATE:
            import torch, torchaudio
            wav_t = torch.from_numpy(waveform).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
            waveform = wav_t.squeeze(0).numpy()
            sr = SAMPLE_RATE

        print(f"\n{path}:")
        conditions = make_conditions(waveform, sr)
        clip_results = {}
        for name, wave in conditions.items():
            scores = score_clip(wave, engine)
            scores["alert"] = scores["peak_risk"] >= args.threshold
            clip_results[name] = scores
            print(f"  {name:20s} peak={scores['peak_risk']:3d} final={scores['final_risk']:3d} alert={scores['alert']}")
        results[path] = clip_results

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
