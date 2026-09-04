"""Extract wav2vec2-base embeddings from individual labeled WAV files, not an
ASVspoof-protocol dataset -- for adding a handful of clips (e.g. a demo
reference recording) as extra training data.

Why this exists: a model trained only on ASVspoof's studio-quality bonafide
recordings learns "genuine = ASVspoof's specific recording conditions" and
flags anything else -- including a real laptop-mic recording -- as
suspicious (see the run where teammate_ref.wav scored the same ~100% risk
as the cloned attack clips). Feeding in a real clip from the ACTUAL
microphone/environment used for the demo teaches the model that condition
is genuine too. Chunks each clip into the same overlapping windows
RiskEngine uses live (WINDOW_SECONDS/HOP_SECONDS), so one clip yields many
training rows through the identical normalization + pooling code path as
src/model.py's VoiceGuardNet.forward.

Usage:
    python -m src.extract_clip_embeddings \
        --clip demo_clips/teammate_ref.wav bonafide \
        --out artifacts/self_calibration_wav2vec.parquet
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Model

from config import SSL_MODEL_NAME, SAMPLE_RATE, WINDOW_SECONDS, HOP_SECONDS
from src.device import get_device
from src.dataset import load_waveform
from src.augment import augment_variants


@torch.no_grad()
def extract_windows_from_waveform(waveform: np.ndarray, label_str: str, ssl, device,
                                   win_samples: int, hop_samples: int):
    rows = []
    pos = 0
    while pos < len(waveform):
        chunk = waveform[pos:pos + win_samples]
        if len(chunk) < win_samples:
            chunk = np.pad(chunk, (0, win_samples - len(chunk)))
        x = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        xn = (x - mean) / torch.sqrt(var + 1e-7)
        feats = ssl(xn).last_hidden_state
        pooled = feats.mean(dim=1)[0].cpu().numpy()
        row = {f"emb_{i}": float(v) for i, v in enumerate(pooled)}
        row["label"] = label_str.lower()
        rows.append(row)
        pos += hop_samples
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", nargs=2, action="append", required=True,
                         metavar=("PATH", "LABEL"),
                         help="a WAV path and its label (bonafide/spoof); repeat --clip for multiple files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ssl-name", default=SSL_MODEL_NAME)
    parser.add_argument("--augment-bonafide", action="store_true",
                         help="also extract noise/reverb variants of each bonafide clip "
                              "(see src/augment.py) -- hardens against real-world mic/room "
                              "conditions the clean original doesn't cover")
    args = parser.parse_args()

    device = get_device()
    print(f"Extracting on device: {device}")
    ssl = Wav2Vec2Model.from_pretrained(args.ssl_name).to(device).eval()
    win_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop_samples = int(HOP_SECONDS * SAMPLE_RATE)

    all_rows = []
    for path, label in args.clip:
        waveform = load_waveform(path, SAMPLE_RATE)
        rows = extract_windows_from_waveform(waveform, label, ssl, device, win_samples, hop_samples)
        print(f"  {path} ({label}): {len(rows)} windows")
        all_rows.extend(rows)

        if args.augment_bonafide and label.lower() == "bonafide":
            for variant_name, variant_wave in augment_variants(waveform, SAMPLE_RATE):
                variant_rows = extract_windows_from_waveform(
                    variant_wave, label, ssl, device, win_samples, hop_samples
                )
                print(f"    + {variant_name}: {len(variant_rows)} windows")
                all_rows.extend(variant_rows)

    df = pd.DataFrame(all_rows)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.out)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
