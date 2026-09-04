"""Self-extract wav2vec2-base embeddings from raw ASVspoof-style audio, using
the EXACT same normalization + pooling logic as VoiceGuardNet.forward (see
src/model.py) -- guaranteeing training and inference are always consistent,
by construction.

Why this exists: the third-party precomputed-embedding dataset
(eminkorkut/deepfakevoice-wac2vec-4datasets) only documents its recipe in
prose ("768-dim wav2vec2 embeddings from 2s segments"), not exact code. A
head trained on those embeddings turned out NOT to transfer to our own
raw-audio pipeline (RiskEngine returned near-zero, undiscriminating risk on
both genuine and cloned demo clips) -- almost certainly because some detail
of their extraction (padding, exact pooling, normalization) doesn't match
ours. Self-extracting removes that unknown entirely.

Usage (inside the Kaggle notebook, after attaching a raw-audio ASVspoof
dataset, e.g. awsaf49/asvpoof-2019-dataset):
    python -m src.extract_embeddings \
        --protocol <path/to/train_protocol.txt> --audio-dir <path/to/train/flac> \
        --out artifacts/self_train_wav2vec.parquet
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from config import SSL_MODEL_NAME, WINDOW_SECONDS
from src.device import get_device
from src.dataset import ASVspoofDataset


@torch.no_grad()
def extract(protocol_path: str, audio_dir: str, out_path: str,
            ssl_name: str = SSL_MODEL_NAME, batch_size: int = 16, seconds: float = WINDOW_SECONDS):
    device = get_device()
    print(f"Extracting on device: {device}")
    ssl = Wav2Vec2Model.from_pretrained(ssl_name).to(device).eval()

    ds = ASVspoofDataset(protocol_path, audio_dir, seconds=seconds)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"{len(ds)} utterances, {seconds}s window (matches RiskEngine's WINDOW_SECONDS)")

    rows = []
    for i, (waveforms, labels) in enumerate(dl):
        waveforms = waveforms.to(device)
        # Identical to VoiceGuardNet.forward's normalization -- this is the
        # whole point: same code path as inference, not a re-derivation of it.
        mean = waveforms.mean(dim=-1, keepdim=True)
        var = waveforms.var(dim=-1, keepdim=True, unbiased=False)
        x = (waveforms - mean) / torch.sqrt(var + 1e-7)
        feats = ssl(x).last_hidden_state
        pooled = feats.mean(dim=1).cpu().numpy()

        for emb, label in zip(pooled, labels.tolist()):
            row = {f"emb_{j}": float(v) for j, v in enumerate(emb)}
            row["label"] = "bonafide" if label == 0 else "spoof"
            rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"  batch {i + 1}/{len(dl)} ({len(rows)} rows so far)")

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(out_path)
    n_bonafide = int((df["label"] == "bonafide").sum())
    n_spoof = int((df["label"] == "spoof").sum())
    print(f"Wrote {len(df)} rows to {out_path} (bonafide={n_bonafide}, spoof={n_spoof})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ssl-name", default=SSL_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=WINDOW_SECONDS)
    args = parser.parse_args()
    extract(args.protocol, args.audio_dir, args.out, args.ssl_name, args.batch_size, args.seconds)
