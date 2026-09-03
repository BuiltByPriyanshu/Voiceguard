"""Audio loading + labels for ASVspoof-style protocol files.

Works against Kaggle-attached datasets (zero local download): point
`protocol_path` and `audio_dir` at the attached dataset's files.

ASVspoof protocol lines look like (whitespace-separated):
    LA_0079 LA_D_1047731 - - bonafide
    LA_0079 LA_D_1047732 - A01 spoof
The label is always the last column ("bonafide" / "spoof"); the audio file
id is column 2 (matches `<audio_dir>/<file_id>.flac` or `.wav`).
"""
import os
import glob

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from config import SAMPLE_RATE, CLIP_SECONDS, LABEL_BONAFIDE, LABEL_SPOOF

EMBEDDING_LABEL_MAP = {
    "bonafide": LABEL_BONAFIDE, "real": LABEL_BONAFIDE,
    "spoof": LABEL_SPOOF, "fake": LABEL_SPOOF,
}


def load_embedding_parquet(path: str):
    """Load a precomputed-embedding parquet (emb_0..emb_N columns + a
    real/fake or bonafide/spoof `label` column) into (X, y) tensors."""
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = torch.tensor(df[emb_cols].values, dtype=torch.float32)
    y = torch.tensor(df["label"].str.lower().map(EMBEDDING_LABEL_MAP).values, dtype=torch.long)
    return X, y

try:
    import torchaudio
except ImportError:  # torchaudio should always be present per requirements
    torchaudio = None


def parse_protocol(protocol_path: str):
    """Return list of (file_id, label) from an ASVspoof protocol file."""
    items = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            file_id = parts[1]
            label_str = parts[-1].lower()
            label = LABEL_BONAFIDE if label_str == "bonafide" else LABEL_SPOOF
            items.append((file_id, label))
    return items


def _find_audio_file(audio_dir: str, file_id: str) -> str:
    for ext in (".flac", ".wav"):
        candidate = os.path.join(audio_dir, file_id + ext)
        if os.path.exists(candidate):
            return candidate
    # fall back to a recursive glob in case files are nested in subfolders
    matches = glob.glob(os.path.join(audio_dir, "**", file_id + ".*"), recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Audio file for id={file_id} not found under {audio_dir}")


def load_waveform(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a wav/flac, downmix to mono, resample to target_sr."""
    waveform, sr = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        if torchaudio is None:
            raise RuntimeError("torchaudio required to resample audio")
        wav_t = torch.from_numpy(waveform).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        waveform = wav_t.squeeze(0).numpy()
    return waveform


def fix_length(waveform: np.ndarray, seconds: float = CLIP_SECONDS, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Trim or zero-pad to a fixed length so batches can be stacked."""
    target_len = int(seconds * sr)
    if len(waveform) >= target_len:
        return waveform[:target_len]
    pad = np.zeros(target_len - len(waveform), dtype=waveform.dtype)
    return np.concatenate([waveform, pad])


class ASVspoofDataset(Dataset):
    def __init__(self, protocol_path: str, audio_dir: str, seconds: float = CLIP_SECONDS,
                 sample_rate: int = SAMPLE_RATE):
        self.items = parse_protocol(protocol_path)
        self.audio_dir = audio_dir
        self.seconds = seconds
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_id, label = self.items[idx]
        path = _find_audio_file(self.audio_dir, file_id)
        waveform = load_waveform(path, self.sample_rate)
        waveform = fix_length(waveform, self.seconds, self.sample_rate)
        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    if len(sys.argv) != 3:
        print("Usage: python -m src.dataset <protocol_path> <audio_dir>")
        sys.exit(1)
    ds = ASVspoofDataset(sys.argv[1], sys.argv[2])
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    wavs, labels = next(iter(dl))
    print("batch waveform shape:", wavs.shape, "labels:", labels.tolist())
