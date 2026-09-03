"""Streaming inference: the real-time engine the Mac runs tomorrow.

Loads a checkpoint once, then scores audio in sliding windows with an EMA-
smoothed running risk score (0-100). Runs on CUDA / MPS / CPU via
src.device.get_device() -- never hardcode .cuda().
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    SSL_MODEL_NAME, WINDOW_SECONDS, HOP_SECONDS, SAMPLE_RATE, EMA_ALPHA,
    CHECKPOINT_PATH, DEFAULT_ALERT_THRESHOLD,
)
from src.device import get_device
from src.model import VoiceGuardNet


class RiskEngine:
    def __init__(self, ckpt=CHECKPOINT_PATH, ssl_name=SSL_MODEL_NAME,
                 win=WINDOW_SECONDS, hop=HOP_SECONDS, sr=SAMPLE_RATE, ema=EMA_ALPHA):
        self.device = get_device()
        self.model = VoiceGuardNet(ssl_name).to(self.device).eval()
        # strict=False: tonight's checkpoint is head-only (trained on
        # precomputed embeddings via src/train_embeddings.py), so only the
        # `head.*` keys match here -- the SSL backbone keeps its pretrained
        # weights from `ssl_name`, loaded above.
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)
        self.win = int(win * sr)
        self.hop = int(hop * sr)
        self.sr = sr
        self.ema = ema
        self.score = 0.0

    @torch.no_grad()
    def push(self, waveform: np.ndarray) -> int:
        """waveform: 1-D np array @ self.sr, length == self.win (pad/trim by caller).
        Returns the smoothed running risk score, 0-100."""
        x = torch.tensor(waveform, dtype=torch.float32, device=self.device).unsqueeze(0)
        p_spoof = F.softmax(self.model(x), dim=-1)[0, 1].item()
        self.score = self.ema * self.score + (1 - self.ema) * p_spoof
        return round(self.score * 100)

    def reset(self):
        self.score = 0.0


def stream_wav_file(path: str, engine: RiskEngine, threshold: int = DEFAULT_ALERT_THRESHOLD):
    """CLI helper: stream a WAV through the engine in hops, print the risk trace."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != engine.sr:
        import torchaudio
        wav_t = torch.from_numpy(audio).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, engine.sr)
        audio = wav_t.squeeze(0).numpy()

    win, hop = engine.win, engine.hop
    pos = 0
    t_idx = 0.0
    while pos < len(audio):
        chunk = audio[pos:pos + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        t0 = time.time()
        risk = engine.push(chunk)
        latency_ms = (time.time() - t0) * 1000
        alert = risk >= threshold
        print(f"t={t_idx:5.1f}s  risk={risk:3d}  alert={alert}  latency={latency_ms:.0f}ms")
        pos += hop
        t_idx += hop / engine.sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream a WAV through the RiskEngine")
    parser.add_argument("wav_path")
    parser.add_argument("--ckpt", default=CHECKPOINT_PATH)
    parser.add_argument("--ssl-name", default=SSL_MODEL_NAME)
    parser.add_argument("--threshold", type=int, default=DEFAULT_ALERT_THRESHOLD)
    args = parser.parse_args()

    engine = RiskEngine(ckpt=args.ckpt, ssl_name=args.ssl_name)
    stream_wav_file(args.wav_path, engine, threshold=args.threshold)
