"""FastAPI backend: /analyze (file), /stream (WebSocket), /config (thresholds).

API contract (also documented in README.md):
- GET  /config           -> {"threshold": int, "high_value_threshold": int}
- POST /config           -> body {"threshold": int} to update the default threshold
- POST /analyze          -> multipart file upload; returns
                             {"risk": 0-100, "verdict": "genuine"|"synthetic",
                              "alert": bool, "reason": str|None, "per_window": [int, ...]}
- WS   /stream            -> client sends raw 16kHz mono PCM16 chunks (bytes);
                             server replies JSON {"risk": int, "alert": bool, "reason": str|None}
                             once per hop.

Mic audio is captured in the browser (Web Audio API) and streamed over the
WebSocket -- this deliberately avoids needing pyaudio/portaudio on the Mac.
"""
import io
import json

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File

from config import (
    DEFAULT_ALERT_THRESHOLD, HIGH_VALUE_ALERT_THRESHOLD, ALERT_REASON,
    SAMPLE_RATE, WINDOW_SECONDS, HOP_SECONDS,
)
from src.infer import RiskEngine

app = FastAPI(title="VoiceGuard API")

# Loaded lazily so the server can boot even before a checkpoint exists
# (e.g. while scaffolding tonight before training finishes).
_engine: RiskEngine | None = None
_state = {"threshold": DEFAULT_ALERT_THRESHOLD, "high_value_threshold": HIGH_VALUE_ALERT_THRESHOLD}


def get_engine() -> RiskEngine:
    global _engine
    if _engine is None:
        _engine = RiskEngine()
    return _engine


@app.get("/config")
async def get_config():
    return _state


@app.post("/config")
async def set_config(body: dict):
    if "threshold" in body:
        _state["threshold"] = int(body["threshold"])
    if "high_value_threshold" in body:
        _state["high_value_threshold"] = int(body["high_value_threshold"])
    return _state


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    engine = get_engine()
    if sr != engine.sr:
        import torch, torchaudio
        wav_t = torch.from_numpy(audio).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, engine.sr)
        audio = wav_t.squeeze(0).numpy()

    engine.reset()
    win, hop = engine.win, engine.hop
    per_window = []
    pos = 0
    while pos < len(audio):
        chunk = audio[pos:pos + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        per_window.append(engine.push(chunk))
        pos += hop

    risk = per_window[-1] if per_window else 0
    alert = risk >= _state["threshold"]
    return {
        "risk": risk,
        "verdict": "synthetic" if alert else "genuine",
        "alert": alert,
        "reason": ALERT_REASON if alert else None,
        "per_window": per_window,
    }


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    engine = get_engine()
    engine.reset()
    buffer = np.zeros(0, dtype=np.float32)
    win, hop = engine.win, engine.hop

    try:
        while True:
            data = await ws.receive_bytes()
            # Client sends 16-bit PCM mono @ SAMPLE_RATE.
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= win:
                window = buffer[:win]
                risk = engine.push(window)
                alert = risk >= _state["threshold"]
                await ws.send_text(json.dumps({
                    "risk": risk,
                    "alert": alert,
                    "reason": ALERT_REASON if alert else None,
                }))
                buffer = buffer[hop:]
    except WebSocketDisconnect:
        pass
