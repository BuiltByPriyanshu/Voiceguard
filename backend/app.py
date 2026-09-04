"""FastAPI backend: /analyze (file), /stream (WebSocket), /config (thresholds),
/context (caller/transaction context for risk fusion).

VoiceGuard is a voice trust / interaction-risk system, not just a voice
detector: voice authenticity (this module's wav2vec2 RiskEngine) is one
evidence source, call/transaction context is another, and the two fuse
into one interaction risk -- see src/risk_fusion.py and HANDOFF.md for
the architecture. "verdict" reflects the voice alone; "alert" and
"decision" reflect the fused interaction risk, which is what should
actually gate a sensitive action.

API contract (also documented in README.md):
- GET  /config    -> {"threshold": int, "high_value_threshold": int}
- POST /config    -> body {"threshold": int} to update the default threshold
- GET  /context   -> {"known_contact": bool, "transaction_value": str}
- POST /context   -> body {"known_contact"?: bool, "transaction_value"?: str}
                      ("none"|"low"|"medium"|"high") to update
- POST /analyze   -> multipart file upload; returns
                      {"voice_authenticity": 0-100, "context_risk": 0-100,
                       "interaction_risk": 0-100, "decision": {"band", "action"},
                       "verdict": "genuine"|"synthetic", "alert": bool,
                       "reason": str|None, "per_window": [int, ...]}
- WS   /stream    -> client sends raw 16kHz mono PCM16 chunks (bytes);
                      server replies JSON with the same risk fields (minus
                      verdict/per_window) once per hop.

Mic audio is captured in the browser (Web Audio API) and streamed over the
WebSocket -- this deliberately avoids needing pyaudio/portaudio on the Mac.
"""
from __future__ import annotations

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
from src.risk_fusion import compute_context_risk, fuse_risk, decision_for

app = FastAPI(title="VoiceGuard API")

# Loaded lazily so the server can boot even before a checkpoint exists
# (e.g. while scaffolding tonight before training finishes).
_engine: RiskEngine | None = None
_state = {"threshold": DEFAULT_ALERT_THRESHOLD, "high_value_threshold": HIGH_VALUE_ALERT_THRESHOLD}
# Context defaults to the neutral/safe case (known contact, no transaction)
# so interaction_risk == voice_authenticity until the client sets otherwise
# -- existing voice-only demo behavior is unchanged by default.
_context = {"known_contact": True, "transaction_value": "none"}


def get_engine() -> RiskEngine:
    global _engine
    if _engine is None:
        _engine = RiskEngine()
    return _engine


def _risk_payload(voice_authenticity: int) -> dict:
    context_risk = compute_context_risk(_context["known_contact"], _context["transaction_value"])
    interaction_risk = fuse_risk(voice_authenticity, context_risk)
    alert = interaction_risk >= _state["threshold"]
    return {
        "voice_authenticity": voice_authenticity,
        "context_risk": context_risk,
        "interaction_risk": interaction_risk,
        "decision": decision_for(interaction_risk),
        # verdict reflects the voice alone (same threshold, applied to the
        # unfused score) -- it can disagree with alert/decision on purpose.
        "verdict": "synthetic" if voice_authenticity >= _state["threshold"] else "genuine",
        "alert": alert,
        "reason": ALERT_REASON if alert else None,
    }


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


@app.get("/context")
async def get_context():
    return _context


@app.post("/context")
async def set_context(body: dict):
    if "known_contact" in body:
        _context["known_contact"] = bool(body["known_contact"])
    if "transaction_value" in body:
        _context["transaction_value"] = str(body["transaction_value"])
    return _context


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

    voice_authenticity = per_window[-1] if per_window else 0
    payload = _risk_payload(voice_authenticity)
    payload["per_window"] = per_window
    return payload


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
                voice_authenticity = engine.push(window)
                await ws.send_text(json.dumps(_risk_payload(voice_authenticity)))
                buffer = buffer[hop:]
    except WebSocketDisconnect:
        pass
