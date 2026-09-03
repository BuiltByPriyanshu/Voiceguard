# VoiceGuard

Real-time voice-cloning / synthetic-speech detection and impersonation
prevention — built for SIH26104 (AICTE Cyber Security Cell).

Detects AI-cloned voices during a live call, computes a running 0-100 risk
score, and fires a pre-transaction warning before a sensitive action (fund
transfer, data disclosure) is approved.

## The train-in-cloud, run-on-Mac rule

**All GPU-heavy work (model training, voice cloning) happens in the cloud
(Kaggle/Colab), never on the Apple Silicon MacBook.** The Mac has no CUDA and
throttles under sustained load, so it only ever runs light inference and the
app. Artifacts (`model.pth`, demo clips) are produced in the cloud and pulled
onto the Mac via GitHub (code) + Google Drive (binaries).

- `torch.cuda` → cloud training tonight
- `torch.mps` → Mac inference/app tomorrow
- Code must be device-agnostic: always use `src/device.py:get_device()`,
  never hardcode `.cuda()`.

## Repo layout

```
voiceguard/
├── requirements_cuda.txt   # cloud/CUDA training env
├── requirements_mac.txt    # Mac/MPS inference + app env
├── config.py                # paths, thresholds, device selection
├── src/
│   ├── device.py             # CUDA -> MPS -> CPU helper
│   ├── dataset.py             # ASVspoof protocol/audio loading
│   ├── model.py                # SSL front-end (wav2vec2/XLS-R) + classifier head
│   ├── train.py                 # training loop (cloud)
│   ├── eval_eer.py               # EER on unseen data
│   └── infer.py                   # streaming RiskEngine (runs on Mac)
├── backend/
│   └── app.py                # FastAPI: /analyze, /stream (WebSocket), /config
├── notebooks/
│   └── train_kaggle.ipynb    # run this in Kaggle tonight
├── demo_clips/               # genuine + cloned WAVs (gitignored, in Drive)
└── artifacts/                # model.pth, metrics.json (gitignored, in Drive)
```

## Setup

Cloud (Kaggle/Colab):
```
pip install -r requirements_cuda.txt
```

Mac (inference + app):
```
pip install -r requirements_mac.txt
python -m src.device   # should print mps
```

## Training + evaluation (cloud)

See `notebooks/train_kaggle.ipynb`. Summary:
```
python -m src.train --protocol <train.trn.txt> --audio-dir <flac_dir> \
  --val-protocol <dev.trl.txt> --val-audio-dir <flac_dir>

python -m src.eval_eer --protocol <unseen_protocol> --audio-dir <unseen_audio> \
  --dataset-name ASVspoof2021-DF
```
Current headline EER: see `artifacts/metrics.json` (filled in after tonight's
training run).

## Streaming inference (Mac)

```
python -m src.infer demo_clips/fraud_en.wav
```
Loads `artifacts/model.pth`, scores in 4s sliding windows (1s hop), prints a
smoothed running risk score per hop.

## Backend

```
uvicorn backend.app:app --reload
```

### API contract

- `GET /config` → `{"threshold": int, "high_value_threshold": int}`
- `POST /config` → body `{"threshold": int}` to update
- `POST /analyze` (multipart file) →
  `{"risk": 0-100, "verdict": "genuine"|"synthetic", "alert": bool, "reason": str|null, "per_window": [int, ...]}`
- `WS /stream` — client sends raw 16kHz mono PCM16 chunks as binary frames;
  server replies once per hop with
  `{"risk": 0-100, "alert": bool, "reason": str|null}`

Pre-transaction warning rule: if `risk >= threshold`, `alert=true` and
`reason="Likely synthetic voice — recommend call-back / MFA before approving."`

Mic capture happens **in the browser** via the Web Audio API — the backend
never touches `pyaudio`/`portaudio`, so there's nothing extra to install on
the Mac.

## Handoff bundle (what the Mac pulls tomorrow)

| Item | Where |
|---|---|
| Code (`src/`, `backend/`, `config.py`, requirements, README) | GitHub |
| `artifacts/model.pth` + `artifacts/ssl_name.txt` | Google Drive |
| `artifacts/metrics.json` (EER) | Google Drive / GitHub |
| `demo_clips/*.wav` (genuine + cloned pairs) | Google Drive |

Tomorrow morning: `git clone`, download the Drive folder into `artifacts/`
and `demo_clips/`, `pip install -r requirements_mac.txt`, confirm
`get_device()` returns `mps`, then `python -m src.infer demo_clips/fraud_en.wav`
to prove the handoff before building anything else.
