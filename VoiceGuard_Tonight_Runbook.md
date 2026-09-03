# VoiceGuard — Tonight's Prep & Cloud-Training Runbook
### A step-by-step implementation plan for Claude Code (SIH26104, hackathon tomorrow)

---

## 0. Context — read this first (for Claude Code)

We are building **VoiceGuard**, a real-time voice-cloning / synthetic-speech detector for SIH26104 (AICTE Cyber Security Cell). The hackathon starts tomorrow. This runbook covers **only tonight's work** and the assets to have ready.

**The hard constraint that shapes everything below:**
- **Tonight:** work happens on *my laptop*, which we use as a **control terminal** to drive a **cloud GPU** (Kaggle/Colab). We do NOT train locally.
- **Tomorrow afternoon:** we move to a teammate's **MacBook Air M5** — Apple Silicon, **MPS backend, NO CUDA**, fanless (throttles under load). So the Mac must only ever do **inference + app-building**, never training.
- **Therefore the golden rule:** *all GPU-heavy work (training, voice cloning) is done in the cloud tonight and saved as artifacts. Tomorrow, laptops just pull those artifacts and run light inference.*

**Tonight's definition of success (what must exist by the time we sleep):**
1. A fine-tuned detection model checkpoint (`model.pth`) with a recorded EER on unseen data.
2. A pack of demo audio clips (genuine + AI-cloned pairs, incl. one Hindi/Indian-accent).
3. A GitHub repo + Google Drive bundle containing the checkpoint, code, clips, and **two** pinned environment files (CUDA + Mac).
4. A device-agnostic **inference module** and a **FastAPI skeleton**, so tomorrow the team starts integrating instead of scaffolding.

Implement the steps below **in order**. Each step states *what it is*, *why it's needed here*, *what to do*, and *definition of done*.

---

## Part A — Have these ready before starting (accounts + assets)

> **Why this part exists:** the single biggest time-waster tonight would be discovering mid-task that an account isn't set up or a dataset won't download. Confirm all of this first.

- [ ] **Kaggle account** + phone-verified (needed to enable GPU + attach datasets). *Primary training environment — more reliable sessions than Colab and hosts the datasets already.*
- [ ] **Google Colab** account (backup training environment).
- [ ] **GitHub account** + an empty repo `voiceguard` created. *Version control + how code reaches the Mac tomorrow.*
- [ ] **Hugging Face account** + access token. *We pull the pretrained wav2vec2/XLS-R front-end from HF; some models need a token.*
- [ ] **Google Drive** with ≥5 GB free. *Where we store large binaries (checkpoints, clips) that don't belong in git.*
- [ ] Identify the **Kaggle-hosted datasets** to attach (search Kaggle Datasets):
  - ASVspoof 2019 **LA** (primary train/dev)
  - ASVspoof 2021 **DF** eval subset (unseen-attack test) — if not on Kaggle, note it and use In-the-Wild as the unseen set
  - In-the-Wild (real-world generalisation test)
  *Why: attaching Kaggle datasets means ZERO download to our disk — solves the space problem entirely.*
- [ ] Note the **baseline model choice** (see Step 5): pragmatic HF-features path is primary; AASIST repo is optional.
- [ ] A **10–20s clean voice sample** of a teammate (for the cloning demo). *XTTS needs a reference clip to clone a target voice.*

---

## Part B — Tonight, step by step

### Step 1 — Scaffold the repo and push to GitHub
**What it is:** the folder skeleton, git init, first push.
**Why here:** a fixed structure lets full-stack and AI/ML work in parallel tomorrow without stepping on each other, and pushing tonight means the Mac just `git clone`s in the afternoon.

**Do this:**
```
voiceguard/
├── README.md
├── requirements_cuda.txt        # cloud/CUDA training env
├── requirements_mac.txt         # Mac/MPS inference + app env
├── config.py                    # paths, thresholds, device selection
├── src/
│   ├── device.py                # device-agnostic helper
│   ├── dataset.py               # audio loading + labels
│   ├── model.py                 # SSL front-end + classifier head
│   ├── train.py                 # training loop (runs in cloud)
│   ├── eval_eer.py              # EER on unseen set
│   └── infer.py                 # streaming inference (runs on Mac)
├── backend/
│   └── app.py                   # FastAPI: /analyze, /stream (WebSocket), /config
├── notebooks/
│   └── train_kaggle.ipynb       # the notebook we run tonight
├── demo_clips/                  # generated genuine + cloned WAVs (gitignored if large)
├── artifacts/                   # model.pth, metrics.json (gitignored; go to Drive)
└── .gitignore
```
`.gitignore` must exclude `artifacts/*.pth`, `demo_clips/*.wav`, `__pycache__`, `*.ckpt`. *Why: git is for code; large binaries go to Drive.*
**Done when:** repo pushed to GitHub with this structure and a README stating the project + the "train-in-cloud, run-on-Mac" rule.

---

### Step 2 — Author the TWO requirements files
**What it is:** two separate pinned dependency lists.
**Why here — critical:** the versions that work on Colab/CUDA are **not** the same as what installs on Apple Silicon/MPS. Dependency conflicts are the #1 hackathon time-sink. Having two tested files kills that risk before it starts. Also: prefer **HuggingFace `transformers`** for wav2vec2 over `fairseq`, which is painful to build on Mac.

**`requirements_cuda.txt` (cloud training):**
```
torch
torchaudio
transformers
datasets
librosa
soundfile
scikit-learn
numpy
pandas
tqdm
TTS                # Coqui XTTS for clip generation
```

**`requirements_mac.txt` (Mac inference + app):**
```
torch              # official wheels support MPS on Apple Silicon
torchaudio
transformers
librosa
soundfile
scikit-learn
numpy
fastapi
uvicorn
websockets
python-multipart
```
> Note for Claude Code: install with the **latest stable** torch/torchaudio first, run the smoke test, and only pin exact versions once they're confirmed working on each platform. On the Mac, do NOT add `fairseq`, `pyaudio`, or `portaudio` — we capture mic audio in the browser instead (see Step 9).
**Done when:** `pip install -r requirements_cuda.txt` succeeds in the Kaggle/Colab runtime, and `requirements_mac.txt` is ready for the teammate to test tonight if reachable.

---

### Step 3 — Device-agnostic config + helper
**What it is:** one function that picks CUDA → MPS → CPU, used everywhere.
**Why here:** we run the *same code* on a CUDA cloud box tonight and an MPS Mac tomorrow. Any hardcoded `.cuda()` will crash on the Mac. This single helper makes the code portable.

**`src/device.py`:**
```python
import os, torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported ops fall back to CPU instead of crashing

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```
Everywhere in the code, use `device = get_device()` and `.to(device)` — never `.cuda()`.
**Done when:** importing and printing `get_device()` returns `cuda` on Kaggle and (verified later) `mps` on the Mac.

---

### Step 4 — Wire data access (no local download)
**What it is:** point the code at Kaggle-attached datasets.
**Why here:** ASVspoof is large; attaching Kaggle datasets means the data sits in `/kaggle/input/...` with zero download, so our disk stays free and training starts immediately.

**Do this:** in the Kaggle notebook, attach the datasets, then in `src/dataset.py` read file lists + protocol/label files from `/kaggle/input/<dataset-name>/...`. Build a `Dataset` that: loads a WAV (`soundfile`/`torchaudio`), resamples to **16 kHz** (wav2vec2's required rate), trims/pads to a fixed length (e.g., 4 s), and returns `(waveform, label)` where `label = 0 bonafide / 1 spoof`.
**Done when:** a `DataLoader` yields a batch of `(waveform, label)` tensors from the attached dataset with correct label mapping (verify a few by hand — ASVspoof protocol files list `bonafide`/`spoof` per utterance).

---

### Step 5 — Model + training script (run in the cloud)
**What it is:** SSL front-end + classifier head, and the loop that fine-tunes it.
**Why here:** this produces the checkpoint that IS tonight's main deliverable. We use a **pretrained self-supervised front-end** because training speech models from scratch is impossible in this timeframe and SSL features generalise far better to unseen attacks — the field's core challenge.

**Recommended pragmatic path (most reliable to finish tonight + installs cleanly on Mac):**
- Front-end: `Wav2Vec2Model` from `facebook/wav2vec2-xls-r-300m` (multilingual → gives us the Indian-accent robustness almost for free) **or** the lighter `facebook/wav2vec2-base` if compute/time is tight.
- Head: attentive/mean pooling → small MLP → 2-class output. (Optional upgrade: an AASIST head from an open repo if the core is done early — treat as stretch.)

**`src/model.py` shape:**
```python
import torch, torch.nn as nn
from transformers import Wav2Vec2Model

class VoiceGuardNet(nn.Module):
    def __init__(self, ssl_name="facebook/wav2vec2-xls-r-300m", freeze_ssl=True):
        super().__init__()
        self.ssl = Wav2Vec2Model.from_pretrained(ssl_name)
        if freeze_ssl:                          # freeze to train fast + avoid overfitting
            for p in self.ssl.parameters(): p.requires_grad = False
        h = self.ssl.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(self, x):                        # x: (B, T) waveform @16kHz
        feats = self.ssl(x).last_hidden_state    # (B, L, H)
        pooled = feats.mean(dim=1)               # simple mean pool
        return self.head(pooled)                 # (B, 2) logits
```
**Training (`src/train.py`) essentials:**
- Loss: `CrossEntropyLoss` (add class weights if bonafide/spoof imbalanced).
- Optimizer: `AdamW`, lr ~1e-4 on the head (freeze SSL first; unfreeze last layers only if time allows).
- **Keep it short:** 2–4 epochs on a subset is enough for a demo-grade model — the goal tonight is a *working, saved* model, not SOTA.
- Save `artifacts/model.pth` (`torch.save(model.state_dict(), ...)`) + the SSL name used.
**Done when:** `model.pth` is saved and training/val loss decreased sensibly.

---

### Step 6 — Cross-dataset evaluation → EER
**What it is:** score the model on a dataset of **attacks it never trained on** and compute Equal Error Rate.
**Why here:** anyone can score well on their own training data. Reporting EER on *unseen* attacks (ASVspoof 2021 DF or In-the-Wild) is the credibility metric judges and AICTE care about, and it's a headline number for the pitch. It also honestly tells us how much to trust the model.

**`src/eval_eer.py`:** run inference over the unseen set → collect spoof-probabilities + true labels → compute EER:
```python
import numpy as np
from sklearn.metrics import roc_curve
def compute_eer(labels, scores):     # scores = P(spoof)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2
```
Write the EER + dataset name to `artifacts/metrics.json`.
**Done when:** `metrics.json` contains an EER number on a clearly-labeled unseen set.

---

### Step 7 — Streaming inference module (the thing the Mac runs tomorrow)
**What it is:** load the checkpoint and score audio in **2–4 s sliding windows** with a smoothed running risk score.
**Why here:** this is the real-time engine behind the demo, and it must run on the Mac's MPS/CPU with low latency. Building it tonight (and testing on CPU) proves it'll work tomorrow.

**`src/infer.py` shape:**
```python
import torch, torch.nn.functional as F, numpy as np
from .device import get_device
from .model import VoiceGuardNet

class RiskEngine:
    def __init__(self, ckpt, ssl_name, win=4.0, hop=1.0, sr=16000, ema=0.6):
        self.device = get_device()
        self.model = VoiceGuardNet(ssl_name).to(self.device).eval()
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.win, self.hop, self.sr, self.ema = int(win*sr), int(hop*sr), sr, ema
        self.score = 0.0
    @torch.no_grad()
    def push(self, waveform):                       # waveform: 1-D np array @16kHz
        x = torch.tensor(waveform, dtype=torch.float32, device=self.device).unsqueeze(0)
        p_spoof = F.softmax(self.model(x), dim=-1)[0, 1].item()
        self.score = self.ema*self.score + (1-self.ema)*p_spoof   # smooth running score
        return round(self.score*100)                # live risk 0–100
```
Add a small CLI that streams a WAV through `push()` in hops and prints the risk trace. **Verify latency per window is < ~1 s on CPU.**
**Done when:** running a genuine clip keeps risk low and a cloned clip drives it high, on CPU.

---

### Step 8 — Generate the demo clip pack (cloud, tonight)
**What it is:** produce all genuine + AI-cloned audio pairs we'll demo with.
**Why here:** running voice cloning (Coqui XTTS) on Apple Silicon tomorrow is exactly the kind of setup that eats an afternoon — so we do it **once, in the cloud, tonight**, and ship the WAVs. This also removes any dependency on live-cloning working in the demo room.

**Do this (in the Kaggle/Colab notebook):**
```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
# clone the teammate's voice from their reference sample:
tts.tts_to_file(text="Hi, please transfer two lakh rupees to this account urgently.",
                speaker_wav="teammate_ref.wav", language="en",
                file_path="demo_clips/fraud_en.wav")
tts.tts_to_file(text="नमस्ते, कृपया इस खाते में तुरंत दो लाख रुपये भेजिए।",
                speaker_wav="teammate_ref.wav", language="hi",
                file_path="demo_clips/fraud_hi.wav")
```
Produce: 2–3 **genuine** recordings, matching **cloned** versions (English + Hindi), and ideally one clip from a **second TTS engine** (e.g., OpenVoice) to prove detection of an attack type not trained on.
**Done when:** `demo_clips/` holds labeled genuine + cloned WAVs and each has been sanity-checked through `RiskEngine` (genuine → low, cloned → high).

---

### Step 9 — FastAPI skeleton + API contract
**What it is:** the backend endpoints and the message format the frontend will use.
**Why here:** locking the API tonight means tomorrow the full-stack member builds the React UI against a real contract from minute one, in parallel with model integration. Mic audio is captured **in the browser** (Web Audio API) and streamed over WebSocket — this deliberately avoids needing `pyaudio`/`portaudio` on the Mac.

**`backend/app.py` shape:**
```python
from fastapi import FastAPI, WebSocket, UploadFile
app = FastAPI()

# GET /config  -> thresholds;  POST /config -> update
# POST /analyze (file)  -> {risk, verdict, per_window[]}
# WS  /stream  -> client sends 16kHz PCM chunks; server replies {risk, alert} per hop

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    # init RiskEngine once; on each received chunk -> engine.push() -> send {risk, alert}
```
Define the JSON contract explicitly in the README: chunk format (16 kHz mono PCM), and the reply `{ "risk": 0-100, "alert": bool, "reason": "..." }`. Add the **pre-transaction warning** rule: if `risk >= threshold`, `alert=true`, `reason="Likely synthetic voice — recommend call-back / MFA before approving."`
**Done when:** the server starts, `/analyze` returns a risk for an uploaded clip, and the WebSocket echoes risk for streamed chunks (test with a short script).

---

### Step 10 — Assemble + upload the handoff bundle
**What it is:** put everything the Mac needs into GitHub (code) + Google Drive (binaries).
**Why here:** this is the bridge between tonight and tomorrow. If this is clean, tomorrow starts with `git clone` + a Drive download and we're building.

**GitHub (push):** all of `src/`, `backend/`, `config.py`, both `requirements_*.txt`, `notebooks/`, README with the API contract + EER number.
**Google Drive (upload, ~2–4 GB):**
- `artifacts/model.pth` (+ note the exact `ssl_name` string used)
- `artifacts/metrics.json`
- `demo_clips/` (all WAVs)
- optionally a small unseen-eval subset (~1 GB) for a live-eval slide
**Done when:** a teammate could, from a clean machine, `git clone` + download the Drive folder and run `infer.py` on a demo clip successfully.

---

## Part C — The handoff bundle (exact contents)

| Item | Goes to | Approx size | Purpose |
|---|---|---|---|
| All code (`src`, `backend`, config, reqs, README) | GitHub | small | Cloned on the Mac tomorrow |
| `model.pth` + SSL name | Drive | ~0.3–2 GB | The trained detector |
| `metrics.json` (EER) | Drive/GitHub | tiny | Pitch headline + trust |
| `demo_clips/` genuine+cloned WAVs | Drive | <200 MB | The live demo, no cloning needed tomorrow |
| Small unseen-eval subset (optional) | Drive | ~1 GB | Live-evaluation slide |
| Two `requirements_*.txt` | GitHub | tiny | No dependency hell tomorrow |

**Total the Mac pulls: ~2–4 GB** — not the 150 GB of raw datasets. The datasets stay in the cloud.

---

## Part D — Tomorrow-morning readiness checklist (both laptops)

- [ ] Mac: `git clone` the repo; `pip install -r requirements_mac.txt`; confirm `get_device()` returns `mps`.
- [ ] Mac: download the Drive bundle into `artifacts/` and `demo_clips/`.
- [ ] Mac: run `infer.py` on a genuine + a cloned clip → confirm low vs high risk. **This proves the whole handoff before you build anything.**
- [ ] Start backend (`uvicorn backend.app:app`) and confirm `/analyze` works.
- [ ] Full-stack: begin the React + Web Audio UI against the locked API contract.
- [ ] If any more training is needed → back to the cloud, never the Mac.

---

## Part E — Traps to avoid (do NOT do these)

- ❌ **Training on the MacBook Air.** No CUDA, fanless throttling — it will be painfully slow and may not finish. Cloud only.
- ❌ **`.cuda()` anywhere.** Use `get_device()`. A single hardcoded CUDA call crashes the Mac.
- ❌ **`fairseq` / `pyaudio` on the Mac.** Use HF `transformers` for the model and browser Web Audio for the mic.
- ❌ **Chasing SOTA tonight.** A working, saved, honestly-evaluated model beats a perfect one that isn't finished. 2–4 epochs is fine.
- ❌ **Training only on clean ASVspoof 2019.** It won't generalise; keep the unseen-set EER honest and mention robustness as ongoing work.
- ❌ **Leaving voice cloning for tomorrow.** Generate every demo clip tonight in the cloud.
- ❌ **Large binaries in git.** Checkpoints and WAVs go to Drive; git stays lean.

---

*Tonight's job in one line: come out of it with a trained, evaluated `model.pth`, a pack of genuine+cloned demo clips, and a device-agnostic inference module + API skeleton — all in GitHub + Drive — so tomorrow the team only integrates and polishes.*
