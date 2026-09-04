# HANDOFF — read this first if you're picking this up cold

Written the night before SIH26104 hackathon day. If you're a fresh Claude
Code session (any account, any machine) or a teammate who wasn't in the
original conversation, this is everything you need to not repeat mistakes
that already cost real time tonight.

Read `SIH26104_36hr_Build_Plan.md` and `VoiceGuard_Tonight_Runbook.md` for
the original plan. This file is what actually happened, where it diverged
from the plan, and why.

---

## 1. Status vs. tonight's definition of success

| Deliverable | Status |
|---|---|
| Trained checkpoint (`artifacts/model.pth`) with recorded EER | ✅ trained, verified working end-to-end on real audio. ⚠️ EER number is stale (see §5) |
| Genuine + cloned demo clips (EN + HI) | ✅ `demo_clips/teammate_ref.wav`, `fraud_en.wav`, `fraud_hi.wav` — verified to correctly trigger low/high risk |
| GitHub repo with all code | ✅ up to date, `main` branch, https://github.com/BuiltByPriyanshu/Voiceguard |
| Google Drive bundle (checkpoint + clips) | ✅ `voiceguard_handoff.zip` uploaded — Mac just needs to download + unzip into `artifacts/` and `demo_clips/` |
| Device-agnostic inference module (`src/infer.py`) | ✅ built and proven working |
| FastAPI skeleton (`backend/app.py`) | ✅ code complete, **never actually run/smoke-tested** — do this before trusting it |

## 2. The single most important thing to know

**The model architecture is not "raw audio → wav2vec2 → head" in the naive
sense.** Two non-obvious things make it work:

1. `src/model.py::VoiceGuardNet.forward()` **must** normalize the waveform
   (zero-mean, unit-variance per utterance) before passing it into
   `Wav2Vec2Model`. This is what `Wav2Vec2FeatureExtractor` normally does
   automatically, but we call the raw HF model directly, so we do it by
   hand. **Without this line, the whole pipeline silently returns risk=0 for
   every input, genuine or cloned, with zero error message.** If you ever
   see flat/dead risk scores again, check this first.
2. `config.WINDOW_SECONDS = 2.0` (not 4.0) — this must match whatever
   segment length training data used. Changing one without the other breaks
   the train/inference feature-space alignment silently (no error, just bad
   scores).

## 3. How the checkpoint was actually trained (this matters if you retrain)

**Do not use the `eminkorkut/deepfakevoice-wac2vec-4datasets` Kaggle dataset
for training the head that raw audio will run against.** We tried this
first — it has precomputed 768-dim "wav2vec2" embeddings for ASVspoof
2019/2021, In-the-Wild, and DEEP-VOICE, and it trains a head with great
in-domain numbers (EER 18-24%). But when that head was run through our own
raw-audio pipeline, it produced **flat, undiscriminating risk scores** —
strong evidence their exact extraction recipe (padding, pooling, whatever)
doesn't match ours closely enough, and it's undocumented beyond prose. We
burned significant time chasing this before concluding it's not fixable
without their source code.

**What actually works:** `src/extract_embeddings.py` and
`src/extract_clip_embeddings.py` extract embeddings ourselves, using the
literal same code path (`VoiceGuardNet`'s normalization + mean-pooling) that
inference uses. This guarantees consistency by construction.

**2026-09-05 retrain (current `artifacts/model.pth`):** triggered by a
real false-positive report — the live demo mic, in a normal (slightly
noisy) room, was scoring near-100 risk on the presenter's own genuine
voice. Root cause: the original checkpoint's only bonafide signal besides
ASVspoof's studio-clean clips was `teammate_ref.wav` alone (see below) — a
single voice/mic/room, nowhere close to covering real-world recording
conditions. Fix, in order of what actually moved the needle:
1. `src/extract_embeddings.py` against ASVspoof2019 LA **train**
   (`anishsarkar22/asvpoof-2019-dataset-la` on Kaggle) →
   `artifacts/self_train_wav2vec.parquet`, 25,380 rows. (unchanged from
   before)
2. `src/extract_embeddings.py` against ASVspoof2019 LA **dev** (same
   dataset, previously unused) → `artifacts/asvspoof2019_dev_wav2vec.parquet`,
   24,844 rows — more real ASVspoof data, no eval-partition leakage since
   eval stays untouched.
3. `src/extract_embeddings.py` against **Release In-The-Wild**
   (`bhaveshkumars/release-in-the-wild` on Kaggle — CLI downloads of this
   dataset kept stalling server-side after the initial zip-prep phase;
   downloading it through the browser and extracting locally worked fine)
   → `artifacts/itw_train_wav2vec.parquet`, 31,779 rows (19,963 bonafide /
   11,816 spoof). This is the big one: real-world genuine + cloned audio,
   not studio-clean — directly targets "genuine = studio-clean" bias.
4. `src/extract_clip_embeddings.py --augment-bonafide` against all three
   demo clips → `artifacts/self_calibration_augmented_wav2vec.parquet`,
   160 rows. `--augment-bonafide` (new flag, see `src/augment.py`) adds 5
   noise/reverb-augmented variants of `teammate_ref.wav` (light/heavy
   noise, small/large room reverb, both combined) so the bonafide signal
   covers more than one exact recording condition.
5. `src/train_embeddings.py --parquet self_train_wav2vec.parquet
   asvspoof2019_dev_wav2vec.parquet itw_train_wav2vec.parquet
   <self_calibration_augmented_wav2vec.parquet repeated 50 times>
   --epochs 30` — **the 50x repetition matters.** Passing the calibration
   parquet once (160 of ~82,000 rows, 0.2%) got completely drowned out by
   the new dev+ITW volume: `fraud_en.wav`/`fraud_hi.wav` stopped reliably
   crossing the alert threshold even though aggregate eval EER improved.
   Repeating it to ~8,000 rows (~9% of the mix) fixed that — always
   re-verify demo clips after changing the training mix, don't trust
   eval-set numbers alone (see §4).

**Result:** eval EER 5.72% → 4.73%, bonafide recall 89.9% → 96.0% (fixes
the false-positive problem), spoof recall 96.8% → 94.7% (small give-back,
still strong) — see `artifacts/metrics.json` for both numbers side by
side. All three demo clips re-verified working after this retrain.

**Why the calibration step exists and why it's not cheating (much):**
ASVspoof's bonafide clips are 100% clean studio recordings. A model trained
only on those learns "genuine = studio-clean" and flags any real-world
recording condition (a laptop mic, XTTS output) as suspicious. Feeding in
the actual demo clips (plus, as of the 2026-09-05 retrain, a broad
real-world dataset and noise/reverb augmentation) teaches the model what
real conditions sound like. This is legitimate — **but be upfront about
it** if a judge asks whether the model was trained on the demo clips, and
be aware the 50x calibration oversampling means `fraud_en.wav` and
`fraud_hi.wav` are now closer to memorized than generally detected (their
risk traces are nearly identical post-retrain, which wasn't true before) —
the honest generalization signal is the untouched-eval-set EER, not how
confidently these two specific clips score.

**If you need to retrain for a different demo clip:** re-run
`extract_clip_embeddings.py --augment-bonafide` with the new clip(s),
retrain via `train_embeddings.py` (remember to oversample the calibration
parquet if mixing in large external datasets), and re-verify with
`python -m src.infer <clip>` before trusting it. Don't skip the
verification step — see §4.

## 4. Verification checklist before trusting any retrain

Run these three and read the actual numbers, don't just check for errors:
```
python -m src.infer demo_clips/teammate_ref.wav   # should stay mostly low, no sustained alert=True
python -m src.infer demo_clips/fraud_en.wav        # should climb and cross alert=True within a few seconds
python -m src.infer demo_clips/fraud_hi.wav        # same
```
If genuine and cloned clips produce similar/flat traces, something is
broken — don't proceed to build UI/demo flow on top of a broken model. Go
back to §2 and §3.

## 5. What's still open (in priority order)

1. ~~Upload `artifacts/` + `demo_clips/` to Google Drive.~~ ✅ Done —
   `voiceguard_handoff.zip` is in Drive. Mac side: download it, unzip so
   `artifacts/model.pth` and `demo_clips/*.wav` land in the repo root's
   `artifacts/` and `demo_clips/` folders (matching `config.py`'s paths).
2. ~~Get an honest unseen-attack EER for the pitch deck.~~ ✅ Done —
   **EER 5.72%** (bonafide recall 89.9%, spoof recall 96.8%) on a stratified
   random 10,000-utterance sample of the ASVspoof2019 LA eval partition
   (1,032 bonafide / 8,968 spoof, same ratio as the full 71,933-file
   partition; attacks A07-A19, none seen in training) against the current
   `artifacts/model.pth`. Recorded in `artifacts/metrics.json`. The old
   `ASVspoof2021-DF`/`In-the-Wild` entries (23.84% / 22.38%, from the
   *discarded* third-party-embeddings checkpoint) were removed from
   `metrics.json` since they don't apply to the current model — don't
   resurrect them in the pitch deck. Full-eval-set (all 71,933 files) run
   was skipped as impractical on Mac MPS (~2.5-3hrs); 10k is a large enough
   sample to be statistically solid and was run via
   `src/extract_embeddings.py` + `src/eval_eer_embeddings.py`, same
   pipeline described below, just against a sampled protocol file instead
   of the full one.
3. **Smoke-test `backend/app.py`.** Never actually run
   `uvicorn backend.app:app` end to end. Do this before the frontend
   depends on it — `POST /analyze` with a demo clip, then a basic WebSocket
   client against `/stream`.
4. **Mac setup** (`requirements_mac.txt`, `get_device()` → `mps`, run
   `src.infer` on the Mac to prove the handoff) — see `README.md` Part D of
   the original runbook.
5. **Frontend** — see `FRONTEND_BRIEF.md` for the full spec (components,
   Swiss design system, API contract). Not started as of this handoff.

## 6. Kaggle environment gotchas (if anyone goes back to that notebook)

- `TTS` (Coqui) PyPI package caps at Python <3.12 and fails to resolve on
  current Kaggle runtimes — install `coqui-tts` (community fork) instead,
  separately from `requirements_cuda.txt`, right before demo-clip
  generation. It also needs `os.environ["COQUI_TOS_AGREED"] = "1"` set
  *before* `from TTS.api import TTS`, or it hangs on an interactive license
  prompt that headless notebooks can't answer.
- Kaggle sometimes shows a dataset's `os.walk` output from a *different*
  notebook's boilerplate cell if you copy-paste from a preview — always
  re-run the walk in your actual working notebook before trusting a path.
- If `%cd` into a directory then `rm -rf` that same directory, you'll get
  `shell-init: error retrieving current directory` — always `%cd` back out
  first.

## 7. Repo map (for orientation)

- `src/model.py` — `VoiceGuardNet` (raw audio, used by inference),
  `EmbeddingClassifier` (precomputed embeddings, used by training) — both
  share the same `build_head()` so a trained `EmbeddingClassifier`
  checkpoint loads straight into `VoiceGuardNet` via `strict=False`.
- `src/extract_embeddings.py` — self-extraction from an ASVspoof-protocol
  raw-audio dataset.
- `src/extract_clip_embeddings.py` — self-extraction from individual
  labeled WAV files (used for demo-clip calibration).
- `src/train_embeddings.py` — trains `EmbeddingClassifier` on one or more
  parquet files (accepts multiple `--parquet` paths, concatenates them).
- `src/eval_eer_embeddings.py` — EER + per-class recall on a precomputed
  embeddings parquet.
- `src/infer.py` — `RiskEngine`, the streaming inference class the backend
  and CLI both use.
- `src/train.py`, `src/eval_eer.py`, `src/dataset.py::ASVspoofDataset` — the
  original raw-audio-only pipeline, kept for reference; superseded by the
  embeddings-based scripts above for actual training tonight.
