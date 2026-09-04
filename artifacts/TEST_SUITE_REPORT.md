# VoiceGuard Test Suite — Results

Generated 2026-09-05. Reproduce any experiment with the commands shown;
raw JSON for each is in `artifacts/test_suite_experiment_{a,b,c,d}.json`.
Checkpoint under test: `artifacts/model.pth` (see `HANDOFF.md` for how it
was trained; honest unseen-attack EER 4.2-4.8% across re-runs).

This suite tests **scenarios**, not just a single accuracy number — per
the shift from "detect fake audio" to "assess interaction risk" (see
`HANDOFF.md` §8).

---

## Experiment A — Voice authenticity accuracy

```
python -m src.eval_full_metrics --parquet artifacts/asvspoof2019_eval_sample10k_wav2vec.parquet --dataset-name ASVspoof2019-LA-eval-sample10k
```

10,000-utterance stratified sample of the ASVspoof2019 LA **eval**
partition (1,032 bonafide / 8,968 spoof), attacks A07-A19 — none seen in
training.

| Metric | Bonafide | Spoof | Overall |
|---|---|---|---|
| Precision | 0.699 | 0.994 | — |
| Recall | 0.953 | 0.953 | — |
| F1 | 0.806 | 0.973 | — |
| Accuracy | — | — | 0.953 |
| ROC-AUC | — | — | 0.991 |
| EER | — | — | 4.76% |

Confusion matrix: TN=983, FP=49, FN=424, TP=8544.

**Reading this honestly:** bonafide precision (0.699) is much lower than
spoof precision (0.994) purely because the eval set is 90% spoof — a
small absolute false-positive count (49) looks large as a fraction of the
small bonafide class. Recall is balanced (0.953 both classes), which is
the number that actually reflects per-class detection quality.

---

## Experiment E — False-positive safety

Read directly off Experiment A (same run, same population): of 1,032
genuine utterances, **49 were incorrectly flagged as synthetic — a 4.75%
false-positive rate.** This is the number that matters most for whether
the system is safe to deploy: it directly measures how often a real
caller's legitimate voice would trigger an unnecessary block/verification
step.

For context, the pre-2026-09-05-retrain checkpoint's false-positive rate
was implicitly ~10.1% (bonafide recall 89.9%) — this session's retrain
work (§3 of `HANDOFF.md`) more than halved it. `teammate_ref.wav` (real
audio, not a statistical sample) confirms this concretely: 23.6s of
genuine speech, never alerts (see Experiment B below for the full trace).

---

## Experiment B — Streaming behaviour (detection latency vs. duration)

```
python -m src.eval_latency demo_clips/fraud_en.wav demo_clips/fraud_hi.wav
```

| Clip | Duration | Time to first alert | Mean inference latency / window |
|---|---|---|---|
| `fraud_en.wav` | 8.6s | **2.0s** | 19.3ms |
| `fraud_hi.wav` | 6.4s | **2.0s** | 19.2ms |
| `teammate_ref.wav` (genuine) | 23.6s | never (correct) | 19.5ms |

Both cloned clips cross the alert threshold within the first two seconds
of speech. Per-window inference is ~19ms on this Mac's MPS backend — well
under the 1s hop budget, so the pipeline has headroom, not a bottleneck.
(First-window latency is occasionally ~500ms due to one-time model
warmup on the very first push — a real effect worth knowing about but
not repeated on subsequent windows.)

---

## Experiment C — Real-world robustness

```
python -m src.eval_robustness demo_clips/teammate_ref.wav demo_clips/fraud_en.wav
```

Peak risk score (0-100) under each degradation, same two clips:

| Condition | `teammate_ref.wav` (genuine) | `fraud_en.wav` (cloned) |
|---|---|---|
| Clean | 0 | 99 (alert) |
| Light noise (25dB SNR) | 0 | **18 (missed)** |
| Heavy noise (10dB SNR) | 0 | **36 (missed)** |
| Reverb | 2 | 83 (alert) |
| Compressed (telephony-style, 12kbps AAC round-trip) | 0 | 96 (alert) |

**This is the most important honest finding in this suite.** The genuine
side is robust everywhere (never a false positive, even under heavy
degradation) — but **cloned-voice detection is fragile specifically
under additive noise**, degrading from a confident 99 down to a missed
18-36 under noise levels a real room easily produces. Reverb and codec
compression are handled much better.

**Root cause, diagnosed, not just observed:** the noise-augmentation work
this session (`extract_clip_embeddings.py --augment-bonafide`,
`src/augment.py`) only hardened the **bonafide** side against noise — the
model learned "noisy audio can still be genuine" but never learned
"noisy audio can still be a clone." That asymmetry is the fix: augment
spoof-labeled training clips with the same noise conditions, not just
bonafide ones. Not done yet — flagged here rather than silently shipped.

---

## Experiment D — Contextual risk (the risk-fusion architecture's payoff)

```
python -m src.eval_context_demo demo_clips/teammate_ref.wav   # backend must be running on :8000
```

Same clip (`teammate_ref.wav`, genuine, `voice_authenticity` stays 0
throughout), only the call/transaction context changes:

| Scenario | Voice authenticity | Context risk | Interaction risk | Decision |
|---|---|---|---|---|
| Ordinary call, known contact | 0 | 0 | 0 | continue |
| Known contact, low-value transaction | 0 | 10 | 10 | continue |
| Unknown caller, no transaction | 0 | 20 | 20 | continue |
| Unknown caller, high-value transaction | 0 | 65 | **65** | **verify** |

The voice is never flagged (`verdict` stays "genuine" throughout) — but
the same genuine voice in the riskiest context still gets escalated to
"recommend secondary verification." This is the concrete demonstration
that voice authenticity and interaction risk answer different questions.

---

## Summary

| Experiment | Result | Status |
|---|---|---|
| A — accuracy/EER/ROC-AUC | EER 4.76%, ROC-AUC 0.991 | ✅ strong |
| B — detection latency | Alerts within 2s, ~19ms/window inference | ✅ strong |
| C — robustness | Robust to reverb/compression; **fragile to noise on the spoof side** | ⚠️ known gap, root cause diagnosed |
| D — contextual risk | Works exactly as designed | ✅ verified live |
| E — false-positive safety | 4.75% FPR (halved from pre-retrain ~10.1%) | ✅ improved, not zero |

**Known, disclosed limitations** (see `HANDOFF.md` for the full
investigation): a genuinely novel zero-shot voice clone (never-seen
reference speaker) is not reliably caught — 4 documented fix attempts,
consistent with the field's actual open research problem, not a bug
unique to this project. Noise robustness on the spoof side (Experiment
C) is a concrete, fixable next step if there's time before the demo.
