# VoiceGuard — Frontend Brief

For whoever's building the React app. This covers what to build, the
components you need, the design language to use, and exactly how it talks to
the backend. The backend team has already built and tested the API this spec
integrates against — see `backend/app.py` and the contract below.

---

## 1. Tech stack

- **React** (Vite, not CRA — faster dev server)
- **Web Audio API** for mic capture — `AudioContext` + `MediaStreamAudioSourceNode`
  + an `AudioWorkletNode` (or `ScriptProcessorNode` if you need it working fast)
  to chunk mic audio into 16kHz mono PCM16 and send it over the WebSocket.
  Do **not** use any Node mic library — everything happens in-browser.
- **Native WebSocket** (`new WebSocket(...)`) for `/stream` — no need for
  socket.io, the backend is plain FastAPI WebSockets.
- **Canvas 2D** for the waveform (not an SVG/chart library — real-time audio
  needs raw canvas drawing for performance).
- No Redux / state library needed. React state + Context is enough for this scope.

---

## 2. The one screen that matters: Call Simulation View

This view **is** the demo. Everything else is secondary. It simulates a live
phone call and shows VoiceGuard's risk score updating in real time as audio
streams in.

**Layout (top to bottom or a clear left/right split):**

1. **Caller panel** — a mocked "incoming call" card: caller name/number,
   a call duration timer (mm:ss, counts up once "connected"), a
   connect/disconnect status dot.
2. **Risk meter** — the single biggest visual element on the page. A live
   number, 0–100, updating every ~1s (matches the backend's hop rate). Color
   state changes as it crosses thresholds (see design system below).
3. **Waveform** — real-time scrolling waveform of whatever audio is currently
   streaming (mic input or a played-back demo clip).
4. **Verdict badge** — small, next to the risk meter: "GENUINE" or "LIKELY
   SYNTHETIC", changes with `verdict` from the API.
5. **Alert log** — a scrolling list below the meter, one row per hop:
   timestamp, risk score, alert boolean. This is what proves to judges the
   system is actually running continuously, not just showing one static number.
6. **Pre-transaction warning modal** — appears (does not replace the page)
   the moment `alert: true` arrives. Shows the exact `reason` string from the
   API verbatim: *"Likely synthetic voice — recommend call-back / MFA before
   approving."* Include two buttons: **Call back** and **Approve anyway**
   (both can be no-ops for the demo — the point is showing the human-in-the-loop
   decision point, not building real banking logic).
7. **Source control** — for the live demo you need to switch between: (a) live
   mic input, (b) playing a cached demo WAV file (genuine or cloned) through
   the same pipeline. A simple toggle/tab between "Live mic" and "Play demo
   clip" (with a dropdown of the WAVs in `demo_clips/`) covers both.

---

## 3. Secondary screens (build only if time allows, in this priority order)

### 3a. Explainability panel
A collapsible panel/drawer off the main view. Shows a static spectrogram
image of the current/last clip with 1-2 text callouts (e.g. "flat pitch
contour", "spectral artifact at 3.2s"). This can be **mocked with a
pre-rendered image + hardcoded annotation** for the demo rather than computed
live — the prosody/explainability model (M3) is a stretch goal, not
guaranteed to exist by demo time. Design it so it still looks intentional
even if the content behind it is static.

### 3b. Upload / analyze view
A drag-and-drop zone that posts a file to `POST /analyze` and shows the same
risk meter + verdict for a single file, no streaming. This is the fallback
path if live mic capture breaks in the demo room — build it defensively, it
might save the presentation.

### 3c. Config panel
A minimal panel (could just be a corner of the main screen) with a threshold
slider bound to `GET`/`POST /config`, and a toggle for "high-value
transaction" mode (switches to the stricter threshold).

---

## 4. Component checklist

| Component | Purpose |
|---|---|
| `RiskMeter` | Big animated 0-100 gauge/number, color-coded by state |
| `WaveformCanvas` | Real-time scrolling waveform, canvas-based |
| `CallerPanel` | Mock incoming-call card + duration timer |
| `VerdictBadge` | Small pill: GENUINE / LIKELY SYNTHETIC |
| `AlertModal` | Pre-transaction warning popup, triggered by `alert:true` |
| `AlertLog` | Scrolling list of past risk events (time, score, alert) |
| `SourceToggle` | Switch between live mic and a cached demo clip |
| `ConnectionStatus` | WebSocket connected/reconnecting/disconnected indicator |
| `ExplainabilityPanel` | Spectrogram + annotation callouts (can be static) |
| `UploadDropzone` | Fallback file-upload analyze view |
| `ThresholdConfig` | Slider bound to `/config` |

---

## 5. Design language: Swiss / International Typographic Style

This is a security tool, not a consumer app — the visual language should say
"instrument panel," not "dashboard with gradients." Swiss style fits because
it's built entirely around **clarity, grid, and typographic hierarchy** — no
decoration, every element earns its place.

**Rules to actually follow:**

1. **Grid, visibly.** Lay the whole page out on a strict column grid (12
   columns is standard). Elements align to grid lines — no free-floating
   centered boxes. Asymmetric composition is fine and encouraged (the risk
   meter doesn't need to be dead-center); alignment to the grid is what
   matters, not symmetry.
2. **Type does the work, not color or icons.** One typeface family only —
   **Inter** or **Helvetica Neue** (both free/system-available; avoid
   anything with personality). Use weight and size contrast aggressively: the
   risk number should be enormous (think 120-180px), labels tiny (11-12px,
   uppercase, letter-spaced). This contrast *is* the visual hierarchy — you
   shouldn't need boxes or borders to separate sections, just type scale and
   whitespace.
3. **Color is functional, not decorative — and there's almost none of it.**
   Base palette is near-monochrome: off-white background (`#F5F5F0` or
   similar, not pure white), near-black text (`#111`), a mid-grey for
   secondary text/labels (`#666`). **Exactly one accent color exists: red**
   (`#E8362B` or similar), and it is reserved *exclusively* for the alert
   state — the risk meter, the alert modal border, the "LIKELY SYNTHETIC"
   badge. Nothing else on the page may use red. Resist the instinct to add a
   green "safe" color too — let the *absence* of red communicate safety; a
   plain black/grey meter in the low-risk state is more Swiss and more
   striking than a traffic-light green.
4. **No shadows, no gradients, no rounded skeuomorphism.** Flat fills, hard
   edges (0px or max 2px border radius), thin 1px hairline rules to separate
   sections instead of card shadows/backgrounds.
5. **Whitespace is structural.** Generous, consistent margins — don't fill
   empty space with filler content. A mostly-empty page with one huge number
   on it is correct Swiss design, not "unfinished."
6. **State changes are sharp, not bouncy.** When the risk meter crosses the
   alert threshold, transition it with a fast, linear or ease-out
   color/scale change (150-250ms) — no springy/bouncy easing, no confetti,
   nothing playful. The tone is clinical.

**Concrete numbers to start from:**
- Background: `#F5F4F0`
- Primary text / meter (safe state): `#111111`
- Secondary text / labels: `#6B6B6B`
- Alert red: `#E8362B`
- Hairline rule: `#D8D6CE`, 1px
- Font: Inter, weights 400/600/800 only
- Risk number: 800 weight, ~160px, tabular-nums
- Labels: 600 weight, 11-12px, uppercase, letter-spacing 0.08em

---

## 6. API contract (already built and tested on the backend)

Full reference: `backend/app.py` and the main `README.md`. Summary:

**`GET /config`** → `{"threshold": 70, "high_value_threshold": 50}`
**`POST /config`** with `{"threshold": int}` → updates it.

**`POST /analyze`** (multipart file upload) →
```json
{
  "risk": 0-100,
  "verdict": "genuine" | "synthetic",
  "alert": true | false,
  "reason": "Likely synthetic voice — recommend call-back / MFA before approving." | null,
  "per_window": [12, 15, 40, 78, ...]
}
```

**`WS /stream`** — client sends raw 16kHz mono PCM16 audio as **binary**
WebSocket frames (not JSON — send the raw bytes from your audio worklet
directly). Server replies with a **text** JSON message once per ~1s hop:
```json
{"risk": 78, "alert": true, "reason": "Likely synthetic voice — ..."}
```
There's no `verdict` or `per_window` field on the streaming reply — only on
`/analyze`. Bind `RiskMeter`/`VerdictBadge`/`AlertModal` state directly off
each incoming `/stream` message.

**Audio chunking detail:** the backend expects PCM16 mono at 16kHz. Convert
`Float32Array` samples from the Web Audio API to `Int16Array` before sending
(`sample * 32767`, clamped to `[-32768, 32767]`). Buffer to whatever chunk
size is convenient client-side — the backend accumulates bytes internally and
only scores once it has a full ~2s window (see `config.WINDOW_SECONDS`).

---

## 7. What "done" looks like for the demo

Minimum bar, matches the actual demo script the team will run tomorrow:
1. Live mic capture works, streams to `/stream`, risk meter updates in real time.
2. Switching to "play demo clip" and picking a cached genuine WAV shows risk
   staying low.
3. Switching to a cached cloned WAV shows risk climbing and the alert modal
   firing, with the exact `reason` text displayed.
4. The whole thing looks intentional even before the explainability/config
   panels exist — those are additive, not required for the core loop to read
   as "finished."

If mic capture proves flaky in testing, the **upload/analyze fallback (3b)**
becomes the safety net — prioritize it over the explainability panel if
you're short on time.
