import { useState } from "react";

// Demo clips are served statically from /demo_clips (see public/demo_clips
// symlink into the repo's top-level demo_clips/ dir -- single source of
// truth, no duplicated/gitignored binaries inside frontend/).
const DEMO_CLIPS = [
  { file: "teammate_ref.wav", label: "teammate_ref.wav (genuine)" },
  { file: "fraud_en.wav", label: "fraud_en.wav (cloned, EN)" },
  { file: "fraud_hi.wav", label: "fraud_hi.wav (cloned, HI)" },
];

export default function SourceToggle({ streaming, onStartMic, onStartClip, onStop }) {
  const [mode, setMode] = useState("mic");
  const [clip, setClip] = useState(DEMO_CLIPS[0].file);

  const handleStart = () => {
    if (mode === "mic") {
      onStartMic();
    } else {
      onStartClip(`/demo_clips/${clip}`);
    }
  };

  return (
    <div className="source-panel">
      <div className="label">Source</div>
      <div className="toggle-group">
        <button
          className={`toggle-btn${mode === "mic" ? " active" : ""}`}
          onClick={() => setMode("mic")}
          disabled={streaming}
        >
          Live mic
        </button>
        <button
          className={`toggle-btn${mode === "clip" ? " active" : ""}`}
          onClick={() => setMode("clip")}
          disabled={streaming}
        >
          Demo clip
        </button>
      </div>

      {mode === "clip" && (
        <select
          className="clip-select"
          value={clip}
          onChange={(e) => setClip(e.target.value)}
          disabled={streaming}
        >
          {DEMO_CLIPS.map((c) => (
            <option key={c.file} value={c.file}>
              {c.label}
            </option>
          ))}
        </select>
      )}

      {streaming ? (
        <button className="action-btn secondary" onClick={onStop}>
          Disconnect
        </button>
      ) : (
        <button className="action-btn" onClick={handleStart}>
          {mode === "mic" ? "Connect call" : "Play clip"}
        </button>
      )}
    </div>
  );
}
