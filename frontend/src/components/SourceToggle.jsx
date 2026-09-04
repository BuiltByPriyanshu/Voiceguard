// Demo clips are served statically from /demo_clips (see public/demo_clips
// symlink into the repo's top-level demo_clips/ dir -- single source of
// truth, no duplicated/gitignored binaries inside frontend/).
export const DEMO_CLIPS = [
  { file: "teammate_ref.wav", label: "teammate_ref.wav (genuine)" },
  { file: "fraud_en.wav", label: "fraud_en.wav (cloned, EN)" },
  { file: "fraud_hi.wav", label: "fraud_hi.wav (cloned, HI)" },
  { file: "fraud_user_clone.wav", label: "fraud_user_clone.wav (cloned, unseen speaker -- known miss)" },
];

// Purely a mode/clip selector -- the mic button is the actual trigger, so
// this only ever sets up what it will do when pressed.
export default function SourceToggle({ mode, setMode, clip, setClip, disabled }) {
  return (
    <div className="source-panel">
      <div className="label">Source</div>
      <div className="toggle-group">
        <button
          className={`toggle-btn${mode === "mic" ? " active" : ""}`}
          onClick={() => setMode("mic")}
          disabled={disabled}
        >
          Live mic
        </button>
        <button
          className={`toggle-btn${mode === "clip" ? " active" : ""}`}
          onClick={() => setMode("clip")}
          disabled={disabled}
        >
          Demo clip
        </button>
      </div>

      {mode === "clip" && (
        <select
          className="clip-select"
          value={clip}
          onChange={(e) => setClip(e.target.value)}
          disabled={disabled}
        >
          {DEMO_CLIPS.map((c) => (
            <option key={c.file} value={c.file}>
              {c.label}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
