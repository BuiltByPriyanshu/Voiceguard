// The primary call-to-action, per the approved Swiss/premium design canvas:
// a large click-to-speak control, not a small toolbar button. Drives both
// source modes -- live mic and demo-clip playback -- through the identical
// three visual states (idle / listening / alert) so the control means the
// same thing regardless of what's actually streaming underneath.
export default function MicButton({ streaming, alert, label, onClick }) {
  const state = !streaming ? "idle" : alert ? "alert" : "listening";
  const stateLabel = state === "idle" ? "Tap to speak" : "Listening...";

  return (
    <div className="mic-cell">
      <button
        type="button"
        className={`mic-wrap ${state}`}
        onClick={onClick}
        aria-label={state === "idle" ? "Start" : "Stop"}
      >
        {state !== "idle" && (
          <>
            <span className="mic-ring" />
            <span className="mic-ring d2" />
          </>
        )}
        <span className="mic-btn">
          <svg viewBox="0 0 24 24">
            <rect x="9" y="2" width="6" height="12" rx="3"></rect>
            <path d="M5 11a7 7 0 0 0 14 0"></path>
            <line x1="12" y1="18" x2="12" y2="22"></line>
            <line x1="8" y1="22" x2="16" y2="22"></line>
          </svg>
        </span>
      </button>
      <div className={`mic-label${state === "alert" ? " alert" : ""}`}>
        {label ?? stateLabel}
      </div>
    </div>
  );
}
