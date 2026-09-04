function formatDuration(totalSeconds) {
  const m = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, "0");
  const s = (totalSeconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

export default function CallerPanel({ connected, alert, elapsed, callerName, callerNumber }) {
  return (
    <div className="caller-panel">
      <div className="label">Incoming call</div>
      <div className="status-row">
        <span
          className={`status-dot${alert ? " alert" : connected ? " connected" : ""}`}
        />
        <span className="label" style={{ color: "var(--text)" }}>
          {connected ? "Connected" : "Idle"}
        </span>
      </div>
      <div className="caller-name">{callerName}</div>
      <div className="caller-number">{callerNumber}</div>
      <div className="call-timer">{formatDuration(elapsed)}</div>
    </div>
  );
}
