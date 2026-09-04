const LABELS = {
  idle: "Idle",
  connecting: "Connecting",
  open: "Connected",
  closed: "Disconnected",
  error: "Connection error",
};

export default function ConnectionStatus({ connection }) {
  return (
    <div className="conn-status">
      <span className={`status-dot${connection === "open" ? " connected" : ""}${connection === "error" ? " alert" : ""}`} />
      WS {LABELS[connection] ?? connection}
    </div>
  );
}
