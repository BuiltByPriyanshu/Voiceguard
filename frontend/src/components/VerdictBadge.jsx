export default function VerdictBadge({ alert, verdict }) {
  return (
    <span className={`verdict-badge${alert ? " synthetic" : ""}`}>
      {verdict}
    </span>
  );
}
