import VerdictBadge from "./VerdictBadge.jsx";

export default function RiskMeter({ risk, alert, verdict }) {
  return (
    <div className="risk-block">
      <div className="label">Risk score</div>
      <div className={`risk-number${alert ? " alert" : ""}`}>
        {risk}
        <span className="risk-scale">/100</span>
      </div>
      <div className="verdict-row">
        <VerdictBadge alert={alert} verdict={verdict} />
      </div>
    </div>
  );
}
