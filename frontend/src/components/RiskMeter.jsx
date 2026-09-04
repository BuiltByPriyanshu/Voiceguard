import VerdictBadge from "./VerdictBadge.jsx";

// Two distinct outputs, deliberately not the same number: interactionRisk
// ("how risky is it to trust this interaction") is the big primary number
// and what should gate a sensitive action; voiceAuthenticity ("how
// suspicious is the voice itself") is a secondary readout beside it. A
// call can be low on voice authenticity alone but still high on
// interaction risk once context is factored in -- see src/risk_fusion.py.
export default function RiskMeter({ interactionRisk, voiceAuthenticity, alert, verdict, decision }) {
  return (
    <div className="risk-block">
      <div className="label">Interaction risk</div>
      <div className={`risk-number${alert ? " alert" : ""}`}>
        {interactionRisk}
        <span className="risk-scale">/100</span>
      </div>
      <div className="verdict-row">
        <VerdictBadge alert={alert} verdict={verdict} />
      </div>
      <div className="decision-band">{decision?.action}</div>
      <div className="voice-authenticity-row">
        <span className="label">Voice authenticity</span>
        <span className="voice-authenticity-value">{voiceAuthenticity}</span>
      </div>
    </div>
  );
}
