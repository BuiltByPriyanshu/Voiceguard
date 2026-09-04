import { useEffect, useState } from "react";
import { getContext, setContext } from "../api.js";

const TXN_LEVELS = ["none", "low", "medium", "high"];

// Lets the demo show the multi-signal architecture's whole point live:
// the voice can stay genuine while interaction risk climbs purely from
// context (unknown caller + high-value transaction), independent of
// anything the voice authenticity model says.
export default function ContextPanel() {
  const [knownContact, setKnownContact] = useState(true);
  const [txnValue, setTxnValue] = useState("none");
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    getContext()
      .then((ctx) => {
        setKnownContact(ctx.known_contact);
        setTxnValue(ctx.transaction_value);
        setLoaded(true);
      })
      .catch(() => setLoaded(true));
  }, []);

  const updateKnownContact = (value) => {
    setKnownContact(value);
    setContext({ known_contact: value }).catch(() => {});
  };

  const updateTxnValue = (value) => {
    setTxnValue(value);
    setContext({ transaction_value: value }).catch(() => {});
  };

  if (!loaded) return null;

  return (
    <div>
      <div className="label" style={{ marginBottom: 8 }}>
        Call context
      </div>
      <div className="context-row">
        <span className="context-field-label">Caller</span>
        <div className="toggle-group">
          <button
            className={`toggle-btn${knownContact ? " active" : ""}`}
            onClick={() => updateKnownContact(true)}
          >
            Known
          </button>
          <button
            className={`toggle-btn${!knownContact ? " active" : ""}`}
            onClick={() => updateKnownContact(false)}
          >
            Unknown
          </button>
        </div>
      </div>
      <div className="context-row">
        <span className="context-field-label">Transaction</span>
        <div className="toggle-group">
          {TXN_LEVELS.map((level) => (
            <button
              key={level}
              className={`toggle-btn${txnValue === level ? " active" : ""}`}
              onClick={() => updateTxnValue(level)}
            >
              {level}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
