import { useEffect, useState } from "react";
import { getConfig, setConfig } from "../api.js";

export default function ThresholdConfig() {
  const [threshold, setThreshold] = useState(70);
  const [highValue, setHighValue] = useState(false);
  const [defaultThreshold, setDefaultThreshold] = useState(70);
  const [highValueThreshold, setHighValueThreshold] = useState(50);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    getConfig()
      .then((cfg) => {
        setDefaultThreshold(cfg.threshold);
        setHighValueThreshold(cfg.high_value_threshold);
        setThreshold(cfg.threshold);
        setLoaded(true);
      })
      .catch(() => setLoaded(true));
  }, []);

  const commit = async (value) => {
    setThreshold(value);
    try {
      await setConfig(value);
    } catch {
      /* non-fatal for the demo -- local state still reflects the intent */
    }
  };

  const toggleHighValue = (checked) => {
    setHighValue(checked);
    commit(checked ? highValueThreshold : defaultThreshold);
  };

  if (!loaded) return null;

  return (
    <div>
      <div className="label" style={{ marginBottom: 8 }}>
        Alert threshold
      </div>
      <div className="threshold-row">
        <input
          type="range"
          min="0"
          max="100"
          value={threshold}
          onChange={(e) => commit(Number(e.target.value))}
        />
        <span className="threshold-value">{threshold}</span>
      </div>
      <label
        className="label"
        style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 16, cursor: "pointer" }}
      >
        <input type="checkbox" checked={highValue} onChange={(e) => toggleHighValue(e.target.checked)} />
        High-value transaction mode
      </label>
    </div>
  );
}
