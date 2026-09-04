function formatTime(ms) {
  const d = new Date(ms);
  return d.toLocaleTimeString("en-GB", { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

export default function AlertLog({ entries }) {
  return (
    <div>
      <div className="label" style={{ marginBottom: 8 }}>
        Alert log
      </div>
      <div className="alert-log">
        {entries.length === 0 ? (
          <div className="empty">No events yet -- start a call to begin logging.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Risk</th>
                <th>Alert</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr key={e.t} className={e.alert ? "alert" : undefined}>
                  <td>{formatTime(e.t)}</td>
                  <td>{e.risk}</td>
                  <td>{e.alert ? "TRUE" : "false"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
