import { useRef, useState } from "react";
import { analyzeFile } from "../api.js";

// Fallback path if live mic capture breaks in the demo room -- posts a file
// straight to POST /analyze and shows the same risk/verdict, no streaming.
export default function UploadDropzone() {
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const runAnalyze = async (file) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const data = await analyzeFile(file);
      setResult(data);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="label" style={{ marginBottom: 8 }}>
        Upload / analyze
      </div>
      <div
        className={`dropzone${dragOver ? " dragover" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          runAnalyze(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
      >
        {busy ? "Analyzing..." : "Drop a WAV file here, or click to choose one"}
        <input
          ref={inputRef}
          type="file"
          accept="audio/wav,audio/x-wav,.wav"
          style={{ display: "none" }}
          onChange={(e) => runAnalyze(e.target.files?.[0])}
        />
      </div>

      {error && (
        <div className="callout" style={{ color: "var(--red)", marginTop: 12 }}>
          {error}
        </div>
      )}

      {result && (
        <div className="upload-result">
          <div className={`risk-number${result.alert ? " alert" : ""}`} style={{ fontSize: 96 }}>
            {result.risk}
            <span className="risk-scale">/100</span>
          </div>
          <div className="verdict-row" style={{ justifyContent: "flex-start", marginTop: 8 }}>
            <span className={`verdict-badge${result.alert ? " synthetic" : ""}`}>
              {result.verdict === "synthetic" ? "Likely synthetic" : "Genuine"}
            </span>
          </div>
          {result.reason && (
            <div className="callout" style={{ marginTop: 16, color: "var(--red)" }}>
              {result.reason}
            </div>
          )}
          <div className="label" style={{ marginTop: 24, marginBottom: 8 }}>
            Per-window trace
          </div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
            {result.per_window.join(", ")}
          </div>
        </div>
      )}
    </div>
  );
}
