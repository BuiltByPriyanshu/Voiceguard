import { useEffect, useRef, useState } from "react";
import { useAudioStream } from "./useAudioStream.js";
import CallerPanel from "./components/CallerPanel.jsx";
import RiskMeter from "./components/RiskMeter.jsx";
import WaveformCanvas from "./components/WaveformCanvas.jsx";
import AlertModal from "./components/AlertModal.jsx";
import AlertLog from "./components/AlertLog.jsx";
import SourceToggle from "./components/SourceToggle.jsx";
import ConnectionStatus from "./components/ConnectionStatus.jsx";
import ExplainabilityPanel from "./components/ExplainabilityPanel.jsx";
import UploadDropzone from "./components/UploadDropzone.jsx";
import ThresholdConfig from "./components/ThresholdConfig.jsx";

export default function App() {
  const [view, setView] = useState("call"); // call | upload
  const [modalOpen, setModalOpen] = useState(false);
  const prevAlertRef = useRef(false);

  const {
    connection,
    streaming,
    risk,
    alert,
    reason,
    log,
    elapsed,
    waveform,
    error,
    startMic,
    startClip,
    stop,
  } = useAudioStream();

  // Fire the pre-transaction warning modal on the false -> true transition
  // only, so it doesn't re-pop every hop while risk stays above threshold.
  useEffect(() => {
    if (alert && !prevAlertRef.current) {
      setModalOpen(true);
    }
    prevAlertRef.current = alert;
  }, [alert]);

  const verdict = alert ? "Likely synthetic" : "Genuine";

  return (
    <div className="app-shell">
      <div className="top-bar">
        <div className="wordmark">
          Voice<span>Guard</span>
        </div>
        <div className="view-tabs">
          <button
            className={`view-tab${view === "call" ? " active" : ""}`}
            onClick={() => setView("call")}
          >
            Call simulation
          </button>
          <button
            className={`view-tab${view === "upload" ? " active" : ""}`}
            onClick={() => setView("upload")}
          >
            Upload / analyze
          </button>
        </div>
        <ConnectionStatus connection={connection} />
      </div>

      {view === "call" ? (
        <div className="grid">
          <div className="col-4">
            <CallerPanel
              connected={streaming}
              alert={alert}
              elapsed={elapsed}
              callerName="Unknown caller"
              callerNumber="+91 98XXX XX210"
            />
          </div>

          <div className="col-5" />

          <div className="col-3">
            <RiskMeter risk={risk} alert={alert} verdict={verdict} />
          </div>

          <div className="col-12">
            <hr className="hairline" />
          </div>

          <div className="col-3">
            <SourceToggle
              streaming={streaming}
              onStartMic={startMic}
              onStartClip={startClip}
              onStop={stop}
            />
            {error && (
              <div className="callout" style={{ color: "var(--red)", marginTop: 16 }}>
                {error}
              </div>
            )}
          </div>

          <div className="col-9">
            <WaveformCanvas data={waveform} alert={alert} />
          </div>

          <div className="col-12">
            <hr className="hairline" />
          </div>

          <div className="col-6">
            <AlertLog entries={log} />
          </div>

          <div className="col-3">
            <ExplainabilityPanel />
          </div>

          <div className="col-3">
            <ThresholdConfig />
          </div>
        </div>
      ) : (
        <div className="grid">
          <div className="col-6">
            <UploadDropzone />
          </div>
        </div>
      )}

      <AlertModal
        open={modalOpen}
        reason={reason}
        onCallBack={() => setModalOpen(false)}
        onApproveAnyway={() => setModalOpen(false)}
      />
    </div>
  );
}
