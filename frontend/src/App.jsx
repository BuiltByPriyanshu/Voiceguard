import { useEffect, useRef, useState } from "react";
import { useAudioStream } from "./useAudioStream.js";
import CallerPanel from "./components/CallerPanel.jsx";
import RiskMeter from "./components/RiskMeter.jsx";
import MicButton from "./components/MicButton.jsx";
import WaveformCanvas from "./components/WaveformCanvas.jsx";
import AlertModal from "./components/AlertModal.jsx";
import AlertLog from "./components/AlertLog.jsx";
import SourceToggle, { DEMO_CLIPS } from "./components/SourceToggle.jsx";
import ConnectionStatus from "./components/ConnectionStatus.jsx";
import ContextPanel from "./components/ContextPanel.jsx";
import ExplainabilityPanel from "./components/ExplainabilityPanel.jsx";
import UploadDropzone from "./components/UploadDropzone.jsx";
import ThresholdConfig from "./components/ThresholdConfig.jsx";

export default function App() {
  const [view, setView] = useState("call"); // call | upload
  const [modalOpen, setModalOpen] = useState(false);
  const [sourceMode, setSourceMode] = useState("mic"); // mic | clip
  const [clip, setClip] = useState(DEMO_CLIPS[0].file);
  const prevAlertRef = useRef(false);

  const {
    connection,
    streaming,
    voiceAuthenticity,
    interactionRisk,
    decision,
    verdict: rawVerdict,
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

  // verdict describes the VOICE alone; alert/decision (interactionRisk)
  // describe the fused interaction -- they can disagree on purpose, e.g. a
  // genuine voice in a risky context. See src/risk_fusion.py.
  const verdict = rawVerdict === "synthetic" ? "Likely synthetic" : "Genuine";

  const handleMicClick = () => {
    if (streaming) {
      stop();
    } else if (sourceMode === "mic") {
      startMic();
    } else {
      startClip(`/demo_clips/${clip}`);
    }
  };

  return (
    <div className="app-shell">
      <div className="top-bar">
        <div className="wordmark">VoiceGuard</div>
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

          <div className="col-8">
            <RiskMeter
              interactionRisk={interactionRisk}
              voiceAuthenticity={voiceAuthenticity}
              alert={alert}
              verdict={verdict}
              decision={decision}
            />
          </div>

          <div className="col-12">
            <hr className="hairline" />
          </div>

          <div className="col-4">
            <MicButton streaming={streaming} alert={alert} onClick={handleMicClick} />
            {error && (
              <div className="callout" style={{ color: "var(--red)", marginTop: 16 }}>
                {error}
              </div>
            )}
          </div>

          <div className="col-8">
            <WaveformCanvas data={waveform} alert={alert} />
          </div>

          <div className="col-12">
            <hr className="hairline" />
          </div>

          <div className="col-3">
            <SourceToggle
              mode={sourceMode}
              setMode={setSourceMode}
              clip={clip}
              setClip={setClip}
              disabled={streaming}
            />
          </div>

          <div className="col-6">
            <AlertLog entries={log} />
          </div>

          <div className="col-3">
            <ContextPanel />
          </div>

          <div className="col-12">
            <hr className="hairline" />
          </div>

          <div className="col-6">
            <ExplainabilityPanel />
          </div>

          <div className="col-6">
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
