import { useState } from "react";

// Mocked per FRONTEND_BRIEF.md 3a -- static spectrogram + hardcoded
// callouts, not computed live. The prosody/explainability model is a
// stretch goal that may not exist by demo time; this still needs to look
// intentional on its own.
export default function ExplainabilityPanel() {
  const [open, setOpen] = useState(false);

  return (
    <div>
      <button className="explain-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "Hide" : "Show"} explainability panel
      </button>
      {open && (
        <div>
          <div className="spectrogram-mock" />
          <div className="callout">
            <strong>0:00-0:02</strong> flat pitch contour -- inconsistent with natural vocal tract resonance
          </div>
          <div className="callout">
            <strong>0:03.2</strong> spectral artifact at high frequency band, consistent with vocoder synthesis
          </div>
        </div>
      )}
    </div>
  );
}
