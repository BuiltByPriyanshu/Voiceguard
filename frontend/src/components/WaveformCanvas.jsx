import { useEffect, useRef } from "react";

// Real-time scrolling waveform. Canvas 2D rather than SVG/chart-lib per
// FRONTEND_BRIEF.md section 1 -- redraws on every incoming buffer, which at
// ~4 chunks/sec (see useAudioStream's worklet chunk size) is cheap for
// direct canvas drawing but would thrash an SVG DOM.
export default function WaveformCanvas({ data, alert }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height || 120));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const mid = h / 2;
    const step = Math.max(1, Math.floor(data.length / w));
    ctx.strokeStyle = alert
      ? getComputedStyle(document.documentElement).getPropertyValue("--red")
      : getComputedStyle(document.documentElement).getPropertyValue("--text");
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const idx = x * step;
      let min = 1;
      let max = -1;
      for (let j = 0; j < step && idx + j < data.length; j++) {
        const v = data[idx + j];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const yMin = mid + min * mid * 0.9;
      const yMax = mid + max * mid * 0.9;
      ctx.moveTo(x + 0.5, yMin);
      ctx.lineTo(x + 0.5, yMax);
    }
    ctx.stroke();
  }, [data, alert]);

  return (
    <div className="waveform-wrap">
      <div className="label" style={{ marginBottom: 8 }}>
        Live waveform
      </div>
      <canvas ref={canvasRef} className="waveform" />
    </div>
  );
}
