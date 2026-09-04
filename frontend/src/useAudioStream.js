import { useCallback, useRef, useState } from "react";

// Same-origin: Vite's dev proxy (see vite.config.js) forwards this to the
// FastAPI backend, so this works in dev and in a same-origin prod deploy
// without hardcoding a host.
const wsUrl = () => {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/stream`;
};

const SAMPLE_RATE = 16000; // must match config.SAMPLE_RATE on the backend
const WAVEFORM_WINDOW = SAMPLE_RATE * 3; // ~3s scrolling window

function floatTo16BitPCM(float32Array) {
  const out = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
}

// Single hook driving both "live mic" and "play demo clip" sources through
// the identical capture -> PCM16 -> WebSocket pipeline, per FRONTEND_BRIEF.md
// section 2.7 (source control) -- this guarantees the demo clip path
// exercises exactly the same code the live-mic path does.
export function useAudioStream() {
  const [connection, setConnection] = useState("idle"); // idle | connecting | open | closed | error
  const [streaming, setStreaming] = useState(false);
  const [risk, setRisk] = useState(0);
  const [alert, setAlert] = useState(false);
  const [reason, setReason] = useState(null);
  const [log, setLog] = useState([]);
  const [elapsed, setElapsed] = useState(0);
  const [waveform, setWaveform] = useState(() => new Float32Array(WAVEFORM_WINDOW));
  const [error, setError] = useState(null);

  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const workletRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const micStreamRef = useRef(null);
  const waveformBufRef = useRef(new Float32Array(WAVEFORM_WINDOW));
  const timerRef = useRef(null);
  const startedAtRef = useRef(null);

  const pushWaveform = useCallback((chunk) => {
    const buf = waveformBufRef.current;
    const n = chunk.length;
    if (n >= buf.length) {
      buf.set(chunk.subarray(n - buf.length));
    } else {
      buf.copyWithin(0, n);
      buf.set(chunk, buf.length - n);
    }
    setWaveform(buf.slice());
  }, []);

  const teardownAudio = useCallback(() => {
    if (workletRef.current) {
      workletRef.current.port.onmessage = null;
      try {
        workletRef.current.disconnect();
      } catch {
        /* already disconnected */
      }
      workletRef.current = null;
    }
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.disconnect();
      } catch {
        /* already disconnected */
      }
      sourceNodeRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    teardownAudio();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStreaming(false);
    setConnection("closed");
    setRisk(0);
    setAlert(false);
    setReason(null);
    setElapsed(0);
  }, [teardownAudio]);

  const openSocket = useCallback(() => {
    return new Promise((resolve, reject) => {
      setConnection("connecting");
      const ws = new WebSocket(wsUrl());
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        setConnection("open");
        resolve(ws);
      };
      ws.onmessage = (evt) => {
        let msg;
        try {
          msg = JSON.parse(evt.data);
        } catch {
          return;
        }
        setRisk(msg.risk);
        setAlert(msg.alert);
        setReason(msg.reason ?? null);
        setLog((prev) =>
          [{ t: Date.now(), risk: msg.risk, alert: msg.alert }, ...prev].slice(0, 200)
        );
      };
      ws.onerror = () => {
        setConnection("error");
        setError("WebSocket connection failed -- is the backend running?");
        reject(new Error("websocket error"));
      };
      ws.onclose = () => setConnection((c) => (c === "error" ? c : "closed"));
      wsRef.current = ws;
    });
  }, []);

  const attachWorklet = useCallback(
    async (sourceNode, audioCtx) => {
      await audioCtx.audioWorklet.addModule("/pcm-worklet.js");
      const worklet = new AudioWorkletNode(audioCtx, "pcm-worklet-processor");
      worklet.port.onmessage = (evt) => {
        const chunk = evt.data;
        pushWaveform(chunk);
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          const pcm16 = floatTo16BitPCM(chunk);
          ws.send(pcm16.buffer);
        }
      };
      sourceNode.connect(worklet);
      workletRef.current = worklet;
    },
    [pushWaveform]
  );

  const startTimer = useCallback(() => {
    startedAtRef.current = Date.now();
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000));
    }, 250);
  }, []);

  const startMic = useCallback(async () => {
    setError(null);
    setLog([]);
    try {
      const ws = await openSocket();
      const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = audioCtx;
      // Browser defaults (echo cancellation, noise suppression, AGC) actively
      // reshape the signal in real time -- none of the training data (ASVspoof,
      // In-the-Wild, demo clips) went through that processing, and it produces
      // exactly the kind of spectral artifacts a spoof classifier keys on.
      // Explicitly disabling all three gets the model raw mic input closer to
      // what it was actually trained on.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      micStreamRef.current = stream;
      if (audioCtx.sampleRate !== SAMPLE_RATE) {
        console.warn(
          `AudioContext sampleRate is ${audioCtx.sampleRate}, not ${SAMPLE_RATE} -- ` +
            "the browser did not honor the requested rate. Audio sent to the " +
            "backend will be mislabeled and scored incorrectly."
        );
      }
      const sourceNode = audioCtx.createMediaStreamSource(stream);
      sourceNodeRef.current = sourceNode;
      await attachWorklet(sourceNode, audioCtx);
      startTimer();
      setStreaming(true);
      void ws;
    } catch (e) {
      setError(e.message || String(e));
      teardownAudio();
      setStreaming(false);
    }
  }, [openSocket, attachWorklet, startTimer, teardownAudio]);

  const startClip = useCallback(
    async (url) => {
      setError(null);
      setLog([]);
      try {
        await openSocket();
        const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
        audioCtxRef.current = audioCtx;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`could not load ${url}: ${res.status}`);
        const arrayBuf = await res.arrayBuffer();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuf);
        const bufferSource = audioCtx.createBufferSource();
        bufferSource.buffer = audioBuffer;
        sourceNodeRef.current = bufferSource;
        await attachWorklet(bufferSource, audioCtx);
        // Audible playback so the room can hear the clip during the demo.
        bufferSource.connect(audioCtx.destination);
        bufferSource.onended = () => stop();
        bufferSource.start();
        startTimer();
        setStreaming(true);
      } catch (e) {
        setError(e.message || String(e));
        teardownAudio();
        setStreaming(false);
      }
    },
    [openSocket, attachWorklet, startTimer, teardownAudio, stop]
  );

  return {
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
  };
}
