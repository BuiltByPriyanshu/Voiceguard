// Runs on the audio rendering thread. Forwards raw mono Float32 samples to
// the main thread in ~4096-sample chunks (~256ms at 16kHz) -- conversion to
// Int16 and the WebSocket send both happen on the main thread, since
// AudioWorkletGlobalScope has no WebSocket access.
class PCMWorkletProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      this.port.postMessage(input[0].slice(0));
    }
    return true;
  }
}

registerProcessor("pcm-worklet-processor", PCMWorkletProcessor);
