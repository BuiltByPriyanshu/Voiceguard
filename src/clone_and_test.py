"""Clone a voice from a reference recording and immediately score it
through the live RiskEngine, in one command -- for the "clone someone's
voice on stage and watch it get flagged live" demo moment
(SIH26104_36hr_Build_Plan.md section 7).

Requires coqui-tts (XTTS-v2) installed -- see requirements_cuda.txt /
HANDOFF.md section 6 for the install gotchas. First run downloads the
~1.8GB XTTS-v2 model; after that, cloning a short sentence takes
roughly 20-90s on this Mac's CPU (no CUDA, no reliable MPS support for
XTTS).

Usage:
    python -m src.clone_and_test --reference demo_clips/teammate_ref.wav \
        --text "Hi, please transfer two lakh rupees to this account urgently." \
        --out demo_clips/live_clone.wav
"""
import argparse
import os

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # skip the interactive license prompt

DEFAULT_TEXT = "Hi, please transfer two lakh rupees to this account urgently."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="WAV of the voice to clone, ~10-20s, clean")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="en")
    parser.add_argument("--out", default="demo_clips/live_clone.wav")
    parser.add_argument("--threshold", type=int, default=None,
                         help="override config.DEFAULT_ALERT_THRESHOLD for this run")
    args = parser.parse_args()

    print("Loading XTTS-v2 (downloads ~1.8GB on first run, cached after)...")
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    print(f"Cloning '{args.reference}' saying: {args.text!r}")
    tts.tts_to_file(text=args.text, speaker_wav=args.reference, language=args.language, file_path=args.out)
    print(f"Wrote {args.out}\n")

    print("Scoring through the live RiskEngine...")
    from config import DEFAULT_ALERT_THRESHOLD
    from src.infer import RiskEngine, stream_wav_file

    engine = RiskEngine()
    stream_wav_file(args.out, engine, threshold=args.threshold or DEFAULT_ALERT_THRESHOLD)


if __name__ == "__main__":
    main()
