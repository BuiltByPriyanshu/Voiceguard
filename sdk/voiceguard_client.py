"""Minimal integration snippet for a bank / contact-center call flow.

Demonstrates the entire integration surface in one function: point
`verify_call()` at a running VoiceGuard backend with a chunk of call audio
(WAV bytes), get back a risk verdict to gate a sensitive action on. This is
the concrete answer to "how would a bank plug this into an existing call
flow" -- see SIH26104_36hr_Build_Plan.md section 5 (M4).

For continuous in-call monitoring (not a one-shot check), stream audio to
the WS /stream endpoint instead -- see README.md's API contract.
"""
import requests


def verify_call(audio_bytes: bytes, filename: str = "call.wav",
                 base_url: str = "http://localhost:8000") -> dict:
    """POST a chunk of call audio to VoiceGuard, return its risk verdict.

    Returns: {"risk": 0-100, "verdict": "genuine"|"synthetic",
              "alert": bool, "reason": str|None, "per_window": [int, ...]}
    Raises `requests.HTTPError` if the backend is unreachable or errors.
    """
    resp = requests.post(
        f"{base_url}/analyze",
        files={"file": (filename, audio_bytes, "audio/wav")},
    )
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m sdk.voiceguard_client <wav_path>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        result = verify_call(f.read(), filename=sys.argv[1])

    print(result)
    if result["alert"]:
        print(f"BLOCK / call back required: {result['reason']}")
    else:
        print("Proceed: voice verified as genuine.")
