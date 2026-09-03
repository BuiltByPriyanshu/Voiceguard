"""Device-agnostic helper: CUDA (cloud tonight) -> MPS (Mac tomorrow) -> CPU.

Use `get_device()` and `.to(device)` everywhere. Never call `.cuda()` directly
-- that crashes on the Mac, which has no CUDA.
"""
import os
import torch

# Unsupported MPS ops fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    print(get_device())
