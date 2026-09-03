"""SSL front-end (wav2vec2 / XLS-R) + small classifier head.

Frozen SSL by default: trains fast, generalises better to unseen attacks,
and avoids overfitting on a few epochs.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from config import SSL_MODEL_NAME, FREEZE_SSL, HIDDEN_DIM, EMBEDDING_DIM


def build_head(in_dim: int, hidden_dim: int = HIDDEN_DIM) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, 2),   # [bonafide, spoof] logits
    )


class VoiceGuardNet(nn.Module):
    """Full raw-audio pipeline: SSL front-end + head. Used at inference time
    (src/infer.py) -- the SSL backbone comes from the pretrained HF checkpoint,
    and only `.head`'s weights are loaded from the trained checkpoint (see
    EmbeddingClassifier below, which shares this exact head architecture)."""

    def __init__(self, ssl_name: str = SSL_MODEL_NAME, freeze_ssl: bool = FREEZE_SSL,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.ssl_name = ssl_name
        self.ssl = Wav2Vec2Model.from_pretrained(ssl_name)
        if freeze_ssl:
            for p in self.ssl.parameters():
                p.requires_grad = False
        h = self.ssl.config.hidden_size
        self.head = build_head(h, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) raw waveform @16kHz -> (B, 2) logits."""
        feats = self.ssl(x).last_hidden_state   # (B, L, H)
        pooled = feats.mean(dim=1)               # mean pool over time
        return self.head(pooled)


class EmbeddingClassifier(nn.Module):
    """Head-only classifier for precomputed embeddings (fast training path).

    Trains in minutes on a parquet of precomputed wav2vec2 embeddings instead
    of hours of raw-audio forward passes. Its `.head` submodule is architecture-
    identical to VoiceGuardNet.head, so `EmbeddingClassifier.state_dict()` can
    be loaded straight into `VoiceGuardNet.load_state_dict(..., strict=False)`
    at inference time -- only the `head.*` keys match and get overwritten; the
    pretrained SSL backbone weights are left untouched.
    """

    def __init__(self, in_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.head = build_head(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) precomputed embedding -> (B, 2) logits."""
        return self.head(x)
