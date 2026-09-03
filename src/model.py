"""SSL front-end (wav2vec2 / XLS-R) + small classifier head.

Frozen SSL by default: trains fast, generalises better to unseen attacks,
and avoids overfitting on a few epochs.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from config import SSL_MODEL_NAME, FREEZE_SSL, HIDDEN_DIM


class VoiceGuardNet(nn.Module):
    def __init__(self, ssl_name: str = SSL_MODEL_NAME, freeze_ssl: bool = FREEZE_SSL,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.ssl_name = ssl_name
        self.ssl = Wav2Vec2Model.from_pretrained(ssl_name)
        if freeze_ssl:
            for p in self.ssl.parameters():
                p.requires_grad = False
        h = self.ssl.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),   # [bonafide, spoof] logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) raw waveform @16kHz -> (B, 2) logits."""
        feats = self.ssl(x).last_hidden_state   # (B, L, H)
        pooled = feats.mean(dim=1)               # mean pool over time
        return self.head(pooled)
