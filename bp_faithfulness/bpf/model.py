"""Small BP-estimation model (Part 3).

We use a compact 2-channel 1D-CNN. The proximal and distal waveforms enter as two
channels, so the very first convolution sees BOTH and can form cross-channel
(delay/alignment) features -- which is where PTT information lives -- and later
layers build morphology features. Global average pooling gives the audited
representation `h`; a small MLP head maps `h` to [SBP, DBP]. A few hundred
thousand parameters.

(An earlier cross-attention variant is thematically appealing -- "PTT alignment
lives in cross-attention" -- but did not reliably fit even the training set at
this scale; T is linearly decodable from the raw signals, so a CNN learns it
robustly. The faithfulness probe below is architecture-agnostic: it only needs
`represent()` and `from_h()`.)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BPModel(nn.Module):
    def __init__(self, d_model=48, heads=2, conv_stride=8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(32, 48, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(48, 48, 7, stride=2, padding=3), nn.ReLU(),
        )
        # flatten (KEEP time) -> linear: preserves position, so a delay (PTT) is
        # representable. Global pooling would be translation-invariant and would
        # throw the delay away.
        self.proj = nn.LazyLinear(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 2))
        self.d_model = d_model

    def represent(self, prox, dist):
        """Audited representation h (batch, d_model)."""
        x = torch.stack([prox, dist], dim=1)        # (B, 2, L) -- both channels together
        feat = self.body(x)                         # (B, C, L')
        return self.norm(self.proj(feat.flatten(1)))

    def forward(self, prox, dist, return_h=False):
        h = self.represent(prox, dist)
        pred = self.head(h)
        return (pred, h) if return_h else pred

    def from_h(self, h):
        """Run only the head on a (possibly edited) representation (ablation/patching)."""
        return self.head(h)
