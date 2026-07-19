"""Small BP-estimation model (Part 3): a compact position-preserving 1D-CNN.

Design note (kept honest, it is a finding):
  We tried a small from-scratch TRANSFORMER (the natural mech-interp substrate).
  At this data scale (~500 training samples) it did not fit the task -- probe R2
  for the true intermediate stayed at chance and BP MAE stayed at mean-prediction
  -- while this CNN learns it cleanly (probe R2 ~ 0.9). Transformers are
  data-hungry; a physiological-scale toy is not where they shine. Crucially, the
  faithfulness battery does NOT depend on the architecture: probing, causal
  ablation, donor-swap activation patching and input saliency all operate on
  `represent()` / `from_h()` and the raw inputs. So we use the CNN and keep the
  audit architecture-agnostic.

The 2-channel input means the first conv sees BOTH proximal and distal, so it can
form cross-channel (delay/alignment == PTT) features; flattening (rather than
global pooling) preserves position so a delay is representable. `h` is the pooled
representation we audit.
"""
from __future__ import annotations

import numpy as np
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
        self.proj = nn.LazyLinear(d_model)          # flatten keeps position -> delay learnable
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 2))
        self.d_model = d_model

    def represent(self, prox, dist):
        x = torch.stack([prox, dist], dim=1)        # (B, 2, L) -- both channels together
        feat = self.body(x)
        return self.norm(self.proj(feat.flatten(1)))

    def forward(self, prox, dist, return_h=False):
        h = self.represent(prox, dist)
        pred = self.head(h)
        return (pred, h) if return_h else pred

    def from_h(self, h):
        return self.head(h)

    def saliency(self, prox, dist):
        """Mean |d SBP / d input| over the batch, per time-step, per channel.
        A model relying on PTT concentrates saliency near the two feet."""
        p = torch.tensor(prox, requires_grad=True)
        d = torch.tensor(dist, requires_grad=True)
        self.forward(p, d)[:, 0].sum().backward()
        return np.abs(p.grad.numpy()).mean(0), np.abs(d.grad.numpy()).mean(0)
