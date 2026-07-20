"""Pure-transformer variant: linear patch embedding + WINDOWED cross-attention.

Tokenizer: one shared Linear per non-overlapping patch (ViT/PatchTST style, no
conv). Positional encoding makes a delay representable. Cross-attention has the
proximal tokens (queries) attend to distal tokens (keys/values) within a LOCAL
band (+/- `window` patches) covering the physiological PTT range. The idea was
that the attention map would become a diagonal whose offset is the transit time.

DOCUMENTED NEGATIVE RESULT.  Even with this designed cross-attention, a linear
patch embedding, ~5x more data, and AdamW+warmup, the model does NOT learn the
task at physiological scale (gate at gamma=0, PEP=0: MAE ~9.5 = mean-prediction,
probe R^2 ~ 0, donor-sign ~ chance). This is the third fair transformer attempt
to fail where a small 1D-CNN succeeds (probe R^2 ~ 0.9). Interpretation: the
informative signal is one sharp upstroke patch per channel; most patches are flat
and embed identically, so content-matching attention has almost nothing to align
on with only hundreds-to-thousands of samples. Convolution's local inductive bias
matters a lot here. We therefore keep the CNN (bpf/model.py) and rely on the fact
that the audit battery is architecture-agnostic. This module is retained as the
record of the attempt; it is not wired into the figures.

Exposes represent()/from_h() so the audit battery *would* work unchanged if a
future, data-richer setting made it learn.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BPModelXAttn(nn.Module):
    def __init__(self, patch=16, d_model=64, heads=4, window=6):
        super().__init__()
        self.patch, self.window, self.d_model = patch, window, d_model
        self.embed = nn.Linear(patch, d_model)          # linear patch embedding (no conv)
        self.chan = nn.Embedding(2, d_model)            # proximal vs distal tag
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
                                nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.proj = nn.LazyLinear(d_model)              # flatten context -> h (keeps position)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 2))

    def _pe(self, n, device):
        pos = torch.arange(n, device=device).unsqueeze(1).float()
        i = torch.arange(0, self.d_model, 2, device=device).float()
        ang = pos / (10000 ** (i / self.d_model))
        pe = torch.zeros(n, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(ang); pe[:, 1::2] = torch.cos(ang[:, : pe[:, 1::2].shape[1]])
        return pe

    def _tokens(self, sig, ch):
        B, L = sig.shape
        L2 = (L // self.patch) * self.patch
        p = sig[:, :L2].reshape(B, L2 // self.patch, self.patch)     # non-overlapping patches
        t = self.embed(p)                                            # one Linear, shared
        return t + self._pe(t.shape[1], t.device) + self.chan.weight[ch]

    def _band(self, n, device):
        idx = torch.arange(n, device=device)
        off = idx[None, :] - idx[:, None]                            # (key j) - (query i)
        allow = (off >= -1) & (off <= self.window)                   # distal lags proximal
        m = torch.zeros(n, n, device=device)
        m[~allow] = float("-inf")
        return m

    def _attend(self, prox, dist, need_w=False):
        P = self._tokens(prox, 0); D = self._tokens(dist, 1)
        m = self._band(P.shape[1], P.device)
        return self.attn(P, D, D, attn_mask=m, need_weights=need_w,
                         average_attn_weights=need_w)

    def represent(self, prox, dist):
        ctx, _ = self._attend(prox, dist)
        ctx = ctx + self.ff(ctx)                                     # residual FF
        return self.norm(self.proj(ctx.flatten(1)))

    def forward(self, prox, dist, return_h=False):
        h = self.represent(prox, dist)
        pred = self.head(h)
        return (pred, h) if return_h else pred

    def from_h(self, h):
        return self.head(h)

    @torch.no_grad()
    def attention_map(self, prox, dist):
        _, w = self._attend(prox, dist, need_w=True)                 # (B, Lq, Lk)
        return w.mean(0).cpu().numpy()


class BPModelHybrid(nn.Module):
    """Hybrid: a SINGLE conv layer as the patch embedder (the inductive bias that
    makes it learn) + windowed cross-attention (proximal queries distal) with a
    PROXIMAL RESIDUAL, so both channels reach the readout -- plus the interpretable
    alignment map. Same represent()/from_h() API as the CNN.

    STATUS (gate at gamma=0, PEP=0; see hybrid_gate.py): with the proximal residual
    it LEARNS well -- MAE ~3.5 mmHg, best of any model here (the pure linear-patch
    transformer and the residual-free hybrid both failed at MAE ~9).

    FAITHFUL, but NONLINEARLY:
       LINEAR  probe R^2 for T   : ~ -0.4   (fails)
       LINEAR  probe R^2 for 1/T : ~ -0.4   (fails)
       NONLINEAR (MLP) probe T   : ~ +0.8   (T IS there)
       donor-swap sign           : ~ 0.74   (PTT used causally)
    The transformer encodes PTT nonlinearly, so a LINEAR probe -- the standard
    faithfulness check -- misses a genuinely faithful model, while a nonlinear
    probe and the causal donor-swap both recover it. (The CNN happens to encode T
    linearly, probe R^2 ~ 0.9.) This is a useful result in its own right: linear
    probing can fail on a faithful model, which argues for causal/nonlinear audits.
    """
    def __init__(self, patch=16, d_model=64, heads=4, window=6):
        super().__init__()
        self.patch, self.window, self.d_model = patch, window, d_model
        self.embed = nn.Conv1d(1, d_model, patch * 2, stride=patch, padding=patch // 2)
        self.chan = nn.Embedding(2, d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
                                nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.proj = nn.LazyLinear(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 2))

    _pe = BPModelXAttn._pe
    _band = BPModelXAttn._band

    def _tokens(self, sig, ch):
        t = self.embed(sig.unsqueeze(1)).transpose(1, 2)             # (B, n_tok, d)
        return t + self._pe(t.shape[1], t.device) + self.chan.weight[ch]

    def _attend(self, prox, dist, need_w=False):
        P = self._tokens(prox, 0); D = self._tokens(dist, 1)
        n = min(P.shape[1], D.shape[1]); P, D = P[:, :n], D[:, :n]
        m = self._band(n, P.device)
        return self.attn(P, D, D, attn_mask=m, need_weights=need_w, average_attn_weights=need_w)

    def represent(self, prox, dist):
        P = self._tokens(prox, 0); D = self._tokens(dist, 1)
        n = min(P.shape[1], D.shape[1]); P, D = P[:, :n], D[:, :n]
        ctx, _ = self.attn(P, D, D, attn_mask=self._band(n, P.device))
        ctx = P + ctx                       # PROXIMAL RESIDUAL: both channels reach the readout
        ctx = ctx + self.ff(ctx)
        return self.norm(self.proj(ctx.flatten(1)))

    def forward(self, prox, dist, return_h=False):
        h = self.represent(prox, dist); pred = self.head(h)
        return (pred, h) if return_h else pred

    def from_h(self, h):
        return self.head(h)

    @torch.no_grad()
    def attention_map(self, prox, dist):
        _, w = self._attend(prox, dist, need_w=True)
        return w.mean(0).cpu().numpy()
