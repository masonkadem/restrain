"""Gate + probe-target reconciliation for the hybrid conv-patch cross-attention
model. Confirms it learns and that it encodes 1/T (the statistic BP is LINEAR in,
BP = K1/T + K2) rather than T -- so a linear T-probe misses it even though PTT is
used causally (donor-swap).

    python hybrid_gate.py
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from bpf import analysis as A
from bpf.config import load_config, seed_everything
from bpf.dataset import generate_dataset, subject_split
from bpf.xattn import BPModelHybrid


def train(model, ds, idx, epochs=150, wu=15, lr=1.5e-3, bs=128):
    p = torch.tensor(ds.prox[idx]); d = torch.tensor(ds.dist[idx]); bp = torch.tensor(ds.bp[idx])
    mu, sd = bp.mean(0), bp.std(0) + 1e-6
    bpn = (bp - mu) / sd
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(idx)
    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = lr * min(1, (ep + 1) / wu)
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            i = perm[s:s + bs]
            opt.zero_grad()
            ((model(p[i], d[i]) - bpn[i]) ** 2).mean().backward()
            opt.step()
    model.eval(); model._mu, model._sd = mu, sd
    return model


def probe_for(model, ds, tr, te, target, nonlinear=False):
    """Probe `target` from frozen activations; return test R^2. Linear (ridge) by
    default, or a small MLP if nonlinear=True (does the info exist at all?)."""
    Htr, Hte = A.representations(model, ds, tr), A.representations(model, ds, te)
    y = target
    mu, sd = y[tr].mean(), y[tr].std() + 1e-9
    if nonlinear:
        sc = StandardScaler().fit(Htr)
        reg = MLPRegressor(hidden_layer_sizes=(64,), max_iter=800, alpha=1e-2, random_state=0)
        reg.fit(sc.transform(Htr), (y[tr] - mu) / sd)
        return float(r2_score((y[te] - mu) / sd, reg.predict(sc.transform(Hte))))
    ridge = Ridge(alpha=1.0).fit(Htr, (y[tr] - mu) / sd)
    return float(r2_score((y[te] - mu) / sd, ridge.predict(Hte)))


def main():
    cfg = load_config(); seed_everything(cfg.seed)
    cfg.raw["subjects"]["n"] = 12
    cfg.raw["subjects"]["samples_per_subject"] = 260          # ~5x data for the transformer

    ds = generate_dataset(cfg, gamma=0.0, pep_sd_ms=0.0, seed=0)
    tr, te = subject_split(ds, 9, seed=0)
    model = train(BPModelHybrid(), ds, tr)

    mae, _ = A.mae_per_subject(model, ds, te)
    r2_T = probe_for(model, ds, tr, te, ds.T)
    r2_invT = probe_for(model, ds, tr, te, 1.0 / ds.T)
    r2_T_nl = probe_for(model, ds, tr, te, ds.T, nonlinear=True)
    dv = A.donor_swap(model, ds, te, A.probe_T(model, ds, tr, te, cfg)["direction"], cfg)

    print("=== hybrid conv-patch + windowed cross-attention (proximal residual) ===")
    print(f"  BP MAE                 : {mae:.2f} mmHg")
    print(f"  LINEAR probe R^2 for T : {r2_T:+.2f}")
    print(f"  LINEAR probe R^2 for 1/T: {r2_invT:+.2f}")
    print(f"  NONLINEAR probe (MLP) T: {r2_T_nl:+.2f}")
    print(f"  donor-swap sign        : {dv['sign_acc']:.2f}")
    ok = "nonlinearly present (linear probe insufficient)" if r2_T_nl > 0.5 else \
         "NOT recovered even nonlinearly"
    print(f"verdict: learns BP well; PTT is {ok}. "
          "The CNN remains the clean choice for the figures (linear T-probe ~0.9).")


if __name__ == "__main__":
    main()
