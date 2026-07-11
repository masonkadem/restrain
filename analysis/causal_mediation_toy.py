"""Toy demonstration: certifying that a model *causally uses* a governing law.

Two governing laws are supported, each with the same two-component structure --
an answer that is only determined when *both* components are used:

* ``saturation``      y = v_i / (v_i + v_j)                (Beer-Lambert ratio)
* ``blood_pressure``  BP = P0 + S * [ 2 ln(L / PTT) - ln(E0 / E_ref) ]
                      (Moens-Korteweg: pulse transit time PTT plus a per-subject
                      arterial-stiffness calibration E0.  Uncalibrated PTT does
                      not determine BP -- the textbook cuffless-BP failure mode.)

A small two-query cross-attention retriever (the CrossSA / q-sparse-averaging
setup) reads a source query that names two slots ``i`` and ``j``, retrieves
their values into separate subspaces, and predicts the answer.

We train two models with the *same architecture* on the *same task*:

* ``law``      -- trained where component ``j`` (the calibration) varies, so it
                  must use both components.
* ``shortcut`` -- trained where ``j`` is (nearly) constant, so reading component
                  ``i`` (PTT) alone predicts the answer.  This is the everyday
                  failure of a confounded / narrow training set: a cuffless-BP
                  model fit on a cohort with similar arterial stiffness.

On the validation distribution a practitioner actually has (``j`` barely varies)
the two models are indistinguishable -- equal predictive error, and a linear
probe decodes the answer equally well from both.  Yet only one generalizes.

Three post-hoc, label-free, frozen-model audits derived from the governing
equation separate them *before any held-out measurement is seen* (the equation
itself is the oracle, evaluated at known component values):

    1. counterfactual sensitivity to component ``j``;
    2. interchange-intervention accuracy (a DAS-style alignment search): is there
       a subspace of the internal state that, swapped between two inputs,
       transfers the answer as the equation predicts?
    3. held-out error where ``j`` varies, revealed afterward to confirm.

Run:

    python analysis/causal_mediation_toy.py --quick            # blood_pressure
    python analysis/causal_mediation_toy.py --law saturation --quick
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

LAW_COLOR = "#2a78d6"       # validated categorical slot 1 (blue)
SHORTCUT_COLOR = "#eb6834"  # validated categorical slot 8 (orange)
ORACLE_COLOR = "#52514e"    # secondary ink for the equation reference


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def positional_encoding(n_slots: int, dim: int) -> np.ndarray:
    positions = np.arange(n_slots, dtype=np.float32)[:, None]
    freqs = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-math.log(10_000.0) / dim))
    enc = np.zeros((n_slots, dim), dtype=np.float32)
    enc[:, 0::2] = np.sin(positions * freqs)
    enc[:, 1::2] = np.cos(positions * freqs[: enc[:, 1::2].shape[1]])
    return enc / np.maximum(np.linalg.norm(enc, axis=1, keepdims=True), 1e-8)


# --------------------------------------------------------------------------- #
# Governing laws
# --------------------------------------------------------------------------- #
@dataclass
class Law:
    name: str
    vi_range: tuple[float, float]     # component i (e.g. PTT)
    vj_range: tuple[float, float]     # component j (e.g. calibration E0)
    j_const: float                    # value j is pinned to in the shortcut regime
    label: Callable[[np.ndarray, np.ndarray], np.ndarray]   # -> network units
    abs_dydvj: Callable[[np.ndarray, np.ndarray], np.ndarray]
    to_display: Callable[[np.ndarray], np.ndarray]          # network units -> plotted units
    display_unit: str
    i_name: str
    j_name: str
    equation: str

    @property
    def value_scale(self) -> tuple[float, float]:
        """Affine (center, half-width) that conditions the token value channel."""
        lo = min(self.vi_range[0], self.vj_range[0])
        hi = max(self.vi_range[1], self.vj_range[1])
        return (lo + hi) / 2.0, (hi - lo) / 2.0


def saturation_law() -> Law:
    return Law(
        name="saturation",
        vi_range=(0.25, 2.0),
        vj_range=(0.25, 2.0),
        j_const=1.0,
        label=lambda vi, vj: vi / (vi + vj),
        abs_dydvj=lambda vi, vj: vi / (vi + vj) ** 2,
        to_display=lambda y: y,
        display_unit="fraction",
        i_name="component i (v_i)",
        j_name="component j (v_j)",
        equation=r"$y = v_i / (v_i + v_j)$",
    )


def blood_pressure_law() -> Law:
    L_ref, E_ref, S, P0 = 0.22, 1.0, 40.0, 100.0

    def label(ptt, e0):  # network units, ~O(1)
        return 2.0 * np.log(L_ref / ptt) - np.log(e0 / E_ref)

    return Law(
        name="blood_pressure",
        vi_range=(0.18, 0.28),   # PTT (s)
        vj_range=(0.5, 2.0),     # E0 (normalized arterial stiffness)
        j_const=1.0,
        label=label,
        abs_dydvj=lambda ptt, e0: 1.0 / e0,          # d/dE0 [ -ln(E0/E_ref) ]
        to_display=lambda y: P0 + S * y,             # -> mmHg
        display_unit="systolic BP (mmHg)",
        i_name="pulse transit time (PTT)",
        j_name="calibration / stiffness (E0)",
        equation=r"$BP = P_0 + S\,[\,2\ln(L/PTT) - \ln(E_0/E_{ref})\,]$",
    )


LAWS = {"saturation": saturation_law, "blood_pressure": blood_pressure_law}


@dataclass
class Batch:
    source: torch.Tensor
    targets: torch.Tensor
    j_index: torch.Tensor
    v_i: torch.Tensor
    v_j: torch.Tensor
    y: torch.Tensor


def _assemble(values, i_idx, j_idx, pos, law) -> Batch:
    n, n_slots = values.shape
    v_i = values[np.arange(n), i_idx]
    v_j = values[np.arange(n), j_idx]
    y = law.label(v_i, v_j).astype(np.float32)
    center, half = law.value_scale
    scaled = ((values - center) / half)[..., None]
    source = np.concatenate([pos[i_idx], pos[j_idx]], axis=-1).astype(np.float32)
    tokens = np.concatenate(
        [scaled, np.broadcast_to(pos, (n, n_slots, pos.shape[1]))], axis=-1
    ).astype(np.float32)
    return Batch(
        source=torch.from_numpy(source),
        targets=torch.from_numpy(tokens),
        j_index=torch.from_numpy(j_idx.astype(np.int64)),
        v_i=torch.from_numpy(v_i.astype(np.float32)),
        v_j=torch.from_numpy(v_j.astype(np.float32)),
        y=torch.from_numpy(y),
    )


def generate(law: Law, n: int, n_slots: int, pos_dim: int, vary_j: bool, seed: int) -> Batch:
    rng = np.random.default_rng(seed)
    pos = positional_encoding(n_slots, pos_dim)
    lo = min(law.vi_range[0], law.vj_range[0])
    hi = max(law.vi_range[1], law.vj_range[1])
    values = rng.uniform(lo, hi, size=(n, n_slots)).astype(np.float32)
    idx = np.stack([rng.permutation(n_slots)[:2] for _ in range(n)])
    i_idx, j_idx = idx[:, 0], idx[:, 1]
    values[np.arange(n), i_idx] = rng.uniform(*law.vi_range, size=n).astype(np.float32)
    if vary_j:
        values[np.arange(n), j_idx] = rng.uniform(*law.vj_range, size=n).astype(np.float32)
    else:
        jitter = rng.normal(scale=0.02, size=n).astype(np.float32)
        values[np.arange(n), j_idx] = np.clip(law.j_const + jitter, *law.vj_range)
    return _assemble(values, i_idx, j_idx, pos, law)


def sweep_batch(law: Law, vi_fixed: float, vj_values: np.ndarray, n_slots: int,
                pos_dim: int, seed: int) -> Batch:
    """Hold component i fixed, sweep component j -- for the response curve."""
    n = len(vj_values)
    rng = np.random.default_rng(seed)
    pos = positional_encoding(n_slots, pos_dim)
    lo = min(law.vi_range[0], law.vj_range[0])
    hi = max(law.vi_range[1], law.vj_range[1])
    values = rng.uniform(lo, hi, size=(n, n_slots)).astype(np.float32)
    i_idx = np.zeros(n, dtype=int)
    j_idx = np.ones(n, dtype=int)
    values[:, 0] = vi_fixed
    values[:, 1] = vj_values.astype(np.float32)
    return _assemble(values, i_idx, j_idx, pos, law)


class Retriever(nn.Module):
    """Two-query cross-attention + scalar head; ``context`` = concat of both retrievals."""

    def __init__(self, source_dim: int, target_dim: int, width: int = 32):
        super().__init__()
        half = source_dim // 2
        self.query_i = nn.Sequential(nn.Linear(half, 64), nn.ReLU(), nn.Linear(64, width))
        self.query_j = nn.Sequential(nn.Linear(half, 64), nn.ReLU(), nn.Linear(64, width))
        self.key = nn.Sequential(nn.Linear(target_dim, 64), nn.ReLU(), nn.Linear(64, width))
        self.value = nn.Sequential(nn.Linear(target_dim, 64), nn.ReLU(), nn.Linear(64, width))
        self.head = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, 1))
        self.width, self.state_dim, self.half = width, 2 * width, half

    def _retrieve(self, query, k, v):
        scores = torch.bmm(query.unsqueeze(1), k.transpose(1, 2)) / math.sqrt(self.width)
        return torch.bmm(scores.softmax(dim=-1), v).squeeze(1)

    def context(self, source, targets):
        k, v = self.key(targets), self.value(targets)
        ctx_i = self._retrieve(self.query_i(source[:, : self.half]), k, v)
        ctx_j = self._retrieve(self.query_j(source[:, self.half :]), k, v)
        return torch.cat([ctx_i, ctx_j], dim=-1)

    def forward(self, source, targets):
        return self.head(self.context(source, targets)).squeeze(-1)


def train(model, data, epochs, device, batch_size=256):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    src, tgt, y = data.source.to(device), data.targets.to(device), data.y.to(device)
    n = len(y)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad()
            loss = torch.mean((model(src[idx], tgt[idx]) - y[idx]) ** 2)
            loss.backward()
            opt.step()
    return float(loss.item())


@torch.no_grad()
def predict(model, data, device):
    return model(data.source.to(device), data.targets.to(device)).cpu().numpy()


def mse(model, data, device):
    return float(np.mean((predict(model, data, device) - data.y.numpy()) ** 2))


@torch.no_grad()
def probe_r2(model, fit, test, device):
    xf = model.context(fit.source.to(device), fit.targets.to(device)).cpu().numpy()
    xt = model.context(test.source.to(device), test.targets.to(device)).cpu().numpy()
    scaler = StandardScaler().fit(xf)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(xf), fit.y.numpy())
    return float(r2_score(test.y.numpy(), ridge.predict(scaler.transform(xt))))


def sensitivity(model, data, law, device):
    model.eval()
    tgt = data.targets.to(device).clone().requires_grad_(True)
    out = model(data.source.to(device), tgt).sum()
    (grad,) = torch.autograd.grad(out, tgt)
    j = data.j_index.to(device)
    rows = torch.arange(len(j), device=device)
    # grad is w.r.t. the scaled channel (v - center)/half; chain-rule back to raw v.
    _, half = law.value_scale
    dydvj = grad[rows, j, 0].abs().cpu().numpy() / half
    analytic = law.abs_dydvj(data.v_i.numpy(), data.v_j.numpy())
    return float(dydvj.mean()), float(analytic.mean())


class Interchange(nn.Module):
    def __init__(self, width, k):
        super().__init__()
        self.raw = nn.Parameter(torch.randn(width, width) * 0.1)
        self.k = k

    def patch(self, ctx_base, ctx_src):
        q, _ = torch.linalg.qr(self.raw)
        zb, zs = ctx_base @ q, ctx_src @ q
        mask = torch.zeros(zb.shape[-1], device=zb.device)
        mask[: self.k] = 1.0
        return (zb * (1 - mask) + zs * mask) @ q.t()


def interchange_accuracy(model, data, device, k, steps):
    """Train only the alignment (model frozen); IIA = R^2 of patched output vs equation y."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        ctx = model.context(data.source.to(device), data.targets.to(device))
    y = data.y.to(device)
    n = len(y)
    rng = np.random.default_rng(0)
    align = Interchange(model.state_dim, k).to(device)
    opt = torch.optim.Adam(align.parameters(), lr=5e-3)
    for _ in range(steps):
        b = torch.from_numpy(rng.integers(0, n, size=min(256, n))).to(device)
        s = torch.from_numpy(rng.integers(0, n, size=min(256, n))).to(device)
        pred = model.head(align.patch(ctx[b], ctx[s])).squeeze(-1)
        opt.zero_grad()
        torch.mean((pred - y[s]) ** 2).backward()
        opt.step()
    with torch.no_grad():
        b = torch.arange(n, device=device)
        s = torch.from_numpy(rng.permutation(n)).to(device)
        pred = model.head(align.patch(ctx[b], ctx[s])).squeeze(-1)
        ss_res = torch.mean((pred - y[s]) ** 2).item()
        ss_tot = torch.var(y[s], unbiased=False).item()
    for p in model.parameters():
        p.requires_grad_(True)
    return float(1.0 - ss_res / max(ss_tot, 1e-8))


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    device = torch.device(args.device)
    law = LAWS[args.law]()
    pos_dim = 8
    source_dim, target_dim = 2 * pos_dim, 1 + pos_dim
    ns = args.n_slots

    law_train = generate(law, args.n_train, ns, pos_dim, vary_j=True, seed=1)
    shortcut_train = generate(law, args.n_train, ns, pos_dim, vary_j=False, seed=1)
    val_indist = generate(law, args.n_eval, ns, pos_dim, vary_j=False, seed=2)
    audit = generate(law, args.n_eval, ns, pos_dim, vary_j=True, seed=3)
    ood = generate(law, args.n_eval, ns, pos_dim, vary_j=True, seed=4)

    vi_mid = float(np.mean(law.vi_range))
    vj_grid = np.linspace(law.vj_range[0], law.vj_range[1], 40)
    sweep = sweep_batch(law, vi_mid, vj_grid, ns, pos_dim, seed=7)
    k_grid = [k for k in (1, 2, 4, 8, 16, 32) if k <= 2 * args.width]

    payload = {
        "law": law.name, "equation": law.equation, "unit": law.display_unit,
        "i_name": law.i_name, "j_name": law.j_name,
        "vj_grid": vj_grid.tolist(), "k_grid": k_grid,
        "sweep_equation": law.to_display(sweep.y.numpy()).tolist(),
        "val_true": law.to_display(val_indist.y.numpy()).tolist(),
        "ood_true": law.to_display(ood.y.numpy()).tolist(),
        "models": {},
    }
    metrics = {}
    for name, train_data in (("law", law_train), ("shortcut", shortcut_train)):
        set_seed(0)
        model = Retriever(source_dim, target_dim, args.width)
        train(model, train_data, args.epochs, device)
        model_dydvj, eq_dydvj = sensitivity(model, audit, law, device)
        iia_curve = [interchange_accuracy(model, audit, device, k, args.das_steps)
                     for k in k_grid]
        payload["models"][name] = {
            "val_pred": law.to_display(predict(model, val_indist, device)).tolist(),
            "ood_pred": law.to_display(predict(model, ood, device)).tolist(),
            "sweep_pred": law.to_display(predict(model, sweep, device)).tolist(),
            "iia_curve": iia_curve,
        }
        metrics[name] = {
            "val_indist_mse": mse(model, val_indist, device),
            "ood_mse": mse(model, ood, device),
            "probe_r2_indist": probe_r2(model, val_indist, val_indist, device),
            "model_abs_dy_dvj": model_dydvj,
            "equation_abs_dy_dvj": eq_dydvj,
            "interchange_accuracy_peak": float(max(iia_curve)),
        }
    payload["metrics"] = metrics
    payload["config"] = {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()}
    return payload


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.4, color="#0b0b0b")


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.3, color="#7a7a76", shrinkA=2, shrinkB=2))


def panel_schematic(ax, p):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("A  Task: retrieve two components, apply the governing law",
                 loc="left", fontsize=9.5, fontweight="bold")
    _box(ax, 0.01, 0.62, 0.22, 0.18, "query names\nslots i, j", "#eef2f7", "#9aa7b4")
    _box(ax, 0.29, 0.74, 0.30, 0.16, f"retrieve i:\n{p['i_name']}", "#dce9f9", LAW_COLOR)
    _box(ax, 0.29, 0.53, 0.30, 0.16, f"retrieve j:\n{p['j_name']}", "#fce4d6", SHORTCUT_COLOR)
    _box(ax, 0.66, 0.62, 0.32, 0.18, "governing law\n(both components)", "#eef2f7", "#9aa7b4")
    _arrow(ax, 0.23, 0.73, 0.29, 0.80)
    _arrow(ax, 0.23, 0.69, 0.29, 0.61)
    _arrow(ax, 0.59, 0.81, 0.66, 0.74)
    _arrow(ax, 0.59, 0.60, 0.66, 0.68)
    ax.text(0.5, 0.45, p["equation"], ha="center", va="center", fontsize=8.6, color="#0b0b0b")
    ax.text(0.01, 0.33,
            "law model — trained with component j varying  → must use both\n"
            "shortcut model — trained with component j fixed → reads i alone",
            ha="left", va="top", fontsize=8.2, color="#52514e")
    ax.plot([0.03, 0.12], [0.14, 0.14], color=LAW_COLOR, lw=3)
    ax.text(0.14, 0.14, "law", va="center", fontsize=8.4, color=LAW_COLOR, fontweight="bold")
    ax.plot([0.30, 0.39], [0.14, 0.14], color=SHORTCUT_COLOR, lw=3)
    ax.text(0.41, 0.14, "shortcut", va="center", fontsize=8.4, color=SHORTCUT_COLOR,
            fontweight="bold")


def panel_scatter(ax, true, law_pred, sc_pred, title, unit, letter):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=9.5, fontweight="bold")
    lo = min(min(true), min(law_pred), min(sc_pred))
    hi = max(max(true), max(law_pred), max(sc_pred))
    ax.plot([lo, hi], [lo, hi], "--", color=ORACLE_COLOR, lw=1.2, zorder=1)
    ax.scatter(true, sc_pred, s=9, color=SHORTCUT_COLOR, alpha=0.5, edgecolor="none",
               label="shortcut", zorder=2)
    ax.scatter(true, law_pred, s=9, color=LAW_COLOR, alpha=0.5, edgecolor="none",
               label="law", zorder=3)
    ax.set_xlabel(f"true {unit}"); ax.set_ylabel(f"predicted {unit}")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_sweep(ax, p):
    ax.set_title("D  Counterfactual sensitivity to the calibration term",
                 loc="left", fontsize=9.5, fontweight="bold")
    x = p["vj_grid"]
    ax.plot(x, p["sweep_equation"], color=ORACLE_COLOR, lw=2, label="governing equation")
    ax.plot(x, p["models"]["law"]["sweep_pred"], color=LAW_COLOR, lw=2, label="law")
    ax.plot(x, p["models"]["shortcut"]["sweep_pred"], color=SHORTCUT_COLOR, lw=2,
            label="shortcut")
    ax.set_xlabel(p["j_name"]); ax.set_ylabel(f"predicted {p['unit']}")
    ax.legend(loc="best", fontsize=7.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_iia(ax, p):
    ax.set_title("E  Interchange-intervention accuracy (causal audit)",
                 loc="left", fontsize=9.5, fontweight="bold")
    x = p["k_grid"]
    ax.plot(x, p["models"]["law"]["iia_curve"], "-o", color=LAW_COLOR, lw=2, ms=5, label="law")
    ax.plot(x, p["models"]["shortcut"]["iia_curve"], "-o", color=SHORTCUT_COLOR, lw=2, ms=5,
            label="shortcut")
    ax.axhline(0.0, color=ORACLE_COLOR, ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.set_xticklabels(x)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel("interchanged subspace dim  k"); ax.set_ylabel("interchange accuracy (R²)")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_scorecard(ax, p):
    ax.set_title("F  Conventional checks agree; the audits separate",
                 loc="left", fontsize=9.5, fontweight="bold")
    m = p["metrics"]

    def skill(mse_val, ref):  # 1 - normalized error, clipped to [0,1]
        return float(np.clip(1 - mse_val / ref, 0, 1))
    ref = max(m["shortcut"]["ood_mse"], 1e-6)
    rows = [
        ("validation accuracy", skill(m["law"]["val_indist_mse"], ref),
         skill(m["shortcut"]["val_indist_mse"], ref), "conventional"),
        ("probe decodes answer", m["law"]["probe_r2_indist"],
         m["shortcut"]["probe_r2_indist"], "conventional"),
        ("interchange audit", m["law"]["interchange_accuracy_peak"],
         m["shortcut"]["interchange_accuracy_peak"], "audit"),
        ("OOD accuracy (revealed)", skill(m["law"]["ood_mse"], ref),
         skill(m["shortcut"]["ood_mse"], ref), "audit"),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (lbl, lv, sv, kind) in zip(ys, rows):
        ax.plot([sv, lv], [y, y], color="#c9c9c4", lw=2, zorder=1)
        ax.scatter(sv, y, s=70, color=SHORTCUT_COLOR, zorder=3)
        ax.scatter(lv, y, s=70, color=LAW_COLOR, zorder=3)
        ax.text(-0.02, y, lbl, ha="right", va="center", fontsize=8.2)
        if kind == "audit":
            ax.text(1.02, y, "separates", ha="left", va="center", fontsize=7.4,
                    color="#0b0b0b", fontweight="bold")
        else:
            ax.text(1.02, y, "agree", ha="left", va="center", fontsize=7.4, color="#8a8a86")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("normalized score  (0 = worst, 1 = best)")
    ax.scatter([], [], s=70, color=LAW_COLOR, label="law")
    ax.scatter([], [], s=70, color=SHORTCUT_COLOR, label="shortcut")
    ax.legend(loc="lower center", ncol=2, fontsize=7.5, frameon=False,
              bbox_to_anchor=(0.5, -0.32))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def make_figure(p, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    law_name = "blood pressure (Moens–Korteweg)" if p["law"] == "blood_pressure" else p["law"]
    fig.suptitle(
        f"Auditing whether a model causally uses the governing law — {law_name}",
        fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    panel_schematic(axes[0, 0], p)
    panel_scatter(axes[0, 1], p["val_true"], p["models"]["law"]["val_pred"],
                  p["models"]["shortcut"]["val_pred"],
                  "Validation (calibration ≈ fixed): both look perfect", p["unit"], "B")
    panel_scatter(axes[0, 2], p["ood_true"], p["models"]["law"]["ood_pred"],
                  p["models"]["shortcut"]["ood_pred"],
                  "New subjects (calibration varies): shortcut fails", p["unit"], "C")
    panel_sweep(axes[1, 0], p)
    panel_iia(axes[1, 1], p)
    panel_scorecard(axes[1, 2], p)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law", choices=list(LAWS), default="blood_pressure")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-train", type=int, default=2500)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--n-slots", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--das-steps", type=int, default=400)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_train, args.n_eval, args.epochs, args.das_steps = 1500, 600, 500, 250
    if args.output_dir is None:
        args.output_dir = Path(f"results/causal_mediation_toy/{args.law}")

    payload = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"metrics": payload["metrics"], "config": payload["config"]}, handle, indent=2)
    make_figure(payload, args.output_dir / "causal_mediation.png")

    m = payload["metrics"]
    print(f"\n=== Governing-law causal audit ({args.law}) ===")
    header = f"{'metric':<34}{'law':>12}{'shortcut':>12}"
    print(header); print("-" * len(header))
    for label, key in (("val MSE (practitioner)", "val_indist_mse"),
                       ("probe R^2 (decodes answer)", "probe_r2_indist"),
                       ("interchange accuracy (audit)", "interchange_accuracy_peak"),
                       ("OOD MSE (revealed)", "ood_mse")):
        print(f"{label:<34}{m['law'][key]:>12.4f}{m['shortcut'][key]:>12.4f}")
    print(f"{'|dy/dvj| model (eq value)':<34}"
          f"{m['law']['model_abs_dy_dvj']:>8.3f} ({m['law']['equation_abs_dy_dvj']:.2f})"
          f"{m['shortcut']['model_abs_dy_dvj']:>6.3f} ({m['shortcut']['equation_abs_dy_dvj']:.2f})")
    print(f"\nwrote {args.output_dir}/metrics.json and causal_mediation.png")


if __name__ == "__main__":
    main()
