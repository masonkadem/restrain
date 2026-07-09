"""Minimal identifiability instrument: a controlled study of when activation
gating can and cannot work.

This is the "toy model" companion to the physiological benchmark. It strips
the idea to the simplest setting where the *identifiability gap* gamma can be
dialed exactly, and asks two questions the messy physiological simulators
could only hint at:

  1. As gamma grows, how does the benefit of probe-based gating scale?
  2. Does *detecting* unanswerability behave the same as *benefiting* from
     abstaining on it?

Construction
------------
Latents z ~ N(0, I) split into a "core" block (always observed) and a
"peripheral" block (sometimes corrupted). The target mixes them:

    y = sqrt(1 - f) * (w_core . z_core) + sqrt(f) * (w_periph . z_periph) + eps

with ||w_core|| = ||w_periph|| = 1 and eps ~ N(0, sigma^2), so Var(y) = 1 +
sigma^2 regardless of f. The observation x concatenates the observed core and
peripheral coordinates (the network is never given a mask indicator).

An input is *answerable* when its peripheral block is intact. It is made
*unanswerable* two ways, at matched f (hence matched error magnitude):

  - "zeroed": z_periph -> 0. The corruption is visible in the input
    (peripheral coordinates are exactly zero) -- the missing-channel case.
  - "resampled": z_periph -> an independent draw. The input stays perfectly
    in-distribution; only its *value* is now decorrelated from the y it
    should predict -- the subtly-wrong-but-complete case.

Because the readout is trained on answerable data only, its best guess for a
corrupted input still uses the (now useless) peripheral pathway, so the
irreducible error is sqrt(f) * w_periph . (z_obs - z_true). The identifiability
gap is therefore

    gamma(f) = Var(y | x, unanswerable) / Var(y | x, answerable)
             = (c * f + sigma^2) / sigma^2

where c = 1 for zeroed (all peripheral signal lost) and c = 2 for resampled
(the wrong value adds its own variance). f = 0 is fully answerable; f -> 1 is
maximally unanswerable.

The punchline
-------------
Detectability tracks whether unanswerability is a *function of the observable
input*, not the error magnitude. Zeroed corruption is input-visible, so a
probe detects it (and gating helps, scaling with gamma). Resampled corruption
leaves the input distribution unchanged, so a single deterministic forward
pass cannot distinguish it -- detection stays at chance and gating cannot help
*no matter how large gamma is*. That dissociation is exactly why the
physiological benchmark's held-out non-missing detection collapsed to chance,
and it is the honest boundary of the whole approach.

    python analysis/toy_identifiability.py
    python analysis/toy_identifiability.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from abstention_utils import (  # noqa: E402
    area_under_risk_coverage,
    risk_coverage_curve,
)


@dataclass
class Instrument:
    """Fixed target-weight vectors defining the linear inverse problem."""

    w_core: np.ndarray
    w_periph: np.ndarray
    sigma: float

    @property
    def k_core(self) -> int:
        return len(self.w_core)

    @property
    def k_periph(self) -> int:
        return len(self.w_periph)


def make_instrument(k_core: int = 8, k_periph: int = 8, sigma: float = 0.1, seed: int = 0) -> Instrument:
    rng = np.random.default_rng(seed)
    w_core = rng.normal(size=k_core)
    w_core /= np.linalg.norm(w_core)
    w_periph = rng.normal(size=k_periph)
    w_periph /= np.linalg.norm(w_periph)
    return Instrument(w_core=w_core, w_periph=w_periph, sigma=sigma)


def gamma_of(f: float, sigma: float, mode: str) -> float:
    """Analytic identifiability gap for a given peripheral-variance fraction."""
    c = 2.0 if mode == "resampled" else 1.0
    return (c * f + sigma**2) / (sigma**2)


def generate(
    inst: Instrument,
    n: int,
    f: float,
    corruption: str,
    seed: int,
) -> dict:
    """Generate a batch. corruption in {"clean", "zeroed", "resampled"}."""
    rng = np.random.default_rng(seed)
    z_core = rng.normal(size=(n, inst.k_core))
    z_periph = rng.normal(size=(n, inst.k_periph))
    eps = rng.normal(scale=inst.sigma, size=n)
    y = (
        np.sqrt(1.0 - f) * (z_core @ inst.w_core)
        + np.sqrt(f) * (z_periph @ inst.w_periph)
        + eps
    )
    x_periph = z_periph.copy()
    answerable = np.ones(n, dtype=int)
    if corruption == "zeroed":
        x_periph[:] = 0.0
        answerable[:] = 0
    elif corruption == "resampled":
        x_periph = rng.normal(size=(n, inst.k_periph))
        answerable[:] = 0
    x = np.concatenate([z_core, x_periph], axis=1).astype(np.float32)
    return {"x": x, "y": y.astype(np.float32), "answerable": answerable}


class Readout(nn.Module):
    """Small MLP; the penultimate activation is what we probe."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, return_features: bool = False):
        h = self.enc(x)
        out = self.head(h).squeeze(-1)
        if return_features:
            return out, h
        return out


def train_readout(model: nn.Module, data: dict, epochs: int, seed: int) -> None:
    torch.manual_seed(seed)
    x = torch.from_numpy(data["x"])
    y = torch.from_numpy(data["y"])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()


def evaluate(model: nn.Module, data: dict) -> dict:
    model.eval()
    with torch.no_grad():
        pred, feat = model(torch.from_numpy(data["x"]), return_features=True)
    pred = pred.numpy()
    return {
        "pred": pred,
        "feat": feat.numpy(),
        "loss": (pred - data["y"]) ** 2,
        "answerable": data["answerable"],
    }


def selective_gain(losses: np.ndarray, confidence: np.ndarray, oracle_conf: np.ndarray, seed: int) -> float:
    """Fraction of the achievable gating benefit that `confidence` captures.

    0 = no better than random abstention; 1 = matches the oracle that knows
    the true answerable label. Normalizing this way makes the benefit
    comparable across gamma levels (where absolute AURC scales trivially).
    """
    rng = np.random.default_rng(seed)
    random_conf = rng.random(len(losses))
    aurc_probe = area_under_risk_coverage(losses, confidence)
    aurc_random = area_under_risk_coverage(losses, random_conf)
    aurc_oracle = area_under_risk_coverage(losses, oracle_conf)
    denom = aurc_random - aurc_oracle
    if abs(denom) < 1e-9:
        return 0.0
    return float((aurc_random - aurc_probe) / denom)


def run_condition(
    inst: Instrument, f: float, corruption: str, cfg: argparse.Namespace, seed: int
) -> dict:
    # Train the readout on answerable data only (matches the frozen-audit protocol).
    train = generate(inst, cfg.n_train, f, "clean", seed * 100 + 1)
    model = Readout(inst.k_core + inst.k_periph, cfg.hidden)
    train_readout(model, train, cfg.epochs, seed)

    # Probe-fitting split: balanced answerable vs corrupted activations.
    probe_ok = generate(inst, cfg.n_probe, f, "clean", seed * 100 + 2)
    probe_bad = generate(inst, cfg.n_probe, f, corruption, seed * 100 + 3)
    ev_ok = evaluate(model, probe_ok)
    ev_bad = evaluate(model, probe_bad)
    probe_feat = np.concatenate([ev_ok["feat"], ev_bad["feat"]])
    probe_lbl = np.concatenate([ev_ok["answerable"], ev_bad["answerable"]])
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probe.fit(probe_feat, probe_lbl)

    # Test split: fresh balanced mix.
    test_ok = evaluate(model, generate(inst, cfg.n_test, f, "clean", seed * 100 + 4))
    test_bad = evaluate(model, generate(inst, cfg.n_test, f, corruption, seed * 100 + 5))
    feat = np.concatenate([test_ok["feat"], test_bad["feat"]])
    answerable = np.concatenate([test_ok["answerable"], test_bad["answerable"]])
    losses = np.concatenate([test_ok["loss"], test_bad["loss"]])
    conf = probe.predict_proba(feat)[:, 1]  # P(answerable) = confidence

    detection_auroc = (
        float(roc_auc_score(answerable, conf)) if np.unique(answerable).size > 1 else float("nan")
    )
    gain = selective_gain(losses, conf, answerable.astype(float), seed)
    return {
        "detection_auroc": detection_auroc,
        "selective_gain": gain,
        "gamma": gamma_of(f, inst.sigma, corruption),
        "mean_loss_answerable": float(test_ok["loss"].mean()),
        "mean_loss_unanswerable": float(test_bad["loss"].mean()),
    }


def risk_coverage_demo(inst: Instrument, cfg: argparse.Namespace, seed: int) -> dict:
    """Risk-coverage curves at one representative gamma, zeroed corruption --
    the canonical 'model restrains itself' visualization."""
    f = cfg.demo_f
    train = generate(inst, cfg.n_train, f, "clean", seed * 100 + 1)
    model = Readout(inst.k_core + inst.k_periph, cfg.hidden)
    train_readout(model, train, cfg.epochs, seed)
    probe_ok = evaluate(model, generate(inst, cfg.n_probe, f, "clean", seed * 100 + 2))
    probe_bad = evaluate(model, generate(inst, cfg.n_probe, f, "zeroed", seed * 100 + 3))
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probe.fit(
        np.concatenate([probe_ok["feat"], probe_bad["feat"]]),
        np.concatenate([probe_ok["answerable"], probe_bad["answerable"]]),
    )
    test_ok = evaluate(model, generate(inst, cfg.n_test, f, "clean", seed * 100 + 4))
    test_bad = evaluate(model, generate(inst, cfg.n_test, f, "zeroed", seed * 100 + 5))
    losses = np.concatenate([test_ok["loss"], test_bad["loss"]])
    answerable = np.concatenate([test_ok["answerable"], test_bad["answerable"]])
    conf = probe.predict_proba(np.concatenate([test_ok["feat"], test_bad["feat"]]))[:, 1]
    rng = np.random.default_rng(seed)
    cov_p, risk_p = risk_coverage_curve(losses, conf)
    cov_r, risk_r = risk_coverage_curve(losses, rng.random(len(losses)))
    cov_o, risk_o = risk_coverage_curve(losses, answerable.astype(float))
    return {
        "coverage_probe": cov_p, "risk_probe": risk_p,
        "coverage_random": cov_r, "risk_random": risk_r,
        "coverage_oracle": cov_o, "risk_oracle": risk_o,
        "conf_answerable": conf[answerable == 1],
        "conf_unanswerable": conf[answerable == 0],
    }


def resampled_scores(inst: Instrument, cfg: argparse.Namespace, seed: int) -> dict:
    """Probe scores for the resampled (in-distribution) corruption at demo_f,
    to sit beside the zeroed histograms and show the non-separation."""
    f = cfg.demo_f
    train = generate(inst, cfg.n_train, f, "clean", seed * 100 + 1)
    model = Readout(inst.k_core + inst.k_periph, cfg.hidden)
    train_readout(model, train, cfg.epochs, seed)
    probe_ok = evaluate(model, generate(inst, cfg.n_probe, f, "clean", seed * 100 + 2))
    probe_bad = evaluate(model, generate(inst, cfg.n_probe, f, "resampled", seed * 100 + 3))
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probe.fit(
        np.concatenate([probe_ok["feat"], probe_bad["feat"]]),
        np.concatenate([probe_ok["answerable"], probe_bad["answerable"]]),
    )
    test_ok = evaluate(model, generate(inst, cfg.n_test, f, "clean", seed * 100 + 4))
    test_bad = evaluate(model, generate(inst, cfg.n_test, f, "resampled", seed * 100 + 5))
    conf_ok = probe.predict_proba(test_ok["feat"])[:, 1]
    conf_bad = probe.predict_proba(test_bad["feat"])[:, 1]
    return {"conf_answerable": conf_ok, "conf_unanswerable": conf_bad}


def sweep(inst: Instrument, cfg: argparse.Namespace) -> dict:
    fs = np.linspace(cfg.f_min, cfg.f_max, cfg.n_f)
    out = {"f": fs.tolist(), "zeroed": [], "resampled": []}
    for mode in ("zeroed", "resampled"):
        for f in fs:
            per_seed = [run_condition(inst, float(f), mode, cfg, s) for s in cfg.seeds]
            out[mode].append({
                "f": float(f),
                "gamma": float(np.mean([r["gamma"] for r in per_seed])),
                "detection_auroc_mean": float(np.mean([r["detection_auroc"] for r in per_seed])),
                "detection_auroc_std": float(np.std([r["detection_auroc"] for r in per_seed])),
                "selective_gain_mean": float(np.mean([r["selective_gain"] for r in per_seed])),
                "selective_gain_std": float(np.std([r["selective_gain"] for r in per_seed])),
            })
    return out


# ── plotting ──────────────────────────────────────────────────────────────────

BLUE, AQUA, YELLOW, VIOLET, MUTE = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#9aa09a"
INK, GRID, SURFACE = "#16191a", "#e1e0d9", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#7c8384", labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)


def plot_figure(sw: dict, rc: dict, rs: dict, output_dir: Path) -> None:
    fig = plt.figure(figsize=(12, 8.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
    fig.text(0.02, 0.975, "The identifiability toy model: when gating can and cannot help",
             fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.952,
             "A linear inverse problem with an exactly-tunable identifiability gap γ. "
             "Detection tracks input-visibility, not error size.",
             fontsize=9.5, color="#454b4c")

    zg = [d["gamma"] for d in sw["zeroed"]]
    rg = [d["gamma"] for d in sw["resampled"]]

    # Panel A: detection AUROC vs gamma
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(zg, [d["detection_auroc_mean"] for d in sw["zeroed"]], "-o", color=BLUE,
            markersize=4, label="zeroed (input-visible)")
    ax.plot(rg, [d["detection_auroc_mean"] for d in sw["resampled"]], "-o", color=YELLOW,
            markersize=4, label="resampled (in-distribution)")
    ax.axhline(0.5, ls="--", color=MUTE, lw=1.2)
    ax.text(ax.get_xlim()[1], 0.5, " chance", va="center", fontsize=7.5, color="#7c8384")
    ax.set_xlabel("identifiability gap  γ", fontsize=9)
    ax.set_ylabel("answerability detection AUROC", fontsize=9)
    ax.set_title("A · Detecting unanswerability", fontsize=10, color=INK, loc="left")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    _style(ax)

    # Panel B: selective gain vs gamma
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(zg, [d["selective_gain_mean"] for d in sw["zeroed"]], "-o", color=BLUE,
            markersize=4, label="zeroed (input-visible)")
    ax.plot(rg, [d["selective_gain_mean"] for d in sw["resampled"]], "-o", color=YELLOW,
            markersize=4, label="resampled (in-distribution)")
    ax.axhline(0.0, ls="--", color=MUTE, lw=1.2)
    ax.text(ax.get_xlim()[1], 0.0, " random", va="center", fontsize=7.5, color="#7c8384")
    ax.set_xlabel("identifiability gap  γ", fontsize=9)
    ax.set_ylabel("gating benefit captured\n(0 = random, 1 = oracle)", fontsize=9)
    ax.set_title("B · Benefiting from abstention", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    _style(ax)

    # Panel C: risk-coverage curves (the canonical gating plot)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(rc["coverage_oracle"], rc["risk_oracle"], color=VIOLET, lw=2, label="oracle (true label)")
    ax.plot(rc["coverage_probe"], rc["risk_probe"], color=BLUE, lw=2, label="activation probe")
    ax.plot(rc["coverage_random"], rc["risk_random"], color=MUTE, lw=2, label="random abstention")
    ax.set_xlabel("coverage  (fraction answered)", fontsize=9)
    ax.set_ylabel("selective risk  (retained MSE)", fontsize=9)
    ax.set_title("C · Risk–coverage: the model restraining itself (zeroed, γ high)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)

    # Panel D: probe-score separation, zeroed vs resampled
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 26)
    ax.hist(rc["conf_answerable"], bins=bins, color=AQUA, alpha=0.6, label="answerable")
    ax.hist(rc["conf_unanswerable"], bins=bins, color=BLUE, alpha=0.6, label="unanswerable · zeroed")
    ax.hist(rs["conf_unanswerable"], bins=bins, color=YELLOW, alpha=0.5,
            label="unanswerable · resampled")
    ax.set_xlabel("probe confidence  P(answerable)", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title("D · Probe scores: zeroed separates, resampled does not",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    _style(ax)

    out = output_dir / "toy_gating.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"[toy] wrote {out}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/toy_identifiability"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--k-core", type=int, default=8)
    p.add_argument("--k-periph", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-probe", type=int, default=600)
    p.add_argument("--n-test", type=int, default=800)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--n-f", type=int, default=9)
    p.add_argument("--f-min", type=float, default=0.02)
    p.add_argument("--f-max", type=float, default=0.85)
    p.add_argument("--demo-f", type=float, default=0.6)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seeds = [0]
        args.n_train, args.n_probe, args.n_test = 400, 120, 160
        args.epochs, args.n_f = 40, 4
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inst = make_instrument(args.k_core, args.k_periph, args.sigma, seed=0)
    print(f"[toy] instrument k_core={inst.k_core} k_periph={inst.k_periph} sigma={inst.sigma}", flush=True)
    sw = sweep(inst, args)
    rc = risk_coverage_demo(inst, args, args.seeds[0])
    rs = resampled_scores(inst, args, args.seeds[0])
    report = {
        "claim": (
            "In a linear instrument with exactly-tunable identifiability gap, "
            "probe-based gating benefit scales with gamma for input-visible "
            "corruption but stays at zero for in-distribution corruption of "
            "identical error magnitude -- detectability tracks input-visibility, "
            "not error size."
        ),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "sweep": sw,
    }
    with open(args.output_dir / "sweep.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    # Curves are large arrays; keep them out of JSON, straight into the figure.
    plot_rc = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rc.items()}
    plot_figure(sw, rc, rs, args.output_dir)
    print(f"[toy] wrote {args.output_dir / 'sweep.json'}", flush=True)


if __name__ == "__main__":
    main()
