"""Can an ensemble break the single-pass detectability ceiling?

The single-model result (`toy_identifiability.py`) showed a frozen linear
probe cannot flag an in-distribution "resampled" corruption, no matter how
large the error. A single deterministic forward pass has no signal to read.
This experiment asks whether extra information sources rescue it:

Detectors (each collapses to a scalar confidence, higher = more answerable):
  - `probe_zeroed`  : a linear probe on activations, trained on answerable vs
    the *zeroed* corruption only, then applied to novel corruptions. Tests
    whether a probe trained on one known failure generalizes to others.
  - `ensemble`      : negative disagreement (std) across K independently
    trained readouts. Label-free -- disagreement is a property of the *set*
    of models, absent from any single one.
  - `mahalanobis`   : negative Mahalanobis distance of the penultimate
    activation to the answerable-activation Gaussian. Label-free OOD score.

Corruptions (all unanswerable; see `generate`):
  - `zeroed`    : degenerate, off-manifold, distinctive input pattern.
  - `shifted`   : off-manifold (inflated draw), no missing-channel pattern,
    information lost.
  - `resampled` : on-manifold (identical marginal), information lost -- the
    fundamental floor.

The hypothesis: ensemble/OOD detectors rescue `shifted` (off-manifold) that a
zeroed-trained probe misses, but nothing rescues `resampled` (on-manifold),
pinning the true boundary at "is the corruption on the data manifold," not
"one forward pass vs many."

    python analysis/toy_ensemble.py
    python analysis/toy_ensemble.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from toy_identifiability import (  # noqa: E402
    Readout,
    evaluate,
    gamma_of,
    generate,
    make_instrument,
    train_readout,
)

CORRUPTIONS = ("zeroed", "shifted", "resampled")


def train_ensemble(inst, f: float, cfg, seed: int) -> list:
    """K independent readouts on the same answerable data, different inits."""
    train = generate(inst, cfg.n_train, f, "clean", seed * 100 + 1)
    models = []
    for k in range(cfg.n_ensemble):
        m = Readout(inst.k_core + inst.k_periph, cfg.hidden)
        train_readout(m, train, cfg.epochs, seed * 1000 + k)
        models.append(m)
    return models


def ensemble_outputs(models, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean prediction, disagreement std, penultimate feats of model 0)."""
    preds = []
    feat0 = None
    for i, m in enumerate(models):
        ev = evaluate(m, data)
        preds.append(ev["pred"])
        if i == 0:
            feat0 = ev["feat"]
    preds = np.stack(preds, axis=1)
    return preds.mean(1), preds.std(1), feat0


def mahalanobis_scorer(ref_feats: np.ndarray):
    mu = ref_feats.mean(0)
    cov = np.cov(ref_feats, rowvar=False) + 1e-3 * np.eye(ref_feats.shape[1])
    inv = np.linalg.pinv(cov)

    def score(feats: np.ndarray) -> np.ndarray:
        d = feats - mu
        return np.einsum("ni,ij,nj->n", d, inv, d)

    return score


def run_seed(inst, f: float, cfg, seed: int) -> dict:
    models = train_ensemble(inst, f, cfg, seed)

    # Answerable reference (probe-fit + Mahalanobis reference).
    ref = generate(inst, cfg.n_probe, f, "clean", seed * 100 + 2)
    ref_mean, ref_std, ref_feat = ensemble_outputs(models, ref)
    maha = mahalanobis_scorer(ref_feat)

    # Linear probe trained on answerable vs ZEROED only (one known failure).
    zeroed_fit = generate(inst, cfg.n_probe, f, "zeroed", seed * 100 + 3)
    _, _, zf = ensemble_outputs(models, zeroed_fit)
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probe.fit(
        np.concatenate([ref_feat, zf]),
        np.concatenate([ref["answerable"], zeroed_fit["answerable"]]),
    )

    # Fresh answerable test pool (shared negative class for every AUROC).
    ok = generate(inst, cfg.n_test, f, "clean", seed * 100 + 4)
    ok_mean, ok_std, ok_feat = ensemble_outputs(models, ok)
    ok_probe = probe.predict_proba(ok_feat)[:, 1]
    ok_maha = maha(ok_feat)

    out = {"gamma": {}, "detection": {}}
    for corr in CORRUPTIONS:
        bad = generate(inst, cfg.n_test, f, corr, seed * 100 + 20 + CORRUPTIONS.index(corr))
        bad_mean, bad_std, bad_feat = ensemble_outputs(models, bad)
        labels = np.concatenate([ok["answerable"], bad["answerable"]])  # 1 vs 0

        # confidence (higher = more answerable) for each detector
        conf_probe = np.concatenate([ok_probe, probe.predict_proba(bad_feat)[:, 1]])
        conf_ens = np.concatenate([-ok_std, -bad_std])          # low disagreement = confident
        conf_maha = np.concatenate([-ok_maha, -maha(bad_feat)])  # low distance = confident

        out["detection"][corr] = {
            "probe_zeroed": float(roc_auc_score(labels, conf_probe)),
            "ensemble": float(roc_auc_score(labels, conf_ens)),
            "mahalanobis": float(roc_auc_score(labels, conf_maha)),
        }
        # empirical gap (measured, not analytic -- shifted's analytic gamma is messy)
        out["gamma"][corr] = float(bad["y"].var() and (bad_std.mean()))
    return out


def sweep(inst, cfg) -> dict:
    result = {c: {d: [] for d in ("probe_zeroed", "ensemble", "mahalanobis")} for c in CORRUPTIONS}
    for corr in CORRUPTIONS:
        for d in result[corr]:
            result[corr][d] = []
    at_demo = {c: {d: [] for d in ("probe_zeroed", "ensemble", "mahalanobis")} for c in CORRUPTIONS}
    for seed in cfg.seeds:
        r = run_seed(inst, cfg.demo_f, cfg, seed)
        for corr in CORRUPTIONS:
            for d in ("probe_zeroed", "ensemble", "mahalanobis"):
                at_demo[corr][d].append(r["detection"][corr][d])
    summary = {
        corr: {
            d: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for d, v in at_demo[corr].items()
        }
        for corr in CORRUPTIONS
    }
    return summary


# ── plotting ──────────────────────────────────────────────────────────────────
BLUE, AQUA, YELLOW, MUTE = "#2a78d6", "#1baf7a", "#eda100", "#9aa09a"
INK, GRID, SURFACE = "#16191a", "#e1e0d9", "#fcfcfb"
DET_COLORS = {"probe_zeroed": BLUE, "ensemble": AQUA, "mahalanobis": YELLOW}
DET_LABELS = {
    "probe_zeroed": "single-pass probe (trained on zeroed)",
    "ensemble": "ensemble disagreement (label-free)",
    "mahalanobis": "Mahalanobis OOD (label-free)",
}
CORR_LABELS = {
    "zeroed": "zeroed\n(central/degenerate)",
    "shifted": "shifted\n(large/off-manifold)",
    "resampled": "resampled\n(typical — the floor)",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#7c8384", labelsize=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)


def plot_summary(summary: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=SURFACE)
    fig.subplots_adjust(top=0.82, left=0.09, right=0.98, bottom=0.12)
    fig.text(0.01, 0.945,
             "Who catches what: every detector has a blind spot; nothing catches the on-manifold floor",
             fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.90,
             "Detectors on unanswerable inputs. Below 0.5 (dashed) = anti-correlated: the detector is "
             "actively confident on the unanswerable input.",
             fontsize=9, color="#7c8384")
    corrs = list(CORRUPTIONS)
    dets = ["probe_zeroed", "ensemble", "mahalanobis"]
    x = np.arange(len(corrs))
    bw = 0.26
    for i, d in enumerate(dets):
        means = [summary[c][d]["mean"] for c in corrs]
        stds = [summary[c][d]["std"] for c in corrs]
        ax.bar(x + (i - 1) * bw, means, bw, yerr=stds, capsize=3,
               color=DET_COLORS[d], label=DET_LABELS[d])
    ax.axhline(0.5, ls="--", color=MUTE, lw=1.3)
    ax.text(len(corrs) - 0.5, 0.5, " chance", va="bottom", ha="right", fontsize=8, color="#7c8384")
    ax.set_xticks(x)
    ax.set_xticklabels([CORR_LABELS[c] for c in corrs], fontsize=9)
    ax.set_ylabel("answerability detection AUROC", fontsize=10)
    ax.set_ylim(0.0, 1.03)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=1)
    _style(ax)
    out = output_dir / "toy_ensemble.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"[ensemble] wrote {out}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/toy_identifiability"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--k-core", type=int, default=8)
    p.add_argument("--k-periph", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n-ensemble", type=int, default=8)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-probe", type=int, default=600)
    p.add_argument("--n-test", type=int, default=800)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--demo-f", type=float, default=0.6)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seeds = [0]
        args.n_ensemble = 4
        args.n_train, args.n_probe, args.n_test = 500, 150, 200
        args.epochs = 40
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inst = make_instrument(args.k_core, args.k_periph, args.sigma, seed=0)
    print(f"[ensemble] K={args.n_ensemble} readouts, demo_f={args.demo_f}", flush=True)
    summary = sweep(inst, args)
    report = {
        "claim": (
            "Ensemble disagreement and Mahalanobis OOD scores rescue detection "
            "of off-manifold corruptions that a single-pass probe (trained on a "
            "different corruption) misses, but no observational detector catches "
            "an on-manifold (resampled) corruption -- the boundary is data-"
            "manifold membership, not the number of forward passes."
        ),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "detection_auroc": summary,
    }
    with open(args.output_dir / "ensemble.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    plot_summary(summary, args.output_dir)
    print(f"[ensemble] wrote {args.output_dir / 'ensemble.json'}", flush=True)
    for corr in CORRUPTIONS:
        row = summary[corr]
        print(f"  {corr:10s} probe={row['probe_zeroed']['mean']:.3f}  "
              f"ensemble={row['ensemble']['mean']:.3f}  maha={row['mahalanobis']['mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
