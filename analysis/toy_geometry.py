"""Geometry of the answerability representation in the toy model.

Given the frozen readout from `toy_identifiability.py`, this asks *how* the
"this input is unanswerable" signal is arranged in activation space, in the
style of mechanistic-interpretability geometry studies:

  1. **Rank** — how many linear directions carry answerability? Measured by
     iterative null-space projection (INLP): fit a linear probe, record its
     detection AUROC, erase that direction, and repeat. The number of
     directions needed to drive AUROC to chance is the effective rank of the
     answerability subspace.

  2. **Concentration vs gamma** — is the signal rank-1? Compare the AUROC of
     the *single* mean-difference direction to the full-probe AUROC across the
     identifiability gap. A ratio near 1 means one direction suffices.

  3. **Geometry** — a 2D projection (answerability axis x top orthogonal PC)
     of answerable, zeroed-unanswerable, and resampled-unanswerable
     activations, showing the separation (zeroed) and the non-separation
     (resampled) directly.

    python analysis/toy_geometry.py
    python analysis/toy_geometry.py --quick
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


def mean_difference_direction(feats: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Unit vector from the unanswerable-class mean to the answerable-class mean."""
    d = feats[labels == 1].mean(0) - feats[labels == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def auroc_along(feats: np.ndarray, labels: np.ndarray, direction: np.ndarray) -> float:
    s = feats @ direction
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, s))


def inlp_rank_curve(
    feats: np.ndarray, labels: np.ndarray, max_dirs: int = 8, seed: int = 0
) -> list[float]:
    """Detection AUROC after iteratively erasing the top linear probe direction.

    Returns [AUROC_full, AUROC_after_1_removed, AUROC_after_2_removed, ...].
    The number of removals needed to reach ~chance is the effective rank.
    """
    x = feats.copy()
    aurocs = []
    n = len(labels)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    for _ in range(max_dirs + 1):
        clf = LogisticRegression(max_iter=2000).fit(x[tr], labels[tr])
        score = clf.decision_function(x[te])
        aurocs.append(float(roc_auc_score(labels[te], score)))
        w = clf.coef_.reshape(-1)
        w = w / (np.linalg.norm(w) + 1e-12)
        # project the whole feature matrix onto the null space of w
        x = x - np.outer(x @ w, w)
    return aurocs


def run_geometry(inst, cfg, seed: int, f: float, corruption: str = "zeroed") -> dict:
    train = generate(inst, cfg.n_train, f, "clean", seed * 100 + 1)
    model = Readout(inst.k_core + inst.k_periph, cfg.hidden)
    train_readout(model, train, cfg.epochs, seed)
    ok = evaluate(model, generate(inst, cfg.n_test, f, "clean", seed * 100 + 4))
    bad = evaluate(model, generate(inst, cfg.n_test, f, corruption, seed * 100 + 5))
    feats = np.concatenate([ok["feat"], bad["feat"]])
    labels = np.concatenate([ok["answerable"], bad["answerable"]])
    scaler = StandardScaler().fit(feats)
    fz = scaler.transform(feats)

    full = LogisticRegression(max_iter=2000).fit(fz, labels)
    full_auroc = float(roc_auc_score(labels, full.decision_function(fz)))
    d1 = mean_difference_direction(fz, labels)
    top1_auroc = auroc_along(fz, labels, d1)
    rank_curve = inlp_rank_curve(fz, labels, max_dirs=min(cfg.hidden, 12), seed=seed)
    # effective rank: removals until AUROC drops below 0.55
    eff_rank = next((i for i, a in enumerate(rank_curve) if a < 0.55), len(rank_curve) - 1)
    return {
        "f": f, "gamma": gamma_of(f, inst.sigma, corruption),
        "full_auroc": full_auroc, "top1_auroc": top1_auroc,
        "concentration": float(top1_auroc / full_auroc) if full_auroc > 0.5 else float("nan"),
        "rank_curve": rank_curve, "effective_rank": eff_rank,
    }


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


def plot_geometry(inst, cfg, output_dir: Path) -> dict:
    seed = cfg.seeds[0]
    # rank curve + concentration sweep (zeroed)
    rank_ref = run_geometry(inst, cfg, seed, cfg.demo_f, "zeroed")
    fs = np.linspace(cfg.f_min, cfg.f_max, cfg.n_f)
    conc = []
    for f in fs:
        per_seed = [run_geometry(inst, cfg, s, float(f), "zeroed") for s in cfg.seeds]
        conc.append({
            "gamma": float(np.mean([r["gamma"] for r in per_seed])),
            "concentration": float(np.mean([r["concentration"] for r in per_seed])),
            "effective_rank": float(np.mean([r["effective_rank"] for r in per_seed])),
        })

    # 2D projection scatter at demo_f
    train = generate(inst, cfg.n_train, cfg.demo_f, "clean", seed * 100 + 1)
    model = Readout(inst.k_core + inst.k_periph, cfg.hidden)
    train_readout(model, train, cfg.epochs, seed)
    ok = evaluate(model, generate(inst, cfg.n_test, cfg.demo_f, "clean", seed * 100 + 4))
    zer = evaluate(model, generate(inst, cfg.n_test, cfg.demo_f, "zeroed", seed * 100 + 5))
    res = evaluate(model, generate(inst, cfg.n_test, cfg.demo_f, "resampled", seed * 100 + 6))
    scaler = StandardScaler().fit(np.concatenate([ok["feat"], zer["feat"]]))
    d1 = mean_difference_direction(
        scaler.transform(np.concatenate([ok["feat"], zer["feat"]])),
        np.concatenate([ok["answerable"], zer["answerable"]]),
    )
    # second axis: top PC of answerable activations orthogonal to d1
    fok = scaler.transform(ok["feat"])
    fok_perp = fok - np.outer(fok @ d1, d1)
    _, _, vt = np.linalg.svd(fok_perp - fok_perp.mean(0), full_matrices=False)
    d2 = vt[0]

    def proj(feat):
        fz = scaler.transform(feat)
        return fz @ d1, fz @ d2

    mean_conc = float(np.mean([c["concentration"] for c in conc]))
    fig = plt.figure(figsize=(12.5, 4.5), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, wspace=0.34, top=0.80, bottom=0.14, left=0.07, right=0.98)
    fig.text(0.02, 0.94, "Geometry of the answerability signal", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.885,
             "One dominant axis carries most of it, but the signal is not rank-1 — "
             "the tail is smeared over ~the corrupted-block dimensionality.",
             fontsize=9, color="#454b4c")

    # Panel A: INLP rank curve
    ax = fig.add_subplot(gs[0, 0])
    rc = rank_ref["rank_curve"]
    ax.plot(range(len(rc)), rc, "-o", color=BLUE, markersize=4)
    ax.axhline(0.5, ls="--", color=MUTE, lw=1.2)
    ax.text(len(rc) - 1, 0.5, " chance", va="bottom", ha="right", fontsize=7.5, color="#7c8384")
    ax.set_xlabel("linear directions erased", fontsize=9)
    ax.set_ylabel("detection AUROC", fontsize=9)
    ax.set_title("A · Not rank-1: a diffuse tail resists erasure", fontsize=9.5, color=INK, loc="left")
    ax.set_ylim(0.45, 1.0)
    _style(ax)

    # Panel B: 2D geometry (draw zeroed first, resampled last so the
    # answerable/resampled overlap — the key comparison — stays visible)
    ax = fig.add_subplot(gs[0, 1])
    xo, yo = proj(ok["feat"]); xz, yz = proj(zer["feat"]); xr, yr = proj(res["feat"])
    ax.scatter(xz, yz, s=9, color=BLUE, alpha=0.45, edgecolors="none", label="unans · zeroed")
    ax.scatter(xo, yo, s=9, color=AQUA, alpha=0.55, edgecolors="none", label="answerable")
    ax.scatter(xr, yr, s=9, color=YELLOW, alpha=0.45, edgecolors="none", label="unans · resampled")
    ax.set_xlabel("answerability axis", fontsize=9)
    ax.set_ylabel("top orthogonal PC", fontsize=9)
    ax.set_title("B · Zeroed shifts left; resampled overlaps answerable", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _style(ax)

    # Panel C: concentration vs gamma
    ax = fig.add_subplot(gs[0, 2])
    g = [c["gamma"] for c in conc]
    ax.plot(g, [c["concentration"] for c in conc], "-o", color=VIOLET, markersize=4)
    ax.axhline(1.0, ls="--", color=MUTE, lw=1.2)
    ax.text(g[-1], 1.0, " rank-1", va="bottom", ha="right", fontsize=7.5, color="#7c8384")
    ax.set_xlabel("identifiability gap  γ", fontsize=9)
    ax.set_ylabel("top-1 AUROC / full AUROC", fontsize=9)
    ax.set_title(f"C · One direction ≈ {mean_conc:.0%} of the signal", fontsize=9.5, color=INK, loc="left")
    ax.set_ylim(0.7, 1.05)
    _style(ax)

    out = output_dir / "toy_geometry.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[geometry] wrote {out}", flush=True)
    return {"rank_reference": rank_ref, "concentration_sweep": conc}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/toy_identifiability"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--k-core", type=int, default=8)
    p.add_argument("--k-periph", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=800)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--n-f", type=int, default=7)
    p.add_argument("--f-min", type=float, default=0.05)
    p.add_argument("--f-max", type=float, default=0.85)
    p.add_argument("--demo-f", type=float, default=0.6)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seeds = [0]
        args.n_train, args.n_test = 500, 200
        args.epochs, args.n_f = 40, 3
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inst = make_instrument(args.k_core, args.k_periph, args.sigma, seed=0)
    print(f"[geometry] instrument k_core={inst.k_core} k_periph={inst.k_periph}", flush=True)
    result = plot_geometry(inst, args, args.output_dir)
    with open(args.output_dir / "geometry.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[geometry] wrote {args.output_dir / 'geometry.json'}", flush=True)


if __name__ == "__main__":
    main()
