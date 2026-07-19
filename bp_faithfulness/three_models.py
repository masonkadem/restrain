"""Act 2 aligned with Act 1: the SAME three-model story on the tube-load BP
simulator. Trains faithful / unfaithful / shortcut, runs the audits, and draws
one figure: the tube-load circuit, the counterfactual response, and the scorecard
(conventional checks agree; only the causal audit isolates the faithful model).

    python three_models.py [--quick]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from bpf import analysis as A
from bpf.config import load_config, seed_everything
from bpf.dataset import generate_dataset, subject_split

SHADE = {"faithful": "#1a1a1a", "unfaithful": "#7b7b7b", "shortcut": "#c0c0c0"}
MARK = {"faithful": "o", "unfaithful": "s", "shortcut": "^"}
clip = lambda v: float(np.clip(v, 0, 1))


def evaluate(cfg, regime, seed):
    """Train one regime, then audit it on a fresh T-varying set at its gamma."""
    g = 0.0  # all at gamma=0; only the TRAINING REGIME differs across the three
    train = generate_dataset(cfg, gamma=g, pep_sd_ms=0, seed=seed,
                             t_fixed=(regime == "shortcut"),
                             wrong_sign=(regime == "unfaithful"))
    tr, te = subject_split(train, cfg["sweep"]["train_subjects"], seed=seed)
    model = A.train_model(train, tr, cfg, seed=seed)
    val_mae, _ = A.mae_per_subject(model, train, te)               # accuracy on its own val

    audit = generate_dataset(cfg, gamma=g, pep_sd_ms=0, seed=seed + 99)  # T varies, true T
    a_tr, a_te = subject_split(audit, cfg["sweep"]["train_subjects"], seed=seed)
    Htr = A.representations(model, audit, a_tr); Hte = A.representations(model, audit, a_te)
    tmu, tsd = audit.T[a_tr].mean(), audit.T[a_tr].std() + 1e-9
    ridge = Ridge(alpha=cfg["probe"]["ridge_alpha"]).fit(Htr, (audit.T[a_tr] - tmu) / tsd)
    probe_r2 = r2_score((audit.T[a_te] - tmu) / tsd, ridge.predict(Hte))
    w = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-9)
    dv = A.donor_swap(model, audit, a_te, w, cfg)
    return dict(val_mae=val_mae, probe_r2=probe_r2,
                slope=dv["slope"], sign=dv["sign_acc"], x=dv["x"], shift=dv["shift"])


def zig(ax, x0, x1, y, n=6, h=0.02):
    xs = np.linspace(x0, x1, 2 * n + 1); ys = np.full_like(xs, y)
    ys[1:-1:2] += h; ys[2:-1:2] -= h
    ax.plot(xs, ys, color="#1a1a1a", lw=1.2)


def draw_circuit(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Tube-load model (electrical analog)", fontsize=10, fontweight="bold", loc="left")
    # source
    ax.add_patch(plt.Circle((0.06, 0.6), 0.035, fill=False, lw=1.3))
    ax.text(0.06, 0.6, "~", ha="center", va="center", fontsize=13)
    ax.text(0.06, 0.51, "heart\n(pressure)", ha="center", va="top", fontsize=7.5)
    # transmission line (tube)
    ax.add_patch(Rectangle((0.16, 0.55), 0.34, 0.10, fill=False, lw=1.3))
    ax.text(0.33, 0.60, "uniform elastic tube\nZc,  one-way delay T", ha="center", va="center", fontsize=8)
    ax.plot([0.095, 0.16], [0.6, 0.6], color="#1a1a1a", lw=1.2)
    # 3-element Windkessel load: R1 series, then R2 || C to ground
    ax.plot([0.50, 0.56], [0.6, 0.6], color="#1a1a1a", lw=1.2)
    zig(ax, 0.56, 0.68, 0.6); ax.text(0.62, 0.66, "R1", ha="center", fontsize=8)
    ax.plot([0.68, 0.74], [0.6, 0.6], color="#1a1a1a", lw=1.2)
    # node -> two parallel branches down to ground
    ax.plot([0.74, 0.74], [0.6, 0.72], color="#1a1a1a", lw=1.2)  # up to R2 branch
    zig(ax, 0.74, 0.86, 0.72); ax.text(0.80, 0.78, "R2", ha="center", fontsize=8)
    ax.plot([0.86, 0.86], [0.72, 0.30], color="#1a1a1a", lw=1.2)
    ax.plot([0.74, 0.74], [0.6, 0.48], color="#1a1a1a", lw=1.2)   # down to C branch
    ax.plot([0.705, 0.775], [0.48, 0.48], color="#1a1a1a", lw=1.6)  # cap plate
    ax.plot([0.715, 0.765], [0.455, 0.455], color="#1a1a1a", lw=1.6)
    ax.text(0.80, 0.47, "C", ha="center", fontsize=8)
    ax.plot([0.74, 0.74], [0.455, 0.30], color="#1a1a1a", lw=1.2)
    ax.plot([0.66, 0.94], [0.30, 0.30], color="#1a1a1a", lw=1.2)  # ground rail
    for gx in (0.74, 0.86):
        ax.plot([gx, gx], [0.30, 0.30], color="#1a1a1a")
    ax.plot([0.78, 0.82], [0.26, 0.26], color="#1a1a1a", lw=1.2)
    ax.plot([0.795, 0.805], [0.23, 0.23], color="#1a1a1a", lw=1.2)
    ax.text(0.80, 0.34, "3-element Windkessel load  Z_L", ha="center", fontsize=7.5)
    # measurement points
    ax.plot([0.16], [0.55], marker="v", color="#3a6ea5", ms=8)
    ax.text(0.16, 0.50, "proximal", color="#3a6ea5", ha="center", fontsize=7.5)
    ax.plot([0.50], [0.55], marker="v", color="#b04a3a", ms=8)
    ax.text(0.50, 0.50, "distal", color="#b04a3a", ha="center", fontsize=7.5)
    ax.text(0.5, 0.14,
            r"$\Gamma=\dfrac{Z_L-Z_c}{Z_L+Z_c}$    "
            r"$H(f)=\dfrac{(1+\Gamma)e^{-j2\pi fT}}{1+\Gamma e^{-j4\pi fT}}$",
            ha="center", va="center", fontsize=10)
    ax.text(0.5, 0.03, "distal = forward wave (arrives at T) + echoes;  BP = K1/T + K2",
            ha="center", fontsize=7.5, color="#555", style="italic")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(); seed_everything(cfg.seed)
    if args.quick:
        cfg.raw["subjects"]["samples_per_subject"] = 60
        cfg.raw["model"]["epochs"] = 40

    print("training faithful / unfaithful / shortcut ...", flush=True)
    R = {r: evaluate(cfg, r, cfg.seed) for r in ("faithful", "unfaithful", "shortcut")}
    for r, v in R.items():
        print(f"  {r:11s} valMAE={v['val_mae']:.2f} probeR2={v['probe_r2']:.2f} "
              f"|slope|={abs(v['slope']):.1f} sign={v['sign']:.2f}")

    # ---- scorecard scores in [0,1] "looks faithful" ----
    ref_slope = abs(R["faithful"]["slope"]) + 1e-9
    ref_mae = max(v["val_mae"] for v in R.values()) + 1e-9
    rows = ["validation accuracy", "probe R² (true PTT)",
            "PTT moves the output? (mag.)", "causal audit: sign vs physics"]
    verdict = {rows[0]: "all high — can't separate", rows[1]: "misses the unfaithful model",
               rows[2]: "misses the unfaithful model", rows[3]: "ISOLATES the faithful model"}
    score = {r: {
        rows[0]: clip(1 - R[r]["val_mae"] / (2 * ref_mae)),
        rows[1]: clip(R[r]["probe_r2"]),
        rows[2]: clip(abs(R[r]["slope"]) / ref_slope),
        rows[3]: clip(R[r]["sign"]),
    } for r in R}

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
    draw_circuit(ax[0])

    # counterfactual: donor-swap shift vs physics for each model
    b = ax[1]
    b.axhline(0, color="#999", lw=.6); b.axvline(0, color="#999", lw=.6)
    for r in R:
        b.scatter(R[r]["x"], R[r]["shift"], s=7, color=SHADE[r], alpha=.5,
                  edgecolor="none", label=r)
    b.set_xlabel("physics  Δ(1/T)  donor − base"); b.set_ylabel("predicted SBP shift")
    b.set_title("Donor-swap: does the output move as physics says?", fontsize=10, fontweight="bold")
    b.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)

    c = ax[2]
    ys = np.arange(len(rows))[::-1]; off = {"faithful": 0.16, "unfaithful": 0.0, "shortcut": -0.16}
    for y, row in zip(ys, rows):
        c.axhspan(y - 0.5, y + 0.5, color="#eef3ee" if row == rows[-1] else "#f6f6f4", zorder=0)
        for r in R:
            c.scatter(score[r][row], y + off[r], s=85, marker=MARK[r], color=SHADE[r],
                      edgecolor="#1a1a1a", lw=.5, zorder=3)
        c.text(-0.03, y, row, ha="right", va="center", fontsize=8.8)
        sep = row == rows[-1]
        c.text(1.03, y, verdict[row], ha="left", va="center", fontsize=7.6,
               fontweight="bold" if sep else "normal", color="#1a1a1a" if sep else "#8a8a86")
    c.set_xlim(-0.03, 1.03); c.set_ylim(-0.6, len(rows) - .4); c.set_yticks([])
    c.set_xlabel("looks faithful  (0 → 1)")
    c.set_title("Only the causal audit separates the three models", fontsize=10, fontweight="bold")
    for s in ("top", "right", "left"):
        c.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], marker=MARK[r], ls="none", color=SHADE[r], mec="#1a1a1a",
                          ms=8, label=r) for r in R]
    c.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.28))

    fig.suptitle("Act 2 — tube-load BP simulator: same three-model story as the abstract task",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    out = Path(cfg["paths"]["results_dir"]); out.mkdir(exist_ok=True)
    fig.savefig(out / "bp_three_models.png", dpi=165, bbox_inches="tight")
    print(f"wrote {out}/bp_three_models.png")


if __name__ == "__main__":
    main()
