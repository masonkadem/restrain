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

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "axes.edgecolor": "#3a3a3a", "axes.linewidth": 0.9,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 180,
})
SHADE = {"faithful": "#2f4b7c", "unfaithful": "#c1543b", "shortcut": "#9aa0a6"}
MARK = {"faithful": "o", "unfaithful": "s", "shortcut": "^"}
INK, MUT, WIRE = "#22262b", "#8b8b8b", "#33383d"
PROX, DIST = "#33475c", "#8199b0"
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
    ax.plot(xs, ys, color="#33383d", lw=1.2)


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
    ax.plot([0.095, 0.16], [0.6, 0.6], color="#33383d", lw=1.2)
    # 3-element Windkessel load: R1 series, then R2 || C to ground
    ax.plot([0.50, 0.56], [0.6, 0.6], color="#33383d", lw=1.2)
    zig(ax, 0.56, 0.68, 0.6); ax.text(0.62, 0.66, "R1", ha="center", fontsize=8)
    ax.plot([0.68, 0.74], [0.6, 0.6], color="#33383d", lw=1.2)
    # node -> two parallel branches down to ground
    ax.plot([0.74, 0.74], [0.6, 0.72], color="#33383d", lw=1.2)  # up to R2 branch
    zig(ax, 0.74, 0.86, 0.72); ax.text(0.80, 0.78, "R2", ha="center", fontsize=8)
    ax.plot([0.86, 0.86], [0.72, 0.30], color="#33383d", lw=1.2)
    ax.plot([0.74, 0.74], [0.6, 0.48], color="#33383d", lw=1.2)   # down to C branch
    ax.plot([0.705, 0.775], [0.48, 0.48], color="#33383d", lw=1.6)  # cap plate
    ax.plot([0.715, 0.765], [0.455, 0.455], color="#33383d", lw=1.6)
    ax.text(0.80, 0.47, "C", ha="center", fontsize=8)
    ax.plot([0.74, 0.74], [0.455, 0.30], color="#33383d", lw=1.2)
    ax.plot([0.66, 0.94], [0.30, 0.30], color="#33383d", lw=1.2)  # ground rail
    for gx in (0.74, 0.86):
        ax.plot([gx, gx], [0.30, 0.30], color="#33383d")
    ax.plot([0.78, 0.82], [0.26, 0.26], color="#33383d", lw=1.2)
    ax.plot([0.795, 0.805], [0.23, 0.23], color="#33383d", lw=1.2)
    ax.text(0.80, 0.34, "3-element Windkessel load  Z_L", ha="center", fontsize=7.5)
    # measurement points
    ax.plot([0.16], [0.55], marker="v", color="#33475c", ms=8)
    ax.text(0.16, 0.50, "proximal", color="#33475c", ha="center", fontsize=7.5)
    ax.plot([0.50], [0.55], marker="v", color="#8199b0", ms=8)
    ax.text(0.50, 0.50, "distal", color="#8199b0", ha="center", fontsize=7.5)
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

    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.4), gridspec_kw={"width_ratios": [1.1, 1, 1.35]})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.16, wspace=0.28)
    draw_circuit(ax[0])

    # ---- counterfactual: donor-swap shift vs physics ----
    b = ax[1]
    b.axhline(0, color="#d0d0d0", lw=1); b.axvline(0, color="#d0d0d0", lw=1)
    for r in ("shortcut", "unfaithful", "faithful"):
        b.scatter(R[r]["x"], R[r]["shift"], s=13, color=SHADE[r], alpha=.55,
                  edgecolor="none", label=r)
    b.set_xlabel("physics  Δ(1/T)   donor − base"); b.set_ylabel("predicted SBP shift")
    b.set_title("Donor-swap: does the output move\nas physics predicts?", loc="left",
                fontweight="bold", pad=10)
    b.legend(frameon=False, fontsize=9, loc="upper left")

    # ---- scorecard (dumbbell), matched to the Act-1 hero ----
    c = ax[2]
    short = {rows[0]: "agree", rows[1]: "misses unfaithful",
             rows[2]: "misses unfaithful", rows[3]: "SEPARATES"}
    label = {rows[0]: "validation accuracy", rows[1]: "probe R²  (decodability)",
             rows[2]: 'gradient  ("uses PTT?")', rows[3]: "interchange audit  (causal)"}
    ys = np.arange(len(rows))[::-1]
    for y, row in zip(ys, rows):
        is_audit = row == rows[-1]
        if is_audit:
            c.axhspan(y - 0.5, y + 0.5, color="#eef1f6", zorder=0)
        vals = [score[r][row] for r in R]
        c.plot([min(vals), max(vals)], [y, y], color="#d7d7d7", lw=2.2, zorder=1,
               solid_capstyle="round")
        for r in R:
            c.scatter(score[r][row], y, s=150, marker=MARK[r], color=SHADE[r],
                      edgecolor="white", linewidth=1.3, zorder=3)
        c.text(-0.02, y, label[row], ha="right", va="center", fontsize=10)
        c.text(1.06, y, short[row], ha="left", va="center",
               fontsize=9.5 if is_audit else 9, fontweight="bold" if is_audit else "normal",
               color=SHADE["faithful"] if is_audit else MUT)
    c.set_xlim(-0.02, 1.45); c.set_ylim(-0.6, len(rows) - .35); c.set_yticks([])
    c.set_xticks([0, 0.5, 1.0]); c.set_xlabel("looks faithful   (0  →  1)")
    c.spines["left"].set_visible(False)
    c.set_title("Conventional checks can't separate them —\nthe causal audit can", loc="left",
                fontweight="bold", pad=10)
    handles = [plt.Line2D([], [], marker=MARK[r], ls="none", color=SHADE[r], mec="white",
                          mew=1.2, ms=11, label=r) for r in R]
    c.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
             bbox_to_anchor=(0.34, -0.26), handletextpad=0.4, columnspacing=1.6)

    fig.suptitle("Act 2 — tube-load BP simulator: the same three-model story on real waveforms",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    out = Path(cfg["paths"]["results_dir"]); out.mkdir(exist_ok=True)
    fig.savefig(out / "bp_three_models.png", bbox_inches="tight")
    print(f"wrote {out}/bp_three_models.png")


if __name__ == "__main__":
    main()
