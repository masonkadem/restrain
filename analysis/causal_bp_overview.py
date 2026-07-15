"""Render a single comprehensive overview figure for the causal-mediation BP audit.

The figure is a self-contained graphical abstract: it draws the data-generating
process, the model and the causal-patching intervention, shows compact
pseudo-code, and embeds the real results loaded from ``causal_bp_summary.json``.

    python analysis/causal_bp_overview.py

Run ``physics_law_credibility.py --experiment causal_bp`` first so the summary
exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from physics_law_credibility import (  # noqa: E402
    LINEAR_BP_A,
    LINEAR_BP_B,
    WIN_LEN,
    build_linear_bp_dataset,
    generate_linear_bp_sample,
)

# Palette (brand-neutral, colour-blind friendly).
C_PTT = "#2f7d4f"       # PTT / causal path (green)
C_CONF = "#d98a1f"      # confound / shortcut (orange)
C_QUERY = "#4b6bbf"     # query stream (blue)
C_CTX = "#6a4ca8"       # context / representation (purple)
C_INK = "#222222"
C_MUTE = "#6b7280"
C_BOX = "#f4f4f5"


def _box(ax, x, y, w, h, text, fc=C_BOX, ec=C_INK, fontsize=9, weight="normal", tc=C_INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, color=tc, weight=weight, zorder=3,
    )


def _arrow(ax, x0, y0, x1, y1, color=C_INK, lw=1.8, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
            linewidth=lw, color=color, linestyle=ls, zorder=1,
            shrinkA=2, shrinkB=2,
        )
    )


# ── Panel A: data-generating process ───────────────────────────────────────────


def panel_data(ax):
    ax.set_title("A · Data-generating process", loc="left", fontsize=12, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sample = generate_linear_bp_sample(
        np.random.default_rng(3), "clean", confound_corr=0.95
    )
    half = WIN_LEN // 2
    t = np.linspace(0.06, 0.62, WIN_LEN)

    def draw_stream(y0, yscale, arr, color, seg=None):
        base = np.asarray(arr, dtype=float)
        base = base - base.mean()
        base = base / (np.abs(base).max() + 1e-8)
        y = y0 + yscale * base
        if seg is None:
            ax.plot(t, y, color=color, lw=1.1)
        else:
            ax.plot(t[:half], y[:half], color=seg[0], lw=1.1)
            ax.plot(t[half:], y[half:], color=seg[1], lw=1.1)

    # Query stream (proximal)
    ax.text(0.02, 0.86, "stream A\n(query)", fontsize=8, color=C_QUERY, va="center")
    draw_stream(0.84, 0.07, sample.proximal, C_QUERY)

    # Value stream (distal): first half = PTT, second half = confound
    ax.text(0.02, 0.60, "stream B\n(values)", fontsize=8, color=C_INK, va="center")
    draw_stream(0.58, 0.07, sample.distal, None, seg=(C_PTT, C_CONF))
    ax.axvline(t[half], 0.30, 0.62, color=C_MUTE, ls=":", lw=1)
    ax.text(0.20, 0.44, "PTT level", fontsize=8, color=C_PTT, ha="center", weight="bold")
    ax.text(0.48, 0.44, "confound level", fontsize=8, color=C_CONF, ha="center", weight="bold")

    # Governing law + confound rule
    ax.text(
        0.70, 0.90,
        f"Governing law\nBP = a·PTT + b\na = {LINEAR_BP_A}  b = {LINEAR_BP_B:.0f}",
        fontsize=9.5, va="top", ha="left", color=C_INK,
        bbox=dict(boxstyle="round", fc="#eef6f0", ec=C_PTT, lw=1.2),
    )
    ax.text(
        0.70, 0.55,
        "Confound z (amplitude of\nstream-B 2nd half):\n"
        "• train: z ≈ PTT  (shortcut)\n• OOD:  z independent of PTT",
        fontsize=9, va="top", ha="left", color=C_INK,
        bbox=dict(boxstyle="round", fc="#fbf1e2", ec=C_CONF, lw=1.2),
    )
    ax.text(
        0.02, 0.14,
        "PTT is the legitimate mediator of BP; the confound is a spurious cue that only\n"
        "predicts BP because it tracks PTT during training. Both live in the value stream\n"
        "(a mean-pooling model can read a channel level, not a sub-sample delay).",
        fontsize=8.2, color=C_MUTE, va="top",
    )


# ── Panel B: train vs OOD coupling ─────────────────────────────────────────────


def panel_coupling(ax):
    ax.set_title("B · Confound coupling: train vs OOD", loc="left", fontsize=12, weight="bold")
    train = build_linear_bp_dataset(220, "clean", confound_corr=0.95, seed=11)
    ood = build_linear_bp_dataset(220, "ood", confound_corr=0.95, seed=12)
    ptt_tr = [s.ptt_ms for s in train]
    z_tr = [s.confound for s in train]
    ptt_o = [s.ptt_ms for s in ood]
    z_o = [s.confound for s in ood]
    ax.scatter(ptt_tr, z_tr, s=12, color=C_CONF, alpha=0.75, label="train  (z ≈ PTT)")
    ax.scatter(ptt_o, z_o, s=12, color=C_MUTE, alpha=0.6, label="OOD  (z independent)")
    ax.set_xlabel("PTT (ms)", fontsize=9)
    ax.set_ylabel("confound z", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.tick_params(labelsize=8)
    ax.text(
        0.5, -0.30,
        "In training the shortcut works; the OOD split decorrelates it while the\n"
        "causal PTT→BP path is untouched. Accuracy on train can hide the reliance.",
        transform=ax.transAxes, ha="center", fontsize=8.2, color=C_MUTE,
    )


# ── Panel C: model + causal-patch schematic ────────────────────────────────────


def panel_pipeline(ax):
    ax.set_title(
        "C · Model, then the causal-fidelity intervention (no test labels)",
        loc="left", fontsize=12, weight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Forward path (top row)
    _box(ax, 0.02, 0.62, 0.15, 0.24,
         "two streams\nA (query)\nB (values:\nPTT | conf)", fc="#eef2fb", ec=C_QUERY, fontsize=8)
    _box(ax, 0.24, 0.64, 0.16, 0.20, "cross-\nattention", fc="#efeaf7", ec=C_CTX, fontsize=9, weight="bold")
    _box(ax, 0.47, 0.66, 0.14, 0.16, "context\nctx  ∈ ℝ^d", fc="#efeaf7", ec=C_CTX, fontsize=9)
    _box(ax, 0.68, 0.66, 0.13, 0.16, "BP head\n(linear)", fc="#f4f4f5", ec=C_INK, fontsize=9)
    _box(ax, 0.87, 0.66, 0.11, 0.16, "BP\n(mmHg)", fc="#eef6f0", ec=C_PTT, fontsize=9, weight="bold")
    for x0, x1 in [(0.17, 0.24), (0.40, 0.47), (0.61, 0.68), (0.81, 0.87)]:
        _arrow(ax, x0, 0.74, x1, 0.74)
    ax.text(0.02, 0.90, "Trained forward pass", fontsize=9, color=C_INK, weight="bold")

    # Intervention path (bottom)
    ax.text(0.02, 0.50, "Causal patch on the model's OWN context", fontsize=9, color=C_INK, weight="bold")
    _box(ax, 0.02, 0.20, 0.20, 0.22,
         "1) probe (on OOD data,\nso PTT independent of conf):\nfind dir_PTT that\nraises decoded PTT\nby Δ, holding conf fixed",
         fc="#eef6f0", ec=C_PTT, fontsize=7.6)
    _box(ax, 0.26, 0.22, 0.19, 0.18,
         "2) ctx′ = ctx + Δ·dir_PTT\n   (patch representation)",
         fc="#efeaf7", ec=C_CTX, fontsize=8)
    _box(ax, 0.49, 0.22, 0.16, 0.18, "3) BP head\n(same weights)", fc="#f4f4f5", ec=C_INK, fontsize=8)
    _box(ax, 0.69, 0.20, 0.29, 0.22,
         "4) causal fidelity =\n(ΔBP / Δ) / a\n\n≈1 → routes through PTT (trust)\n≈0 → ignores PTT (shortcut)",
         fc="#eef6f0", ec=C_PTT, fontsize=8, weight="bold")
    for x0, x1 in [(0.22, 0.26), (0.45, 0.49), (0.65, 0.69)]:
        _arrow(ax, x0, 0.31, x1, 0.31, color=C_CTX)
    # link ctx down into the intervention
    _arrow(ax, 0.54, 0.66, 0.12, 0.42, color=C_MUTE, ls="--", lw=1.4)
    ax.text(0.30, 0.545, "reuse ctx", fontsize=7.5, color=C_MUTE, style="italic")
    # control note
    ax.text(
        0.02, 0.06,
        "Control: a random direction (matched step size) must give ≈0 fidelity — "
        "confirms the signal is the PTT subspace, not perturbation size.",
        fontsize=8, color=C_MUTE, va="center",
    )


# ── Panel D: pseudo-code ───────────────────────────────────────────────────────

PSEUDO = r"""# ---- generate ------------------------------------------------
for each example:
    PTT   ~ Uniform(100, 240 ms)
    BP    = a*PTT + b + noise          # governing law, a<0
    z_train = corr*PTT + sqrt(1-corr^2)*noise   # shortcut (train)
    z_ood   = independent noise                 # shortcut broken
    streamB = [ level(PTT) | level(z) ]         # PTT & confound
    streamA = query carrier

# ---- train ---------------------------------------------------
model = CrossAttention(A -> attends -> B -> ctx -> BP head)
fit model on train split (minimise |BP_pred - BP|)

# ---- audit (NO test labels) ----------------------------------
ctx = model.context(OOD_probe)               # decorrelated probe
dir_PTT, dir_z = partialled_directions(ctx, PTT, z)   # isolate PTT
slope = mean_Δ[ (head(ctx + Δ*dir_PTT) - head(ctx)) / Δ ]
causal_fidelity = slope / a                   # 1=trust, 0=shortcut
random_slope    = along a matched random dir  # must be ~0

# ---- verdict -------------------------------------------------
# low causal_fidelity  ==>  model uses the shortcut
#                      ==>  predicted to FAIL out-of-distribution"""


def panel_pseudocode(ax):
    ax.set_title("D · Method in pseudo-code", loc="left", fontsize=12, weight="bold")
    ax.axis("off")
    ax.text(
        0.0, 0.98, PSEUDO, fontsize=8.1, family="monospace", va="top", ha="left",
        color=C_INK,
        bbox=dict(boxstyle="round,pad=0.6", fc="#0f172a10", ec=C_MUTE, lw=1),
    )


# ── Panels E/F: real results ───────────────────────────────────────────────────


def panel_scatter(ax, records):
    ax.set_title("E · Causal fidelity predicts OOD failure", loc="left", fontsize=12, weight="bold")
    fid = np.array([r["ptt_causal_fidelity"] for r in records])
    ood = np.array([r["ood_mae"] for r in records])
    corr = np.array([r["confound_corr"] for r in records])
    sc = ax.scatter(fid, ood, c=corr, cmap="viridis", s=70, edgecolor="k", zorder=3)
    ax.axvline(1.0, ls="--", color=C_MUTE, lw=1)
    ax.axvline(0.0, ls=":", color="#b23b3b", lw=1)
    ax.text(1.0, ax.get_ylim()[1], " trust", color=C_MUTE, fontsize=8, va="top")
    ax.text(0.0, ax.get_ylim()[1], " shortcut", color="#b23b3b", fontsize=8, va="top")
    r = np.corrcoef(fid, ood)[0, 1]
    ax.text(
        0.96, 0.06, f"Pearson r = {r:+.2f}", transform=ax.transAxes,
        ha="right", fontsize=9, weight="bold",
        bbox=dict(boxstyle="round", fc="white", ec=C_INK, lw=1),
    )
    ax.set_xlabel("PTT causal fidelity  (recovered slope / a)", fontsize=9)
    ax.set_ylabel("OOD MAE (mmHg)", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("train confound corr", fontsize=8)
    cb.ax.tick_params(labelsize=7)


def panel_gap(ax, records):
    ax.set_title("F · Accuracy is blind to the shortcut", loc="left", fontsize=12, weight="bold")
    corrs = sorted(set(r["confound_corr"] for r in records))
    id_mean, ood_mean = [], []
    for c in corrs:
        sub = [r for r in records if r["confound_corr"] == c]
        id_mean.append(np.mean([r["in_distribution_mae"] for r in sub]))
        ood_mean.append(np.mean([r["ood_mae"] for r in sub]))
    ax.plot(corrs, id_mean, "o-", color=C_QUERY, label="in-distribution MAE")
    ax.plot(corrs, ood_mean, "s-", color="#b23b3b", label="out-of-distribution MAE")
    ax.fill_between(corrs, id_mean, ood_mean, color="#b23b3b", alpha=0.10)
    ax.set_xlabel("train confound correlation (z vs PTT)", fontsize=9)
    ax.set_ylabel("MAE (mmHg)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=8)
    ax.text(
        0.5, -0.30,
        "As the shortcut strengthens, in-distribution error stays low (looks fine)\n"
        "while OOD error climbs — only causal fidelity flags the failure early.",
        transform=ax.transAxes, ha="center", fontsize=8.2, color=C_MUTE,
    )


def build_figure(summary_path: Path, out_path: Path) -> Path:
    summary = json.loads(summary_path.read_text())
    report = summary["report"]
    records = report["records"]
    decision = report["decision"]

    fig = plt.figure(figsize=(15.5, 20.5))
    gs = fig.add_gridspec(
        6, 2, height_ratios=[0.42, 1.15, 1.35, 1.35, 1.25, 0.5],
        hspace=0.42, wspace=0.2,
        left=0.055, right=0.965, top=0.965, bottom=0.02,
    )

    # Banner
    banner = fig.add_subplot(gs[0, :])
    banner.axis("off")
    banner.text(
        0.0, 0.78, "Causal-mediation audit for BP = a·PTT",
        fontsize=20, weight="bold", color=C_INK,
    )
    banner.text(
        0.0, 0.30,
        "Is a blood-pressure model trustworthy because it computes the governing "
        "mediator (PTT), or because it latched onto a shortcut?\n"
        "Patch the model's internal PTT representation and check its BP output moves "
        "by the physical slope a — a label-free pre-deployment test.",
        fontsize=11, color=C_MUTE, va="top",
    )

    panel_data(fig.add_subplot(gs[1, 0]))
    panel_coupling(fig.add_subplot(gs[1, 1]))
    panel_pipeline(fig.add_subplot(gs[2, :]))
    panel_pseudocode(fig.add_subplot(gs[3, :]))
    panel_scatter(fig.add_subplot(gs[4, 0]), records)
    panel_gap(fig.add_subplot(gs[4, 1]), records)

    # Verdict strip
    verdict = fig.add_subplot(gs[5, :])
    verdict.axis("off")

    def flag(name, ok):
        return f"{'✓' if ok else '✗'} {name}"

    line = "   ".join([
        flag("fidelity → OOD  (r = %+.2f)" % decision["fidelity_vs_ood_pearson"],
             decision["fidelity_predicts_ood_failure"]),
        flag("shortcut raises OOD", decision["shortcut_reliance_raises_ood"]),
        flag("accuracy hides it", decision["in_distribution_accuracy_hides_failure"]),
        flag("random dir ≈ 0", decision["random_direction_is_null"]),
    ])
    verdict.text(
        0.5, 0.72, line, ha="center", fontsize=10.5, weight="bold", color=C_PTT,
    )
    verdict.text(
        0.5, 0.28,
        "in-distribution MAE %.1f mmHg   vs   OOD MAE %.1f mmHg   "
        "|   %d models over %d confound levels"
        % (
            decision["mean_in_distribution_mae"], decision["mean_ood_mae"],
            len(records), len(set(r["confound_corr"] for r in records)),
        ),
        ha="center", fontsize=9.5, color=C_MUTE,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    base = Path("results/physics_credibility/causal_bp")
    summary_path = base / "causal_bp_summary.json"
    if not summary_path.exists():
        raise SystemExit(
            "Missing %s — run "
            "`python analysis/physics_law_credibility.py --experiment causal_bp` first."
            % summary_path
        )
    out = build_figure(summary_path, base / "causal_bp_overview.png")
    print(f"[overview] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
