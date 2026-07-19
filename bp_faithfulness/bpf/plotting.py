"""Comprehensive multi-panel figure (Part 5 headline + Part 4 diagnostics)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUT, ACC = "#1a1a1a", "#7b7b7b", "#b04a3a"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def make_figure(R, path):
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))

    # A. example waveform pair
    a = ax[0, 0]; t = np.arange(len(R["ex_prox"])) / R["fs"]
    a.plot(t, R["ex_prox"], color=INK, lw=1.2, label="proximal")
    a.plot(t, R["ex_dist"], color=MUT, lw=1.2, label="distal")
    a.set_title("A  Simulated waveform pair"); a.set_xlabel("time (s)"); a.set_ylabel("pressure")
    a.legend(frameon=False, fontsize=7.5); _clean(a)

    # B. foot-to-foot vs true T (physics: foot ~ T, reflection-robust)
    b = ax[0, 1]
    b.plot(R["T_grid"] * 1e3, R["T_grid"] * 1e3, "--", color=MUT, lw=1, label="identity")
    b.plot(R["T_grid"] * 1e3, R["ftf_g0"] * 1e3, "-o", color=INK, ms=3, label="γ=0")
    b.plot(R["T_grid"] * 1e3, R["ftf_g08"] * 1e3, "-s", color=ACC, ms=3, label="γ=0.8")
    b.set_title("B  Foot-to-foot ≈ T (reflection-robust)")
    b.set_xlabel("true T (ms)"); b.set_ylabel("foot-to-foot (ms)")
    b.legend(frameon=False, fontsize=7.5); _clean(b)

    # C. probe R2 vs shuffle floor (true positive), with bootstrap CI + threshold
    c = ax[0, 2]
    r2, lo, hi = R["probe_r2"], R["probe_ci"][0], R["probe_ci"][1]
    c.bar([0], [r2], color=INK, width=0.5, yerr=[[r2 - lo], [hi - r2]], capsize=4)
    c.bar([1], [R["probe_shuffle"]], color=MUT, width=0.5)
    c.axhline(R["r2_threshold"], color=ACC, ls=":", lw=1)
    c.text(1.5, R["r2_threshold"], " pre-registered\n threshold", color=ACC, va="center", fontsize=7)
    c.set_xticks([0, 1]); c.set_xticklabels(["probe\n(true T)", "shuffle\ncontrol"])
    c.set_ylabel("R²"); c.set_title("C  Probe fires (γ=0, PEP=0)"); _clean(c)

    # D. causal ablation
    d = ax[0, 3]
    d.bar([0, 1], [R["mae_intact"], R["mae_ablated"]], color=[INK, ACC], width=0.5)
    d.set_xticks([0, 1]); d.set_xticklabels(["intact", "PTT dir\nablated"])
    d.set_ylabel("BP MAE (mmHg)")
    d.set_title(f"D  Ablation Δ={R['mae_ablated']-R['mae_intact']:+.2f} mmHg"); _clean(d)

    # E. donor-swap patching: predicted shift vs physics sign
    e = ax[1, 0]
    x = np.array(R["donor_x"]); s = np.array(R["donor_shift"])
    e.axhline(0, color=MUT, lw=0.6); e.axvline(0, color=MUT, lw=0.6)
    e.scatter(x, s, s=6, color=INK, alpha=0.4, edgecolor="none")
    e.set_xlabel("physics Δ(1/T) donor−base"); e.set_ylabel("predicted SBP shift")
    e.set_title(f"E  Donor swap: sign acc {R['donor_sign']:.0%}"); _clean(e)

    # F. gamma sweep
    _sweep_panel(ax[1, 1], R["gamma_x"], R["gamma_r2"], R["gamma_shuf"], R["gamma_mae"],
                 R["gamma_abst"], "F  γ sweep (PEP=0): recoverability knob?", "reflection γ")
    # G. pep sweep
    _sweep_panel(ax[1, 2], R["pep_x"], R["pep_r2"], R["pep_shuf"], R["pep_mae"],
                 R["pep_abst"], "G  PEP sweep (γ=0): the collapse", "PEP sd (ms)")

    # H. per-subject MAE
    h = ax[1, 3]
    subs = list(R["per_subject_mae"].keys())
    h.bar(range(len(subs)), [R["per_subject_mae"][s] for s in subs], color=MUT, width=0.6)
    h.axhline(R["mae_intact"], color=INK, ls="--", lw=1, label="overall")
    h.set_xlabel("held-out subject"); h.set_ylabel("BP MAE (mmHg)")
    h.set_title("H  Per-subject MAE (LOSO)"); h.legend(frameon=False, fontsize=7.5); _clean(h)

    fig.suptitle("Faithfulness verification for cuffless BP: does the model encode PTT?",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _sweep_panel(ax, x, r2, shuf, mae, abst, title, xlabel):
    x = np.array(x)
    ax.plot(x, r2, "-o", color=INK, ms=4, label="probe R²")
    ax.plot(x, shuf, "-o", color=MUT, ms=3, label="shuffle")
    ax.plot(x, abst, "-^", color=ACC, ms=4, label="abstention")
    ax.set_ylim(-0.15, 1.05); ax.set_xlabel(xlabel); ax.set_ylabel("R² / abstention rate")
    ax.set_title(title); _clean(ax)
    tw = ax.twinx()
    tw.plot(x, mae, "-s", color="#3a6ea5", ms=4, label="BP MAE")
    tw.set_ylabel("BP MAE (mmHg)", color="#3a6ea5")
    tw.tick_params(axis="y", colors="#3a6ea5")
    lines = ax.get_lines() + tw.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=6.8, loc="center left")
