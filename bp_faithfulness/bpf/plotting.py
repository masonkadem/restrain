"""Simplified, self-explaining figure. Six panels; every panel carries a one-line
plain-language point AND a tiny pseudocode of exactly what it computes."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUT, ACC, BLU, GRN = "#1a1a1a", "#8a8a8a", "#b04a3a", "#3a6ea5", "#4a7a4a"


def _fmt(ax, title, point, code):
    # title (bold) and plain-language point stacked ABOVE the axes; pseudocode below
    ax.text(0.0, 1.17, title, transform=ax.transAxes, fontsize=9.3, fontweight="bold", va="bottom")
    ax.text(0.0, 1.04, point, transform=ax.transAxes, fontsize=7.3, color="#555",
            va="bottom", style="italic")
    ax.text(0.0, -0.34, code, transform=ax.transAxes, fontsize=6.7, color=MUT,
            family="monospace", va="top")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def make_figure(R, path):
    plt.rcParams.update({"font.size": 8.5})
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.6))

    # --- A: waveform pair + PAT = PTT + PEP ---
    a = ax[0, 0]; t = np.arange(len(R["ex_prox"])) / R["fs"]
    a.plot(t, R["ex_prox"], color=INK, lw=1.1, label="proximal (ECG-like marker)")
    a.plot(t, R["ex_dist"], color=BLU, lw=1.1, label="distal (PPG)")
    a.set_xlabel("time (s)"); a.set_ylabel("pressure"); a.legend(fontsize=6.8, frameon=False)
    _fmt(a, "A  What the model sees",
         "two channels; the proximal->distal lag carries PAT = PTT + PEP",
         "distal = tube_load(prox, T, gamma)   # first arrival at T")

    # --- B: probe fires (true positive) ---
    b = ax[0, 1]
    r2, lo, hi = R["probe_r2"], R["probe_ci"][0], R["probe_ci"][1]
    b.bar([0], [r2], width=.5, color=INK, yerr=[[r2 - lo], [hi - r2]], capsize=4)
    b.bar([1], [R["probe_shuffle"]], width=.5, color=MUT)
    b.axhline(R["r2_threshold"], color=ACC, ls=":", lw=1)
    b.set_xticks([0, 1]); b.set_xticklabels(["probe\n(true T)", "shuffle"])
    b.set_ylabel("R²"); b.set_ylim(min(-0.2, R["probe_shuffle"] - .05), 1.0)
    _fmt(b, "B  Probe fires  (gamma=0, PEP=0)",
         "T is decodable from activations, far above the shuffle floor",
         "R2 = fit Ridge(h -> true_T); fire if R2 > thr and > shuffle")

    # --- C: causal checks ---
    c = ax[0, 2]
    x = np.array(R["donor_x"]); s = np.array(R["donor_shift"])
    c.axhline(0, color=MUT, lw=.5); c.axvline(0, color=MUT, lw=.5)
    c.scatter(x, s, s=6, color=INK, alpha=.4, edgecolor="none")
    c.set_xlabel("physics  Δ(1/T)  donor − base"); c.set_ylabel("predicted SBP shift")
    _fmt(c, f"C  Causal check: donor-swap  ({R['donor_sign']:.0%} sign)",
         "patch the PTT direction from a donor -> BP moves as physics predicts",
         "h_patch = h + proj_w(h_donor - h);  shift = head(h_patch) - head(h)")

    # --- D: THE BENCHMARK -- PEP sweep, do the methods agree? ---
    d = ax[1, 0]
    px = np.array(R["pep_x"])
    d.axvspan(px.min(), px.min() + (px.max() - px.min()) * 0.33, color=GRN, alpha=.06)
    d.axvspan(px.max() - (px.max() - px.min()) * 0.33, px.max(), color=ACC, alpha=.06)
    d.plot(px, R["pep_r2"], "-o", color=INK, ms=4, label="probe R²")
    d.plot(px, R["pep_donor"], "-s", color=ACC, ms=4, label="donor sign acc")
    d.plot(px, R["pep_abst"], "-^", color=BLU, ms=4, label="abstention")
    d.set_ylim(-0.1, 1.05); d.set_xlabel("PEP jitter sd (ms)")
    d.set_ylabel("audit verdict (0–1)"); d.legend(fontsize=6.8, frameon=False, loc="center left")
    d.text(px.min(), 1.02, "PTT encoded", color=GRN, fontsize=6.8)
    d.text(px.max(), 1.02, "unrecoverable", color=ACC, fontsize=6.8, ha="right")
    _fmt(d, "D  BENCHMARK: recoverability sweep (PEP)",
         "as PEP hides PTT, every audit correctly stops firing -> discriminant validity",
         "for pep: retrain; run all audits; do verdicts track known ground truth?")

    # --- E: BP MAE across the same sweep (is it the null branch?) ---
    e = ax[1, 1]
    e.plot(px, R["pep_mae"], "-s", color=BLU, ms=4)
    e.set_xlabel("PEP jitter sd (ms)"); e.set_ylabel("BP MAE (mmHg)")
    _fmt(e, "E  BP error over the same sweep",
         "MAE rises with PEP: here accuracy DEPENDS on PTT (not the null branch)",
         "MAE flat while D collapses would flag: accurate WITHOUT encoding PTT")

    # --- F: saliency -- where in the waveform the model looks ---
    f = ax[1, 2]
    ts = np.arange(len(R["sal_prox"])) / R["fs"]
    f.plot(ts, R["sal_prox"] / R["sal_prox"].max(), color=INK, lw=1, label="proximal")
    f.plot(ts, R["sal_dist"] / R["sal_dist"].max(), color=BLU, lw=1, label="distal")
    f.set_xlabel("time (s)"); f.set_ylabel("|d SBP / d input| (norm.)")
    f.legend(fontsize=6.8, frameon=False)
    _fmt(f, "F  Where the model looks (saliency)",
         "reliance concentrates at the upstroke feet -- the PTT timing landmarks",
         "saliency = |grad(SBP_pred wrt input)|, averaged over samples")

    fig.suptitle("Faithfulness verification for cuffless BP: does the model encode PTT?",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.08, hspace=0.95, wspace=0.28)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
