"""The one hero figure (Act 1): only the causal audit separates a faithful model
from one that fakes the law (unfaithful) or ignores it (shortcut).

    python analysis/make_hero_figure.py
"""
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- shared aesthetic ----
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "axes.edgecolor": "#3a3a3a", "axes.linewidth": 0.9,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 180,
})
C = {"faithful": "#2f4b7c", "unfaithful": "#c1543b", "shortcut": "#9aa0a6"}
M = {"faithful": "o", "unfaithful": "s", "shortcut": "^"}
INK, MUT = "#22262b", "#8b8b8b"

P = json.load(open(Path(__file__).resolve().parent.parent /
                   "results/causal_mediation_toy/blood_pressure/payload.json"))
m, rep, regimes = P["metrics"], P["rep"], ["faithful", "unfaithful", "shortcut"]
_key = {"faithful": "law", "unfaithful": "unfaithful", "shortcut": "shortcut"}  # payload names
ref = max(m["shortcut"]["ood_mse"]["mean"], 1e-6)
clip = lambda v: float(np.clip(v, 0, 1))


def score(r):
    k = _key[r]
    return {
        "validation accuracy": clip(1 - m[k]["val_indist_mse"]["mean"] / ref),
        "probe R²  (decodability)": clip(m[k]["probe_r2_indist"]["mean"]),
        'gradient  ("does it use $j$?")': clip(m[k]["uses_j_score"]["mean"]),
        "interchange audit  (causal)": clip(m[k]["interchange_accuracy_peak"]["mean"]),
    }


rows = list(score("faithful"))
S = {r: score(r) for r in regimes}
verdict = {rows[0]: "agree", rows[1]: "agree", rows[2]: "misses unfaithful",
           rows[3]: "SEPARATES"}

fig, ax = plt.subplots(1, 2, figsize=(15.2, 5.6), gridspec_kw={"width_ratios": [1.55, 1]})
fig.subplots_adjust(left=0.005, right=0.985, top=0.82, bottom=0.14, wspace=0.30)

# ---------------- LEFT: scorecard ----------------
a = ax[0]
ys = np.arange(len(rows))[::-1]
for y, row in zip(ys, rows):
    is_audit = row == rows[-1]
    if is_audit:
        a.axhspan(y - 0.5, y + 0.5, color="#eef1f6", zorder=0)          # highlight the audit row
    vals = [S[r][row] for r in regimes]
    a.plot([min(vals), max(vals)], [y, y], color="#d7d7d7", lw=2.2, zorder=1,
           solid_capstyle="round")                                      # dumbbell connector
    for r in regimes:
        a.scatter(S[r][row], y, s=150, marker=M[r], color=C[r], edgecolor="white",
                  linewidth=1.3, zorder=3)
    a.text(-0.02, y, row, ha="right", va="center", fontsize=10)
    a.text(1.06, y, verdict[row], ha="left", va="center",
           fontsize=9.5 if is_audit else 9,
           fontweight="bold" if is_audit else "normal",
           color=C["faithful"] if is_audit else MUT)
a.set_xlim(-0.02, 1.45); a.set_ylim(-0.6, len(rows) - 0.35); a.set_yticks([])
a.set_xticks([0, 0.5, 1.0]); a.set_xlabel("looks faithful   (0  →  1)")
a.spines["left"].set_visible(False)
a.set_title("Conventional checks can't separate the three models —\nthe causal audit can",
            loc="left", fontweight="bold", pad=12)
handles = [plt.Line2D([], [], marker=M[r], ls="none", color=C[r], mec="white", mew=1.2,
                      ms=11, label=r) for r in regimes]
a.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
         bbox_to_anchor=(0.36, -0.24), handletextpad=0.4, columnspacing=1.6)

# ---------------- RIGHT: counterfactual response ----------------
b = ax[1]
x = np.array(P["vj_grid"])
b.plot(x, rep["sweep_equation"], color="#c9c9c9", lw=6, solid_capstyle="round",
       zorder=1, label="true law")
for r in regimes:
    b.plot(x, rep["models"][_key[r]]["sweep_pred"], color=C[r], lw=2.2, marker=M[r],
           markevery=7, ms=6, mec="white", mew=0.8, zorder=3, label=r)
b.set_xlabel(P["j_name"]); b.set_ylabel(f"predicted {P['unit']}")
b.set_title("Why:  unfaithful bends the wrong way;\nshortcut is flat; faithful tracks the law",
            loc="left", fontweight="bold", pad=12)
b.legend(frameon=False, fontsize=9, loc="upper right")

fig.suptitle("Only the causal audit tells a faithful model from an impostor",
             x=0.005, ha="left", fontsize=14, fontweight="bold")
out = Path(__file__).resolve().parent.parent / "results/causal_mediation_toy/hero.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
