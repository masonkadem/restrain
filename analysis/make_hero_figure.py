"""The one hero figure: only the causal audit separates a faithful model from one
that fakes the law (unfaithful) or ignores it (shortcut). Reads the cached
three-model results from causal_mediation_toy.py (blood-pressure law).

    python analysis/make_hero_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = json.load(open(Path(__file__).resolve().parent.parent /
                   "results/causal_mediation_toy/blood_pressure/payload.json"))
m, rep = P["metrics"], P["rep"]
regimes = P["regimes"]                                  # law, unfaithful, shortcut
SHADE = {"law": "#1a1a1a", "unfaithful": "#7b7b7b", "shortcut": "#c0c0c0"}
MARK = {"law": "o", "unfaithful": "s", "shortcut": "^"}
ref = max(m["shortcut"]["ood_mse"]["mean"], 1e-6)
clip = lambda v: float(np.clip(v, 0, 1))


def looks_faithful(r):
    return {
        "validation accuracy": clip(1 - m[r]["val_indist_mse"]["mean"] / ref),
        "probe R² (decodability)": clip(m[r]["probe_r2_indist"]["mean"]),
        'gradient "does it use j?"': clip(m[r]["uses_j_score"]["mean"]),
        "interchange audit (causal)": clip(m[r]["interchange_accuracy_peak"]["mean"]),
    }


rows = list(looks_faithful("law").keys())
verdict = {"validation accuracy": "all high — can't separate",
           "probe R² (decodability)": "all high — can't separate",
           'gradient "does it use j?"': "misses the unfaithful model",
           "interchange audit (causal)": "ISOLATES the faithful model"}
scores = {r: looks_faithful(r) for r in regimes}

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})

# ---- LEFT: the scorecard (the punchline) ----
a = ax[0]
ys = np.arange(len(rows))[::-1]
off = {"law": 0.16, "unfaithful": 0.0, "shortcut": -0.16}
for y, row in zip(ys, rows):
    vals = [scores[r][row] for r in regimes]
    sep = scores["law"][row] - max(scores["u" if False else rr][row] for rr in regimes if rr != "law") > 0.3
    a.axhspan(y - 0.5, y + 0.5, color="#f6f6f4" if row != rows[-1] else "#eef3ee", zorder=0)
    for r in regimes:
        a.scatter(scores[r][row], y + off[r], s=90, marker=MARK[r], color=SHADE[r],
                  edgecolor="#1a1a1a", linewidth=0.5, zorder=3)
    a.text(-0.03, y, row, ha="right", va="center", fontsize=9.5)
    a.text(1.03, y, verdict[row], ha="left", va="center", fontsize=8,
           fontweight="bold" if sep else "normal", color="#1a1a1a" if sep else "#8a8a86")
a.set_xlim(-0.03, 1.03); a.set_ylim(-0.6, len(rows) - 0.4); a.set_yticks([])
a.set_xlabel("looks faithful   (0 → 1)", fontsize=9.5)
a.set_title("Only the causal audit separates the three models", fontsize=10.5, fontweight="bold")
for s in ("top", "right", "left"):
    a.spines[s].set_visible(False)
handles = [plt.Line2D([], [], marker=MARK[r], ls="none", color=SHADE[r], mec="#1a1a1a",
                      ms=8, label=r) for r in regimes]
a.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.28))

# ---- RIGHT: the intuition (counterfactual response) ----
b = ax[1]
x = np.array(P["vj_grid"])
b.plot(x, rep["sweep_equation"], "--", color="#555", lw=2.4, label="true law")
for r in regimes:
    b.plot(x, rep["models"][r]["sweep_pred"], color=SHADE[r], lw=2,
           marker=MARK[r], markevery=6, ms=5, label=r)
b.set_xlabel(P["j_name"], fontsize=9.5); b.set_ylabel(f"predicted {P['unit']}", fontsize=9.5)
b.set_title("Why: unfaithful uses j the WRONG way; shortcut ignores it",
            fontsize=10.5, fontweight="bold")
b.legend(frameon=False, fontsize=8.5)
for s in ("top", "right"):
    b.spines[s].set_visible(False)

fig.suptitle("Conventional checks agree on all three models — the causal audit does not",
             x=0.5, fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
out = Path(__file__).resolve().parent.parent / "results/causal_mediation_toy/hero.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print("wrote", out)
