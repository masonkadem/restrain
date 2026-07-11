"""Render a precise black-and-white 'audit map': the forward computation of the
two-query cross-attention retriever, and exactly where each of the three audits
reads from or intervenes on it.

    python analysis/audit_map.py

Writes results/causal_mediation_toy/audit_map.png.  No training; static schematic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#111111"
GRID = "#8a8a8a"
FILL = "#f4f4f2"
STATE_FILL = "#dcdcdc"
AUDIT_FILL = "#ffffff"


def box(ax, x, y, w, h, text, fill=FILL, fs=8.6, ec=INK, lw=1.2, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor=fill, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight=weight)


def arrow(ax, x0, y0, x1, y1, style="-|>", color=INK, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, shrinkA=1, shrinkB=1,
                                 linestyle=ls))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.975, "Where the audit reads and intervenes",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(0.01, 0.945,
            "Two-query cross-attention retriever (weights frozen).  "
            r"$p$ = positional dim, $w$ = attention width, $S$ = number of slots.",
            fontsize=9, color="#333333")

    # ------------------------------------------------------------------ forward
    ax.text(0.02, 0.90, "FORWARD COMPUTATION", fontsize=9.5, fontweight="bold", color=INK)
    box(ax, 0.02, 0.80, 0.20, 0.065,
        r"source  $s\in\mathbb{R}^{2p}$" + "\n(codes of slots i, j)")
    box(ax, 0.25, 0.80, 0.22, 0.065,
        r"targets  $T\in\mathbb{R}^{S\times(1+p)}$" + "\n(value channel + code)")

    box(ax, 0.02, 0.685, 0.20, 0.07,
        r"$q_i=W_Q^{i}(s_{1:p})$" + "\n" + r"$q_j=W_Q^{j}(s_{p+1:2p})$")
    box(ax, 0.25, 0.685, 0.22, 0.07, r"$K=W_K(T),\;\;V=W_V(T)$" + "\n" + r"$\in\mathbb{R}^{S\times w}$")

    box(ax, 0.02, 0.565, 0.45, 0.075,
        r"$a_i=\mathrm{softmax}\!\left(q_iK^{\top}/\sqrt{w}\right),$"
        r"$\quad c_i=a_iV\in\mathbb{R}^{w}$" + "\n"
        r"$a_j=\mathrm{softmax}\!\left(q_jK^{\top}/\sqrt{w}\right),$"
        r"$\quad c_j=a_jV\in\mathbb{R}^{w}$")

    box(ax, 0.09, 0.445, 0.31, 0.065,
        r"$h=[\,c_i\,;\,c_j\,]\in\mathbb{R}^{2w}$    (audited state)",
        fill=STATE_FILL, fs=9.4, lw=1.8, weight="bold")

    box(ax, 0.13, 0.345, 0.23, 0.055, r"$\hat y=W_O(h)\in\mathbb{R}$")

    arrow(ax, 0.12, 0.80, 0.12, 0.757)
    arrow(ax, 0.36, 0.80, 0.36, 0.757)
    arrow(ax, 0.12, 0.685, 0.20, 0.642)
    arrow(ax, 0.36, 0.685, 0.28, 0.642)
    arrow(ax, 0.245, 0.565, 0.245, 0.512)
    arrow(ax, 0.245, 0.445, 0.245, 0.401)

    # ground-truth reference
    box(ax, 0.02, 0.235, 0.45, 0.07,
        r"governing law (oracle):  $y=g(v_i,v_j)$" + "\n"
        r"$BP=P_0+S\,[\,2\ln(L/PTT)-\ln(E_0/E_{ref})\,]$",
        fill="#ffffff", fs=8.6, ec=GRID)
    ax.text(0.245, 0.205, "the audits compare the model to $g$; no measured labels used",
            ha="center", fontsize=7.6, color="#555555", style="italic")

    # ------------------------------------------------------------------ audits
    xa = 0.53
    ax.text(xa, 0.90, "THREE AUDITS", fontsize=9.5, fontweight="bold", color=INK)

    # 1 probe (read h)
    box(ax, xa, 0.735, 0.45, 0.135, "", fill=AUDIT_FILL, ec=INK, lw=1.2)
    ax.text(xa + 0.02, 0.848, "1  Decodability probe  —  reads $h$   (conventional)",
            fontsize=9, fontweight="bold", color=INK)
    ax.text(xa + 0.02, 0.812,
            r"fit $\beta$ to minimize $\sum \| \beta^{\top}h - y\|^2$;  report $R^2$.",
            fontsize=8.6, color=INK)
    ax.text(xa + 0.02, 0.766,
            "asks: is $y$ linearly present in $h$?\n"
            "(present $\\neq$ used — passes all three models)", fontsize=8.0, color="#444444")

    # 2 sensitivity (perturb input v_j)
    box(ax, xa, 0.560, 0.45, 0.150, "", fill=AUDIT_FILL, ec=INK, lw=1.2)
    ax.text(xa + 0.02, 0.685, "2  Counterfactual sensitivity  —  perturbs input $v_j$",
            fontsize=9, fontweight="bold", color=INK)
    ax.text(xa + 0.02, 0.648,
            r"compare $\left|\partial \hat y/\partial v_j\right|$"
            r"  to  $\left|\partial g/\partial v_j\right|$.", fontsize=8.6, color=INK)
    ax.text(xa + 0.02, 0.598,
            "asks: does the output move with $v_j$?\n"
            "a magnitude check is FOOLED by a wrong-sign\n"
            "(unfaithful) model — it still moves with $v_j$.",
            fontsize=8.0, color="#444444")

    # 3 interchange / DAS (intervene on h)
    box(ax, xa, 0.315, 0.45, 0.220, "", fill=AUDIT_FILL, ec=INK, lw=1.4)
    ax.text(xa + 0.02, 0.510, "3  Interchange intervention (DAS)  —  intervenes on $h$",
            fontsize=9, fontweight="bold", color=INK)
    ax.text(xa + 0.02, 0.470,
            r"orthonormal $R$;   $z=R^{\top}h$." + "\n"
            r"base $b$, source $s$:  swap top-$k$ coords of $z$:", fontsize=8.4, color=INK)
    ax.text(xa + 0.02, 0.417,
            r"$\tilde z=[\,z^{(s)}_{1:k}\,;\,z^{(b)}_{k+1:2w}\,],$"
            r"$\;\;\hat y_{\mathrm{patch}}=W_O(R\tilde z)$", fontsize=8.6, color=INK)
    ax.text(xa + 0.02, 0.372,
            r"learn $R$:  $\min_R \sum \| \hat y_{\mathrm{patch}}-y^{(s)}\|^2;$"
            r"  IIA $=R^2$.", fontsize=8.4, color=INK)
    ax.text(xa + 0.02, 0.335,
            "asks: is a subspace of $h$ the causal carrier of $y$?",
            fontsize=8.0, color="#444444", style="italic")

    # connectors from audited nodes to audit boxes
    arrow(ax, 0.40, 0.478, xa, 0.80, style="-|>", color=GRID, lw=1.1, ls=(0, (4, 3)))  # h->probe
    arrow(ax, 0.36, 0.812, xa, 0.635, style="-|>", color=GRID, lw=1.1, ls=(0, (4, 3)))  # T->sens
    arrow(ax, 0.40, 0.470, xa, 0.45, style="-|>", color=GRID, lw=1.1, ls=(0, (4, 3)))  # h->DAS

    # verdict strip
    box(ax, xa, 0.235, 0.45, 0.055,
        "verdict:  probe & sensitivity pass the unfaithful model;\n"
        "only interchange (audit 3) isolates the faithful model.",
        fill="#efefec", ec=GRID, fs=8.0)

    fig.tight_layout()
    out = Path("results/causal_mediation_toy/audit_map.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
