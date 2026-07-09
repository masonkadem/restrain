"""Preview the raw simulated signals behind the physics-law credibility benchmark.

Renders example Beer-Lambert PPG windows and Moens-Korteweg pulse-transit
windows under clean and law-violating conditions, plus the derived-statistic
scatter (ratio_R vs SpO2, PTT vs BP) an answerability probe would see. This is
a sanity-check view of the data, not a results plot -- see
``physics_law_credibility.py`` for the benchmark itself.

    python analysis/plot_data_preview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics_law_credibility import (
    FS,
    generate_beer_lambert_sample,
    generate_moens_korteweg_sample,
)

# Validated palette (analysis/../references/palette.md): fixed categorical
# slots + reserved status colors. Never cycled, never reused across roles.
BLUE = "#2a78d6"
RED = "#e34948"
AQUA = "#1baf7a"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

BEER_PREVIEW_SCENARIOS = ("clean", "missing_ir", "motion_noise", "wavelength_mismatch")
MK_PREVIEW_SCENARIOS = ("clean", "missing_distal", "temporal_shift", "model_discrepancy")


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(SECONDARY_INK)
    ax.xaxis.label.set_color(SECONDARY_INK)


def _status_badge(ax, answerable: bool) -> None:
    color, text = (GOOD, "answerable") if answerable else (CRITICAL, "unanswerable")
    ax.text(
        0.98, 0.94, text, transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color=color, fontweight="bold",
        bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5),
    )


def plot_beer_waveforms(fig, gridspec_slice, rng: np.random.Generator) -> None:
    sub = gridspec_slice.subgridspec(1, len(BEER_PREVIEW_SCENARIOS), wspace=0.15)
    axes = sub.subplots(sharey=True)
    t = np.arange(128) / FS
    for ax, scenario in zip(axes, BEER_PREVIEW_SCENARIOS):
        sample = generate_beer_lambert_sample(rng, scenario)
        ax.plot(t, sample.red, color=RED, linewidth=1.4, label="Red")
        ax.plot(t, sample.ir, color=BLUE, linewidth=1.4, label="IR")
        ax.set_title(scenario.replace("_", " "), fontsize=9, color=INK, loc="left")
        ax.set_xlabel("time (s)", fontsize=8)
        _status_badge(ax, sample.answerable)
        ax.grid(True, color=GRID, linewidth=0.6)
        _style_axes(ax)
    axes[0].set_ylabel("intensity (a.u.)", fontsize=8)
    axes[-1].legend(frameon=True, framealpha=0.85, facecolor=SURFACE, edgecolor="none",
                     fontsize=7.5, loc="lower right", labelcolor=SECONDARY_INK)


def plot_mk_waveforms(fig, gridspec_slice, rng: np.random.Generator) -> None:
    sub = gridspec_slice.subgridspec(1, len(MK_PREVIEW_SCENARIOS), wspace=0.15)
    axes = sub.subplots(sharey=True)
    t = np.arange(128) / FS
    for ax, scenario in zip(axes, MK_PREVIEW_SCENARIOS):
        sample = generate_moens_korteweg_sample(rng, scenario)
        ax.plot(t, sample.proximal, color=BLUE, linewidth=1.4, label="Proximal")
        ax.plot(t, sample.distal, color=AQUA, linewidth=1.4, label="Distal")
        ax.set_title(scenario.replace("_", " "), fontsize=9, color=INK, loc="left")
        ax.set_xlabel("time (s)", fontsize=8)
        _status_badge(ax, sample.answerable)
        ax.grid(True, color=GRID, linewidth=0.6)
        _style_axes(ax)
    axes[0].set_ylabel("pulse amplitude (a.u.)", fontsize=8)
    axes[-1].legend(frameon=True, framealpha=0.85, facecolor=SURFACE, edgecolor="none",
                     fontsize=7.5, loc="lower right", labelcolor=SECONDARY_INK)


def plot_beer_scatter(ax, rng: np.random.Generator, n_per_scenario: int = 120) -> None:
    scenarios = ("clean", "missing_red", "missing_ir", "wavelength_mismatch",
                 "calibration_shift", "pigmentation_shift", "motion_noise")
    ratio_r, spo2, answerable = [], [], []
    for scenario in scenarios:
        for _ in range(n_per_scenario):
            s = generate_beer_lambert_sample(rng, scenario)
            ratio_r.append(s.ratio_r)
            spo2.append(s.spo2)
            answerable.append(s.answerable)
    ratio_r, spo2, answerable = np.array(ratio_r), np.array(spo2), np.array(answerable)
    ok = answerable.astype(bool)
    # A zeroed IR channel drives the AC/DC ratio toward zero, so ratio_R
    # diverges -- that blow-up *is* the identifiability collapse, but a few
    # such points would otherwise swamp the linear axis. Clip for display and
    # say so, rather than let one degenerate sample rescale the whole plot.
    clip_hi = 3.0
    clipped = int(np.sum(ratio_r > clip_hi))
    ratio_r_display = np.clip(ratio_r, 0.0, clip_hi)
    ax.scatter(ratio_r_display[ok], spo2[ok], s=14, color=GOOD, alpha=0.55,
               edgecolors="none", label="answerable")
    ax.scatter(ratio_r_display[~ok], spo2[~ok], s=14, color=CRITICAL, alpha=0.55,
               edgecolors="none", label="unanswerable")
    xlabel = "ratio R  (AC/DC red ÷ AC/DC IR)"
    if clipped:
        xlabel += f"  [{clipped} pts >{clip_hi:g} clipped: missing-IR ratio diverges]"
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("SpO2 (%)", fontsize=8)
    ax.set_title("Beer-Lambert: identifiability collapses ratio R", fontsize=9, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=SECONDARY_INK)


def plot_mk_scatter(ax, rng: np.random.Generator, n_per_scenario: int = 120) -> None:
    scenarios = ("clean", "missing_path_length", "missing_stiffness_cal", "missing_distal",
                 "temporal_shift", "anatomy_shift", "viscoelastic_shift")
    ptt, bp, answerable = [], [], []
    for scenario in scenarios:
        for _ in range(n_per_scenario):
            s = generate_moens_korteweg_sample(rng, scenario)
            ptt.append(s.ptt_ms)
            bp.append(s.bp)
            answerable.append(s.answerable)
    ptt, bp, answerable = np.array(ptt), np.array(bp), np.array(answerable)
    ok = answerable.astype(bool)
    ax.scatter(ptt[ok], bp[ok], s=14, color=GOOD, alpha=0.55,
               edgecolors="none", label="answerable")
    ax.scatter(ptt[~ok], bp[~ok], s=14, color=CRITICAL, alpha=0.55,
               edgecolors="none", label="unanswerable")
    ax.set_xlabel("pulse transit time (ms)", fontsize=8)
    ax.set_ylabel("blood pressure (mmHg)", fontsize=8)
    ax.set_title("Moens-Korteweg: uncalibrated PTT does not determine BP", fontsize=9, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=SECONDARY_INK)


def main() -> None:
    output_dir = Path("results/physics_credibility")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(13, 9), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.55, wspace=0.3)

    fig.text(0.02, 0.985, "What the simulators generate", fontsize=12,
              fontweight="bold", color=INK)
    fig.text(
        0.02, 0.965,
        "Same underlying physical law; missing channels/calibration make the "
        "inverse problem underdetermined even though the waveform still looks plausible.",
        fontsize=8.5, color=SECONDARY_INK,
    )

    plot_beer_waveforms(fig, gs[0, :], np.random.default_rng(1))
    plot_mk_waveforms(fig, gs[1, :], np.random.default_rng(2))

    ax_beer_scatter = fig.add_subplot(gs[2, 0])
    ax_mk_scatter = fig.add_subplot(gs[2, 1])
    plot_beer_scatter(ax_beer_scatter, np.random.default_rng(3))
    plot_mk_scatter(ax_mk_scatter, np.random.default_rng(4))

    out_path = output_dir / "data_preview.png"
    fig.savefig(out_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"[data-preview] wrote {out_path}")


if __name__ == "__main__":
    main()
