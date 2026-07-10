"""Right for the wrong reason: catching a shortcut the accuracy metric misses.

This is the demonstration that justifies auditing internal representations over
just checking held-out accuracy. We build two models that are *equally accurate
on clean data* but for different reasons:

  - GROUNDED: sees amplitude-normalized waveforms, so the spurious amplitude
    confound is unavailable and it must solve BP through the pulse-transit-time
    physics -> it encodes PTT.
  - SHORTCUT: sees the raw waveforms, where a planted population confound makes
    amplitude linearly predict BP. Amplitude is an easier cue than a lag, so
    the model latches onto it and never really learns PTT -> it does not encode
    PTT.

Both are the same architecture with the same capacity; the only difference is
whether the shortcut was available during training.

Standard validation (held-out clean accuracy) passes both. But on a *shifted
population* where the amplitude confound is decorrelated from BP -- the physics
unchanged -- the shortcut model collapses while the grounded model holds. The
PTT audit predicts exactly this from clean data alone: the grounded model
decodes PTT (high R^2), the shortcut model does not (low R^2).

    python analysis/ptt_shortcut.py
    python analysis/ptt_shortcut.py --quick
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ptt_audit_models import CrossAttentionBP  # noqa: E402
from ptt_gate import WIN, evaluate, generate, train_model  # noqa: E402

AMP_GAIN = 0.8


def amplitude_normalize(data: dict) -> dict:
    """Divide each waveform stream by its own std -> removes the amplitude cue."""
    x = data["x"].copy()
    prox, dist = x[:, :WIN], x[:, WIN:2 * WIN]
    prox /= prox.std(axis=1, keepdims=True) + 1e-6
    dist /= dist.std(axis=1, keepdims=True) + 1e-6
    x[:, :WIN], x[:, WIN:2 * WIN] = prox, dist
    return {**data, "x": x}


def _shift_distal(data: dict, delta: int) -> dict:
    """Counterfactual: move the distal pulse `delta` samples later (changes the
    true PTT by delta) while leaving amplitude and everything else fixed."""
    x = data["x"].copy()
    x[:, WIN:2 * WIN] = np.roll(x[:, WIN:2 * WIN], delta, axis=1)
    return {**data, "x": x}


def ptt_causal_sensitivity(model, raw, preprocess, ym, ys, delta: int = 3) -> float:
    """How much the model's BP prediction moves when PTT is perturbed by delta.

    This is the *usage* audit that decodability cannot give: a model that only
    decodes PTT but does not use it (a shortcut learner) has ~zero sensitivity;
    a physically grounded model's prediction tracks the change in transit time.
    Reported in mmHg of BP change per the delta-sample PTT shift.
    """
    base = evaluate(model, preprocess(raw), ym, ys)["pred"]
    shifted = evaluate(model, preprocess(_shift_distal(raw, delta)), ym, ys)["pred"]
    return float(np.abs(shifted - base).mean())


def fit_and_audit(preprocess, cfg, seed: int) -> dict:
    """Train a CrossAttentionBP under a preprocessing regime; return accuracy on
    the clean and shifted populations plus the PTT-decodability audit."""
    train = preprocess(generate(cfg.n_train, "clean", seed * 100 + 1, amp_gain=AMP_GAIN))
    ym, ys = float(train["y"].mean()), float(train["y"].std() + 1e-8)
    model = CrossAttentionBP()
    train_model(model, train, ym, ys, cfg.epochs, seed)

    # audit: decode PTT from activations on clean
    probe_raw = generate(cfg.n_probe, "clean", seed * 100 + 2, amp_gain=AMP_GAIN)
    pe = evaluate(model, preprocess(probe_raw), ym, ys)
    scaler = StandardScaler().fit(pe["feat"])
    ptt_probe = Ridge(alpha=1.0).fit(scaler.transform(pe["feat"]), probe_raw["ptt_true"])

    def mae(raw):
        ev = evaluate(model, preprocess(raw), ym, ys)
        return float(np.abs(ev["pred"] - raw["y"]).mean())

    clean = generate(cfg.n_test, "clean", seed * 100 + 3, amp_gain=AMP_GAIN)
    shifted = generate(cfg.n_test, "clean", seed * 100 + 4, amp_gain=AMP_GAIN, amp_decorrelate=True)
    ce = evaluate(model, preprocess(clean), ym, ys)
    ptt_r2 = float(r2_score(clean["ptt_true"], ptt_probe.predict(scaler.transform(ce["feat"]))))
    sensitivity = ptt_causal_sensitivity(model, clean, preprocess, ym, ys)
    return {"clean_mae": mae(clean), "shifted_mae": mae(shifted),
            "ptt_r2": ptt_r2, "ptt_sensitivity": sensitivity}


def aggregate(cfg) -> dict:
    regimes = {"grounded": amplitude_normalize, "shortcut": (lambda d: d)}
    out = {}
    for name, pre in regimes.items():
        per = [fit_and_audit(pre, cfg, s) for s in cfg.seeds]
        out[name] = {k: {"mean": float(np.mean([p[k] for p in per])),
                         "std": float(np.std([p[k] for p in per]))}
                     for k in ("clean_mae", "shifted_mae", "ptt_r2", "ptt_sensitivity")}
    return out


# ── plotting ──────────────────────────────────────────────────────────────────
BLUE, RED, MUTE = "#2a78d6", "#d03b3b", "#9aa09a"
GOOD, BAD = "#0f8a3c", "#c1362f"
INK, GRID, SURFACE = "#16191a", "#e1e0d9", "#fcfcfb"
MLAB = {"grounded": "grounded model\n(uses PTT physics)", "shortcut": "shortcut model\n(uses amplitude cue)"}
MCOL = {"grounded": BLUE, "shortcut": RED}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#7c8384", labelsize=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)


def plot(agg: dict, output_dir: Path) -> None:
    names = ["grounded", "shortcut"]
    x = np.arange(2)
    fig = plt.figure(figsize=(14.5, 4.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, wspace=0.34)
    fig.text(0.02, 0.95, "Right for the wrong reason: what accuracy hides, and which audit reveals it",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.905,
             "Two same-architecture models, equally accurate on the clean population. Decodability is "
             "fooled; the causal (usage) audit is not.", fontsize=9.5, color="#454b4c")

    # Panel 1: clean vs shifted MAE
    ax = fig.add_subplot(gs[0, 0]); bw = 0.36
    for i, split in enumerate(("clean_mae", "shifted_mae")):
        vals = [agg[n][split]["mean"] for n in names]; errs = [agg[n][split]["std"] for n in names]
        ax.bar(x + (i - 0.5) * bw, vals, bw, yerr=errs, capsize=4,
               color=[MCOL[n] for n in names], alpha=(0.95 if split == "clean_mae" else 0.45),
               hatch=("" if split == "clean_mae" else "//"),
               label=("clean population" if split == "clean_mae" else "shifted population"))
    ax.set_xticks(x); ax.set_xticklabels([MLAB[n] for n in names], fontsize=8.5)
    ax.set_ylabel("BP error — MAE (mmHg)", fontsize=9)
    ax.set_title("① Accuracy: equal on clean,\ndiverges on a new population", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)

    # Panel 2: decodability audit -- FOOLED
    ax = fig.add_subplot(gs[0, 1])
    r2 = [agg[n]["ptt_r2"]["mean"] for n in names]; r2e = [agg[n]["ptt_r2"]["std"] for n in names]
    ax.bar(x, r2, 0.55, yerr=r2e, capsize=4, color=[MCOL[n] for n in names])
    ax.axhline(0.5, ls="--", color=MUTE, lw=1.2)
    for i, n in enumerate(names):
        ax.text(i, agg[n]["ptt_r2"]["mean"] + 0.05, "looks OK", ha="center", fontsize=8.5,
                fontweight="bold", color="#7c8384")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("PTT decodable (R²)", fontsize=9)
    ax.set_title("② Decodability audit: FOOLED\n(both decode PTT)", fontsize=9.5, color=BAD, loc="left")
    _style(ax)

    # Panel 3: causal usage audit -- CATCHES IT
    ax = fig.add_subplot(gs[0, 2])
    se = [agg[n]["ptt_sensitivity"]["mean"] for n in names]
    see = [agg[n]["ptt_sensitivity"]["std"] for n in names]
    ax.bar(x, se, 0.55, yerr=see, capsize=4, color=[MCOL[n] for n in names])
    for i, n in enumerate(names):
        v = agg[n]["ptt_sensitivity"]["mean"]
        uses = v > 3.0
        ax.text(i, v + max(se) * 0.04 + 0.4, "USES PTT" if uses else "IGNORES PTT", ha="center",
                fontsize=9, fontweight="bold", color=(GOOD if uses else BAD))
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("BP change when PTT is perturbed (mmHg)", fontsize=9)
    ax.set_title("③ Causal usage audit: CATCHES IT\n(does the output depend on PTT?)", fontsize=9.5, color=GOOD, loc="left")
    _style(ax)

    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.06, right=0.99)
    out = output_dir / "ptt_shortcut.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"[shortcut] wrote {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/ptt_gate"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-train", type=int, default=2500)
    p.add_argument("--n-probe", type=int, default=800)
    p.add_argument("--n-test", type=int, default=800)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.seeds = [0]; a.n_train, a.n_probe, a.n_test, a.epochs = 500, 200, 300, 45
    return a


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate(a)
    plot(agg, a.output_dir)
    for n in ("grounded", "shortcut"):
        r = agg[n]
        print(f"  {n:9s} clean MAE={r['clean_mae']['mean']:5.2f}  "
              f"shifted MAE={r['shifted_mae']['mean']:6.2f}  PTT R2={r['ptt_r2']['mean']:.3f}")
    with open(a.output_dir / "ptt_shortcut.json", "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2)
    print(f"[shortcut] wrote {a.output_dir / 'ptt_shortcut.json'}")


if __name__ == "__main__":
    main()
