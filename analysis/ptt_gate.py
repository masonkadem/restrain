"""Auditing a model by decoding a physically-required intermediate: PTT.

Platform idea: a user submits a trained model that maps pulse waveforms (+
subject calibration) to blood pressure. We want to certify whether the model
actually encodes the physics it must -- specifically the pulse-transit-time
(PTT), the inter-stream lag the Moens-Korteweg law is built on. Crucially the
audit needs *no access to the model's original training data*: PTT is defined
by the governing law, so we generate synthetic waveforms, run them through the
submitted model, and probe its activations.

Two gates, both per-input, both operating only on the frozen model:

  1. THRESHOLD gate. Linearly decode PTT from the activations; if the decoded
     PTT is outside the physiological range, the model's internal state is
     nonsense -> distrust. Catches gross/garbage failures. Cannot catch a
     PTT that is wrong but still plausible.

  2. CONSISTENCY gate (the stronger one). PTT is also *measurable directly
     from the raw waveforms* (the lag between proximal and distal, via
     cross-correlation) -- an independent ground truth that needs no model.
     Compare decoded PTT (from activations) with measured PTT (from signal).
     Agreement -> the model is physically grounded. Disagreement -> the model
     misread a signal whose PTT was actually well-defined -> distrust. This
     catches a plausible-but-wrong internal PTT that the threshold misses.

The honest boundary this exposes: both gates test whether the model correctly
represents PTT. Neither can catch a *calibration* error -- if the waveforms
(hence PTT) are fine but the subject calibration is wrong, the BP is wrong yet
both gates pass. Calibration is not observable from the waveform, so it is the
fundamental floor, exactly as the identifiability analysis predicts.

    python analysis/ptt_gate.py
    python analysis/ptt_gate.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

FS = 50.0                        # chosen so physiological PTT lands in a
WIN = 128                        # resolvable lag range that fits the window
PTT_LO, PTT_HI = 5, 40           # physiological lag band, in samples
CONDITIONS = ("clean", "missing_distal", "morphology_shift", "wrong_calibration")


def _pulse(win, hr, fs, harmonic=0.0, phase=0.0):
    t = np.arange(win) / fs
    f = hr / 60.0
    w = np.sin(2 * np.pi * f * t + phase) + 0.25 * np.sin(4 * np.pi * f * t + 0.3)
    if harmonic:
        w = w + harmonic * np.sin(6 * np.pi * f * t + 1.1)   # extra dicrotic-like component
    return w


def generate(n: int, condition: str, seed: int,
             amp_gain: float = 0.0, amp_decorrelate: bool = False) -> dict:
    """Waveform Moens-Korteweg. Returns proximal+distal streams, calibration,
    BP, plus the true and signal-measured PTT (in samples).

    amp_gain > 0 injects a *spurious* population confound: the waveform
    amplitude is scaled by a factor linear in BP, so amplitude carries BP
    information that has nothing to do with the pulse-transit-time physics. A
    model can exploit it to get good accuracy for the wrong reason. Setting
    amp_decorrelate=True (the shifted population) drives the amplitude factor
    from an independent BP draw, so the confound no longer predicts BP -- any
    model that relied on it collapses. Default amp_gain=0 leaves the base
    task untouched.
    """
    rng = np.random.default_rng(seed)
    bp = rng.uniform(90.0, 150.0, size=n)
    K = rng.uniform(0.8, 2.2, size=n)
    alpha = rng.uniform(0.8, 1.5, size=n)
    L = rng.uniform(0.6, 0.95, size=n)
    hr = rng.uniform(55, 95, size=n)

    pwv = np.sqrt(K * np.exp(alpha * bp / 100.0))
    ptt_ms = 1000.0 * L / pwv
    lag = np.clip(ptt_ms * FS / 1000.0, PTT_LO, PTT_HI)      # continuous (fractional) lag

    prox = np.zeros((n, WIN), np.float32)
    dist = np.zeros((n, WIN), np.float32)
    cal = np.stack([K, alpha, L], axis=1).astype(np.float32)
    answerable = np.ones(n, dtype=int)
    grid = np.arange(WIN)

    for i in range(n):
        p = _pulse(WIN, hr[i], FS)
        damp = rng.uniform(0.85, 0.95)
        noise = rng.normal(0, 0.03, WIN)
        d_shape = p
        if condition == "morphology_shift":
            d_shape = _pulse(WIN, hr[i], FS, harmonic=rng.uniform(0.6, 1.0), phase=0.4)
        # fractional shift right by lag[i] via interpolation -> continuous PTT
        d = np.interp(grid - lag[i], grid, d_shape, left=0.0, right=0.0) * damp
        prox[i] = (p + noise).astype(np.float32)
        dist[i] = (d + 0.5 * noise).astype(np.float32)

    if amp_gain > 0.0:
        driver = rng.uniform(90.0, 150.0, size=n) if amp_decorrelate else bp
        amp = (1.0 + amp_gain * (driver - 120.0) / 60.0).astype(np.float32)
        prox *= amp[:, None]
        dist *= amp[:, None]

    if condition == "missing_distal":
        dist[:] = 0.0
        answerable[:] = 0
    elif condition == "morphology_shift":
        answerable[:] = 0                          # lag intact, but morphology is novel
    elif condition == "wrong_calibration":
        cal = np.stack([rng.uniform(0.8, 2.2, n), rng.uniform(0.8, 1.5, n),
                        rng.uniform(0.6, 0.95, n)], axis=1).astype(np.float32)
        answerable[:] = 0                          # waves fine, calibration wrong

    ptt_measured = np.array([_xcorr_lag(prox[i], dist[i]) for i in range(n)], dtype=np.float32)
    x = np.concatenate([prox, dist, cal], axis=1).astype(np.float32)
    return {
        "x": x, "y": bp.astype(np.float32), "answerable": answerable,
        "ptt_true": lag.astype(np.float32), "ptt_measured": ptt_measured,
    }


def _xcorr_lag(a: np.ndarray, b: np.ndarray) -> float:
    """Lag (samples) that best aligns b to a; the signal-measured PTT."""
    a = a - a.mean()
    b = b - b.mean()
    if b.std() < 1e-6:
        return -1.0                                # no distal signal -> undefined
    corr = np.correlate(b, a, mode="full")
    lag = int(np.argmax(corr) - (len(a) - 1))
    return float(lag)


class BPModel(nn.Module):
    """Conv encoder per stream (captures lag) + calibration -> BP.
    The penultimate activation is what the audit probes."""

    def __init__(self, cal_dim: int = 3, hidden: int = 96):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2, 16, 7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 16, 7, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
        )
        self.mix = nn.Sequential(
            nn.Linear(16 * 8 + cal_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, prox, dist, cal, return_features=False):
        s = torch.stack([prox, dist], dim=1)                   # (B, 2, WIN)
        wave_feat = self.enc(s)                                # waveform-only features
        feat = self.mix(torch.cat([wave_feat, cal], dim=1))
        out = self.head(feat).squeeze(-1)
        if return_features:
            # Probe the WAVEFORM encoder, where PTT physically lives and which
            # never sees the calibration -- so the PTT gate is, by construction,
            # blind to a calibration error (that is a separate, non-observable
            # failure mode, not a PTT-competence failure).
            return out, wave_feat
        return out


def _split(x):
    prox = torch.from_numpy(x[:, :WIN])
    dist = torch.from_numpy(x[:, WIN:2 * WIN])
    cal = torch.from_numpy(x[:, 2 * WIN:])
    return prox, dist, cal


def train_model(model, data, ym, ys, epochs, seed):
    torch.manual_seed(seed)
    prox, dist, cal = _split(data["x"])
    y = torch.from_numpy(((data["y"] - ym) / ys).astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(prox, dist, cal) - y) ** 2).mean()
        loss.backward()
        opt.step()


def evaluate(model, data, ym, ys):
    model.eval()
    prox, dist, cal = _split(data["x"])
    with torch.no_grad():
        pred, feat = model(prox, dist, cal, return_features=True)
    return {"pred": pred.numpy() * ys + ym, "feat": feat.numpy()}


def run(cfg, seed: int) -> dict:
    train = generate(cfg.n_train, "clean", seed * 100 + 1)
    ym, ys = float(train["y"].mean()), float(train["y"].std() + 1e-8)
    model = BPModel().to("cpu")
    train_model(model, train, ym, ys, cfg.epochs, seed)

    # Fit the PTT decoder (linear, on clean activations) -- the audit probe.
    probe_data = generate(cfg.n_probe, "clean", seed * 100 + 2)
    ev = evaluate(model, probe_data, ym, ys)
    scaler = StandardScaler().fit(ev["feat"])
    ptt_probe = Ridge(alpha=1.0).fit(scaler.transform(ev["feat"]), probe_data["ptt_true"])

    # Held-out clean check: is PTT decodable at all?
    okd = generate(cfg.n_test, "clean", seed * 100 + 3)
    ok_ev = evaluate(model, okd, ym, ys)
    ptt_dec_ok = ptt_probe.predict(scaler.transform(ok_ev["feat"]))
    ptt_r2 = float(r2_score(okd["ptt_true"], ptt_dec_ok))
    ok_bp_mae = float(np.abs(ok_ev["pred"] - okd["y"]).mean())

    per_cond, points = {}, {}
    for cond in CONDITIONS:
        d = generate(cfg.n_test, cond, seed * 100 + 10 + CONDITIONS.index(cond))
        ev = evaluate(model, d, ym, ys)
        ptt_dec = ptt_probe.predict(scaler.transform(ev["feat"]))
        # gate scores: higher = more trustworthy
        thr_conf = -(np.maximum(0, PTT_LO - ptt_dec) + np.maximum(0, ptt_dec - PTT_HI))
        cons_conf = -np.abs(ptt_dec - d["ptt_measured"])
        per_cond[cond] = {
            "bp_mae": float(np.abs(ev["pred"] - d["y"]).mean()),
            "ptt_decoded_mean": float(ptt_dec.mean()),
            "ptt_measured_mean": float(d["ptt_measured"].mean()),
            "thr_conf_mean": float(thr_conf.mean()),
            "cons_conf_mean": float(cons_conf.mean()),
        }
        points[cond] = {"ptt_measured": d["ptt_measured"], "ptt_decoded": ptt_dec}

    # detection AUROC: clean vs each corruption, per gate
    detection = {}
    ok_thr = -(np.maximum(0, PTT_LO - ptt_dec_ok) + np.maximum(0, ptt_dec_ok - PTT_HI))
    ok_cons = -np.abs(ptt_dec_ok - okd["ptt_measured"])
    for cond in CONDITIONS:
        if cond == "clean":
            continue
        d = generate(cfg.n_test, cond, seed * 100 + 30 + CONDITIONS.index(cond))
        ev = evaluate(model, d, ym, ys)
        ptt_dec = ptt_probe.predict(scaler.transform(ev["feat"]))
        thr = -(np.maximum(0, PTT_LO - ptt_dec) + np.maximum(0, ptt_dec - PTT_HI))
        cons = -np.abs(ptt_dec - d["ptt_measured"])
        labels = np.concatenate([np.ones(len(okd["y"]), int), np.zeros(len(d["y"]), int)])
        detection[cond] = {
            "threshold": float(roc_auc_score(labels, np.concatenate([ok_thr, thr]))),
            "consistency": float(roc_auc_score(labels, np.concatenate([ok_cons, cons]))),
        }
    return {"ptt_r2": ptt_r2, "clean_bp_mae": ok_bp_mae, "per_condition": per_cond,
            "detection": detection, "points": points}


def aggregate(cfg) -> dict:
    per = [run(cfg, s) for s in cfg.seeds]
    conds = [c for c in CONDITIONS if c != "clean"]
    det = {c: {g: {"mean": float(np.mean([p["detection"][c][g] for p in per])),
                   "std": float(np.std([p["detection"][c][g] for p in per]))}
               for g in ("threshold", "consistency")} for c in conds}
    bp = {c: float(np.mean([p["per_condition"][c]["bp_mae"] for p in per])) for c in CONDITIONS}
    return {
        "ptt_r2": float(np.mean([p["ptt_r2"] for p in per])),
        "clean_bp_mae": float(np.mean([p["clean_bp_mae"] for p in per])),
        "detection": det, "bp_mae": bp, "points_seed0": per[0]["points"],
    }


# ── plotting ──────────────────────────────────────────────────────────────────
BLUE, AQUA, YELLOW, RED, MUTE = "#2a78d6", "#1baf7a", "#eda100", "#d03b3b", "#9aa09a"
INK, GRID, SURFACE = "#16191a", "#e1e0d9", "#fcfcfb"
COND_COLOR = {"clean": AQUA, "missing_distal": RED, "morphology_shift": YELLOW, "wrong_calibration": BLUE}
COND_LABEL = {"clean": "clean", "missing_distal": "missing distal",
              "morphology_shift": "morphology shift", "wrong_calibration": "wrong calibration"}


def _style(ax, ygrid=False):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#7c8384", labelsize=8)
    ax.grid(True, axis="y" if ygrid else "both", color=GRID, linewidth=0.6)


def plot(agg: dict, output_dir: Path) -> None:
    fig = plt.figure(figsize=(12.5, 5.2), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.25)
    fig.text(0.02, 0.95, "Auditing a model by decoding PTT from its activations",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.905,
             f"No original data used — synthetic waveforms only. PTT decodable on clean: "
             f"R² = {agg['ptt_r2']:.2f}. Clean BP MAE = {agg['clean_bp_mae']:.1f} mmHg.",
             fontsize=9.5, color="#454b4c")

    # Left: decoded vs measured PTT scatter
    ax = fig.add_subplot(gs[0, 0])
    pts = agg["points_seed0"]
    lim = (0, 55)
    ax.axhspan(PTT_LO, PTT_HI, color=GRID, alpha=0.5, zorder=0)
    ax.plot(lim, lim, "--", color=MUTE, lw=1.3, zorder=1)
    for cond in CONDITIONS:
        p = pts[cond]
        m = np.clip(p["ptt_measured"], *lim); dcd = np.clip(p["ptt_decoded"], *lim)
        ax.scatter(m, dcd, s=12, color=COND_COLOR[cond], alpha=0.5, edgecolors="none",
                   label=COND_LABEL[cond], zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("PTT measured from raw signal (samples)", fontsize=9)
    ax.set_ylabel("PTT decoded from activations (samples)", fontsize=9)
    ax.set_title("On the diagonal = model agrees with the signal\n"
                 "(shaded = physiological band)", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)

    # Right: what each gate catches
    ax = fig.add_subplot(gs[0, 1])
    conds = [c for c in CONDITIONS if c != "clean"]
    x = np.arange(len(conds)); bw = 0.36
    for i, g in enumerate(("threshold", "consistency")):
        means = [agg["detection"][c][g]["mean"] for c in conds]
        stds = [agg["detection"][c][g]["std"] for c in conds]
        ax.bar(x + (i - 0.5) * bw, means, bw, yerr=stds, capsize=3,
               color=(YELLOW if g == "threshold" else BLUE),
               label=("threshold gate" if g == "threshold" else "consistency gate"))
    ax.axhline(0.5, ls="--", color=MUTE, lw=1.2)
    ax.text(len(conds) - 0.5, 0.5, " chance", va="bottom", ha="right", fontsize=8, color="#7c8384")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{COND_LABEL[c]}\nBP err {agg['bp_mae'][c]:.0f}" for c in conds], fontsize=8.5)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("detection AUROC (clean vs corrupted)", fontsize=9)
    ax.set_title("What each gate catches", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    _style(ax, ygrid=True)

    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.07, right=0.98)
    out = output_dir / "ptt_gate.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"[ptt-gate] wrote {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/ptt_gate"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-probe", type=int, default=800)
    p.add_argument("--n-test", type=int, default=800)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.seeds = [0]; a.n_train, a.n_probe, a.n_test, a.epochs = 500, 200, 300, 40
    return a


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate(a)
    plot(agg, a.output_dir)
    print(f"[ptt-gate] PTT R2(clean)={agg['ptt_r2']:.3f}  clean BP MAE={agg['clean_bp_mae']:.2f}")
    for c in [c for c in CONDITIONS if c != "clean"]:
        d = agg["detection"][c]
        print(f"  {c:18s} BP_err={agg['bp_mae'][c]:5.1f}  "
              f"threshold={d['threshold']['mean']:.3f}  consistency={d['consistency']['mean']:.3f}")
    with open(a.output_dir / "ptt_gate.json", "w", encoding="utf-8") as fh:
        json.dump({"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                   "ptt_r2": agg["ptt_r2"], "clean_bp_mae": agg["clean_bp_mae"],
                   "detection": agg["detection"], "bp_mae": agg["bp_mae"]}, fh, indent=2)
    print(f"[ptt-gate] wrote {a.output_dir / 'ptt_gate.json'}")


if __name__ == "__main__":
    main()
