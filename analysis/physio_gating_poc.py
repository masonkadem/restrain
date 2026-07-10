"""Physics-grounded proof of concept: gating a model on its own activations.

This is the clean toy-task design from `toy_identifiability.py`, but with the
abstract linear map replaced by a *real governing physical law*. The point:
demonstrate self-restraint (the model abstaining when the physics is
non-identifiable) in a recognizable, motivated setting, while keeping the
controlled structure that makes the result interpretable.

Two laws, each with an exact identifiability boundary and three corruption
families that map to real clinical failures:

  Beer-Lambert (pulse oximetry, SpO2):
    SpO2 is read from the ratio-of-ratios R = (AC/DC)_red / (AC/DC)_ir, so it
    needs *both* wavelengths. The IR channel is the identifiability-critical
    block. Corruptions:
      - missing_ir   (zeroed):    IR channel dead              -> off-manifold
      - saturated_ir (shifted):   IR channel railed/motion     -> off-manifold
      - mismatched_ir(resampled): IR from a different moment    -> IN-distribution
        (each channel looks normal, but the pair implies the wrong SpO2)

  Moens-Korteweg (cuffless blood pressure, BP):
    BP is read from pulse-transit-time PTT plus a per-subject stiffness
    calibration. Uncalibrated PTT does not determine BP. Corruptions:
      - missing_cal  (zeroed):    no calibration                -> off-manifold
      - distorted_cal(shifted):   calibration railed            -> off-manifold
      - wrong_subject(resampled): calibration from another person-> IN-distribution

The clean toy result predicts, and this reproduces: probe-based gating
reduces selective risk (the model restrains itself); a supervised probe plus
label-free ensemble/OOD scores between them catch the off-manifold failures;
but the in-distribution "plausible-but-wrong" failure (mismatched IR / wrong
subject) is caught by nothing -- the information needed is absent from the
input. That boundary is the honest, decision-relevant contribution.

    python analysis/physio_gating_poc.py
    python analysis/physio_gating_poc.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from abstention_utils import area_under_risk_coverage, risk_coverage_curve  # noqa: E402
from toy_identifiability import Readout  # noqa: E402


# ── physical instruments ──────────────────────────────────────────────────────
# Each returns a dict {x, y, answerable, critical} where `critical` marks the
# columns of x that are the identifiability-critical block (the channel a
# corruption attacks).


def generate_beer_lambert(n: int, corruption: str, seed: int) -> dict:
    """SpO2 from a two-wavelength ratio-of-ratios (standard R->SpO2 calibration).

    ratio_ir depends on perfusion (a nuisance), ratio_red = R * ratio_ir, so
    SpO2 = 110 - 25 * ratio_red/ratio_ir is identifiable only with both.
    """
    rng = np.random.default_rng(seed)
    spo2 = rng.uniform(85.0, 99.0, size=n)
    perfusion = rng.uniform(0.4, 1.6, size=n)
    r = (110.0 - spo2) / 25.0                       # standard linear calibration, R in ~[0.44, 1.0]
    ratio_ir = 0.6 * perfusion * (1.0 + 0.03 * rng.normal(size=n))
    ratio_red = r * ratio_ir * (1.0 + 0.03 * rng.normal(size=n))

    answerable = np.ones(n, dtype=int)
    if corruption == "missing_ir":
        ratio_ir = np.zeros(n)
        answerable[:] = 0
    elif corruption == "saturated_ir":
        ratio_ir = ratio_ir * rng.uniform(4.0, 8.0, size=n)   # railed / motion, off-manifold
        answerable[:] = 0
    elif corruption == "mismatched_ir":
        perf2 = rng.uniform(0.4, 1.6, size=n)                 # IR from a different moment
        ratio_ir = 0.6 * perf2 * (1.0 + 0.03 * rng.normal(size=n))
        answerable[:] = 0

    x = np.stack([ratio_red, ratio_ir], axis=1).astype(np.float32)
    return {
        "x": x, "y": spo2.astype(np.float32), "answerable": answerable,
        "critical": np.array([False, True]),   # the IR column is critical
    }


def generate_moens_korteweg(n: int, corruption: str, seed: int) -> dict:
    """BP from pulse-transit-time plus a per-subject stiffness calibration.

    PWV^2 = K * exp(alpha * BP / 100); PTT = L / PWV. BP is identifiable from
    PTT only with the calibration block (K, alpha, L).
    """
    rng = np.random.default_rng(seed)
    bp = rng.uniform(90.0, 150.0, size=n)
    K = rng.uniform(0.8, 2.2, size=n)                # lumped subject stiffness (scaled)
    alpha = rng.uniform(0.8, 1.5, size=n)
    L = rng.uniform(0.5, 0.9, size=n)                # path length (scaled)
    pwv = np.sqrt(K * np.exp(alpha * bp / 100.0))
    ptt = L / pwv * (1.0 + 0.02 * rng.normal(size=n))

    cal = np.stack([K, alpha, L], axis=1)
    answerable = np.ones(n, dtype=int)
    if corruption == "missing_cal":
        cal = np.zeros((n, 3))
        answerable[:] = 0
    elif corruption == "distorted_cal":
        cal = cal * rng.uniform(4.0, 8.0, size=(n, 1))
        answerable[:] = 0
    elif corruption == "wrong_subject":
        cal = np.stack([                              # calibration from another person
            rng.uniform(0.8, 2.2, size=n),
            rng.uniform(0.8, 1.5, size=n),
            rng.uniform(0.5, 0.9, size=n),
        ], axis=1)
        answerable[:] = 0

    x = np.concatenate([ptt[:, None], cal], axis=1).astype(np.float32)
    return {
        "x": x, "y": bp.astype(np.float32), "answerable": answerable,
        "critical": np.array([False, True, True, True]),   # PTT observed, calibration critical
    }


LAWS = {
    "beer_lambert": {
        "gen": generate_beer_lambert,
        "target": "SpO2 (%)",
        "corruptions": ("missing_ir", "saturated_ir", "mismatched_ir"),
        "corr_labels": {
            "missing_ir": "missing IR\n(dead sensor)",
            "saturated_ir": "saturated IR\n(motion / rail)",
            "mismatched_ir": "mismatched IR\n(wrong moment — hardest)",
        },
        "title": "Beer–Lambert · pulse oximetry (SpO₂)",
    },
    "moens_korteweg": {
        "gen": generate_moens_korteweg,
        "target": "BP (mmHg)",
        "corruptions": ("missing_cal", "distorted_cal", "wrong_subject"),
        "corr_labels": {
            "missing_cal": "missing cal\n(uncalibrated)",
            "distorted_cal": "distorted cal\n(bad values)",
            "wrong_subject": "wrong subject\n(other person — hardest)",
        },
        "title": "Moens–Korteweg · cuffless blood pressure (BP)",
    },
}


# ── training / detectors (law-agnostic, on standardized features) ──────────────


@dataclass
class Scaler:
    xm: np.ndarray
    xs: np.ndarray
    ym: float
    ys: float


def fit_scaler(data: dict) -> Scaler:
    x = data["x"]
    return Scaler(x.mean(0), x.std(0) + 1e-8, float(data["y"].mean()), float(data["y"].std() + 1e-8))


def train_readout_scaled(model, data, sc: Scaler, epochs: int, seed: int) -> None:
    torch.manual_seed(seed)
    x = torch.from_numpy(((data["x"] - sc.xm) / sc.xs).astype(np.float32))
    y = torch.from_numpy(((data["y"] - sc.ym) / sc.ys).astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def eval_scaled(model, data, sc: Scaler) -> dict:
    model.eval()
    with torch.no_grad():
        pred, feat = model(torch.from_numpy(((data["x"] - sc.xm) / sc.xs).astype(np.float32)),
                           return_features=True)
    pred = pred.numpy() * sc.ys + sc.ym          # back to real units
    pred = np.clip(pred, sc.ym - 8 * sc.ys, sc.ym + 8 * sc.ys)
    return {"pred": pred, "feat": feat.numpy(), "loss": (pred - data["y"]) ** 2}


def mahalanobis_scorer(ref_feats):
    mu = ref_feats.mean(0)
    cov = np.cov(ref_feats, rowvar=False) + 1e-3 * np.eye(ref_feats.shape[1])
    inv = np.linalg.pinv(cov)
    return lambda f: np.einsum("ni,ij,nj->n", f - mu, inv, f - mu)


def run_law(gen, corruptions, cfg, seed: int) -> dict:
    train = gen(cfg.n_train, "clean", seed * 100 + 1)
    sc = fit_scaler(train)
    models = [Readout(train["x"].shape[1], cfg.hidden) for _ in range(cfg.n_ensemble)]
    for k, m in enumerate(models):
        train_readout_scaled(m, train, sc, cfg.epochs, seed * 1000 + k)

    def ens(data):
        evs = [eval_scaled(m, data, sc) for m in models]
        preds = np.stack([e["pred"] for e in evs], axis=1)
        return preds.mean(1), preds.std(1), evs[0]["feat"], evs[0]["loss"]

    # answerable reference
    ref = gen(cfg.n_probe, "clean", seed * 100 + 2)
    _, _, ref_feat, _ = ens(ref)
    maha = mahalanobis_scorer(ref_feat)

    # probe trained on answerable vs the FIRST (missing) corruption only
    known = gen(cfg.n_probe, corruptions[0], seed * 100 + 3)
    _, _, kf, _ = ens(known)
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probe.fit(np.concatenate([ref_feat, kf]),
              np.concatenate([ref["answerable"], known["answerable"]]))

    # test answerable pool
    ok = gen(cfg.n_test, "clean", seed * 100 + 4)
    ok_mean, ok_std, ok_feat, ok_loss = ens(ok)
    ok_probe = probe.predict_proba(ok_feat)[:, 1]
    ok_maha = maha(ok_feat)

    detection, risk_curves = {}, {}
    for corr in corruptions:
        bad = gen(cfg.n_test, corr, seed * 100 + 30 + corruptions.index(corr))
        bad_mean, bad_std, bad_feat, bad_loss = ens(bad)
        labels = np.concatenate([ok["answerable"], bad["answerable"]])
        detection[corr] = {
            "probe": float(roc_auc_score(labels, np.concatenate([ok_probe, probe.predict_proba(bad_feat)[:, 1]]))),
            "ensemble": float(roc_auc_score(labels, np.concatenate([-ok_std, -bad_std]))),
            "mahalanobis": float(roc_auc_score(labels, np.concatenate([-ok_maha, -maha(bad_feat)]))),
        }
        # risk-coverage on the answerable+corrupted mix, gating by the probe
        losses = np.concatenate([ok_loss, bad_loss])
        conf = np.concatenate([ok_probe, probe.predict_proba(bad_feat)[:, 1]])
        rng = np.random.default_rng(seed)
        cov_p, risk_p = risk_coverage_curve(losses, conf)
        cov_r, risk_r = risk_coverage_curve(losses, rng.random(len(losses)))
        cov_o, risk_o = risk_coverage_curve(losses, labels.astype(float))
        risk_curves[corr] = {
            "coverage": cov_p.tolist(), "probe": risk_p.tolist(),
            "random": np.interp(cov_p, cov_r, risk_r).tolist(),
            "oracle": np.interp(cov_p, cov_o, risk_o).tolist(),
            "aurc_probe": area_under_risk_coverage(losses, conf),
            "aurc_random": area_under_risk_coverage(losses, rng.random(len(losses))),
        }
    clean_mae = float(np.abs(ok_mean - ok["y"]).mean())
    return {"detection": detection, "risk_curves": risk_curves, "clean_mae": clean_mae}


def aggregate(gen, corruptions, cfg) -> dict:
    per = [run_law(gen, corruptions, cfg, s) for s in cfg.seeds]
    det = {
        c: {d: {"mean": float(np.mean([p["detection"][c][d] for p in per])),
                "std": float(np.std([p["detection"][c][d] for p in per]))}
            for d in ("probe", "ensemble", "mahalanobis")}
        for c in corruptions
    }
    return {
        "detection": det,
        "clean_mae": float(np.mean([p["clean_mae"] for p in per])),
        "risk_curves_seed0": per[0]["risk_curves"],   # one representative seed for the curve
    }


# ── plotting ──────────────────────────────────────────────────────────────────
BLUE, AQUA, YELLOW, VIOLET, MUTE = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#9aa09a"
INK, GRID, SURFACE = "#16191a", "#e1e0d9", "#fcfcfb"
DET_COLORS = {"probe": BLUE, "ensemble": AQUA, "mahalanobis": YELLOW}
DET_LABELS = {"probe": "supervised probe", "ensemble": "ensemble (label-free)",
              "mahalanobis": "OOD distance (label-free)"}


def _style(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#7c8384", labelsize=8)
    ax.grid(True, axis="y" if ygrid else "both", color=GRID, linewidth=0.6)


def plot_law(law_key: str, agg: dict, output_dir: Path) -> None:
    meta = LAWS[law_key]
    corrs = list(meta["corruptions"])
    fig = plt.figure(figsize=(12.5, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.26)
    fig.text(0.02, 0.955, meta["title"], fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.915,
             f"Gating a frozen model on its own activations. Clean-task MAE = {agg['clean_mae']:.2f}.",
             fontsize=9.5, color="#454b4c")

    # Left: risk-coverage on the hardest DETECTABLE corruption (the first/missing one)
    ax = fig.add_subplot(gs[0, 0])
    rc = agg["risk_curves_seed0"][corrs[0]]
    cov = rc["coverage"]
    ax.plot(cov, rc["oracle"], color=VIOLET, lw=2, label="oracle (true label)")
    ax.plot(cov, rc["probe"], color=BLUE, lw=2, label="activation probe")
    ax.plot(cov, rc["random"], color=MUTE, lw=2, label="random abstention")
    ax.set_xlabel("coverage (fraction answered)", fontsize=9)
    ax.set_ylabel(f"selective risk — retained {meta['target']} error²", fontsize=9)
    ax.set_title("The model restraining itself (clean + missing-channel mix)",
                 fontsize=9.5, color=INK, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _style(ax, ygrid=False)

    # Right: who-catches-what across the three corruption families
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(corrs)); bw = 0.26
    for i, d in enumerate(("probe", "ensemble", "mahalanobis")):
        means = [agg["detection"][c][d]["mean"] for c in corrs]
        stds = [agg["detection"][c][d]["std"] for c in corrs]
        ax.bar(x + (i - 1) * bw, means, bw, yerr=stds, capsize=3, color=DET_COLORS[d], label=DET_LABELS[d])
    ax.axhline(0.5, ls="--", color=MUTE, lw=1.2)
    ax.text(len(corrs) - 0.5, 0.5, " chance", va="bottom", ha="right", fontsize=8, color="#7c8384")
    ax.set_xticks(x); ax.set_xticklabels([meta["corr_labels"][c] for c in corrs], fontsize=8.5)
    ax.set_ylabel("answerability detection AUROC", fontsize=9)
    ax.set_ylim(0.0, 1.03)
    ax.set_title("Who catches which failure", fontsize=9.5, color=INK, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center")
    _style(ax)

    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98)
    out = output_dir / f"poc_{law_key}.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"[poc] wrote {out}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/physio_poc"))
    p.add_argument("--laws", nargs="+", default=["beer_lambert", "moens_korteweg"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-ensemble", type=int, default=6)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-probe", type=int, default=800)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seeds = [0]; args.n_ensemble = 3
        args.n_train, args.n_probe, args.n_test = 600, 200, 300
        args.epochs = 50
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for law in args.laws:
        meta = LAWS[law]
        print(f"[poc] {law}", flush=True)
        agg = aggregate(meta["gen"], meta["corruptions"], args)
        plot_law(law, agg, args.output_dir)
        report[law] = {"detection": agg["detection"], "clean_mae": agg["clean_mae"]}
        for c in meta["corruptions"]:
            r = agg["detection"][c]
            print(f"  {c:16s} probe={r['probe']['mean']:.3f} ensemble={r['ensemble']['mean']:.3f} "
                  f"maha={r['mahalanobis']['mean']:.3f}", flush=True)
    with open(args.output_dir / "poc.json", "w", encoding="utf-8") as fh:
        json.dump({"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   "results": report}, fh, indent=2)
    print(f"[poc] wrote {args.output_dir / 'poc.json'}", flush=True)


if __name__ == "__main__":
    main()
