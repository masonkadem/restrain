"""End-to-end driver: baseline probe battery + gamma/PEP sweeps -> comprehensive
figure. Saves all results to disk so the figure regenerates without retraining.

    python run_demo.py           # full
    python run_demo.py --quick   # smaller/faster
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bpf import analysis as A
from bpf.config import load_config, seed_everything
from bpf.dataset import generate_dataset, subject_split
from bpf.plotting import make_figure
from bpf.simulator import (apply_tube_load, foot_to_foot_delay, make_proximal_waveform)


def foot_to_foot_vs_T(fs=250, hr=66, n_beats=8):
    """Clean illustration: foot-to-foot vs true T at gamma 0 and 0.8."""
    Tg = np.linspace(0.10, 0.22, 9)
    out = {0.0: [], 0.8: []}
    for g in (0.0, 0.8):
        for T in Tg:
            _, p = make_proximal_waveform(fs, hr, n_beats)
            d = apply_tube_load(p, fs, T, g)
            out[g].append(foot_to_foot_delay(p, d, fs, hr, n_beats))
    return Tg, np.array(out[0.0]), np.array(out[0.8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg.seed)
    if args.quick:
        cfg.raw["subjects"]["samples_per_subject"] = 50
        cfg.raw["model"]["epochs"] = 40
        cfg.raw["sweep"]["seeds"] = [0]
    out = Path(cfg["paths"]["results_dir"]); out.mkdir(exist_ok=True)

    # ---- baseline (gamma=0, pep=0): the true-positive regime ----
    print("[baseline] training + probe battery ...", flush=True)
    ds = generate_dataset(cfg, gamma=0.0, pep_sd_ms=0.0, seed=cfg.seed)
    tr, te = subject_split(ds, cfg["sweep"]["train_subjects"], seed=cfg.seed)
    model = A.train_model(ds, tr, cfg, seed=cfg.seed)
    mae, per_sub = A.mae_per_subject(model, ds, te)
    pr = A.probe_T(model, ds, tr, te, cfg)
    ab = A.ablation(model, ds, te, pr["direction"])
    dv = A.donor_swap(model, ds, te, pr["direction"], cfg)

    Tg, ftf0, ftf08 = foot_to_foot_vs_T()

    print("[sweep] gamma ...", flush=True)
    gsw = A.run_sweep(cfg, "gamma")
    print("[sweep] pep ...", flush=True)
    psw = A.run_sweep(cfg, "pep")

    ex_p, ex_d = ds.prox[te][0], ds.dist[te][0]
    L_ex = int(np.nonzero(np.abs(ex_p) > 1e-9)[0].max()) + 1   # drop trailing zero-pad
    R = dict(
        fs=cfg["simulator"]["fs"],
        ex_prox=ex_p[:L_ex], ex_dist=ex_d[:L_ex],
        T_grid=Tg, ftf_g0=ftf0, ftf_g08=ftf08,
        probe_r2=pr["r2"], probe_ci=pr["ci"], probe_shuffle=pr["r2_shuffle"],
        r2_threshold=cfg["probe"]["r2_significance"],
        mae_intact=ab["mae_intact"], mae_ablated=ab["mae_ablated"],
        donor_x=dv["x"], donor_shift=dv["shift"], donor_sign=dv["sign_acc"],
        per_subject_mae=per_sub,
        gamma_x=[r["x"] for r in gsw], gamma_r2=[r["r2"] for r in gsw],
        gamma_shuf=[r["r2_shuffle"] for r in gsw], gamma_mae=[r["mae"] for r in gsw],
        gamma_abst=[r["abstention"] for r in gsw],
        pep_x=[r["x"] for r in psw], pep_r2=[r["r2"] for r in psw],
        pep_shuf=[r["r2_shuffle"] for r in psw], pep_mae=[r["mae"] for r in psw],
        pep_abst=[r["abstention"] for r in psw],
    )
    np.savez(out / "demo_results.npz", **{k: np.array(v, dtype=object) if isinstance(v, (list, dict)) else v
                                          for k, v in R.items()})
    with open(out / "summary.json", "w") as f:
        json.dump({"baseline": {"mae": mae, "probe_r2": pr["r2"], "probe_shuffle": pr["r2_shuffle"],
                                "ablation_delta": ab["mae_ablated"] - ab["mae_intact"],
                                "donor_sign_acc": dv["sign_acc"], "donor_bounded": dv["bounded_ok"]},
                   "gamma_sweep": gsw, "pep_sweep": psw}, f, indent=2)
    make_figure(R, out / "comprehensive.png")
    print(f"\nBaseline: MAE={mae:.2f}  probeR2={pr['r2']:.2f} (shuffle {pr['r2_shuffle']:.2f})  "
          f"ablationΔ={ab['mae_ablated']-ab['mae_intact']:+.2f}  donorSign={dv['sign_acc']:.0%}")
    print(f"wrote {out}/comprehensive.png, demo_results.npz, summary.json")


if __name__ == "__main__":
    main()
