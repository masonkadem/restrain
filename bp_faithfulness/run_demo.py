"""End-to-end driver: baseline probe battery + PEP recoverability sweep ->
self-explaining figure. Results are cached so the figure regenerates without
retraining.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg.seed)
    if args.quick:
        cfg.raw["subjects"]["samples_per_subject"] = 60
        cfg.raw["model"]["epochs"] = 40
        cfg.raw["sweep"]["seeds"] = [0]
    out = Path(cfg["paths"]["results_dir"]); out.mkdir(exist_ok=True)

    # ---- baseline (gamma=0, pep=0): the true-positive regime ----
    print("[baseline] train + probe battery ...", flush=True)
    ds = generate_dataset(cfg, gamma=0.0, pep_sd_ms=0.0, seed=cfg.seed)
    tr, te = subject_split(ds, cfg["sweep"]["train_subjects"], seed=cfg.seed)
    model = A.train_model(ds, tr, cfg, seed=cfg.seed)
    mae, per_sub = A.mae_per_subject(model, ds, te)
    pr = A.probe_T(model, ds, tr, te, cfg)
    ab = A.ablation(model, ds, te, pr["direction"])
    dv = A.donor_swap(model, ds, te, pr["direction"], cfg)
    sal_p, sal_d = model.saliency(ds.prox[te], ds.dist[te])

    print("[sweep] PEP (recoverability) ...", flush=True)
    psw = A.run_sweep(cfg, "pep")

    ex_p, ex_d = ds.prox[te][0], ds.dist[te][0]
    L_ex = int(np.nonzero(np.abs(ex_p) > 1e-9)[0].max()) + 1
    R = dict(
        fs=cfg["simulator"]["fs"], ex_prox=ex_p[:L_ex], ex_dist=ex_d[:L_ex],
        probe_r2=pr["r2"], probe_ci=pr["ci"], probe_shuffle=pr["r2_shuffle"],
        r2_threshold=cfg["probe"]["r2_significance"],
        donor_x=dv["x"], donor_shift=dv["shift"], donor_sign=dv["sign_acc"],
        sal_prox=sal_p, sal_dist=sal_d,
        pep_x=[r["x"] for r in psw], pep_r2=[r["r2"] for r in psw],
        pep_donor=[r["donor_sign"] for r in psw], pep_abst=[r["abstention"] for r in psw],
        pep_mae=[r["mae"] for r in psw],
    )
    np.savez(out / "demo_results.npz",
             **{k: (np.array(v, dtype=object) if isinstance(v, (list, tuple)) else v)
                for k, v in R.items()})
    with open(out / "summary.json", "w") as f:
        json.dump({"baseline": {"mae": mae, "probe_r2": pr["r2"], "probe_shuffle": pr["r2_shuffle"],
                                "ablation_delta": ab["mae_ablated"] - ab["mae_intact"],
                                "donor_sign_acc": dv["sign_acc"], "donor_bounded": dv["bounded_ok"]},
                   "pep_sweep": psw}, f, indent=2)
    make_figure(R, out / "comprehensive.png")
    print(f"\nBaseline: MAE={mae:.2f}  probeR2={pr['r2']:.2f} (shuffle {pr['r2_shuffle']:.2f})  "
          f"donorSign={dv['sign_acc']:.0%}")
    print("PEP sweep probe R²:", [round(r["r2"], 2) for r in psw])
    print(f"wrote {out}/comprehensive.png, demo_results.npz, summary.json")


if __name__ == "__main__":
    main()
