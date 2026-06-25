"""
eda.py — exploratory data analysis for the BP dataset, logged to W&B.

Run once (not per-experiment).  Produces:
  - Violin plots of SBP/DBP across train/val/test  (distribution-shift check)
  - Overlaid histograms (SBP, DBP, pulse pressure)
  - SBP-vs-DBP joint distribution
  - Per-subject mean-BP spread  (inter-subject variability — the core difficulty)
  - Summary-statistics table per split
  - ACC/AHA BP-category breakdown per split
  - KS distribution-shift test (train vs val, train vs test)
  - Example raw waveforms (PPG/ECG/RESP)

Usage:
    python eda.py [DATA_ROOT]
"""
import os, sys, glob
import numpy as np
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False

try:
    import wandb
    WB = bool(os.environ.get("WANDB_API_KEY"))
except ImportError:
    WB = False

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
CHANNELS  = ["ppg", "ecg", "resp"]
FS        = 125
SPLITS    = ["train", "val", "test"]
COLORS    = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}


def read_subject_list(path):
    with open(path) as f:
        content = f.read().strip()[1:-1]
    return [s.strip().strip("'") for s in content.split(",") if s.strip()]


def _load(subject, modality):
    p = os.path.join(DATA_ROOT, modality, f"{subject}_{modality}.npy")
    return np.load(p) if os.path.exists(p) else None


def load_split(split):
    """Return labels (N,2) and per-subject mean BP, matching the model's subject filter."""
    subs = read_subject_list(os.path.join(DATA_ROOT, f"{split}_subjects.txt"))
    all_lbls, subj_means = [], []
    for s in subs:
        lbl = _load(s, "labels")
        ppg = os.path.exists(os.path.join(DATA_ROOT, "ppg", f"{s}_ppg.npy"))
        if lbl is None or not ppg:
            continue
        all_lbls.append(lbl)
        subj_means.append(lbl.mean(axis=0))
    return np.concatenate(all_lbls), np.array(subj_means), len(subj_means)


def categorize(sbp, dbp):
    """ACC/AHA 2017 BP categories."""
    cats = np.empty(len(sbp), dtype=object)
    cats[:] = "Stage 2 / Crisis"
    cats[(sbp < 140) & (dbp < 90)] = "Stage 1"
    cats[(sbp < 130) & (dbp < 80)] = "Elevated"
    cats[(sbp < 120) & (dbp < 80)] = "Normal"
    return cats


def main():
    data = {sp: load_split(sp) for sp in SPLITS}
    for sp in SPLITS:
        lbl, _, n_subj = data[sp]
        print(f"{sp:5s}: {len(lbl):6d} segments  {n_subj:4d} subjects  "
              f"SBP {lbl[:,0].mean():.1f}+/-{lbl[:,0].std():.1f}  "
              f"DBP {lbl[:,1].mean():.1f}+/-{lbl[:,1].std():.1f}")

    run = None
    if WB:
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "bp-estimation"),
                         name="eda", job_type="eda", reinit=True)
    wb_logs = {}

    # ── Violin plots (SBP, DBP across splits) ─────────────────────────────────
    if MPL:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, ti, name in zip(axes, [0, 1], ["SBP", "DBP"]):
            vdata = [data[sp][0][:, ti] for sp in SPLITS]
            parts = ax.violinplot(vdata, showmeans=True, showmedians=True)
            for pc, sp in zip(parts["bodies"], SPLITS):
                pc.set_facecolor(COLORS[sp]); pc.set_alpha(0.6)
            ax.set_xticks([1, 2, 3]); ax.set_xticklabels(SPLITS)
            ax.set_ylabel(f"{name} (mmHg)")
            ax.set_title(f"{name} distribution by split")
        fig.tight_layout()
        wb_logs["eda/violin_bp"] = wandb.Image(fig) if WB else None
        fig.savefig(os.path.join(DATA_ROOT, "eda_violin_bp.png"), dpi=120)
        plt.close(fig)

        # ── Overlaid histograms ───────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        titles = ["SBP (mmHg)", "DBP (mmHg)", "Pulse Pressure SBP-DBP (mmHg)"]
        for ax, comp, title in zip(axes, ["sbp", "dbp", "pp"], titles):
            for sp in SPLITS:
                lbl = data[sp][0]
                vals = (lbl[:, 0] if comp == "sbp" else
                        lbl[:, 1] if comp == "dbp" else lbl[:, 0] - lbl[:, 1])
                ax.hist(vals, bins=60, density=True, alpha=0.5,
                        label=sp, color=COLORS[sp])
            ax.set_xlabel(title); ax.set_ylabel("density"); ax.legend()
        fig.tight_layout()
        wb_logs["eda/histograms"] = wandb.Image(fig) if WB else None
        fig.savefig(os.path.join(DATA_ROOT, "eda_histograms.png"), dpi=120)
        plt.close(fig)

        # ── Joint SBP-DBP (train) ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 5))
        tr = data["train"][0]
        hb = ax.hexbin(tr[:, 0], tr[:, 1], gridsize=50, cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="count")
        ax.set_xlabel("SBP (mmHg)"); ax.set_ylabel("DBP (mmHg)")
        ax.set_title("Joint SBP-DBP (train)")
        fig.tight_layout()
        wb_logs["eda/joint_sbp_dbp"] = wandb.Image(fig) if WB else None
        fig.savefig(os.path.join(DATA_ROOT, "eda_joint.png"), dpi=120)
        plt.close(fig)

        # ── Per-subject mean BP spread (inter-subject variability) ────────────
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, ti, name in zip(axes, [0, 1], ["SBP", "DBP"]):
            for sp in SPLITS:
                means = data[sp][1][:, ti]
                ax.hist(means, bins=30, density=True, alpha=0.5,
                        label=sp, color=COLORS[sp])
            ax.set_xlabel(f"Per-subject mean {name} (mmHg)")
            ax.set_ylabel("density"); ax.legend()
            ax.set_title(f"Inter-subject {name} variability")
        fig.tight_layout()
        wb_logs["eda/per_subject_bp"] = wandb.Image(fig) if WB else None
        fig.savefig(os.path.join(DATA_ROOT, "eda_per_subject.png"), dpi=120)
        plt.close(fig)

        # ── Example raw waveforms (first train subject, first segment) ────────
        subs = read_subject_list(os.path.join(DATA_ROOT, "train_subjects.txt"))
        for s in subs:
            sigs = {c: _load(s, c) for c in CHANNELS}
            if sigs["ppg"] is not None:
                fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
                t = np.arange(sigs["ppg"].shape[1]) / FS
                for ax, c in zip(axes, CHANNELS):
                    if sigs[c] is not None:
                        ax.plot(t, sigs[c][0], lw=0.7, color=COLORS["train"])
                    ax.set_ylabel(c.upper())
                axes[-1].set_xlabel("time (s)")
                axes[0].set_title(f"Example raw waveforms — subject {s}, segment 0")
                fig.tight_layout()
                wb_logs["eda/example_waveforms"] = wandb.Image(fig) if WB else None
                fig.savefig(os.path.join(DATA_ROOT, "eda_waveforms.png"), dpi=120)
                plt.close(fig)
                break

    # ── Summary statistics table ──────────────────────────────────────────────
    stat_rows = []
    for sp in SPLITS:
        lbl, _, n_subj = data[sp]
        for ti, name in [(0, "SBP"), (1, "DBP")]:
            v = lbl[:, ti]
            stat_rows.append([sp, name, len(v), n_subj,
                              round(float(v.mean()), 1), round(float(v.std()), 1),
                              round(float(v.min()), 1), round(float(np.percentile(v, 25)), 1),
                              round(float(np.median(v)), 1), round(float(np.percentile(v, 75)), 1),
                              round(float(v.max()), 1)])
    print("\nSummary statistics:")
    for r in stat_rows:
        print(f"  {r[0]:5s} {r[1]} n={r[2]:6d} mean={r[4]:.1f} std={r[5]:.1f} "
              f"range=[{r[6]:.0f},{r[10]:.0f}]")

    # ── BP category breakdown ─────────────────────────────────────────────────
    cat_order = ["Normal", "Elevated", "Stage 1", "Stage 2 / Crisis"]
    cat_rows = []
    for sp in SPLITS:
        lbl = data[sp][0]
        cats = categorize(lbl[:, 0], lbl[:, 1])
        total = len(cats)
        for c in cat_order:
            n = int((cats == c).sum())
            cat_rows.append([sp, c, n, round(100 * n / total, 1)])
    print("\nBP categories (ACC/AHA):")
    for r in cat_rows:
        print(f"  {r[0]:5s} {r[1]:18s} {r[2]:6d} ({r[3]:.1f}%)")

    # ── Distribution shift (KS test vs train) ─────────────────────────────────
    shift_rows = []
    for sp in ["val", "test"]:
        for ti, name in [(0, "SBP"), (1, "DBP")]:
            ks, p = stats.ks_2samp(data["train"][0][:, ti], data[sp][0][:, ti])
            shift_rows.append([f"train vs {sp}", name, round(float(ks), 4),
                               round(float(p), 4),
                               "SHIFT" if p < 0.05 else "ok"])
    print("\nDistribution shift (KS test vs train):")
    for r in shift_rows:
        print(f"  {r[0]:14s} {r[1]} KS={r[2]:.4f} p={r[3]:.4f}  {r[4]}")

    # ── Log tables to W&B ─────────────────────────────────────────────────────
    if WB and run:
        wb_logs = {k: v for k, v in wb_logs.items() if v is not None}
        wb_logs["eda/summary_stats"] = wandb.Table(
            columns=["split", "target", "n", "n_subjects", "mean", "std",
                     "min", "p25", "median", "p75", "max"], data=stat_rows)
        wb_logs["eda/bp_categories"] = wandb.Table(
            columns=["split", "category", "count", "pct"], data=cat_rows)
        wb_logs["eda/distribution_shift"] = wandb.Table(
            columns=["comparison", "target", "ks_stat", "p_value", "flag"], data=shift_rows)
        wandb.log(wb_logs)
        run.finish()
        print("\nLogged EDA to W&B run 'eda'.")
    else:
        print("\nW&B not active — PNGs saved to data root.")


if __name__ == "__main__":
    main()
