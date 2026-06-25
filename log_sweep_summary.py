"""
log_sweep_summary.py — post a sweep comparison table to W&B.
Reads all *_best.json from results/ and logs a wandb.Table to a 'sweep_summary' run.

Usage:
    python log_sweep_summary.py [DATA_ROOT]
"""
import os, sys, json, glob

try:
    import wandb
except ImportError:
    print("wandb not installed, skipping summary"); sys.exit(0)

if not os.environ.get("WANDB_API_KEY"):
    print("WANDB_API_KEY not set, skipping summary"); sys.exit(0)

DATA_ROOT   = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(DATA_ROOT, "results")

rows = []

# ── Deep model results (single JSON per run) ─────────────────────────────────
for path in sorted(glob.glob(os.path.join(results_dir, "*_best.json"))):
    fname = os.path.basename(path)
    # Skip lgbm (handled separately) and smoke/debug runs
    if any(x in fname for x in ("_sbp_", "_dbp_", "smoke")):
        continue

    with open(path) as f:
        d = json.load(f)

    name = d.get("name", fname.replace("_best.json", ""))
    if name.startswith("lgbm"):
        continue

    m  = d.get("metrics", {})
    hp = d.get("hyperparams", {})
    rows.append({
        "model":          name,
        "hz":             hp.get("fs", 125),
        "test_mae_sbp":   round(m.get("test_mae_sbp",  float("nan")), 2),
        "test_mae_dbp":   round(m.get("test_mae_dbp",  float("nan")), 2),
        "test_rmse_sbp":  round(m.get("test_rmse_sbp", float("nan")), 2),
        "test_rmse_dbp":  round(m.get("test_rmse_dbp", float("nan")), 2),
        "n_params":       d.get("n_params", 0),
        "best_epoch":     d.get("best_epoch", -1),
        "train_time_min": round(d.get("total_train_time_min", 0), 1),
    })

# ── LGBM results (pair sbp + dbp JSONs) ──────────────────────────────────────
pairs: dict = {}
for path in sorted(glob.glob(os.path.join(results_dir, "lgbm_*_best.json"))):
    with open(path) as f:
        d = json.load(f)
    target = d.get("target", "")           # "sbp" or "dbp"
    name   = d.get("name", "")             # e.g. "lgbm_ppg_ecg_sbp"
    key    = name[: name.rfind(f"_{target}")]  # "lgbm_ppg_ecg"
    hp     = d.get("hyperparams", {})
    pairs.setdefault(key, {"hp": hp, "data": d.get("data", {})})
    pairs[key][target] = d.get("metrics", {})

for key, val in sorted(pairs.items()):
    sbp = val.get("sbp", {}); dbp = val.get("dbp", {})
    rows.append({
        "model":          key,
        "hz":             val["data"].get("seq_len", "?"),  # seq_len as proxy
        "test_mae_sbp":   round(sbp.get("mae", float("nan")), 2),
        "test_mae_dbp":   round(dbp.get("mae", float("nan")), 2),
        "test_rmse_sbp":  round(sbp.get("rmse", float("nan")), 2),
        "test_rmse_dbp":  round(dbp.get("rmse", float("nan")), 2),
        "n_params":       0,
        "best_epoch":     -1,
        "train_time_min": 0.0,
    })

if not rows:
    print("No result JSONs found — nothing to log."); sys.exit(0)

# Sort by test SBP MAE
rows.sort(key=lambda r: r["test_mae_sbp"])

# ── Log to W&B ────────────────────────────────────────────────────────────────
run = wandb.init(
    project = os.environ.get("WANDB_PROJECT", "bp-estimation"),
    name    = "sweep_summary",
    job_type= "summary",
    reinit  = True,
)

cols  = ["model", "hz", "test_mae_sbp", "test_mae_dbp",
         "test_rmse_sbp", "test_rmse_dbp", "n_params", "best_epoch", "train_time_min"]
table = wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])

wandb.log({"sweep/comparison_table": table})

# Also log bar charts for quick visual comparison
bar_sbp = wandb.plot.bar(
    wandb.Table(columns=["model", "mae_sbp"], data=[[r["model"], r["test_mae_sbp"]] for r in rows]),
    "model", "mae_sbp", title="Test MAE SBP (mmHg) — lower is better")
bar_dbp = wandb.plot.bar(
    wandb.Table(columns=["model", "mae_dbp"], data=[[r["model"], r["test_mae_dbp"]] for r in rows]),
    "model", "mae_dbp", title="Test MAE DBP (mmHg) — lower is better")
wandb.log({"sweep/bar_sbp": bar_sbp, "sweep/bar_dbp": bar_dbp})

run.finish()
print(f"Logged {len(rows)} runs to W&B sweep_summary.")
for r in rows:
    print(f"  {r['model']:<35} SBP={r['test_mae_sbp']:.2f}  DBP={r['test_mae_dbp']:.2f} mmHg")
