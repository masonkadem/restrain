"""
train.py — single-experiment runner for BP estimation.

Usage:
    python train.py --model transformer  --channels ppg,ecg,resp
    python train.py --model dual_stream  --channels ppg,ecg
    python train.py --model tri_stream   --channels ppg,ecg,resp
    python train.py --model s4           --channels ppg,ecg,resp
    python train.py --model lgbm         --channels ppg,ecg,resp

    # GPU profiles (controls window length + model size)
    python train.py --model transformer --channels ppg,ecg --gpu_profile 3080
    python train.py --model transformer --channels ppg,ecg --gpu_profile h100

    # Smoke test (3 epochs, 400 samples)
    python train.py --model transformer --channels ppg --smoke

All results are logged to W&B if WANDB_API_KEY is set in the environment.
Each run saves  <run_name>_best.pt  and  <run_name>_best.json  alongside.
"""

import argparse, json, os, math, random, time, warnings
try:
    from torchinfo import summary as torch_summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = MATPLOTLIB_AVAILABLE
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
import numpy as np
from scipy import stats
from scipy.signal import welch, find_peaks
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── GPU profiles ──────────────────────────────────────────────────────────────
# Labels = mean beat-wise SBP/DBP across the full 30 s ABP waveform.
# Transformer attention is O(L^2) so window length is GPU-limited.
#
#  Profile  | window     | seq_len | d_model | layers | batch | 3080 | H100
#  ---------|-----------|---------|---------|--------|-------|------|------
#  fast     |  8 s      |  1000   |   64    |   2    |  64   |  OK  |  OK
#  3080     | 15 s      |  1875   |  128    |   4    |  32   |  OK  |  OK
#  h100     | 30 s      |  3750   |  256    |   6    |  32   | OOM  |  OK
_PROFILES = {
    #              seq_len  seg_start  d_model  n_heads  n_layers  batch
    "fast":       ( 1000,    375,       64,       4,       2,        64),
    "3080":       ( 1875,    375,      128,       8,       4,        32),
    "h100":       ( 3750,      0,      256,       8,       6,        32),
}

FS          = 125
CHANNEL_MAP = {"ppg": 0, "ecg": 1, "resp": 2}
_ROOT       = os.path.dirname(os.path.abspath(__file__))

# Bump when model architectures change so the sweep skip-logic re-runs stale results.
# v2 = sinusoidal PE everywhere, redesigned dual/tri stream, fixed S4 cross-channel fusion.
ARCH_VERSION = "v2"

# ── Data loading ──────────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def read_subject_list(path):
    with open(path) as f:
        content = f.read().strip()[1:-1]
    return np.array([s.strip().strip("'") for s in content.split(",") if s.strip()])


def _load(root, subject, modality):
    p = os.path.join(root, modality, f"{subject}_{modality}.npy")
    return np.load(p) if os.path.exists(p) else None


def extract_segments(subjects, root, seq_len, seg_start):
    sl = slice(seg_start, seg_start + seq_len)
    sigs, lbls, missing = [], [], 0
    for subj in subjects:
        ppg_raw = _load(root, subj, "ppg")
        lbl_raw = _load(root, subj, "labels")
        if ppg_raw is None or lbl_raw is None:
            missing += 1; continue
        raws = [ppg_raw, _load(root, subj, "ecg"), _load(root, subj, "resp")]
        for j in range(len(ppg_raw)):
            channels = []
            for raw in raws:
                if raw is not None:
                    s = raw[j][sl].astype(np.float32)
                    s = (s - s.mean()) / (s.std() + 1e-8)
                else:
                    s = np.zeros(seq_len, dtype=np.float32)
                channels.append(s)
            sigs.append(np.stack(channels, axis=-1))
            lbls.append(lbl_raw[j].astype(np.float32))
    if missing:
        print(f"Warning: {missing} subjects skipped (missing PPG/labels)")
    return np.array(sigs), np.array(lbls)


# ── PyTorch utilities ─────────────────────────────────────────────────────────

class BPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, channels, cfg):
    ch_idx = [CHANNEL_MAP[c] for c in channels]
    sel    = lambda X: np.ascontiguousarray(X[:, :, ch_idx])
    g = torch.Generator(); g.manual_seed(cfg["seed"])
    tr = DataLoader(BPDataset(sel(X_tr), y_tr), cfg["batch_size"],
                    shuffle=True,  num_workers=0, generator=g)
    va = DataLoader(BPDataset(sel(X_va), y_va), cfg["batch_size"],
                    shuffle=False, num_workers=0)
    te = DataLoader(BPDataset(sel(X_te), y_te), cfg["batch_size"],
                    shuffle=False, num_workers=0)
    return tr, va, te, len(ch_idx)


def cosine_lr(opt, epoch, total, warmup, base_lr):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        p  = (epoch - warmup) / (total - warmup)
        lr = base_lr * (1 + math.cos(math.pi * p)) / 2
    for pg in opt.param_groups: pg["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, tgts = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        tgts.append(y.numpy())
    preds = np.concatenate(preds); tgts = np.concatenate(tgts)
    return {
        "mae_sbp":  mean_absolute_error(tgts[:, 0], preds[:, 0]),
        "mae_dbp":  mean_absolute_error(tgts[:, 1], preds[:, 1]),
        "rmse_sbp": mean_squared_error(tgts[:, 0], preds[:, 0]) ** 0.5,
        "rmse_dbp": mean_squared_error(tgts[:, 1], preds[:, 1]) ** 0.5,
    }


@torch.no_grad()
def predict(model, loader, device):
    """Return raw (preds, targets) arrays of shape (N, 2) for diagnostics."""
    model.eval()
    preds, tgts = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        tgts.append(y.numpy())
    return np.concatenate(preds), np.concatenate(tgts)


def test_diagnostics(preds, tgts, use_wb=False):
    """Compute clinical BP diagnostics and (optionally) build W&B figures.

    Returns (metrics_dict, wandb_log_dict).  Metrics follow AAMI/BHS reporting:
      ME (mean error / bias), SD of error, and % of predictions within 5/10/15 mmHg.
    Figures: predicted-vs-true scatter and Bland-Altman, per target.
    """
    metrics, wb_logs = {}, {}
    for i, t in enumerate(["sbp", "dbp"]):
        p, g  = preds[:, i], tgts[:, i]
        err   = p - g
        me    = float(err.mean())
        sd    = float(err.std())
        mae   = float(np.abs(err).mean())
        r     = float(np.corrcoef(p, g)[0, 1]) if len(p) > 1 else 0.0
        w5    = float((np.abs(err) <= 5).mean()  * 100)
        w10   = float((np.abs(err) <= 10).mean() * 100)
        w15   = float((np.abs(err) <= 15).mean() * 100)
        metrics.update({
            f"test/{t}_me":        me,    # bias (AAMI wants |ME| <= 5)
            f"test/{t}_sd":        sd,    # precision (AAMI wants SD <= 8)
            f"test/{t}_mae":       mae,
            f"test/{t}_corr":      r,
            f"test/{t}_within5":   w5,    # BHS grade A: >=60%
            f"test/{t}_within10":  w10,   # BHS grade A: >=85%
            f"test/{t}_within15":  w15,   # BHS grade A: >=95%
        })

        if use_wb and MATPLOTLIB_AVAILABLE:
            # Predicted-vs-true scatter (calibration / range-compression check)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(g, p, s=6, alpha=0.3)
            lo, hi = float(min(g.min(), p.min())), float(max(g.max(), p.max()))
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_xlabel(f"True {t.upper()} (mmHg)")
            ax.set_ylabel(f"Predicted {t.upper()} (mmHg)")
            ax.set_title(f"{t.upper()}  MAE={mae:.2f}  r={r:.2f}")
            wb_logs[f"diag/{t}_scatter"] = wandb.Image(fig); plt.close(fig)

            # Bland-Altman (clinical agreement: bias +/- 1.96 SD)
            fig, ax = plt.subplots(figsize=(5, 5))
            mean_bp = (p + g) / 2
            ax.scatter(mean_bp, err, s=6, alpha=0.3)
            ax.axhline(me,             color="k",  lw=1)
            ax.axhline(me + 1.96 * sd, color="r", ls="--", lw=1)
            ax.axhline(me - 1.96 * sd, color="r", ls="--", lw=1)
            ax.set_xlabel(f"Mean {t.upper()} (mmHg)")
            ax.set_ylabel("Pred - True (mmHg)")
            ax.set_title(f"{t.upper()} Bland-Altman  bias={me:.2f}  LoA=+/-{1.96*sd:.1f}")
            wb_logs[f"diag/{t}_bland_altman"] = wandb.Image(fig); plt.close(fig)

    return metrics, wb_logs


def _save_checkpoint(name, weights, cfg, metrics, n_params, extra=None):
    """Save model weights (.pt) + companion JSON with all hyperparameters and results."""
    results_dir = cfg.get("results_dir", ".")
    os.makedirs(results_dir, exist_ok=True)
    torch.save(weights, os.path.join(results_dir, f"{name}_best.pt"))
    meta = {
        "name":        name,
        "n_params":    n_params,
        "hyperparams": {k: v for k, v in cfg.items() if k not in ("channel_map", "results_dir")},
        "metrics":     {k: float(v) for k, v in metrics.items()},
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(results_dir, f"{name}_best.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def train_deep(model, train_loader, val_loader, test_loader, cfg, run_name, device):
    seed_everything(cfg["seed"])
    opt      = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg.get("weight_decay", 1e-3))
    loss_fn  = nn.L1Loss()
    n_params = sum(p.numel() for p in model.parameters())
    use_wb   = WANDB_AVAILABLE and bool(os.environ.get("WANDB_API_KEY"))

    run = None
    if use_wb:
        wb_cfg = {k: v for k, v in cfg.items() if k != "channel_map"}
        run = wandb.init(project=cfg["wandb_project"], entity=cfg.get("wandb_entity"),
                         name=run_name,
                         config={**wb_cfg, "model": run_name, "n_params": n_params},
                         reinit=True)

    if TORCHINFO_AVAILABLE:
        # infer input shape from first batch of train_loader
        sample_x, _ = next(iter(train_loader))
        arch_summary = torch_summary(
            model, input_data=sample_x[:1].to(next(model.parameters()).device),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            verbose=0)
        print(arch_summary)
        if use_wb and run:
            wandb.log({"architecture": wandb.Html(
                f"<pre>{arch_summary}</pre>")}, commit=False)

    best_mae, best_w, best_epoch = float("inf"), None, 0
    no_improve   = 0
    patience     = cfg.get("early_stop_patience", 20)
    train_start  = time.time()

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        model.train()
        lr = cosine_lr(opt, epoch, cfg["epochs"], cfg["warmup_epochs"], cfg["lr"])
        total_loss, total_gnorm, n_batches = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); total_gnorm += gnorm.item(); n_batches += 1

        epoch_time = time.time() - epoch_start
        val_m      = evaluate(model, val_loader,  device)
        avg_loss   = total_loss  / n_batches
        avg_gnorm  = total_gnorm / n_batches
        val_mae    = (val_m["mae_sbp"] + val_m["mae_dbp"]) / 2

        if use_wb and run:
            wandb.log({
                "epoch":             epoch,
                # loss/* superimposed: train MAE vs val MAE on one chart
                "loss/train":        avg_loss,
                "loss/val_sbp":      val_m["mae_sbp"],
                "loss/val_dbp":      val_m["mae_dbp"],
                "loss/val_mean":     val_mae,
                "val/rmse_sbp":      val_m["rmse_sbp"],
                "val/rmse_dbp":      val_m["rmse_dbp"],
                "train/grad_norm":   avg_gnorm,
                "train/lr":          lr,
                "perf/epoch_time_s": epoch_time,
                "perf/elapsed_min":  (time.time() - train_start) / 60,
            })

        if val_mae < best_mae:
            best_mae = val_mae; best_epoch = epoch; no_improve = 0
            best_w   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(run_name, best_w, cfg, val_m, n_params,
                             extra={"best_epoch": best_epoch})
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"[{run_name}] Early stop at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            if use_wb and run:
                wandb.log({"early_stop_epoch": epoch + 1})
            break

        if (epoch + 1) % max(1, cfg["epochs"] // 10) == 0:
            elapsed = (time.time() - train_start) / 60
            print(f"[{run_name}] ep {epoch+1:3d}/{cfg['epochs']} | "
                  f"loss={avg_loss:.4f} | gnorm={avg_gnorm:.3f} | "
                  f"MAE SBP={val_m['mae_sbp']:.2f} DBP={val_m['mae_dbp']:.2f} mmHg | "
                  f"{epoch_time:.1f}s/ep | {elapsed:.1f}min elapsed")

    total_time = time.time() - train_start
    model.load_state_dict(best_w)
    val_final  = evaluate(model, val_loader,  device)
    test_p, test_g = predict(model, test_loader, device)
    test_final = {
        "mae_sbp":  mean_absolute_error(test_g[:, 0], test_p[:, 0]),
        "mae_dbp":  mean_absolute_error(test_g[:, 1], test_p[:, 1]),
        "rmse_sbp": mean_squared_error(test_g[:, 0], test_p[:, 0]) ** 0.5,
        "rmse_dbp": mean_squared_error(test_g[:, 1], test_p[:, 1]) ** 0.5,
    }
    diag_metrics, diag_figs = test_diagnostics(test_p, test_g, use_wb=bool(use_wb and run))
    _save_checkpoint(run_name, best_w, cfg,
                     {"val_" + k: v for k, v in val_final.items()} |
                     {"test_" + k: v for k, v in test_final.items()},
                     n_params,
                     extra={"best_epoch": best_epoch,
                            "total_train_time_s":  round(total_time, 1),
                            "total_train_time_min": round(total_time / 60, 2),
                            "diagnostics": diag_metrics})
    if use_wb and run:
        if diag_figs:
            wandb.log(diag_figs)
        # Test MAE bar chart (quick visual comparison of SBP vs DBP)
        mae_table = wandb.Table(columns=["target", "test_mae"],
                                data=[["SBP", test_final["mae_sbp"]],
                                      ["DBP", test_final["mae_dbp"]]])
        wandb.log({"test/mae_bar": wandb.plot.bar(mae_table, "target", "test_mae",
                                                  title="Test MAE (mmHg)")})
        run.summary.update({
            "best/val_mae_sbp":     val_final["mae_sbp"],
            "best/val_mae_dbp":     val_final["mae_dbp"],
            "best/test_mae_sbp":    test_final["mae_sbp"],
            "best/test_mae_dbp":    test_final["mae_dbp"],
            "best/test_rmse_sbp":   test_final["rmse_sbp"],
            "best/test_rmse_dbp":   test_final["rmse_dbp"],
            "best/epoch":           best_epoch,
            "n_params":             n_params,
            "total_train_time_s":   total_time,
            "total_train_time_min": total_time / 60,
            **diag_metrics,
        })
        run.finish()
    print(f"[{run_name}] Done — {total_time/60:.1f} min | best epoch {best_epoch+1} | "
          f"val MAE SBP={val_final['mae_sbp']:.2f}  DBP={val_final['mae_dbp']:.2f} | "
          f"test MAE SBP={test_final['mae_sbp']:.2f}  DBP={test_final['mae_dbp']:.2f} mmHg  "
          f"({n_params:,} params)")
    return test_final


# ── Model definitions ─────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).
    No learned parameters — generalizes to unseen sequence lengths."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)          # not a learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]          # (B, L, d_model)


class BPTransformer(nn.Module):
    def __init__(self, in_channels, d_model=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pe   = SinusoidalPE(d_model)
        self.enc  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4,
                                       dropout, batch_first=True, norm_first=True),
            num_layers=n_layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.enc(self.pe(self.proj(x))).transpose(1, 2))


class _ModalFusionBlock(nn.Module):
    """All-pairs bidirectional cross-attention for N modalities + per-modality FFN.
    Each modality queries all others (concatenated in seq dim), matching the
    BiDirectionalFusionBlock pattern from the prior cross-site project."""
    def __init__(self, n_mod, d_model, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_mod)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mod)])
        self.ffns  = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model),
                          nn.Linear(d_model, d_model * 4), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
            for _ in range(n_mod)])

    def forward(self, embs):
        n, out = len(embs), []
        for i in range(n):
            others   = torch.cat([embs[j] for j in range(n) if j != i], dim=1)
            attended, _ = self.cross_attn[i](embs[i], others, others)
            x = self.norms[i](embs[i] + attended)
            out.append(x + self.ffns[i](x))
        return out


class BPDualStreamTransformer(nn.Module):
    """PPG+ECG: per-modality sinusoidal PE → 3 cross-attention fusion blocks → shared encoder.
    Requires channels=['ppg','ecg']."""
    def __init__(self, d_model=128, n_heads=8, n_layers=2, n_fusion_blocks=3, dropout=0.1):
        super().__init__()
        n_mod = 2
        self.proj   = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_mod)])
        self.pos    = nn.ModuleList([SinusoidalPE(d_model) for _ in range(n_mod)])
        self.fusion = nn.ModuleList([_ModalFusionBlock(n_mod, d_model, n_heads, dropout)
                                     for _ in range(n_fusion_blocks)])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout,
                                       batch_first=True, norm_first=True),
            num_layers=n_layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):
        embs = [self.pos[c](self.proj[c](x[:, :, c:c+1])) for c in range(2)]
        for block in self.fusion:
            embs = block(embs)
        fused = torch.stack(embs, dim=0).mean(0)
        return self.head(self.encoder(fused).transpose(1, 2))


class BPTriStreamTransformer(nn.Module):
    """PPG+ECG+RESP: per-modality sinusoidal PE → 3 cross-attention fusion blocks → shared encoder.
    Requires channels=['ppg','ecg','resp']."""
    def __init__(self, d_model=128, n_heads=8, n_layers=2, n_fusion_blocks=3, dropout=0.1):
        super().__init__()
        n_mod = 3
        self.proj   = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_mod)])
        self.pos    = nn.ModuleList([SinusoidalPE(d_model) for _ in range(n_mod)])
        self.fusion = nn.ModuleList([_ModalFusionBlock(n_mod, d_model, n_heads, dropout)
                                     for _ in range(n_fusion_blocks)])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout,
                                       batch_first=True, norm_first=True),
            num_layers=n_layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):
        embs = [self.pos[c](self.proj[c](x[:, :, c:c+1])) for c in range(3)]
        for block in self.fusion:
            embs = block(embs)
        fused = torch.stack(embs, dim=0).mean(0)
        return self.head(self.encoder(fused).transpose(1, 2))


class _NoiseGenerator(nn.Module):
    """On-the-fly physiological noise augmentation (training only).
    Gaussian noise + baseline drift + motion artifact spikes + EMG — same
    corruption types as NoiseRobustTransformer in the prior cross-site project."""
    def __init__(self, fs: int = 25):
        super().__init__()
        self.fs = fs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, L, C = x.shape
        dev = x.device
        out = x + torch.randn_like(x) * 0.10                          # Gaussian
        t   = torch.linspace(0, L / self.fs, L, device=dev)
        out = out + 0.08 * torch.sin(2 * math.pi * 0.1 * t).view(1, L, 1)  # drift
        mask = torch.rand(B, L, C, device=dev) < 0.01                 # motion spikes
        out = out + torch.randn_like(x) * 0.5 * mask.float()
        out = out + torch.randn_like(x) * 0.03                        # EMG
        return out


class BPNoiseRobustTransformer(nn.Module):
    """Clean-vs-noisy dual stream with bidirectional cross-attention — adapted from
    NoiseRobustTransformer in the prior cross-site project.

    During training the noise generator corrupts a copy of the input; the
    _ModalFusionBlock ×3 forces the model to reconcile clean and noisy representations,
    acting as structured on-the-fly data augmentation.  At test time noisy = clean so
    the architecture degenerates to a single encoder path with no overhead.

    Works with any number of input channels.
    """
    def __init__(self, n_channels, d_model=128, n_heads=8, n_layers=2,
                 n_fusion_blocks=3, dropout=0.1):
        super().__init__()
        self.noise     = _NoiseGenerator()
        # Separate projections so clean/noisy can learn different representations
        self.clean_proj = nn.Linear(n_channels, d_model)
        self.noisy_proj = nn.Linear(n_channels, d_model)
        self.pe         = SinusoidalPE(d_model)
        # Bidirectional cross-attention: [clean, noisy] each attends to the other
        self.fusion     = nn.ModuleList([
            _ModalFusionBlock(2, d_model, n_heads, dropout)
            for _ in range(n_fusion_blocks)])
        # Adaptive gate: learn how much to trust clean vs noisy at each position
        self.gate       = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, 1), nn.Sigmoid())
        self.encoder    = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout,
                                       batch_first=True, norm_first=True),
            num_layers=n_layers)
        self.head       = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):
        noisy = self.noise(x)
        clean = self.pe(self.clean_proj(x))
        noisy = self.pe(self.noisy_proj(noisy))
        for block in self.fusion:
            [clean, noisy] = block([clean, noisy])
        w     = self.gate(torch.cat([clean, noisy], dim=-1))   # (B, L, 1)
        fused = w * clean + (1 - w) * noisy
        return self.head(self.encoder(fused).transpose(1, 2))


class S4DLayer(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        N = d_state // 2
        self.log_dt = nn.Parameter(torch.rand(d_model) * 2 - 4)
        n = torch.arange(N, dtype=torch.float32)
        self.A_real = nn.Parameter(-0.5 * torch.ones(d_model, N))
        self.A_imag = nn.Parameter(math.pi * n.unsqueeze(0).expand(d_model, N))
        self.B_re = nn.Parameter(torch.randn(d_model, N) * 0.5)
        self.B_im = nn.Parameter(torch.randn(d_model, N) * 0.5)
        self.C_re = nn.Parameter(torch.randn(d_model, N) * 0.5)
        self.C_im = nn.Parameter(torch.randn(d_model, N) * 0.5)
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)

    def _kernel(self, L):
        dt = torch.exp(self.log_dt)
        A  = -torch.exp(self.A_real) + 1j * self.A_imag
        B  = self.B_re + 1j * self.B_im; C = self.C_re + 1j * self.C_im
        A_bar = torch.exp(A * dt.unsqueeze(-1))
        B_bar = (A_bar - 1) / (A + 1e-8) * B
        l = torch.arange(L, device=dt.device, dtype=torch.float32)
        return 2 * (C.unsqueeze(-1) * A_bar.unsqueeze(-1)**l * B_bar.unsqueeze(-1)).real.sum(1)

    def forward(self, x):
        B_sz, L, H = x.shape; u = x.transpose(1, 2); n = 2 * L
        y = torch.fft.irfft(
            torch.fft.rfft(self._kernel(L), n=n).unsqueeze(0) * torch.fft.rfft(u, n=n), n=n
        )[:, :, :L]
        return self.drop(self.norm((y + self.D.unsqueeze(-1) * u).transpose(1, 2) + x))


class BPS4(nn.Module):
    def __init__(self, in_channels, d_model=128, d_state=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.proj   = nn.Linear(in_channels, d_model)
        self.layers = nn.ModuleList([S4DLayer(d_model, d_state, dropout) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers: x = layer(x)
        return self.head(self.norm(x).transpose(1, 2))


class BPS4CrossChannel(nn.Module):
    """Per-channel S4 encoder → full-sequence cross-channel fusion → temporal pool → head.

    Cross-attention runs on the FULL sequence (before any temporal pooling) so that
    inter-channel *timing* relationships survive — e.g. a PPG foot at t can attend to
    the ECG R-peak a few samples earlier, implicitly learning pulse transit time.
    Pooling before fusion (the naive design) would average that timing away.

    S4 handles within-channel dynamics; _ModalFusionBlock handles cross-channel
    relationships — the modality analogue of cross-site fusion in the prior project.
    S4 is inherently position-aware (its kernel is a causal convolution), so no
    explicit positional encoding is added before fusion.
    """
    def __init__(self, n_channels, d_model=256, d_state=128, n_layers=6,
                 n_heads=8, n_fusion_blocks=3, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        # Independent S4 stack per channel (each channel is its own site)
        self.projs     = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_channels)])
        self.s4_stacks = nn.ModuleList([
            nn.ModuleList([S4DLayer(d_model, d_state, dropout) for _ in range(n_layers)])
            for _ in range(n_channels)])
        self.ch_norms  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_channels)])

        # Full-sequence cross-channel fusion (all-pairs bidirectional cross-attention)
        self.fusion = nn.ModuleList([
            _ModalFusionBlock(n_channels, d_model, n_heads, dropout)
            for _ in range(n_fusion_blocks)])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):
        # Stage 1 — independent temporal encoding per channel, keep full sequence
        embs = []
        for c in range(self.n_channels):
            h = self.projs[c](x[:, :, c:c+1])
            for layer in self.s4_stacks[c]:
                h = layer(h)
            embs.append(self.ch_norms[c](h))               # (B, L, d_model)

        # Stage 2 — cross-channel fusion on full sequence (preserves timing for PTT)
        for block in self.fusion:
            embs = block(embs)

        fused = torch.stack(embs, dim=0).mean(0)            # (B, L, d_model)
        return self.head(fused.transpose(1, 2))             # head pools over time


# ── LightGBM ──────────────────────────────────────────────────────────────────

try:
    _TRAPZ = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _TRAPZ = np.trapz       # NumPy < 2.0

_FEATS_PER_CH = [
    # statistical
    "mean", "std", "skew", "kurt", "min", "max", "median", "rms",
    # spectral — total + HRV-relevant bands (VLF/LF/HF) + peak frequency
    "psd_total", "psd_vlf", "psd_lf", "psd_hf", "lf_hf_ratio", "peak_freq",
    # fractal complexity
    "higuchi_fd", "petrosian_fd",
    # peak morphology
    "peak_rate", "ipi_mean", "ipi_std", "ipi_rmssd", "rise_time",
    # amplitude
    "pulse_amp",          # peak-to-peak; direct hemodynamic indicator
]  # 22 features per channel
_FEATS_CROSS = ["ptt_mean", "ptt_std", "ptt_cv"]


def _higuchi_fd(x, kmax=5):
    """Higuchi fractal dimension via log-log regression of curve length vs interval k."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    Lk = []
    for k in range(1, kmax + 1):
        Lm_vals = []
        for m in range(1, k + 1):
            n_m = int((N - m) / k)
            if n_m < 1:
                continue
            idx = np.arange(m - 1, m - 1 + (n_m + 1) * k, k)[:n_m + 1]
            Lm = np.sum(np.abs(np.diff(x[idx]))) * (N - 1) / (k * n_m)
            Lm_vals.append(Lm)
        if Lm_vals:
            Lk.append(np.mean(Lm_vals))
    if len(Lk) < 2:
        return 1.0
    ks = np.arange(1, len(Lk) + 1, dtype=np.float64)
    coeffs = np.polyfit(np.log(ks), np.log(np.maximum(Lk, 1e-10)), 1)
    return float(-coeffs[0])


def _petrosian_fd(x):
    """Petrosian fractal dimension — O(N) approximation."""
    diffs = np.diff(x)
    n_delta = int(np.sum(diffs[:-1] * diffs[1:] < 0))
    N = len(x)
    return float(np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * n_delta + 1e-9))))


def _peak_features(s, fs):
    """Peak-based morphological features: rate, IPI stats, rise time."""
    min_dist = max(int(0.3 * fs), 1)
    peaks, _ = find_peaks(s, distance=min_dist, prominence=0.3)
    if len(peaks) < 2:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    ipi = np.diff(peaks) / fs
    peak_rate  = float(60.0 / np.mean(ipi))
    ipi_mean   = float(np.mean(ipi))
    ipi_std    = float(np.std(ipi))
    ipi_rmssd  = float(np.sqrt(np.mean(np.diff(ipi) ** 2))) if len(ipi) > 1 else 0.0
    rise_times = []
    for pk in peaks:
        lo   = max(0, pk - min_dist)
        foot = lo + int(np.argmin(s[lo:pk])) if pk > lo else pk
        rise_times.append((pk - foot) / fs)
    return [peak_rate, ipi_mean, ipi_std, ipi_rmssd, float(np.mean(rise_times))]


def _ptt_features(ecg, ppg, fs):
    """Pulse transit time: time from ECG R-peak to the following PPG foot."""
    min_dist = max(int(0.3 * fs), 1)
    r_peaks, _ = find_peaks(ecg, distance=min_dist, prominence=0.5)
    ptts = []
    for rp in r_peaks:
        lo = rp + max(int(0.05 * fs), 1)
        hi = min(rp + int(0.50 * fs), len(ppg))
        if lo >= hi:
            continue
        foot = lo + int(np.argmin(ppg[lo:hi]))
        ptt  = (foot - rp) / fs
        if 0.05 < ptt < 0.5:
            ptts.append(ptt)
    if len(ptts) < 2:
        return [0.0, 0.0, 0.0]
    mu = float(np.mean(ptts))
    sd = float(np.std(ptts))
    return [mu, sd, sd / mu if mu > 1e-9 else 0.0]


def _feature_names(channels):
    names = [f"{ch}_{f}" for ch in channels for f in _FEATS_PER_CH]
    if "ppg" in channels and "ecg" in channels:
        names += _FEATS_CROSS
    return names


def extract_features(X, fs=FS, channels=None):
    """(N, L, C) -> (N, n_features) float32.  channels list tells us which col is which."""
    N, L, C  = X.shape
    ch_list  = list(channels) if channels is not None else [f"ch{i}" for i in range(C)]
    has_ptt  = "ppg" in ch_list and "ecg" in ch_list
    ppg_ci   = ch_list.index("ppg") if has_ptt else None
    ecg_ci   = ch_list.index("ecg") if has_ptt else None
    rows = []
    for i in range(N):
        row = []
        for ci in range(C):
            s    = X[i, :, ci]
            f, p = welch(s, fs=fs, nperseg=min(256, L))
            # spectral bands aligned with HRV standard (Task Force 1996)
            m_vlf = (f >= 0.003) & (f < 0.04)
            m_lf  = (f >= 0.04)  & (f < 0.15)
            m_hf  = (f >= 0.15)  & (f < 0.4)
            psd_lf  = float(_TRAPZ(p[m_lf], f[m_lf])) if m_lf.any() else 0.0
            psd_hf  = float(_TRAPZ(p[m_hf], f[m_hf])) if m_hf.any() else 0.0
            row += [
                float(s.mean()), float(s.std()),
                float(stats.skew(s)), float(stats.kurtosis(s)),
                float(s.min()), float(s.max()),
                float(np.median(s)), float(np.sqrt(np.mean(s ** 2))),
                float(_TRAPZ(p, f)),
                float(_TRAPZ(p[m_vlf], f[m_vlf])) if m_vlf.any() else 0.0,
                psd_lf, psd_hf,
                psd_lf / (psd_hf + 1e-10),   # LF/HF ratio
                float(f[np.argmax(p)]),
            ]
            row += [_higuchi_fd(s), _petrosian_fd(s)]
            row += _peak_features(s, fs)
            row += [float(s.max() - s.min())]  # pulse_amp
        if has_ptt:
            row += _ptt_features(X[i, :, ecg_ci], X[i, :, ppg_ci], fs)
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def get_features(X, channels, cfg, split_name):
    """Extract or load cached feature matrix. Cache: features/<split>_<ch>_seq<L>_v2.npy"""
    ch_idx    = [CHANNEL_MAP[c] for c in channels]
    ch_tag    = "_".join(channels)
    cache_dir = os.path.join(cfg["data_root"], "features")
    os.makedirs(cache_dir, exist_ok=True)
    ds_tag     = f"_ds{cfg.get('downsample', 1)}" if cfg.get("downsample", 1) > 1 else ""
    cache_path = os.path.join(cache_dir,
                              f"{split_name}_{ch_tag}_seq{cfg['seq_len']}{ds_tag}_v3.npy")
    if os.path.exists(cache_path):
        F = np.load(cache_path)
        print(f"  Loaded cached features: {os.path.basename(cache_path)}  {F.shape}")
    else:
        print(f"  Computing features [{channels}] for {split_name} ...")
        F = extract_features(X[:, :, ch_idx], cfg["fs"], channels=channels)
        np.save(cache_path, F)
        print(f"  Saved -> {cache_path}  {F.shape}")
    return F


def _fmt_tree(node, fnames, lines, depth=0):
    """Recursively format a LightGBM tree node into a list of lines."""
    pad = "  " * depth
    if "leaf_value" in node:
        lines.append(f"{pad}-> {node['leaf_value']:.3f} mmHg  (n={node.get('leaf_count','?')})")
    else:
        feat   = fnames[node["split_feature"]] if fnames else f"f{node['split_feature']}"
        thresh = node["threshold"]
        lines.append(f"{pad}if {feat} <= {thresh:.4f}:")
        _fmt_tree(node["left_child"],  fnames, lines, depth + 1)
        lines.append(f"{pad}else:  # {feat} > {thresh:.4f}")
        _fmt_tree(node["right_child"], fnames, lines, depth + 1)


def train_lgbm(F_tr, y_tr, F_va, y_va, F_te, y_te, cfg, run_name, channels=None):
    use_wb     = WANDB_AVAILABLE and bool(os.environ.get("WANDB_API_KEY"))
    results    = {}
    fnames     = _feature_names(channels) if channels else [f"f{i}" for i in range(F_tr.shape[1])]
    smoke      = cfg.get("smoke", False)

    for target_idx, target_name in enumerate(["sbp", "dbp"]):
        rname = f"{run_name}_{target_name}"
        run   = None
        if use_wb:
            wb_cfg = {k: v for k, v in cfg.items() if k != "channel_map"}
            run = wandb.init(project=cfg["wandb_project"], entity=cfg.get("wandb_entity"),
                             name=rname,
                             config={**wb_cfg, "model": run_name, "target": target_name},
                             reinit=True)

        # ── Optuna hyperparameter search on val set ───────────────────────────
        if OPTUNA_AVAILABLE and not smoke:
            print(f"  Optuna search [{target_name.upper()}] (50 trials) ...")
            def _objective(trial):
                p = dict(
                    n_estimators      = 2000,
                    learning_rate     = trial.suggest_float("learning_rate",    0.01, 0.2,  log=True),
                    num_leaves        = trial.suggest_int(  "num_leaves",       15,   127),
                    max_depth         = trial.suggest_int(  "max_depth",        3,    8),
                    min_child_samples = trial.suggest_int(  "min_child_samples",5,    50),
                    subsample         = trial.suggest_float("subsample",        0.6,  1.0),
                    colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5,  1.0),
                    reg_alpha         = trial.suggest_float("reg_alpha",        1e-8, 1.0, log=True),
                    reg_lambda        = trial.suggest_float("reg_lambda",       1e-8, 1.0, log=True),
                    random_state=cfg["seed"], verbose=-1)
                tm = lgb.LGBMRegressor(**p)
                tm.fit(F_tr, y_tr[:, target_idx],
                       eval_set=[(F_va, y_va[:, target_idx])],
                       callbacks=[lgb.early_stopping(30, verbose=False)])
                return mean_absolute_error(y_va[:, target_idx], tm.predict(F_va))
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
            study.optimize(_objective, n_trials=50, show_progress_bar=False)
            best_hp = study.best_params
            print(f"  Best val MAE={study.best_value:.3f}  params={best_hp}")
        else:
            best_hp = dict(learning_rate=cfg["lgbm_lr"], num_leaves=63, max_depth=7,
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.0, reg_lambda=0.0)
            study   = None

        # ── Final model with best hyperparams ────────────────────────────────
        evals_result = {}
        t0 = time.time()
        m  = lgb.LGBMRegressor(
            n_estimators=cfg["lgbm_n_estimators"],
            random_state=cfg["seed"], verbose=-1, **best_hp)
        m.fit(F_tr, y_tr[:, target_idx],
              eval_set=[(F_tr, y_tr[:, target_idx]), (F_va, y_va[:, target_idx])],
              eval_names=["train", "val"],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(100),
                         lgb.record_evaluation(evals_result)])
        train_time = time.time() - t0
        preds_te   = m.predict(F_te)
        mae        = mean_absolute_error(y_te[:, target_idx], preds_te)
        rmse       = mean_squared_error( y_te[:, target_idx], preds_te) ** 0.5
        best_iter  = m.best_iteration_ or cfg["lgbm_n_estimators"]

        imp_sorted = sorted(zip(fnames, m.feature_importances_.tolist()),
                            key=lambda x: x[1], reverse=True)

        # ── Single-estimator decision tree ───────────────────────────────────
        tree_m = lgb.LGBMRegressor(n_estimators=1, max_depth=4,
                                    random_state=cfg["seed"], verbose=-1)
        tree_m.fit(F_tr, y_tr[:, target_idx])
        tree_mae  = mean_absolute_error(y_te[:, target_idx], tree_m.predict(F_te))
        tree_node = tree_m.booster_.dump_model()["tree_info"][0]["tree_structure"]
        tree_lines: list = []
        _fmt_tree(tree_node, fnames, tree_lines)
        tree_text = "\n".join(tree_lines)
        print(f"\n  Decision tree [{target_name.upper()}]  (single-tree test MAE={tree_mae:.2f}):")
        print(tree_text, "\n")

        # ── Save JSON ─────────────────────────────────────────────────────────
        results_dir = cfg.get("results_dir", ".")
        os.makedirs(results_dir, exist_ok=True)
        meta = {
            "name": rname, "target": target_name,
            "arch_version": cfg.get("arch_version", "v2"),
            "hyperparams": {**best_hp, "n_estimators": cfg["lgbm_n_estimators"],
                            "best_iteration": best_iter},
            "data": {"channels": channels or [], "n_features": len(fnames),
                     "n_train": len(F_tr), "n_val": len(F_va), "n_test": len(F_te),
                     "seq_len": cfg.get("seq_len"), "downsample": cfg.get("downsample", 1)},
            "metrics": {"mae": float(mae), "rmse": float(rmse)},
            "train_time_s": round(train_time, 1),
            "top20_features": [{"rank": i+1, "feature": f, "importance": int(v)}
                               for i, (f, v) in enumerate(imp_sorted[:20])],
        }
        with open(os.path.join(results_dir, f"{rname}_best.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        # ── W&B logging ───────────────────────────────────────────────────────
        if use_wb and run:
            # Training curves
            tr_l1 = evals_result.get("train", {}).get("l1", [])
            va_l1 = evals_result.get("val",   {}).get("l1", [])
            for rnd, (tr, va) in enumerate(zip(tr_l1, va_l1)):
                wandb.log({"lgbm/round": rnd, "lgbm/train_mae": tr,
                           "lgbm/val_mae": va}, step=rnd)

            # Feature importance — ordered by rank, so chart is sorted
            imp_table = wandb.Table(
                columns=["rank", "feature", "importance"],
                data=[[i+1, f, v] for i, (f, v) in enumerate(imp_sorted[:20])])
            wandb.log({"lgbm/feature_importance":
                       wandb.plot.bar(imp_table, "feature", "importance",
                                      title=f"Top-20 Features ({target_name.upper()})")})

            # Decision tree as HTML (readable in W&B UI)
            tree_html = (f"<h3>Single-tree {target_name.upper()} "
                         f"— test MAE {tree_mae:.2f} mmHg</h3>"
                         f"<pre style='font-size:13px'>{tree_text}</pre>")
            wandb.log({"lgbm/decision_tree": wandb.Html(tree_html)})

            # Optuna trial table
            if study is not None:
                opt_cols = ["trial", "val_mae"] + list(best_hp.keys())
                opt_data = [[t.number, t.value] + [t.params.get(k) for k in best_hp]
                            for t in study.trials if t.value is not None]
                wandb.log({"lgbm/optuna_trials": wandb.Table(columns=opt_cols, data=opt_data)})

            # SHAP waterfall
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(m)
                    shap_vals = explainer.shap_values(F_te[:200])
                    shap_exp  = shap.Explanation(
                        values=shap_vals[0], base_values=float(explainer.expected_value),
                        data=F_te[0], feature_names=fnames)
                    fig, _ = plt.subplots(figsize=(10, 7))
                    shap.plots.waterfall(shap_exp, show=False)
                    wandb.log({"lgbm/shap_waterfall": wandb.Image(fig)})
                    plt.close(fig)
                except Exception as e:
                    print(f"  SHAP skipped: {e}")

            # Consistent naming with deep models: best/test_mae_sbp / best/test_mae_dbp
            run.summary.update({
                f"best/test_mae_{target_name}":  mae,
                f"best/test_rmse_{target_name}": rmse,
                "best_iteration": best_iter,
                "train_time_s":   train_time,
            })
            run.finish()

        print(f"  LightGBM {target_name.upper()}: MAE={mae:.2f}  RMSE={rmse:.2f} mmHg "
              f"({best_iter} trees, {train_time:.1f}s)")
        results[f"mae_{target_name}"]  = mae
        results[f"rmse_{target_name}"] = rmse

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BP estimation experiment runner")
    parser.add_argument("--model",    required=True,
                        choices=["transformer", "dual_stream", "tri_stream",
                                 "noise_robust", "s4", "s4_cross", "lgbm"])
    parser.add_argument("--channels", required=True,
                        help="Comma-separated channels, e.g. ppg,ecg,resp")
    parser.add_argument("--gpu_profile", default="3080",
                        choices=list(_PROFILES.keys()),
                        help="Controls window length + model size (fast / 3080 / h100)")
    parser.add_argument("--smoke",      action="store_true",
                        help="3 epochs / small data subset for sanity checks")
    parser.add_argument("--epochs",              type=int,   default=None)
    parser.add_argument("--batch_size",          type=int,   default=None)
    parser.add_argument("--d_model",             type=int,   default=None)
    parser.add_argument("--n_heads",             type=int,   default=None)
    parser.add_argument("--n_layers",            type=int,   default=None)
    parser.add_argument("--d_state",             type=int,   default=64,
                        help="S4 state dimension (default 64; try 128/256 to scale up)")
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--weight_decay",        type=float, default=1e-3)
    parser.add_argument("--early_stop_patience", type=int,   default=20,
                        help="Stop if val MAE doesn't improve for N epochs (0=disabled)")
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--downsample",   type=int,   default=1, metavar="F",
                        help="Downsample factor applied after loading (e.g. 5 for 125→25 Hz)")
    parser.add_argument("--wandb_project", default="bp-estimation")
    parser.add_argument("--data_root",    default=_ROOT)
    args = parser.parse_args()

    channels = [c.strip() for c in args.channels.split(",")]
    if args.model == "dual_stream" and channels != ["ppg", "ecg"]:
        parser.error("--model dual_stream requires exactly --channels ppg,ecg")
    if args.model == "tri_stream" and channels != ["ppg", "ecg", "resp"]:
        parser.error("--model tri_stream requires exactly --channels ppg,ecg,resp")

    seq_len, seg_start, d_model, n_heads, n_layers, batch_size = _PROFILES[args.gpu_profile]
    epochs     = args.epochs     or (3  if args.smoke else 100)
    batch_size = args.batch_size or (32 if args.smoke else batch_size)
    d_model    = args.d_model    or d_model
    n_heads    = args.n_heads    or n_heads
    n_layers   = args.n_layers   or n_layers

    cfg = {
        "seed":              args.seed,
        "data_root":         args.data_root,
        "channel_map":       CHANNEL_MAP,
        "fs":                FS,
        "seq_len":           seq_len,
        "gpu_profile":       args.gpu_profile,
        "batch_size":        batch_size,
        "d_model":           d_model,
        "n_heads":           n_heads,
        "n_layers":          n_layers,
        "d_state":              args.d_state,
        "dropout":              0.1,
        "lr":                   args.lr,
        "weight_decay":         args.weight_decay,
        "early_stop_patience":  args.early_stop_patience,
        "epochs":               epochs,
        "warmup_epochs":        max(1, epochs // 10),
        "lgbm_n_estimators":    50  if args.smoke else 2000,
        "lgbm_lr":              0.1 if args.smoke else 0.03,
        "downsample":           args.downsample,
        "arch_version":         ARCH_VERSION,
        "wandb_project":     args.wandb_project,
        "wandb_entity":      None,
    }

    cfg["results_dir"] = os.path.join(args.data_root, "results")

    seed_everything(cfg["seed"])
    run_name = f"{args.model}_{'_'.join(channels)}"
    mode_tag = " [SMOKE]" if args.smoke else ""
    print(f"\n{'='*65}")
    print(f"  Experiment: {run_name}{mode_tag}")
    print(f"  Profile: {args.gpu_profile}  |  window: {seg_start/FS:.0f}-"
          f"{(seg_start+seq_len)/FS:.0f}s ({seq_len} samples)")
    print(f"  d_model={d_model}  n_layers={n_layers}  batch={batch_size}  epochs={epochs}")
    print(f"{'='*65}")

    root      = args.data_root
    train_sub = read_subject_list(os.path.join(root, "train_subjects.txt"))
    val_sub   = read_subject_list(os.path.join(root, "val_subjects.txt"))
    test_sub  = read_subject_list(os.path.join(root, "test_subjects.txt"))

    if args.smoke:
        train_sub = train_sub[:4]; val_sub = val_sub[:1]; test_sub = test_sub[:2]

    print("Loading data...")
    train_sig, train_lbl = extract_segments(train_sub, root, seq_len, seg_start)
    val_sig,   val_lbl   = extract_segments(val_sub,   root, seq_len, seg_start)
    test_sig,  test_lbl  = extract_segments(test_sub,  root, seq_len, seg_start)

    X_train, y_train = train_sig, train_lbl
    X_val,   y_val   = val_sig,   val_lbl
    X_test,  y_test  = test_sig,  test_lbl

    if args.downsample > 1:
        from scipy.signal import decimate as sp_decimate
        f = args.downsample
        X_train = sp_decimate(X_train.astype(np.float64), q=f, axis=1, zero_phase=True).astype(np.float32)
        X_val   = sp_decimate(X_val.astype(  np.float64), q=f, axis=1, zero_phase=True).astype(np.float32)
        X_test  = sp_decimate(X_test.astype( np.float64), q=f, axis=1, zero_phase=True).astype(np.float32)
        cfg["seq_len"] = X_train.shape[1]   # use actual post-decimate length
        cfg["fs"]      = FS // f
        print(f"Downsampled {f}x -> {cfg['fs']} Hz  seq_len={cfg['seq_len']}")

    print(f"X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")

    if args.model == "lgbm":
        F_train = get_features(X_train, channels, cfg, "train")
        F_val   = get_features(X_val,   channels, cfg, "val")
        F_test  = get_features(X_test,  channels, cfg, "test")
        metrics = train_lgbm(F_train, y_train, F_val, y_val, F_test, y_test,
                             cfg, run_name, channels=channels)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        train_loader, val_loader, test_loader, n_ch = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test, channels, cfg)

        if args.model == "transformer":
            model = BPTransformer(n_ch, d_model, n_heads, n_layers, 0.1).to(device)
        elif args.model == "dual_stream":
            model = BPDualStreamTransformer(d_model, n_heads, n_layers, dropout=0.1).to(device)
        elif args.model == "tri_stream":
            model = BPTriStreamTransformer(d_model, n_heads, n_layers, dropout=0.1).to(device)
        elif args.model == "noise_robust":
            model = BPNoiseRobustTransformer(n_ch, d_model, n_heads, n_layers,
                                             dropout=0.1).to(device)
        elif args.model == "s4":
            model = BPS4(n_ch, d_model, cfg["d_state"], n_layers, 0.1).to(device)
        elif args.model == "s4_cross":
            model = BPS4CrossChannel(n_ch, d_model, cfg["d_state"], n_layers,
                                     n_heads, 0.1).to(device)

        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        metrics = train_deep(model, train_loader, val_loader, test_loader, cfg, run_name, device)

    print(f"\nResults [{run_name}]:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f} mmHg")


if __name__ == "__main__":
    main()
