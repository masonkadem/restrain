"""Two architectures, one audit: which model actually encodes PTT?

The platform claim is discrimination -- given a submitted model, can we certify
whether it encodes the physics it must? This shows the audit separating two
models on a *principled* basis, not an arbitrary one.

PTT is intrinsically a CROSS-stream quantity: the lag between the proximal and
distal waveforms. So the physics predicts:

  - CrossAttentionBP relates the two streams position-by-position -> it can
    recover the lag -> should encode PTT (audit passes, and it solves the task).
  - SelfAttentionPoolBP self-attends within each stream, then mean-pools and
    concatenates. Pooling a shifted signal is shift-invariant, and the streams
    are never related position-wise -> it structurally cannot recover the lag
    -> should NOT encode PTT (audit fails, and it cannot solve the task).

The audit (linear PTT decode from the waveform representation + consistency
against the signal-measured PTT) confirms exactly this.

Honesty note: here the ungrounded model also has poor accuracy, so accuracy
alone would flag it too. The audit's unique value is the shortcut-learner case
(good accuracy without the physics), which needs a planted spurious cue -- a
natural follow-up. This module makes the principled can/can't point cleanly.

    python analysis/ptt_audit_models.py
    python analysis/ptt_audit_models.py --quick
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
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ptt_gate import WIN, _split, evaluate, generate, train_model  # noqa: E402


def _positional_encoding(n, dim):
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq[: pe[:, 1::2].shape[1]])
    return pe


class CrossAttentionBP(nn.Module):
    """Proximal attends to distal -> can align positions -> recovers the lag."""

    def __init__(self, cal_dim=3, width=48, pos_dim=16):
        super().__init__()
        self.register_buffer("pe", _positional_encoding(WIN, pos_dim))
        self.q = nn.Sequential(nn.Linear(1 + pos_dim, width), nn.ReLU(), nn.Linear(width, width))
        self.k = nn.Sequential(nn.Linear(1 + pos_dim, width), nn.ReLU(), nn.Linear(width, width))
        self.v = nn.Sequential(nn.Linear(1 + pos_dim, width), nn.ReLU(), nn.Linear(width, width))
        self.width = width
        self.mix = nn.Sequential(nn.Linear(width + cal_dim, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU())
        self.head = nn.Linear(96, 1)

    def _tok(self, s):
        pe = self.pe.unsqueeze(0).expand(s.shape[0], -1, -1)
        return torch.cat([s.unsqueeze(-1), pe], dim=-1)

    def forward(self, prox, dist, cal, return_features=False):
        q = self.q(self._tok(prox)); k = self.k(self._tok(dist)); v = self.v(self._tok(dist))
        attn = (torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.width)).softmax(-1)
        wave_feat = torch.bmm(attn, v).mean(dim=1)          # cross-stream aligned context
        out = self.head(self.mix(torch.cat([wave_feat, cal], dim=1))).squeeze(-1)
        if return_features:
            return out, wave_feat
        return out


class SelfAttentionPoolBP(nn.Module):
    """Each stream self-attends independently, then mean-pool + concat.

    No positional encoding and no cross-stream interaction -> permutation- and
    shift-invariant per stream -> the inter-stream lag (PTT) is unrecoverable.
    """

    def __init__(self, cal_dim=3, width=48):
        super().__init__()
        self.embed = nn.Linear(1, width)
        self.qa = nn.Linear(width, width); self.ka = nn.Linear(width, width); self.va = nn.Linear(width, width)
        self.qb = nn.Linear(width, width); self.kb = nn.Linear(width, width); self.vb = nn.Linear(width, width)
        self.width = width
        self.mix = nn.Sequential(nn.Linear(2 * width + cal_dim, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU())
        self.head = nn.Linear(96, 1)

    def _self(self, s, q, k, v):
        e = self.embed(s.unsqueeze(-1))                     # no positional encoding
        attn = (torch.bmm(q(e), k(e).transpose(1, 2)) / math.sqrt(self.width)).softmax(-1)
        return torch.bmm(attn, v(e)).mean(dim=1)            # pooled -> shift-invariant

    def forward(self, prox, dist, cal, return_features=False):
        ra = self._self(prox, self.qa, self.ka, self.va)
        rb = self._self(dist, self.qb, self.kb, self.vb)
        wave_feat = torch.cat([ra, rb], dim=1)              # two independent summaries
        out = self.head(self.mix(torch.cat([wave_feat, cal], dim=1))).squeeze(-1)
        if return_features:
            return out, wave_feat
        return out


MODELS = {"cross_attention": CrossAttentionBP, "self_attention_pool": SelfAttentionPoolBP}


def audit_model(factory, cfg, seed: int) -> dict:
    train = generate(cfg.n_train, "clean", seed * 100 + 1)
    ym, ys = float(train["y"].mean()), float(train["y"].std() + 1e-8)
    model = factory()
    train_model(model, train, ym, ys, cfg.epochs, seed)

    probe_data = generate(cfg.n_probe, "clean", seed * 100 + 2)
    ev = evaluate(model, probe_data, ym, ys)
    scaler = StandardScaler().fit(ev["feat"])
    ptt_probe = Ridge(alpha=1.0).fit(scaler.transform(ev["feat"]), probe_data["ptt_true"])

    test = generate(cfg.n_test, "clean", seed * 100 + 3)
    te = evaluate(model, test, ym, ys)
    ptt_dec = ptt_probe.predict(scaler.transform(te["feat"]))
    return {
        "ptt_r2": float(r2_score(test["ptt_true"], ptt_dec)),
        "bp_mae": float(np.abs(te["pred"] - test["y"]).mean()),
        "ptt_measured": test["ptt_measured"], "ptt_decoded": ptt_dec,
    }


def aggregate(cfg) -> dict:
    out = {}
    for name, factory in MODELS.items():
        per = [audit_model(factory, cfg, s) for s in cfg.seeds]
        out[name] = {
            "ptt_r2_mean": float(np.mean([p["ptt_r2"] for p in per])),
            "ptt_r2_std": float(np.std([p["ptt_r2"] for p in per])),
            "bp_mae_mean": float(np.mean([p["bp_mae"] for p in per])),
            "bp_mae_std": float(np.std([p["bp_mae"] for p in per])),
            "points": {"ptt_measured": per[0]["ptt_measured"], "ptt_decoded": per[0]["ptt_decoded"]},
        }
    return out


# ── plotting ──────────────────────────────────────────────────────────────────
BLUE, RED, MUTE = "#2a78d6", "#d03b3b", "#9aa09a"
GOOD, BAD = "#0f8a3c", "#c1362f"
INK, GRID, SURFACE = "#16191a", "#e1e0d9", "#fcfcfb"
MODEL_LABEL = {"cross_attention": "cross-attention\n(relates the two streams)",
               "self_attention_pool": "self-attention + pool\n(per-stream, no cross)"}
MODEL_COLOR = {"cross_attention": BLUE, "self_attention_pool": RED}


def _style(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#7c8384", labelsize=8)
    ax.grid(True, axis="y" if ygrid else "both", color=GRID, linewidth=0.6)


def plot(agg: dict, output_dir: Path) -> None:
    names = list(MODELS)
    fig = plt.figure(figsize=(12.5, 5), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.28)
    fig.text(0.02, 0.95, "One audit, two models: does the model encode PTT?",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.905,
             "PTT is a cross-stream lag. The audit certifies the architecture that can represent it and "
             "rejects the one that structurally cannot.",
             fontsize=9.5, color="#454b4c")

    # Left: the audit metric (PTT R2) with BP MAE annotated
    ax = fig.add_subplot(gs[0, 0])
    r2 = [agg[n]["ptt_r2_mean"] for n in names]
    r2e = [agg[n]["ptt_r2_std"] for n in names]
    bars = ax.bar([0, 1], r2, 0.55, yerr=r2e, capsize=4,
                  color=[MODEL_COLOR[n] for n in names])
    ax.axhline(0.5, ls="--", color=MUTE, lw=1.2)
    ax.text(1.4, 0.5, " audit\n threshold", va="center", fontsize=7.5, color="#7c8384")
    for i, n in enumerate(names):
        verdict = "GROUNDED" if agg[n]["ptt_r2_mean"] > 0.5 else "NOT GROUNDED"
        col = GOOD if agg[n]["ptt_r2_mean"] > 0.5 else BAD
        ax.text(i, max(agg[n]["ptt_r2_mean"], 0.04) + 0.05, verdict, ha="center",
                fontsize=9, fontweight="bold", color=col)
        ax.text(i, -0.14, f"BP MAE {agg[n]['bp_mae_mean']:.1f}", ha="center", fontsize=8.5, color="#454b4c")
    ax.set_xticks([0, 1]); ax.set_xticklabels([MODEL_LABEL[n] for n in names], fontsize=8.5)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("PTT decodable from activations (R²)", fontsize=9)
    ax.set_title("The audit metric", fontsize=10, color=INK, loc="left")
    _style(ax)

    # Right: decoded vs measured scatter for both models
    ax = fig.add_subplot(gs[0, 1])
    lim = (0, 45)
    ax.plot(lim, lim, "--", color=MUTE, lw=1.3, zorder=1)
    for n in names:
        p = agg[n]["points"]
        m = np.clip(p["ptt_measured"], *lim); d = np.clip(p["ptt_decoded"], *lim)
        ax.scatter(m, d, s=12, color=MODEL_COLOR[n], alpha=0.45, edgecolors="none",
                   label=MODEL_LABEL[n].replace("\n", " "), zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("PTT measured from raw signal (samples)", fontsize=9)
    ax.set_ylabel("PTT decoded from activations (samples)", fontsize=9)
    ax.set_title("On the diagonal = the model really tracks PTT", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, ygrid=False)

    fig.subplots_adjust(top=0.83, bottom=0.17, left=0.08, right=0.98)
    out = output_dir / "ptt_audit_models.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"[audit-models] wrote {out}", flush=True)


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
    for n in MODELS:
        print(f"  {n:20s} PTT R2={agg[n]['ptt_r2_mean']:.3f}  BP MAE={agg[n]['bp_mae_mean']:.2f}")
    with open(a.output_dir / "ptt_audit_models.json", "w", encoding="utf-8") as fh:
        json.dump({n: {k: v for k, v in agg[n].items() if k != "points"} for n in MODELS}, fh, indent=2)
    print(f"[audit-models] wrote {a.output_dir / 'ptt_audit_models.json'}")


if __name__ == "__main__":
    main()
