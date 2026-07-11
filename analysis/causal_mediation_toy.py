"""Toy demonstration: certifying that a model *causally uses* a governing law.

The governing equation is a two-component saturation, analogous to deriving
SpO2 from a ratio of two absorbances (Beer-Lambert):

    y = v_i / (v_i + v_j)          (both components strictly positive)

A small cross-attention retriever (the CrossSA / q-sparse-averaging setup from
``representational_capacity.ipynb``) reads a source query that names two slots
``i`` and ``j``, retrieves their values, and predicts ``y``.

We train two models with the *same architecture* on the *same task*:

* ``law``      -- trained on data where component ``j`` varies, so it must use
                  both components.
* ``shortcut`` -- trained on data where component ``j`` is (nearly) constant, so
                  reading component ``i`` alone already predicts ``y``.  This is
                  the everyday failure of a confounded / narrow training set.

The point of the demo: on the validation distribution a practitioner actually
has (``j`` barely varies), the two models are indistinguishable ---

    * equal predictive error, and
    * a linear probe decodes the sufficient statistic equally well from both.

Yet only one of them will generalize.  Three *post-hoc, label-free, frozen-model*
audits derived from the governing equation separate them before any OOD label is
seen:

    1. counterfactual sensitivity to component ``j`` (the equation says
       dy/dv_j != 0; the shortcut model is insensitive);
    2. interchange-intervention accuracy (DAS): is there a subspace of the
       internal state that, when swapped between two inputs, transfers ``y`` as
       the equation predicts?  A model that merely *encodes* the ratio fails
       this; a model that *causally routes* through it passes;
    3. we then reveal held-out error where ``j`` varies, confirming the audit's
       verdict without having used those labels to reach it.

Run:

    python analysis/causal_mediation_toy.py
    python analysis/causal_mediation_toy.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def positional_encoding(n_slots: int, dim: int) -> np.ndarray:
    positions = np.arange(n_slots, dtype=np.float32)[:, None]
    freqs = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-math.log(10_000.0) / dim))
    enc = np.zeros((n_slots, dim), dtype=np.float32)
    enc[:, 0::2] = np.sin(positions * freqs)
    enc[:, 1::2] = np.cos(positions * freqs[: enc[:, 1::2].shape[1]])
    return enc / np.maximum(np.linalg.norm(enc, axis=1, keepdims=True), 1e-8)


@dataclass
class Batch:
    source: torch.Tensor       # (N, 2 * pos_dim)   query naming slots i, j
    targets: torch.Tensor      # (N, n_slots, 1 + pos_dim)
    value_j_index: torch.Tensor  # (N,) slot index of component j (for sensitivity)
    v_i: torch.Tensor          # (N,)
    v_j: torch.Tensor          # (N,)
    y: torch.Tensor            # (N,)


def generate(
    n: int,
    n_slots: int = 8,
    pos_dim: int = 8,
    vary_j: bool = True,
    j_const: float = 1.0,
    seed: int = 0,
) -> Batch:
    """One (i, j) query per example.  ``vary_j`` toggles law vs shortcut regime."""
    rng = np.random.default_rng(seed)
    pos = positional_encoding(n_slots, pos_dim)

    values = rng.uniform(0.25, 2.0, size=(n, n_slots)).astype(np.float32)
    idx = np.stack([rng.permutation(n_slots)[:2] for _ in range(n)]).astype(np.int64)
    i_idx, j_idx = idx[:, 0], idx[:, 1]

    if not vary_j:
        # Component j is pinned near a constant: reading v_i alone determines y.
        jitter = rng.normal(scale=0.02, size=n).astype(np.float32)
        values[np.arange(n), j_idx] = np.clip(j_const + jitter, 0.25, 2.0)

    v_i = values[np.arange(n), i_idx]
    v_j = values[np.arange(n), j_idx]
    y = v_i / (v_i + v_j)

    source = np.concatenate([pos[i_idx], pos[j_idx]], axis=-1).astype(np.float32)
    tokens = np.concatenate(
        [values[..., None], np.broadcast_to(pos, (n, n_slots, pos_dim))], axis=-1
    ).astype(np.float32)

    return Batch(
        source=torch.from_numpy(source),
        targets=torch.from_numpy(tokens),
        value_j_index=torch.from_numpy(j_idx),
        v_i=torch.from_numpy(v_i),
        v_j=torch.from_numpy(v_j),
        y=torch.from_numpy(y),
    )


class Retriever(nn.Module):
    """Two-query cross-attention + scalar head.

    The source names slots ``i`` and ``j``; a separate query for each retrieves
    its value into its own subspace, so the head *can* form the ratio.  The
    audited ``context`` is the concatenation of the two retrieved states (width
    ``2 * width``): component ``i`` lives in the first half, component ``j`` in
    the second.
    """

    def __init__(self, source_dim: int, target_dim: int, width: int = 32):
        super().__init__()
        half = source_dim // 2
        self.query_i = nn.Sequential(nn.Linear(half, 64), nn.ReLU(), nn.Linear(64, width))
        self.query_j = nn.Sequential(nn.Linear(half, 64), nn.ReLU(), nn.Linear(64, width))
        self.key = nn.Sequential(nn.Linear(target_dim, 64), nn.ReLU(), nn.Linear(64, width))
        self.value = nn.Sequential(nn.Linear(target_dim, 64), nn.ReLU(), nn.Linear(64, width))
        self.head = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, 1))
        self.width = width
        self.state_dim = 2 * width
        self.half = half

    def _retrieve(self, query: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.bmm(query.unsqueeze(1), k.transpose(1, 2)) / math.sqrt(self.width)
        return torch.bmm(scores.softmax(dim=-1), v).squeeze(1)

    def context(self, source: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        k, v = self.key(targets), self.value(targets)
        ctx_i = self._retrieve(self.query_i(source[:, : self.half]), k, v)
        ctx_j = self._retrieve(self.query_j(source[:, self.half :]), k, v)
        return torch.cat([ctx_i, ctx_j], dim=-1)     # (N, 2W)

    def forward(self, source: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.head(self.context(source, targets)).squeeze(-1)


def train(model: nn.Module, data: Batch, epochs: int, device: torch.device) -> float:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    src, tgt, y = data.source.to(device), data.targets.to(device), data.y.to(device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(src, tgt) - y) ** 2)
        loss.backward()
        opt.step()
    return float(loss.item())


@torch.no_grad()
def mse(model: nn.Module, data: Batch, device: torch.device) -> float:
    pred = model(data.source.to(device), data.targets.to(device)).cpu()
    return float(torch.mean((pred - data.y) ** 2))


@torch.no_grad()
def probe_r2(model: nn.Module, fit: Batch, test: Batch, device: torch.device) -> float:
    """Linear decodability of the sufficient statistic y from the context state."""
    xf = model.context(fit.source.to(device), fit.targets.to(device)).cpu().numpy()
    xt = model.context(test.source.to(device), test.targets.to(device)).cpu().numpy()
    scaler = StandardScaler().fit(xf)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(xf), fit.y.numpy())
    return float(r2_score(test.y.numpy(), ridge.predict(scaler.transform(xt))))


def sensitivity(model: nn.Module, data: Batch, device: torch.device) -> dict:
    """Counterfactual |dy/dv_j| via autograd, vs the equation's analytic value."""
    model.eval()
    src = data.source.to(device)
    tgt = data.targets.to(device).clone().requires_grad_(True)
    out = model(src, tgt).sum()
    (grad,) = torch.autograd.grad(out, tgt)
    j = data.value_j_index.to(device)
    rows = torch.arange(len(j), device=device)
    dydvj = grad[rows, j, 0].abs().cpu()             # value channel is index 0
    analytic = (data.v_i / (data.v_i + data.v_j) ** 2).abs()
    return {
        "model_abs_dy_dvj": float(dydvj.mean()),
        "equation_abs_dy_dvj": float(analytic.mean()),
    }


class Interchange(nn.Module):
    """DAS: learn an orthogonal basis in which a k-dim subspace carries the ratio.

    We swap that subspace between a base and a source input and require the frozen
    head to output the *source's* y.  If a faithful causal variable exists, IIA is
    high; if the model only correlates with the ratio, no such subspace exists.
    """

    def __init__(self, width: int, k: int):
        super().__init__()
        self.raw = nn.Parameter(torch.randn(width, width) * 0.1)
        self.k = k

    def basis(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.raw)
        return q

    def patch(self, ctx_base: torch.Tensor, ctx_src: torch.Tensor) -> torch.Tensor:
        q = self.basis()
        zb = ctx_base @ q                            # rotate into aligned basis
        zs = ctx_src @ q
        mask = torch.zeros(zb.shape[-1], device=zb.device)
        mask[: self.k] = 1.0
        zp = zb * (1 - mask) + zs * mask             # swap the causal subspace
        return zp @ q.t()                            # rotate back


def interchange_accuracy(
    model: nn.Module, data: Batch, device: torch.device, k: int, steps: int
) -> float:
    """Train the alignment (model frozen); report IIA as R^2 of patched output vs y_src."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        ctx = model.context(data.source.to(device), data.targets.to(device))
    y = data.y.to(device)
    n = len(y)
    rng = np.random.default_rng(0)

    align = Interchange(model.state_dim, k).to(device)
    opt = torch.optim.Adam(align.parameters(), lr=5e-3)
    for _ in range(steps):
        b = torch.from_numpy(rng.integers(0, n, size=min(256, n))).to(device)
        s = torch.from_numpy(rng.integers(0, n, size=min(256, n))).to(device)
        patched = align.patch(ctx[b], ctx[s])
        pred = model.head(patched).squeeze(-1)
        opt.zero_grad()
        loss = torch.mean((pred - y[s]) ** 2)        # target: the source's y
        loss.backward()
        opt.step()

    with torch.no_grad():
        b = torch.arange(n, device=device)
        s = torch.from_numpy(rng.permutation(n)).to(device)
        pred = model.head(align.patch(ctx[b], ctx[s])).squeeze(-1)
        ss_res = torch.mean((pred - y[s]) ** 2).item()
        ss_tot = torch.var(y[s], unbiased=False).item()
    return float(1.0 - ss_res / max(ss_tot, 1e-8))


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    pos_dim = 8
    source_dim, target_dim = 2 * pos_dim, 1 + pos_dim

    # --- data ---------------------------------------------------------------
    law_train = generate(args.n_train, vary_j=True, seed=1)
    shortcut_train = generate(args.n_train, vary_j=False, seed=1)
    # Validation the practitioner has: component j barely varies (shortcut-like).
    val_indist = generate(args.n_eval, vary_j=False, seed=2)
    # Audit / counterfactual set: component j is varied (never used for labels below).
    audit = generate(args.n_eval, vary_j=True, seed=3)
    # Held-out reality, revealed only to confirm the audit.
    ood = generate(args.n_eval, vary_j=True, seed=4)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    report = {"config": config | {"device": str(device)}, "models": {}}
    for name, train_data in (("law", law_train), ("shortcut", shortcut_train)):
        set_seed(0)
        model = Retriever(source_dim, target_dim, args.width)
        train(model, train_data, args.epochs, device)
        report["models"][name] = {
            "val_indist_mse": mse(model, val_indist, device),
            "ood_mse": mse(model, ood, device),
            "probe_r2_indist": probe_r2(model, val_indist, val_indist, device),
            "sensitivity_to_j": sensitivity(model, audit, device),
            "interchange_accuracy": interchange_accuracy(
                model, audit, device, args.k, args.das_steps
            ),
        }
    return report


def plot(report: dict, path: Path) -> None:
    names = ["law", "shortcut"]
    metrics = report["models"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    axes[0].bar(names, [metrics[n]["val_indist_mse"] for n in names], color="#4c72b0")
    axes[0].bar(
        names, [metrics[n]["ood_mse"] for n in names], color="#c44e52", alpha=0.55,
        label="OOD (j varies)",
    )
    axes[0].set_title("MSE: validation vs OOD")
    axes[0].set_ylabel("MSE")
    axes[0].legend(["val (j~const)", "OOD (j varies)"])

    axes[1].bar(names, [metrics[n]["probe_r2_indist"] for n in names], color="#55a868")
    axes[1].set_title("Decodability probe R^2\n(cannot separate the models)")
    axes[1].set_ylim(0, 1)

    axes[2].bar(
        names, [metrics[n]["interchange_accuracy"] for n in names], color="#8172b3"
    )
    axes[2].set_title("Interchange-intervention accuracy\n(the audit that separates them)")
    axes[2].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/causal_mediation_toy"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-eval", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--k", type=int, default=2, help="interchange subspace dim")
    parser.add_argument("--das-steps", type=int, default=800)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_train, args.n_eval, args.epochs, args.das_steps = 800, 400, 300, 200

    report = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    plot(report, args.output_dir / "causal_mediation.png")

    m = report["models"]
    print("\n=== Governing-law causal audit (toy) ===")
    header = f"{'metric':<34}{'law':>12}{'shortcut':>12}"
    print(header)
    print("-" * len(header))
    rows = [
        ("val MSE (j~const, practitioner)", "val_indist_mse"),
        ("probe R^2 (decodes ratio)", "probe_r2_indist"),
        ("interchange accuracy (audit)", "interchange_accuracy"),
        ("OOD MSE (j varies, revealed)", "ood_mse"),
    ]
    for label, key in rows:
        print(f"{label:<34}{m['law'][key]:>12.4f}{m['shortcut'][key]:>12.4f}")
    print(f"{'|dy/dvj| model vs equation':<34}"
          f"{m['law']['sensitivity_to_j']['model_abs_dy_dvj']:>6.3f} / "
          f"{m['law']['sensitivity_to_j']['equation_abs_dy_dvj']:<4.3f}"
          f"{m['shortcut']['sensitivity_to_j']['model_abs_dy_dvj']:>6.3f} / "
          f"{m['shortcut']['sensitivity_to_j']['equation_abs_dy_dvj']:<4.3f}")
    print(f"\nwrote {args.output_dir}/metrics.json and causal_mediation.png")


if __name__ == "__main__":
    main()
