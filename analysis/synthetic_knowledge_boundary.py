"""Synthetic proof of concept for activation-based abstention.

The task extends the CrossSA retrieval-and-averaging experiment from
``representational_capacity.ipynb``.  A source token names q target slots and
the model must return their mean value.  At evaluation time selected targets
may be absent, making the requested value information-theoretically
underdetermined.  A linear probe, trained on frozen activations, predicts
whether each source query remains answerable.

Example (from ``supervised_baselines``):

    python analysis/synthetic_knowledge_boundary.py --device cuda

Use ``--quick`` for a short smoke run.  The default runs three random seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch import nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from abstention_utils import (  # noqa: E402
    area_under_risk_coverage,
    binary_detection_metrics,
    fit_answerability_probe,
    risk_coverage_curve,
    risks_at_coverages,
)


@dataclass
class CrossRetrievalBatch:
    source: torch.Tensor
    targets: torch.Tensor
    labels: torch.Tensor
    answerable: torch.Tensor
    selection: torch.Tensor
    condition: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_positional_encoding(n_slots: int, dimension: int) -> np.ndarray:
    """Deterministic sinusoidal encodings with unit-normalized rows."""
    positions = np.arange(n_slots, dtype=np.float32)[:, None]
    frequencies = np.exp(
        np.arange(0, dimension, 2, dtype=np.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = np.zeros((n_slots, dimension), dtype=np.float32)
    encoding[:, 0::2] = np.sin(positions * frequencies)
    if dimension > 1:
        encoding[:, 1::2] = np.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
    return encoding / np.maximum(np.linalg.norm(encoding, axis=1, keepdims=True), 1e-8)


def generate_cross_retrieval_data(
    n_examples: int,
    n_slots: int = 15,
    q: int = 3,
    value_dim: int = 4,
    pos_dim: int = 8,
    missing_rate: float = 0.0,
    positional_noise: float = 0.0,
    seed: int = 0,
    condition: str = "clean",
) -> CrossRetrievalBatch:
    """Generate clean, missing-target, or shifted-but-answerable examples."""
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1).")
    if not 1 <= q <= n_slots:
        raise ValueError("q must be between 1 and n_slots.")
    rng = np.random.default_rng(seed)
    base_pos = make_positional_encoding(n_slots, pos_dim)
    values = rng.normal(size=(n_examples, n_slots, value_dim)).astype(np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)

    selection = np.empty((n_examples, n_slots, q), dtype=np.int64)
    for batch_index in range(n_examples):
        for source_index in range(n_slots):
            selection[batch_index, source_index] = rng.choice(
                n_slots, size=q, replace=False
            )

    batch_indices = np.arange(n_examples)[:, None, None]
    selected_values = values[batch_indices, selection]
    labels = selected_values.mean(axis=2)

    availability = np.ones((n_examples, n_slots), dtype=bool)
    n_missing = int(round(missing_rate * n_slots))
    if missing_rate > 0:
        n_missing = max(1, n_missing)
        for batch_index in range(n_examples):
            unavailable = rng.choice(n_slots, size=n_missing, replace=False)
            availability[batch_index, unavailable] = False
    observed_values = values * availability[..., None]

    # A per-example perturbation shared between references and target keys keeps
    # the task answerable while moving it off the clean PE distribution.
    target_pos = np.broadcast_to(base_pos, (n_examples, n_slots, pos_dim)).copy()
    if positional_noise > 0:
        target_pos += rng.normal(
            scale=positional_noise, size=target_pos.shape
        ).astype(np.float32)
        target_pos /= np.maximum(
            np.linalg.norm(target_pos, axis=-1, keepdims=True), 1e-8
        )
    source_own = np.broadcast_to(base_pos, (n_examples, n_slots, pos_dim))
    selected_pos = target_pos[batch_indices, selection].reshape(
        n_examples, n_slots, q * pos_dim
    )
    source = np.concatenate([source_own, selected_pos], axis=-1)
    target = np.concatenate(
        [observed_values, target_pos, availability[..., None].astype(np.float32)],
        axis=-1,
    )
    selected_available = availability[
        np.arange(n_examples)[:, None, None], selection
    ]
    answerable = selected_available.all(axis=2)

    return CrossRetrievalBatch(
        source=torch.from_numpy(source.astype(np.float32)),
        targets=torch.from_numpy(target.astype(np.float32)),
        labels=torch.from_numpy(labels.astype(np.float32)),
        answerable=torch.from_numpy(answerable.astype(np.int64)),
        selection=torch.from_numpy(selection),
        condition=condition,
    )


class CrossAttentionRetriever(nn.Module):
    def __init__(self, source_dim: int, target_dim: int, value_dim: int, width: int):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(source_dim, 64), nn.ReLU(), nn.Linear(64, width))
        self.key = nn.Sequential(nn.Linear(target_dim, 64), nn.ReLU(), nn.Linear(64, width))
        self.value = nn.Sequential(nn.Linear(target_dim, 64), nn.ReLU(), nn.Linear(64, width))
        self.output = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, value_dim)
        )
        self.width = width

    def forward(
        self, source: torch.Tensor, targets: torch.Tensor, return_features: bool = False
    ):
        query = self.query(source)
        key = self.key(targets)
        value = self.value(targets)
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.width)
        attention = scores.softmax(dim=-1)
        context = torch.bmm(attention, value)
        prediction = self.output(context)
        if return_features:
            return prediction, context, attention
        return prediction


class GlobalPoolRetriever(nn.Module):
    """Control that sees the target bank but cannot retrieve by target identity."""

    def __init__(self, source_dim: int, target_dim: int, value_dim: int, width: int = 64):
        super().__init__()
        self.source_encoder = nn.Sequential(
            nn.Linear(source_dim, 64), nn.ReLU(), nn.Linear(64, width)
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, 64), nn.ReLU(), nn.Linear(64, width)
        )
        self.output = nn.Sequential(
            nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, value_dim)
        )

    def forward(
        self, source: torch.Tensor, targets: torch.Tensor, return_features: bool = False
    ):
        source_features = self.source_encoder(source)
        pooled = self.target_encoder(targets).mean(dim=1, keepdim=True)
        pooled = pooled.expand(-1, source.shape[1], -1)
        context = torch.cat([source_features, pooled], dim=-1)
        prediction = self.output(context)
        if return_features:
            return prediction, context, None
        return prediction


def train_model(
    model: nn.Module,
    data: CrossRetrievalBatch,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float = 1e-3,
) -> list[float]:
    model.to(device)
    source, targets, labels = data.source, data.targets, data.labels
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(source))
        total = 0.0
        for start in range(0, len(source), batch_size):
            indices = order[start : start + batch_size]
            prediction = model(
                source[indices].to(device), targets[indices].to(device)
            )
            loss = torch.mean((prediction - labels[indices].to(device)) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        scheduler.step()
        losses.append(total / len(source))
    return losses


def evaluate_model(
    model: nn.Module, data: CrossRetrievalBatch, device: torch.device, batch_size: int
) -> dict[str, np.ndarray]:
    predictions, activations, attentions = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data.source), batch_size):
            output, features, attention = model(
                data.source[start : start + batch_size].to(device),
                data.targets[start : start + batch_size].to(device),
                return_features=True,
            )
            predictions.append(output.cpu().numpy())
            activations.append(features.cpu().numpy())
            if attention is not None:
                attentions.append(attention.cpu().numpy())
    prediction = np.concatenate(predictions)
    result = {
        "prediction": prediction,
        "activation": np.concatenate(activations),
        "loss": np.mean((prediction - data.labels.numpy()) ** 2, axis=-1),
        "answerable": data.answerable.numpy(),
    }
    if attentions:
        attention = np.concatenate(attentions)
        result["attention_confidence"] = 1.0 - (
            -(attention * np.log(np.maximum(attention, 1e-12))).sum(axis=-1)
            / math.log(attention.shape[-1])
        )
    return result


def _flatten(parts: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    arrays = [part[key].reshape(-1, part[key].shape[-1]) for part in parts]
    return np.concatenate(arrays)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def summarize_decision(all_metrics: dict) -> dict:
    seed_metrics = list(all_metrics.values())
    clean_mse = {
        model_name: float(
            np.mean(
                [
                    metrics[model_name]["condition_mse"]["clean"]
                    for metrics in seed_metrics
                ]
            )
        )
        for model_name in ("cross_attention", "low_capacity", "no_retrieval")
    }
    within_missing_auroc = float(
        np.mean(
            [
                metrics["cross_attention"][
                    "answerability_within_missing_condition"
                ]["auroc"]
                for metrics in seed_metrics
            ]
        )
    )
    probe_aurc = float(
        np.mean(
            [
                metrics["cross_attention"]["selective"][
                    "probe_clean_and_missing"
                ]["aurc"]
                for metrics in seed_metrics
            ]
        )
    )
    random_aurc = float(
        np.mean(
            [
                metrics["cross_attention"]["selective"][
                    "random_clean_and_missing"
                ]["aurc"]
                for metrics in seed_metrics
            ]
        )
    )
    return {
        "mean_clean_mse": clean_mse,
        "mean_within_missing_answerability_auroc": within_missing_auroc,
        "mean_probe_aurc": probe_aurc,
        "mean_random_aurc": random_aurc,
        "adequate_model_beats_both_controls": bool(
            clean_mse["cross_attention"] < clean_mse["low_capacity"]
            and clean_mse["cross_attention"] < clean_mse["no_retrieval"]
        ),
        "answerability_detection_above_chance": bool(within_missing_auroc > 0.5),
        "probe_abstention_beats_random": bool(probe_aurc < random_aurc),
    }


def run_seed(
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict]:
    set_seed(seed)
    pos_dim = args.pos_dim
    source_dim = pos_dim * (1 + args.q)
    target_dim = args.value_dim + pos_dim + 1
    model_factories = {
        "cross_attention": lambda: CrossAttentionRetriever(
            source_dim, target_dim, args.value_dim, args.width
        ),
        "low_capacity": lambda: CrossAttentionRetriever(
            source_dim, target_dim, args.value_dim, args.low_width
        ),
        "no_retrieval": lambda: GlobalPoolRetriever(
            source_dim, target_dim, args.value_dim, args.width
        ),
    }

    train = generate_cross_retrieval_data(
        args.n_train,
        args.n_slots,
        args.q,
        args.value_dim,
        pos_dim,
        seed=seed * 100 + 1,
    )
    validation_batches = [
        generate_cross_retrieval_data(
            args.n_probe,
            args.n_slots,
            args.q,
            args.value_dim,
            pos_dim,
            seed=seed * 100 + 2,
            condition="clean",
        ),
        generate_cross_retrieval_data(
            args.n_probe,
            args.n_slots,
            args.q,
            args.value_dim,
            pos_dim,
            missing_rate=args.missing_rate,
            seed=seed * 100 + 3,
            condition="missing",
        ),
        generate_cross_retrieval_data(
            args.n_probe,
            args.n_slots,
            args.q,
            args.value_dim,
            pos_dim,
            positional_noise=args.positional_noise,
            seed=seed * 100 + 4,
            condition="shifted",
        ),
    ]
    test_batches = [
        generate_cross_retrieval_data(
            args.n_test,
            args.n_slots,
            args.q,
            args.value_dim,
            pos_dim,
            seed=seed * 100 + 5,
            condition="clean",
        ),
        generate_cross_retrieval_data(
            args.n_test,
            args.n_slots,
            args.q,
            args.value_dim,
            pos_dim,
            missing_rate=args.missing_rate,
            seed=seed * 100 + 6,
            condition="missing",
        ),
        generate_cross_retrieval_data(
            args.n_test,
            args.n_slots,
            args.q,
            args.value_dim,
            pos_dim,
            positional_noise=args.positional_noise * 1.5,
            seed=seed * 100 + 7,
            condition="shifted",
        ),
    ]

    seed_metrics, plot_payload = {}, {}
    for model_name, factory in model_factories.items():
        model = factory()
        train_losses = train_model(
            model, train, device, args.epochs, args.batch_size
        )
        validation = [
            evaluate_model(model, batch, device, args.batch_size)
            for batch in validation_batches
        ]
        test = [
            evaluate_model(model, batch, device, args.batch_size)
            for batch in test_batches
        ]

        probe_x = _flatten(validation, "activation")
        probe_y = np.concatenate(
            [part["answerable"].reshape(-1) for part in validation]
        )
        answerability_probe = fit_answerability_probe(probe_x, probe_y, seed)
        test_x = _flatten(test, "activation")
        probabilities = answerability_probe.predict_proba(test_x)
        test_answerable = np.concatenate(
            [part["answerable"].reshape(-1) for part in test]
        )
        test_losses = np.concatenate([part["loss"].reshape(-1) for part in test])
        missing_x = test[1]["activation"].reshape(
            -1, test[1]["activation"].shape[-1]
        )
        missing_probabilities = answerability_probe.predict_proba(missing_x)
        missing_answerable = test[1]["answerable"].reshape(-1)
        missing_losses = test[1]["loss"].reshape(-1)
        clean_probabilities = answerability_probe.predict_proba(
            test[0]["activation"].reshape(
                -1, test[0]["activation"].shape[-1]
            )
        )
        primary_losses = np.concatenate(
            [test[0]["loss"].reshape(-1), missing_losses]
        )
        primary_probabilities = np.concatenate(
            [clean_probabilities, missing_probabilities]
        )

        clean_activation = validation[0]["activation"].reshape(
            -1, validation[0]["activation"].shape[-1]
        )
        clean_target = validation_batches[0].labels.numpy().reshape(
            -1, args.value_dim
        )
        statistic_scaler = StandardScaler().fit(clean_activation)
        statistic_probe = Ridge(alpha=10.0).fit(
            statistic_scaler.transform(clean_activation), clean_target
        )
        clean_test_activation = test[0]["activation"].reshape(
            -1, test[0]["activation"].shape[-1]
        )
        statistic_r2 = r2_score(
            test_batches[0].labels.numpy().reshape(-1, args.value_dim),
            statistic_probe.predict(statistic_scaler.transform(clean_test_activation)),
            multioutput="variance_weighted",
        )

        random_confidence = np.random.default_rng(seed).random(len(test_losses))
        primary_random_confidence = np.random.default_rng(seed + 10_000).random(
            len(primary_losses)
        )
        model_metrics = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "final_train_mse": train_losses[-1],
            "condition_mse": {
                batch.condition: float(part["loss"].mean())
                for batch, part in zip(test_batches, test)
            },
            "retrieval_statistic_probe_r2": float(statistic_r2),
            "answerability": binary_detection_metrics(
                test_answerable, probabilities
            ),
            "answerability_within_missing_condition": binary_detection_metrics(
                missing_answerable, missing_probabilities
            ),
            "selective": {
                "probe": {
                    "aurc": area_under_risk_coverage(test_losses, probabilities),
                    "risk": risks_at_coverages(test_losses, probabilities),
                },
                "random": {
                    "aurc": area_under_risk_coverage(
                        test_losses, random_confidence
                    ),
                    "risk": risks_at_coverages(
                        test_losses, random_confidence
                    ),
                },
                "probe_clean_and_missing": {
                    "aurc": area_under_risk_coverage(
                        primary_losses, primary_probabilities
                    ),
                    "risk": risks_at_coverages(
                        primary_losses, primary_probabilities
                    ),
                },
                "random_clean_and_missing": {
                    "aurc": area_under_risk_coverage(
                        primary_losses, primary_random_confidence
                    ),
                    "risk": risks_at_coverages(
                        primary_losses, primary_random_confidence
                    ),
                },
                "probe_within_missing_condition": {
                    "aurc": area_under_risk_coverage(
                        missing_losses, missing_probabilities
                    ),
                    "risk": risks_at_coverages(
                        missing_losses, missing_probabilities
                    ),
                },
            },
        }
        if "attention_confidence" in test[0]:
            attention_confidence = np.concatenate(
                [part["attention_confidence"].reshape(-1) for part in test]
            )
            model_metrics["selective"]["attention_entropy"] = {
                "aurc": area_under_risk_coverage(
                    test_losses, attention_confidence
                ),
                "risk": risks_at_coverages(test_losses, attention_confidence),
            }

        seed_metrics[model_name] = model_metrics
        if model_name == "cross_attention":
            fpr, tpr, _ = roc_curve(
                missing_answerable, missing_probabilities
            )
            probe_curve = risk_coverage_curve(
                primary_losses, primary_probabilities
            )
            random_curve = risk_coverage_curve(
                primary_losses, primary_random_confidence
            )
            plot_payload = {
                "fpr": fpr,
                "tpr": tpr,
                "probe_coverage": probe_curve[0],
                "probe_risk": probe_curve[1],
                "random_coverage": random_curve[0],
                "random_risk": random_curve[1],
            }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return seed_metrics, plot_payload


def plot_results(all_metrics: dict, payloads: list[dict], output_dir: Path) -> None:
    model_names = ["cross_attention", "low_capacity", "no_retrieval"]
    labels = ["Cross-attention", "Low capacity", "No retrieval"]
    clean_means = [
        np.mean(
            [
                seed_metrics[name]["condition_mse"]["clean"]
                for seed_metrics in all_metrics.values()
            ]
        )
        for name in model_names
    ]
    clean_stds = [
        np.std(
            [
                seed_metrics[name]["condition_mse"]["clean"]
                for seed_metrics in all_metrics.values()
            ]
        )
        for name in model_names
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, clean_means, yerr=clean_stds, capsize=4)
    ax.set_ylabel("Clean test MSE")
    ax.set_title("Capacity and retrieval controls")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_capacity.png", dpi=180)
    plt.close(fig)

    roc_grid = np.linspace(0.0, 1.0, 201)
    mean_tpr = np.mean(
        [np.interp(roc_grid, payload["fpr"], payload["tpr"]) for payload in payloads],
        axis=0,
    )
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(roc_grid, mean_tpr, label="Activation probe")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate")
    ax.set_title("Answerability detection (mean over seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_unknown_detection.png", dpi=180)
    plt.close(fig)

    coverage_grid = np.linspace(0.01, 1.0, 200)
    mean_probe_risk = np.mean(
        [
            np.interp(
                coverage_grid, payload["probe_coverage"], payload["probe_risk"]
            )
            for payload in payloads
        ],
        axis=0,
    )
    mean_random_risk = np.mean(
        [
            np.interp(
                coverage_grid, payload["random_coverage"], payload["random_risk"]
            )
            for payload in payloads
        ],
        axis=0,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coverage_grid, mean_probe_risk, label="Activation probe")
    ax.plot(coverage_grid, mean_random_risk, label="Random abstention")
    ax.set(xlabel="Coverage", ylabel="Retained prediction MSE")
    ax.set_title("Selective risk (mean over seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_risk_coverage.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/physics_abstention/synthetic"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-probe", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-slots", type=int, default=15)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--value-dim", type=int, default=4)
    parser.add_argument("--pos-dim", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--low-width", type=int, default=4)
    parser.add_argument("--missing-rate", type=float, default=0.25)
    parser.add_argument("--positional-noise", type=float, default=0.15)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = [0]
        args.n_train = min(args.n_train, 128)
        args.n_probe = min(args.n_probe, 48)
        args.n_test = min(args.n_test, 64)
        args.epochs = min(args.epochs, 3)
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    all_metrics, payloads = {}, []
    for seed in args.seeds:
        print(f"[synthetic] seed={seed} device={device}", flush=True)
        seed_metrics, payload = run_seed(seed, args, device)
        all_metrics[str(seed)] = seed_metrics
        payloads.append(payload)
    report = {
        "claim": (
            "A frozen activation probe detects answerability and supports "
            "selective prediction; it is an external monitor, not intrinsic awareness."
        ),
        "config": vars(args),
        "seeds": all_metrics,
        "decision": summarize_decision(all_metrics),
    }
    report["config"]["output_dir"] = str(report["config"]["output_dir"])
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(_json_ready(report), handle, indent=2)
    curve_payload = {
        f"seed_{seed}_{key}": value
        for seed, payload in zip(args.seeds, payloads)
        for key, value in payload.items()
    }
    np.savez_compressed(output_dir / "curves.npz", **curve_payload)
    plot_results(all_metrics, payloads, output_dir)
    print(f"[synthetic] wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
