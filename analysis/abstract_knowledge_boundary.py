"""Abstract parametric-knowledge boundary test for activation-based abstention.

The synthetic cross-retrieval task probes *contextual* unanswerability: the
evidence is absent from the provided input, so unanswerability is in principle
computable from the input alone.  This experiment removes that cue entirely.

A universe of random key->value associations is generated, then split into a
"known" half that the model is trained to memorize and an "unknown" half whose
values are never shown.  Known and unknown keys are drawn from the identical
distribution *before* the split, so no surface feature of a query marks it as
unanswerable.  The only way to tell that a query is outside the model's
competence is whether the model itself has learned the association -- the
abstract analogue of asking about a fact that was never in the training data.

A linear probe on frozen activations, fit and evaluated on *disjoint* sets of
keys, predicts whether the queried association was trained.  A falsification
control retrains the same architecture on every key: if the probe still
"detects" the nominally unknown keys there, the signal is a protocol artifact
rather than model knowledge.

Run from the repository root:

    python analysis/abstract_knowledge_boundary.py --device cuda

Use ``--quick`` for a short smoke run.  The default runs three random seeds.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch import nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from abstention_utils import (  # noqa: E402
    area_under_risk_coverage,
    binary_detection_metrics,
    binary_ranking_metrics,
    fit_answerability_probe,
    risk_coverage_curve,
    risks_at_coverages,
)


@dataclass
class AssociationUniverse:
    """Fixed key->value codebook with a trained/never-trained split."""

    keys: np.ndarray
    values: np.ndarray
    known: np.ndarray


@dataclass
class AssociationBatch:
    inputs: torch.Tensor
    labels: torch.Tensor
    key_index: torch.Tensor
    known: torch.Tensor
    condition: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def generate_universe(
    n_keys: int,
    key_dim: int,
    value_dim: int,
    known_fraction: float,
    seed: int,
) -> AssociationUniverse:
    """Draw keys and values first, then split, so the split leaves no input cue."""
    if not 0.0 < known_fraction < 1.0:
        raise ValueError("known_fraction must be in (0, 1).")
    rng = np.random.default_rng(seed)
    keys = _unit_rows(rng.normal(size=(n_keys, key_dim)).astype(np.float32))
    values = _unit_rows(rng.normal(size=(n_keys, value_dim)).astype(np.float32))
    known = np.zeros(n_keys, dtype=bool)
    n_known = int(round(known_fraction * n_keys))
    n_known = min(max(n_known, 1), n_keys - 1)
    known[rng.choice(n_keys, size=n_known, replace=False)] = True
    return AssociationUniverse(keys=keys, values=values, known=known)


def make_queries(
    universe: AssociationUniverse,
    key_indices: np.ndarray,
    queries_per_key: int,
    input_noise: float,
    seed: int,
    condition: str = "clean",
) -> AssociationBatch:
    """Noisy queries against the codebook; labels are always the true values."""
    key_indices = np.asarray(key_indices, dtype=np.int64)
    repeated = np.repeat(key_indices, queries_per_key)
    rng = np.random.default_rng(seed)
    inputs = universe.keys[repeated]
    if input_noise > 0:
        inputs = inputs + rng.normal(
            scale=input_noise, size=inputs.shape
        ).astype(np.float32)
    labels = universe.values[repeated]
    known = universe.known[repeated].astype(np.int64)
    return AssociationBatch(
        inputs=torch.from_numpy(inputs.astype(np.float32)),
        labels=torch.from_numpy(labels.astype(np.float32)),
        key_index=torch.from_numpy(repeated),
        known=torch.from_numpy(known),
        condition=condition,
    )


def split_key_indices(
    indices: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint halves so the probe never sees its evaluation keys."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("Need at least two keys to split.")
    shuffled = np.random.default_rng(seed).permutation(indices)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


class AssociationMemorizer(nn.Module):
    """MLP that must store the trained key->value associations in its weights."""

    def __init__(self, key_dim: int, value_dim: int, width: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(key_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.head = nn.Linear(width, value_dim)

    def forward(self, inputs: torch.Tensor, return_features: bool = False):
        features = self.encoder(inputs)
        prediction = self.head(features)
        if return_features:
            return prediction, features
        return prediction


def train_model(
    model: nn.Module,
    data: AssociationBatch,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float = 1e-3,
) -> list[float]:
    model.to(device)
    inputs, labels = data.inputs, data.labels
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(inputs))
        total = 0.0
        for start in range(0, len(inputs), batch_size):
            indices = order[start : start + batch_size]
            prediction = model(inputs[indices].to(device))
            loss = torch.mean((prediction - labels[indices].to(device)) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        scheduler.step()
        losses.append(total / len(inputs))
    return losses


def evaluate_model(
    model: nn.Module, data: AssociationBatch, device: torch.device, batch_size: int
) -> dict[str, np.ndarray]:
    predictions, activations = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data.inputs), batch_size):
            output, features = model(
                data.inputs[start : start + batch_size].to(device),
                return_features=True,
            )
            predictions.append(output.cpu().numpy())
            activations.append(features.cpu().numpy())
    prediction = np.concatenate(predictions)
    return {
        "prediction": prediction,
        "activation": np.concatenate(activations),
        "loss": np.mean((prediction - data.labels.numpy()) ** 2, axis=-1),
        "known": data.known.numpy(),
    }


def nearest_key_confidence(
    queries: np.ndarray, trained_keys: np.ndarray
) -> np.ndarray:
    """Input-space familiarity baseline: negative distance to the training set.

    Unlike the activation probe, this baseline needs privileged access to the
    training keys; it bounds what input-side novelty detection can achieve.
    """
    differences = queries[:, None, :] - trained_keys[None, :, :]
    return -np.sqrt(np.square(differences).sum(axis=-1)).min(axis=1)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run_seed(seed: int, args: argparse.Namespace, device: torch.device) -> tuple[dict, dict]:
    set_seed(seed)
    universe = generate_universe(
        args.n_keys, args.key_dim, args.value_dim, args.known_fraction, seed * 100 + 1
    )
    known_indices = np.flatnonzero(universe.known)
    unknown_indices = np.flatnonzero(~universe.known)
    known_fit, known_test = split_key_indices(known_indices, seed * 100 + 2)
    unknown_fit, unknown_test = split_key_indices(unknown_indices, seed * 100 + 3)

    train_known = make_queries(
        universe, known_indices, args.train_queries, args.input_noise, seed * 100 + 4
    )
    train_all = make_queries(
        universe,
        np.arange(args.n_keys),
        args.train_queries,
        args.input_noise,
        seed * 100 + 5,
    )
    probe_fit_batch = make_queries(
        universe,
        np.concatenate([known_fit, unknown_fit]),
        args.probe_queries,
        args.input_noise,
        seed * 100 + 6,
        condition="probe_fit",
    )
    test_batch = make_queries(
        universe,
        np.concatenate([known_test, unknown_test]),
        args.test_queries,
        args.input_noise,
        seed * 100 + 7,
        condition="test",
    )
    noisy_known_batch = make_queries(
        universe,
        known_test,
        args.test_queries,
        args.input_noise * args.noise_stress_factor,
        seed * 100 + 8,
        condition="noisy_known",
    )

    model_setups = {
        "adequate": (
            lambda: AssociationMemorizer(args.key_dim, args.value_dim, args.width),
            train_known,
        ),
        "low_capacity": (
            lambda: AssociationMemorizer(args.key_dim, args.value_dim, args.low_width),
            train_known,
        ),
        # Falsification control: trained on every key, so the "unknown" labels
        # no longer describe a real knowledge boundary.
        "no_boundary": (
            lambda: AssociationMemorizer(args.key_dim, args.value_dim, args.width),
            train_all,
        ),
    }

    seed_metrics, plot_payload = {}, {}
    for model_name, (factory, train_data) in model_setups.items():
        model = factory()
        train_losses = train_model(
            model, train_data, device, args.epochs, args.batch_size
        )
        probe_fit = evaluate_model(model, probe_fit_batch, device, args.batch_size)
        test = evaluate_model(model, test_batch, device, args.batch_size)
        noisy_known = evaluate_model(
            model, noisy_known_batch, device, args.batch_size
        )

        probe = fit_answerability_probe(
            probe_fit["activation"], probe_fit["known"], seed
        )
        probabilities = probe.predict_proba(test["activation"])
        noisy_probabilities = probe.predict_proba(noisy_known["activation"])
        random_confidence = np.random.default_rng(seed + 10_000).random(
            len(test["loss"])
        )
        nn_confidence = nearest_key_confidence(
            test_batch.inputs.numpy(), universe.keys[known_indices]
        )
        unknown_mask = test["known"] == 0
        stress_labels = np.concatenate(
            [np.ones(len(noisy_known["known"]), dtype=int), np.zeros(unknown_mask.sum(), dtype=int)]
        )
        stress_probabilities = np.concatenate(
            [noisy_probabilities, probabilities[unknown_mask]]
        )

        model_metrics = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "final_train_mse": train_losses[-1],
            "condition_mse": {
                "known": float(test["loss"][test["known"] == 1].mean()),
                "unknown": float(test["loss"][unknown_mask].mean()),
                "noisy_known": float(noisy_known["loss"].mean()),
            },
            "familiarity_detection": binary_detection_metrics(
                test["known"], probabilities
            ),
            "familiarity_under_input_noise": binary_detection_metrics(
                stress_labels, stress_probabilities
            ),
            "input_nn_baseline": binary_ranking_metrics(
                test["known"], nn_confidence
            ),
            "selective": {
                "probe": {
                    "aurc": area_under_risk_coverage(test["loss"], probabilities),
                    "risk": risks_at_coverages(test["loss"], probabilities),
                },
                "random": {
                    "aurc": area_under_risk_coverage(
                        test["loss"], random_confidence
                    ),
                    "risk": risks_at_coverages(test["loss"], random_confidence),
                },
                "input_nn_baseline": {
                    "aurc": area_under_risk_coverage(test["loss"], nn_confidence),
                    "risk": risks_at_coverages(test["loss"], nn_confidence),
                },
            },
        }
        seed_metrics[model_name] = model_metrics
        if model_name == "adequate":
            fpr, tpr, _ = roc_curve(test["known"], probabilities)
            probe_curve = risk_coverage_curve(test["loss"], probabilities)
            random_curve = risk_coverage_curve(test["loss"], random_confidence)
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


def summarize_decision(all_metrics: dict) -> dict:
    seed_metrics = list(all_metrics.values())

    def mean_of(model_name: str, *path: str) -> float:
        collected = []
        for metrics in seed_metrics:
            value = metrics[model_name]
            for key in path:
                value = value[key]
            collected.append(value)
        return float(np.mean(collected))

    known_mse = mean_of("adequate", "condition_mse", "known")
    unknown_mse = mean_of("adequate", "condition_mse", "unknown")
    low_capacity_known_mse = mean_of("low_capacity", "condition_mse", "known")
    probe_auroc = mean_of("adequate", "familiarity_detection", "auroc")
    no_boundary_auroc = mean_of("no_boundary", "familiarity_detection", "auroc")
    probe_aurc = mean_of("adequate", "selective", "probe", "aurc")
    random_aurc = mean_of("adequate", "selective", "random", "aurc")
    return {
        "mean_known_mse": known_mse,
        "mean_unknown_mse": unknown_mse,
        "mean_low_capacity_known_mse": low_capacity_known_mse,
        "mean_probe_auroc": probe_auroc,
        "mean_no_boundary_probe_auroc": no_boundary_auroc,
        "mean_probe_aurc": probe_aurc,
        "mean_random_aurc": random_aurc,
        "adequate_memorizes_known_associations": bool(
            known_mse < unknown_mse and known_mse < low_capacity_known_mse
        ),
        "familiarity_detection_above_chance": bool(probe_auroc > 0.5),
        "probe_abstention_beats_random": bool(probe_aurc < random_aurc),
        "no_boundary_control_collapses": bool(probe_auroc > no_boundary_auroc),
    }


def plot_results(all_metrics: dict, payloads: list[dict], output_dir: Path) -> None:
    model_names = ["adequate", "low_capacity", "no_boundary"]
    labels = ["Adequate", "Low capacity", "No boundary"]
    conditions = ["known", "unknown"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(model_names))
    bar_width = 0.35
    for offset, condition in enumerate(conditions):
        means = [
            np.mean(
                [
                    seed_metrics[name]["condition_mse"][condition]
                    for seed_metrics in all_metrics.values()
                ]
            )
            for name in model_names
        ]
        stds = [
            np.std(
                [
                    seed_metrics[name]["condition_mse"][condition]
                    for seed_metrics in all_metrics.values()
                ]
            )
            for name in model_names
        ]
        ax.bar(
            x + (offset - 0.5) * bar_width,
            means,
            bar_width,
            yerr=stds,
            capsize=4,
            label=f"{condition} keys",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test MSE")
    ax.set_title("Memorization gap and controls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "abstract_memorization.png", dpi=180)
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
    ax.set_title("Trained-association detection (mean over seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "abstract_unknown_detection.png", dpi=180)
    plt.close(fig)

    coverage_grid = np.linspace(0.01, 1.0, 200)
    mean_probe_risk = np.mean(
        [
            np.interp(coverage_grid, payload["probe_coverage"], payload["probe_risk"])
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
    fig.savefig(output_dir / "abstract_risk_coverage.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/physics_abstention/abstract")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-keys", type=int, default=240)
    parser.add_argument("--known-fraction", type=float, default=0.5)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--value-dim", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--low-width", type=int, default=8)
    parser.add_argument("--train-queries", type=int, default=40)
    parser.add_argument("--probe-queries", type=int, default=20)
    parser.add_argument("--test-queries", type=int, default=20)
    parser.add_argument("--input-noise", type=float, default=0.1)
    parser.add_argument("--noise-stress-factor", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = [0]
        args.n_keys = min(args.n_keys, 40)
        args.train_queries = min(args.train_queries, 6)
        args.probe_queries = min(args.probe_queries, 4)
        args.test_queries = min(args.test_queries, 4)
        args.epochs = min(args.epochs, 5)
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    all_metrics, payloads = {}, []
    for seed in args.seeds:
        print(f"[abstract] seed={seed} device={device}", flush=True)
        seed_metrics, payload = run_seed(seed, args, device)
        all_metrics[str(seed)] = seed_metrics
        payloads.append(payload)
    report = {
        "claim": (
            "With known and unknown queries drawn from the same input "
            "distribution, a frozen activation probe detects whether the "
            "queried association was ever trained; it is an external monitor, "
            "not intrinsic awareness."
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
    print(f"[abstract] wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
