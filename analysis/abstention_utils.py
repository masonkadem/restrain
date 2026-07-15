"""Shared metrics and probe utilities for physics-aware abstention experiments.

All confidence scores in this module follow the convention "larger is more
confident".  Probe fitting and confidence calibration are deliberately kept
separate so callers can enforce train/validation/test isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """Return equal-width binary expected calibration error."""
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(labels), 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        include = (probabilities >= lo) & (
            probabilities <= hi if hi == 1.0 else probabilities < hi
        )
        if include.any():
            error += include.sum() / total * abs(
                probabilities[include].mean() - labels[include].mean()
            )
    return float(error)


def binary_detection_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Metrics for probability that an example is answerable/clean."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.unique(labels).size < 2:
        raise ValueError("Binary detection metrics require both classes.")
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
    }


def binary_ranking_metrics(
    labels: np.ndarray, confidence: np.ndarray
) -> dict[str, float]:
    """AUROC/AUPRC for a confidence score without calling it a probability."""
    labels = np.asarray(labels, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    if np.unique(labels).size < 2:
        raise ValueError("Binary ranking metrics require both classes.")
    return {
        "auroc": float(roc_auc_score(labels, confidence)),
        "auprc": float(average_precision_score(labels, confidence)),
    }


def risk_coverage_curve(
    losses: np.ndarray, confidence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return selective risk after retaining highest-confidence examples."""
    losses = np.asarray(losses, dtype=float).reshape(-1)
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    if losses.shape != confidence.shape or losses.size == 0:
        raise ValueError("losses and confidence must be non-empty equal-length arrays")
    finite = np.isfinite(losses) & np.isfinite(confidence)
    losses, confidence = losses[finite], confidence[finite]
    if losses.size == 0:
        raise ValueError("No finite loss/confidence pairs.")
    order = np.argsort(-confidence, kind="stable")
    sorted_losses = losses[order]
    risks = np.cumsum(sorted_losses) / np.arange(1, len(sorted_losses) + 1)
    coverage = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    return coverage, risks


# numpy 2.0 renamed trapz to trapezoid.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def area_under_risk_coverage(losses: np.ndarray, confidence: np.ndarray) -> float:
    coverage, risk = risk_coverage_curve(losses, confidence)
    # Include coverage=0 with the first retained-example risk.
    return float(_trapezoid(np.r_[risk[0], risk], np.r_[0.0, coverage]))


def risks_at_coverages(
    losses: np.ndarray,
    confidence: np.ndarray,
    coverages: Iterable[float] = (0.5, 0.8, 1.0),
) -> dict[str, float]:
    """Return mean retained loss at requested coverage fractions."""
    losses = np.asarray(losses, dtype=float).reshape(-1)
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    order = np.argsort(-confidence, kind="stable")
    result = {}
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError("Coverage must be in (0, 1].")
        retained = max(1, int(np.ceil(coverage * len(losses))))
        result[f"{int(round(100 * coverage))}%"] = float(losses[order[:retained]].mean())
    return result


@dataclass
class LinearAnswerabilityProbe:
    scaler: StandardScaler
    classifier: LogisticRegression

    def predict_proba(self, activations: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.scaler.transform(activations))[:, 1]


def fit_answerability_probe(
    activations: np.ndarray, labels: np.ndarray, seed: int = 0
) -> LinearAnswerabilityProbe:
    """Fit a balanced logistic probe to frozen activations."""
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).size < 2:
        raise ValueError("Answerability probe requires both classes.")
    scaler = StandardScaler().fit(activations)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=seed,
    ).fit(scaler.transform(activations), labels)
    return LinearAnswerabilityProbe(scaler, classifier)


@dataclass
class BootstrapRidgeProbe:
    """Bootstrap Ridge ensemble with activation-support confidence."""

    scaler: StandardScaler
    models: list[Ridge]
    distance_center: float = 0.0
    distance_scale: float = 1.0
    disagreement_center: float = 0.0
    disagreement_scale: float = 1.0

    def predict_ensemble(self, activations: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(activations)
        return np.stack([model.predict(x) for model in self.models], axis=1)

    def components(self, activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.scaler.transform(activations)
        disagreement = self.predict_ensemble(activations).std(axis=1)
        # Diagonal standardized distance is stable when D is comparable to N.
        distance = np.sqrt(np.mean(np.square(x), axis=1))
        return disagreement, distance

    def confidence(self, activations: np.ndarray) -> np.ndarray:
        disagreement, distance = self.components(activations)
        disagreement_z = (
            disagreement - self.disagreement_center
        ) / self.disagreement_scale
        distance_z = (distance - self.distance_center) / self.distance_scale
        return -(disagreement_z + distance_z)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return self.predict_ensemble(activations).mean(axis=1)

    def calibrate_confidence(self, validation_activations: np.ndarray) -> None:
        """Calibrate score location/scale using clean validation activations only."""
        disagreement, distance = self.components(validation_activations)
        self.disagreement_center, self.disagreement_scale = _robust_location_scale(
            disagreement
        )
        self.distance_center, self.distance_scale = _robust_location_scale(distance)


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(values))
    return center, max(scale, 1e-8)


def fit_bootstrap_ridge(
    activations: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int = 20,
    alpha: float = 10.0,
    seed: int = 0,
) -> BootstrapRidgeProbe:
    """Fit Ridge probes on bootstrap samples of frozen activations."""
    activations = np.asarray(activations, dtype=float)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    finite = np.isfinite(targets) & np.isfinite(activations).all(axis=1)
    activations, targets = activations[finite], targets[finite]
    if len(targets) < 4:
        raise ValueError("At least four finite examples are required for a probe.")
    scaler = StandardScaler().fit(activations)
    x = scaler.transform(activations)
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(targets), size=len(targets))
        models.append(Ridge(alpha=alpha).fit(x[indices], targets[indices]))
    return BootstrapRidgeProbe(scaler=scaler, models=models)
