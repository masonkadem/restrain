"""Deterministic smoke tests for synthetic abstention experiments."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from abstention_utils import risk_coverage_curve, risks_at_coverages
from synthetic_knowledge_boundary import (
    CrossAttentionRetriever,
    evaluate_model,
    generate_cross_retrieval_data,
)


class PhysicsAbstentionTests(unittest.TestCase):
    def test_missing_target_labels_match_observed_availability(self) -> None:
        batch = generate_cross_retrieval_data(
            12, n_slots=7, q=3, missing_rate=0.4, seed=8
        )
        availability = batch.targets[..., -1].numpy().astype(bool)
        selected = availability[
            np.arange(len(batch.source))[:, None, None], batch.selection.numpy()
        ]
        np.testing.assert_array_equal(batch.answerable.numpy(), selected.all(axis=2))
        self.assertTrue((~batch.answerable.bool()).any())
        missing_per_example = (~availability).sum(axis=1)
        self.assertEqual(np.unique(missing_per_example).size, 1)

    def test_shifted_task_remains_answerable(self) -> None:
        batch = generate_cross_retrieval_data(
            5, n_slots=7, q=2, positional_noise=0.25, seed=3
        )
        self.assertTrue(batch.answerable.bool().all())

    def test_synthetic_activation_shapes(self) -> None:
        batch = generate_cross_retrieval_data(4, n_slots=7, q=2, seed=2)
        model = CrossAttentionRetriever(
            source_dim=batch.source.shape[-1],
            target_dim=batch.targets.shape[-1],
            value_dim=batch.labels.shape[-1],
            width=6,
        )
        outputs = evaluate_model(model, batch, torch.device("cpu"), batch_size=2)
        self.assertEqual(outputs["prediction"].shape, tuple(batch.labels.shape))
        self.assertEqual(outputs["activation"].shape, (4, 7, 6))
        self.assertEqual(outputs["loss"].shape, (4, 7))

    def test_selective_metrics_retain_high_confidence_first(self) -> None:
        losses = np.array([4.0, 1.0, 3.0, 2.0])
        confidence = np.array([0.1, 0.9, 0.2, 0.8])
        coverage, risk = risk_coverage_curve(losses, confidence)
        np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(risk, [1.0, 1.5, 2.0, 2.5])
        self.assertEqual(
            risks_at_coverages(losses, confidence, [0.5])["50%"], 1.5
        )


if __name__ == "__main__":
    unittest.main()
