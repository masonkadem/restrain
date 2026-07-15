"""Deterministic smoke tests for the abstract parametric-knowledge task."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from abstract_knowledge_boundary import (
    AssociationMemorizer,
    evaluate_model,
    generate_universe,
    make_queries,
    nearest_key_confidence,
    split_key_indices,
)


class AbstractKnowledgeBoundaryTests(unittest.TestCase):
    def test_split_leaves_no_input_cue(self) -> None:
        """Keys and values must not depend on which keys end up known."""
        half = generate_universe(30, key_dim=6, value_dim=3, known_fraction=0.5, seed=7)
        quarter = generate_universe(
            30, key_dim=6, value_dim=3, known_fraction=0.25, seed=7
        )
        np.testing.assert_array_equal(half.keys, quarter.keys)
        np.testing.assert_array_equal(half.values, quarter.values)
        self.assertEqual(half.known.sum(), 15)
        self.assertEqual(quarter.known.sum(), 8)
        np.testing.assert_allclose(
            np.linalg.norm(half.keys, axis=1), 1.0, atol=1e-5
        )
        np.testing.assert_allclose(
            np.linalg.norm(half.values, axis=1), 1.0, atol=1e-5
        )

    def test_queries_carry_true_values_and_known_flags(self) -> None:
        universe = generate_universe(
            20, key_dim=5, value_dim=2, known_fraction=0.5, seed=3
        )
        indices = np.array([0, 4, 9])
        batch = make_queries(
            universe, indices, queries_per_key=3, input_noise=0.0, seed=1
        )
        repeated = np.repeat(indices, 3)
        np.testing.assert_array_equal(batch.key_index.numpy(), repeated)
        np.testing.assert_allclose(batch.inputs.numpy(), universe.keys[repeated])
        np.testing.assert_allclose(batch.labels.numpy(), universe.values[repeated])
        np.testing.assert_array_equal(
            batch.known.numpy(), universe.known[repeated].astype(np.int64)
        )

    def test_probe_key_split_is_disjoint_and_complete(self) -> None:
        indices = np.arange(11)
        fit, test = split_key_indices(indices, seed=5)
        self.assertEqual(len(np.intersect1d(fit, test)), 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([fit, test])), indices
        )

    def test_model_activation_shapes(self) -> None:
        universe = generate_universe(
            10, key_dim=4, value_dim=2, known_fraction=0.5, seed=2
        )
        batch = make_queries(
            universe, np.arange(10), queries_per_key=2, input_noise=0.05, seed=4
        )
        model = AssociationMemorizer(key_dim=4, value_dim=2, width=6)
        outputs = evaluate_model(model, batch, torch.device("cpu"), batch_size=8)
        self.assertEqual(outputs["prediction"].shape, (20, 2))
        self.assertEqual(outputs["activation"].shape, (20, 6))
        self.assertEqual(outputs["loss"].shape, (20,))
        self.assertEqual(outputs["known"].shape, (20,))

    def test_nearest_key_confidence_prefers_trained_neighborhood(self) -> None:
        trained = np.eye(3, dtype=np.float32)
        near = trained[0] + 0.01
        far = -trained.sum(axis=0)
        confidence = nearest_key_confidence(np.stack([near, far]), trained)
        self.assertGreater(confidence[0], confidence[1])


if __name__ == "__main__":
    unittest.main()
