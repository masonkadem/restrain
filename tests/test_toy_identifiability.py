"""Tests for the identifiability toy model and its central dissociation."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from toy_identifiability import (  # noqa: E402
    gamma_of,
    generate,
    make_instrument,
    run_condition,
)
from toy_geometry import (  # noqa: E402
    auroc_along,
    inlp_rank_curve,
    mean_difference_direction,
)


class ToyIdentifiabilityTests(unittest.TestCase):
    def test_gamma_monotone_and_modes(self) -> None:
        # gamma is 1 at f=0 and increases with f; resampled (c=2) exceeds zeroed.
        self.assertAlmostEqual(gamma_of(0.0, 0.1, "zeroed"), 1.0, places=6)
        self.assertGreater(gamma_of(0.5, 0.1, "zeroed"), gamma_of(0.1, 0.1, "zeroed"))
        self.assertGreater(gamma_of(0.5, 0.1, "resampled"), gamma_of(0.5, 0.1, "zeroed"))

    def test_generate_shapes_and_masking(self) -> None:
        inst = make_instrument(k_core=4, k_periph=4, sigma=0.1, seed=0)
        clean = generate(inst, 20, 0.5, "clean", seed=1)
        zeroed = generate(inst, 20, 0.5, "zeroed", seed=1)
        resampled = generate(inst, 20, 0.5, "resampled", seed=1)
        self.assertEqual(clean["x"].shape, (20, 8))
        self.assertTrue((clean["answerable"] == 1).all())
        self.assertTrue((zeroed["answerable"] == 0).all())
        # zeroed masks the peripheral block to exactly zero; clean does not.
        self.assertTrue(np.allclose(zeroed["x"][:, 4:], 0.0))
        self.assertFalse(np.allclose(clean["x"][:, 4:], 0.0))
        # resampled keeps the peripheral block non-zero and in-distribution.
        self.assertFalse(np.allclose(resampled["x"][:, 4:], 0.0))

    def test_target_variance_roughly_constant_in_f(self) -> None:
        # y variance should stay ~1 + sigma^2 regardless of f, by construction.
        inst = make_instrument(seed=0)
        v_low = generate(inst, 4000, 0.1, "clean", seed=2)["y"].var()
        v_high = generate(inst, 4000, 0.8, "clean", seed=2)["y"].var()
        self.assertAlmostEqual(v_low, v_high, delta=0.1)

    def test_zeroed_more_detectable_than_resampled(self) -> None:
        # The central claim: at matched f, input-visible (zeroed) corruption is
        # detectable while in-distribution (resampled) corruption is not.
        inst = make_instrument(seed=0)
        cfg = argparse.Namespace(
            n_train=800, n_probe=200, n_test=300, hidden=32, epochs=60,
        )
        zeroed = run_condition(inst, f=0.5, corruption="zeroed", cfg=cfg, seed=0)
        resampled = run_condition(inst, f=0.5, corruption="resampled", cfg=cfg, seed=0)
        self.assertGreater(zeroed["detection_auroc"], 0.75)
        self.assertLess(resampled["detection_auroc"], 0.6)
        # And gating benefit follows detectability, not error magnitude.
        self.assertGreater(zeroed["selective_gain"], 0.3)


class ToyGeometryTests(unittest.TestCase):
    def _separable(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        pos = rng.normal(loc=2.0, size=(80, 5))
        neg = rng.normal(loc=-2.0, size=(80, 5))
        feats = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones(80, int), np.zeros(80, int)])
        return feats, labels

    def test_mean_difference_direction_is_unit(self) -> None:
        feats, labels = self._separable()
        d = mean_difference_direction(feats, labels)
        self.assertAlmostEqual(float(np.linalg.norm(d)), 1.0, places=6)

    def test_auroc_along_separates(self) -> None:
        feats, labels = self._separable()
        d = mean_difference_direction(feats, labels)
        self.assertGreater(auroc_along(feats, labels, d), 0.95)

    def test_inlp_erases_a_rank1_signal(self) -> None:
        # A signal that lives in exactly one direction should be erased by one
        # INLP removal (AUROC drops from high to ~chance).
        feats, labels = self._separable()
        curve = inlp_rank_curve(feats, labels, max_dirs=3, seed=0)
        self.assertGreater(curve[0], 0.9)
        self.assertLess(curve[1], 0.7)


if __name__ == "__main__":
    unittest.main()
