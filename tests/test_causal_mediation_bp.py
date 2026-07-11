"""Deterministic smoke tests for the causal-mediation BP audit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from physics_law_credibility import (
    LINEAR_BP_A,
    CausalBPConfig,
    TwoStreamCrossAttention,
    WIN_LEN,
    build_linear_bp_dataset,
    generate_linear_bp_sample,
    linear_bp_to_tensors,
    measure_causal_fidelity,
    run_causal_bp_audit,
    train_regressor,
)


class CausalMediationBPTests(unittest.TestCase):
    def test_confound_correlation_regimes(self) -> None:
        """Confound tracks PTT in train, is independent under OOD."""
        rng = np.random.default_rng(0)
        train = [
            generate_linear_bp_sample(rng, "train", confound_corr=0.99)
            for _ in range(400)
        ]
        ptt = np.array([s.ptt_ms for s in train])
        z = np.array([s.confound for s in train])
        self.assertGreater(abs(np.corrcoef(ptt, z)[0, 1]), 0.8)

        rng = np.random.default_rng(1)
        ood = [
            generate_linear_bp_sample(rng, "ood", confound_corr=0.99)
            for _ in range(400)
        ]
        ptt_o = np.array([s.ptt_ms for s in ood])
        z_o = np.array([s.confound for s in ood])
        self.assertLess(abs(np.corrcoef(ptt_o, z_o)[0, 1]), 0.2)

    def test_bp_follows_linear_law(self) -> None:
        rng = np.random.default_rng(2)
        samples = [
            generate_linear_bp_sample(rng, "train", confound_corr=0.5)
            for _ in range(600)
        ]
        ptt = np.array([s.ptt_ms for s in samples])
        bp = np.array([s.bp for s in samples])
        slope = np.polyfit(ptt, bp, 1)[0]
        self.assertAlmostEqual(slope, LINEAR_BP_A, delta=0.05)

    def test_tensor_shapes(self) -> None:
        data = linear_bp_to_tensors(
            build_linear_bp_dataset(10, "train", 0.5, seed=3)
        )
        self.assertEqual(data["stream_a"].shape, (10, WIN_LEN))
        self.assertEqual(data["stream_b"].shape, (10, WIN_LEN))
        self.assertEqual(data["target"].shape, (10,))
        self.assertEqual(data["ptt_ms"].shape, (10,))
        self.assertEqual(data["confound"].shape, (10,))

    def test_causal_fidelity_keys_and_random_null(self) -> None:
        """A trained model's random-direction slope should be near zero."""
        torch.manual_seed(0)
        train = linear_bp_to_tensors(
            build_linear_bp_dataset(96, "train", 0.5, seed=4)
        )
        model = TwoStreamCrossAttention(WIN_LEN, width=32)
        train_regressor(model, train, torch.device("cpu"), epochs=30, batch_size=32)
        probe = linear_bp_to_tensors(
            build_linear_bp_dataset(96, "ood", 0.5, seed=5)
        )
        result = measure_causal_fidelity(
            model, probe, torch.device("cpu"), a=LINEAR_BP_A, seed=0
        )
        for key in (
            "ptt_causal_slope",
            "ptt_causal_fidelity",
            "confound_causal_slope",
            "random_direction_slope",
        ):
            self.assertIn(key, result)
        # Random direction carries far less BP sensitivity than the PTT subspace.
        self.assertLess(
            abs(result["random_direction_slope"]),
            abs(result["ptt_causal_slope"]) + 1e-6,
        )

    def test_quick_audit_runs_and_reports_decision(self) -> None:
        cfg = CausalBPConfig(
            n_train=64, n_eval=48, n_ood=48, epochs=5,
            confound_corrs=(0.0, 0.95), seeds=(0,),
        )
        report = run_causal_bp_audit(cfg, torch.device("cpu"))
        self.assertEqual(len(report["records"]), 2)
        self.assertIn("fidelity_predicts_ood_failure", report["decision"])
        for record in report["records"]:
            self.assertIn("ptt_causal_fidelity", record)
            self.assertIn("ood_mae", record)


if __name__ == "__main__":
    unittest.main()
