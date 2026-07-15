"""Tests for the PTT decodability gates (model auditing)."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from ptt_gate import CONDITIONS, PTT_HI, PTT_LO, _xcorr_lag, aggregate, generate  # noqa: E402


class PttGateTests(unittest.TestCase):
    def test_generate_shapes_and_flags(self) -> None:
        for cond in CONDITIONS:
            d = generate(30, cond, seed=0)
            self.assertEqual(d["x"].shape[1], 128 + 128 + 3)   # prox + dist + cal
            expected = 1 if cond == "clean" else 0
            self.assertTrue((d["answerable"] == expected).all())
        # missing distal really zeroes the distal stream
        self.assertTrue(np.allclose(generate(20, "missing_distal", 0)["x"][:, 128:256], 0.0))

    def test_measured_ptt_tracks_true_lag_on_clean(self) -> None:
        # Cross-correlation of the raw waveforms recovers the true lag.
        d = generate(300, "clean", seed=1)
        corr = np.corrcoef(d["ptt_measured"], d["ptt_true"])[0, 1]
        self.assertGreater(corr, 0.9)

    def test_wrong_calibration_leaves_waveform_ptt_intact(self) -> None:
        # The calibration corruption must NOT change the signal-measured PTT
        # (that is why it is the floor for a PTT gate).
        clean = generate(300, "clean", seed=2)
        wrong = generate(300, "wrong_calibration", seed=2)
        # same seed -> same waveforms; only calibration columns differ
        np.testing.assert_allclose(clean["x"][:, :256], wrong["x"][:, :256], atol=1e-5)
        self.assertFalse(np.allclose(clean["x"][:, 256:], wrong["x"][:, 256:]))

    def test_xcorr_undefined_without_distal(self) -> None:
        prox = np.sin(np.linspace(0, 6, 128)).astype(np.float32)
        self.assertEqual(_xcorr_lag(prox, np.zeros(128, np.float32)), -1.0)

    def test_consistency_gate_catches_ptt_failures_not_calibration(self) -> None:
        # End to end: the consistency gate should catch missing-distal well
        # above chance and wrong-calibration near chance.
        cfg = argparse.Namespace(seeds=[0], n_train=800, n_probe=300, n_test=400, epochs=120)
        agg = aggregate(cfg)
        self.assertGreater(agg["ptt_r2"], 0.8)                       # PTT is decodable
        self.assertGreater(agg["detection"]["missing_distal"]["consistency"]["mean"], 0.9)
        self.assertLess(agg["detection"]["wrong_calibration"]["consistency"]["mean"], 0.65)


if __name__ == "__main__":
    unittest.main()
