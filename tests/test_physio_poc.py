"""Tests for the physics-grounded gating proof of concept."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from physio_gating_poc import (  # noqa: E402
    LAWS,
    aggregate,
    generate_beer_lambert,
    generate_moens_korteweg,
)


class PhysioInstrumentTests(unittest.TestCase):
    def test_beer_clean_is_identifiable_by_closed_form(self) -> None:
        # SpO2 = 110 - 25 * ratio_red/ratio_ir must recover the true SpO2 on
        # clean data (the identifiability the benchmark rests on).
        d = generate_beer_lambert(2000, "clean", seed=0)
        ratio_red, ratio_ir = d["x"][:, 0], d["x"][:, 1]
        recovered = 110.0 - 25.0 * ratio_red / ratio_ir
        corr = np.corrcoef(recovered, d["y"])[0, 1]
        self.assertGreater(corr, 0.95)

    def test_beer_red_alone_is_not_identifiable(self) -> None:
        # ratio_red alone should NOT determine SpO2 (perfusion confounds it) --
        # that is why the IR channel is identifiability-critical.
        d = generate_beer_lambert(2000, "clean", seed=0)
        corr = abs(np.corrcoef(d["x"][:, 0], d["y"])[0, 1])
        self.assertLess(corr, 0.6)

    def test_beer_corruptions_flag_unanswerable(self) -> None:
        self.assertTrue((generate_beer_lambert(50, "clean", 0)["answerable"] == 1).all())
        for corr in ("missing_ir", "saturated_ir", "mismatched_ir"):
            d = generate_beer_lambert(50, corr, 0)
            self.assertTrue((d["answerable"] == 0).all())
        # missing_ir zeroes the IR column; mismatched keeps it in-distribution.
        self.assertTrue(np.allclose(generate_beer_lambert(50, "missing_ir", 0)["x"][:, 1], 0.0))
        self.assertFalse(np.allclose(generate_beer_lambert(50, "mismatched_ir", 0)["x"][:, 1], 0.0))

    def test_moens_clean_is_identifiable(self) -> None:
        # BP recoverable from PTT + calibration via the closed-form inverse.
        d = generate_moens_korteweg(2000, "clean", seed=0)
        ptt, K, alpha, L = d["x"][:, 0], d["x"][:, 1], d["x"][:, 2], d["x"][:, 3]
        pwv = L / ptt
        bp = (100.0 / alpha) * np.log((pwv ** 2) / K)
        corr = np.corrcoef(bp, d["y"])[0, 1]
        self.assertGreater(corr, 0.95)

    def test_moens_ptt_alone_is_not_identifiable(self) -> None:
        d = generate_moens_korteweg(2000, "clean", seed=0)
        corr = abs(np.corrcoef(d["x"][:, 0], d["y"])[0, 1])
        self.assertLess(corr, 0.6)

    def test_poc_gating_beats_random_on_missing_channel(self) -> None:
        # End-to-end: on the clean+missing mix, probe AURC < random AURC
        # (gating on activations reduces selective risk).
        cfg = argparse.Namespace(
            n_ensemble=3, n_train=800, n_probe=200, n_test=300, hidden=32, epochs=60, seeds=[0],
        )
        meta = LAWS["beer_lambert"]
        agg = aggregate(meta["gen"], meta["corruptions"], cfg)
        rc = agg["risk_curves_seed0"]["missing_ir"]
        self.assertLess(rc["aurc_probe"], rc["aurc_random"])
        # and the fully-missing channel is detected near-perfectly
        self.assertGreater(agg["detection"]["missing_ir"]["probe"]["mean"], 0.9)


if __name__ == "__main__":
    unittest.main()
