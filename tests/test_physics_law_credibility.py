"""Tests for physics-law credibility simulators and audit invariants."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from physics_law_credibility import (  # noqa: E402
    BEER_SCENARIOS_TEST,
    MK_FS,
    MK_SCENARIOS_TEST,
    LawAuditConfig,
    _moens_korteweg_pwv,
    build_beer_dataset,
    build_mk_dataset,
    empirical_clean_confidence,
    export_summary,
    generate_beer_lambert_sample,
    generate_moens_korteweg_sample,
    intervene_activation,
    paired_significance,
    probe_direction,
    random_direction,
    run_beer_audit,
    run_mk_audit,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class PhysicsLawCredibilityTests(unittest.TestCase):
    def test_beer_missing_wavelength_is_unanswerable(self) -> None:
        rng = np.random.default_rng(0)
        clean = generate_beer_lambert_sample(rng, "clean")
        missing = generate_beer_lambert_sample(rng, "missing_ir")
        self.assertTrue(clean.answerable)
        self.assertFalse(missing.answerable)

    def test_beer_ratio_monotonic_with_spo2_trend(self) -> None:
        rng = np.random.default_rng(1)
        high = generate_beer_lambert_sample(rng, "clean")
        rng2 = np.random.default_rng(1)
        high.spo2 = 98.0
        low = generate_beer_lambert_sample(rng2, "clean")
        self.assertGreater(high.ratio_r, 0)
        self.assertGreater(low.ratio_r, 0)

    def test_beer_ratio_carries_spo2_signal(self) -> None:
        # Regression test for the exponent-underflow bug: mu*path_length
        # used to reach ~20-30, driving exp(-(dc+ac)) below the additive
        # motion-noise floor so ratio_R was statistically independent of
        # SpO2 (|corr| ~ 0.03). Guard against that silently coming back.
        rng = np.random.default_rng(0)
        samples = [generate_beer_lambert_sample(rng, "clean") for _ in range(500)]
        ratio_r = np.array([s.ratio_r for s in samples])
        spo2 = np.array([s.spo2 for s in samples])
        corr = np.corrcoef(ratio_r, spo2)[0, 1]
        self.assertLess(abs(corr), 1.0)
        self.assertGreater(abs(corr), 0.5)

    def test_mk_shift_resolves_ptt_range(self) -> None:
        # Regression test for the FS=25Hz bug: at a 40ms sample period, the
        # whole 5-43ms physiological PTT range rounded to exactly 1 sample
        # for every example, so the distal waveform's time-shift carried no
        # BP information regardless of the true value.
        rng = np.random.default_rng(0)
        samples = [generate_moens_korteweg_sample(rng, "clean") for _ in range(200)]
        shift_samples = np.array(
            [round(s.ptt_ms * MK_FS / 1000.0) for s in samples]
        )
        self.assertGreater(len(np.unique(shift_samples)), 5)

    def test_mk_pwv_units_and_ptt_ms(self) -> None:
        pwv = _moens_korteweg_pwv(1.0e6, 0.0005, 1060.0, 0.004)
        self.assertGreater(pwv, 0)
        ptt_ms = 1000.0 * 0.35 / pwv
        self.assertGreater(ptt_ms, 0)

    def test_mk_missing_calibration_unanswerable(self) -> None:
        rng = np.random.default_rng(2)
        sample = generate_moens_korteweg_sample(rng, "missing_stiffness_cal")
        self.assertFalse(sample.answerable)

    def test_mk_calibration_vector_matches_answerability(self) -> None:
        rng = np.random.default_rng(3)
        clean = generate_moens_korteweg_sample(rng, "clean")
        missing_path = generate_moens_korteweg_sample(rng, "missing_path_length")
        missing_cal = generate_moens_korteweg_sample(rng, "missing_stiffness_cal")
        self.assertTrue(clean.answerable)
        self.assertTrue(np.any(clean.calibration != 0))
        self.assertEqual(clean.calibration[-1], 1.0)
        for sample in (missing_path, missing_cal):
            self.assertFalse(sample.answerable)
            np.testing.assert_array_equal(sample.calibration, np.zeros(4, dtype=np.float32))

    def test_mk_calibration_recovers_bp_given_true_lag(self) -> None:
        # Given a perfect PTT read-out (from classical cross-correlation, not
        # the network) plus the calibration vector, BP should be recoverable
        # via the closed-form inversion this benchmark's identifiability
        # claim rests on: PWV^2 = K*exp(alpha*BP/100).
        rng = np.random.default_rng(4)
        errors = []
        for _ in range(50):
            s = generate_moens_korteweg_sample(rng, "clean")
            path_m, k_scaled, alpha, _avail = s.calibration
            pwv = path_m / (s.ptt_ms / 1000.0)
            bp_pred = (100.0 / alpha) * np.log((pwv ** 2) / (k_scaled * 100.0))
            errors.append(abs(bp_pred - s.bp))
        self.assertLess(np.mean(errors), 1e-2)

    def test_matched_scenario_counts(self) -> None:
        beer = build_beer_dataset(5, BEER_SCENARIOS_TEST, seed=0)
        counts = {}
        for s in beer:
            counts[s.scenario] = counts.get(s.scenario, 0) + 1
        self.assertTrue(all(c == 5 for c in counts.values()))

    def test_empirical_confidence_bounds(self) -> None:
        ref = np.array([-1.0, 0.0, 1.0])
        conf = empirical_clean_confidence(ref, np.array([-2.0, 0.5, 3.0]))
        np.testing.assert_allclose(conf, [0.0, 2 / 3, 1.0])

    def test_intervention_directions(self) -> None:
        act = np.random.randn(10, 4).astype(np.float32)
        direction = np.array([1.0, 0, 0, 0], dtype=np.float32)
        out = intervene_activation(act, direction)
        self.assertLess(out[:, 0].mean(), act[:, 0].mean())

    def test_probe_direction_normalized(self) -> None:
        scaler = StandardScaler().fit(np.random.randn(20, 3))
        ridge = Ridge().fit(scaler.transform(np.random.randn(20, 3)), np.random.randn(20))
        d = probe_direction(ridge, scaler, 3)
        self.assertAlmostEqual(np.linalg.norm(d), 1.0, places=5)

    def test_random_direction_matches_norm(self) -> None:
        d = random_direction(8, 2.5, seed=0)
        self.assertAlmostEqual(np.linalg.norm(d), 2.5, places=5)

    def test_quick_audit_runs(self) -> None:
        import torch

        cfg = LawAuditConfig(
            n_train=32, n_probe=16, n_val=16, n_test=8,
            epochs=2, n_bootstrap=3, seed=0,
        )
        report = run_beer_audit(cfg, torch.device("cpu"))
        self.assertEqual(report["law"], "beer_lambert")
        self.assertIn("cross_attention", report["models"])

    def test_paired_significance_detects_consistent_gap(self) -> None:
        rng = np.random.default_rng(0)
        a = 1.0 + rng.normal(0, 0.05, 10)
        b = 1.5 + rng.normal(0, 0.05, 10)
        result = paired_significance(a, b, seed=0)
        self.assertTrue(result["significant"])
        self.assertLess(result["ci_high"], 0)
        self.assertLess(result["wilcoxon_p"], 0.05)

    def test_paired_significance_rejects_noise(self) -> None:
        rng = np.random.default_rng(1)
        a = 1.0 + rng.normal(0, 1.0, 6)
        b = 1.0 + rng.normal(0, 1.0, 6)
        result = paired_significance(a, b, seed=0)
        self.assertFalse(result["significant"])

    def test_export_summary_schema(self) -> None:
        import torch

        cfg = LawAuditConfig(
            n_train=24, n_probe=12, n_val=12, n_test=8,
            epochs=2, n_bootstrap=3, seed=0,
        )
        reports = [
            run_beer_audit(cfg, torch.device("cpu")),
            run_mk_audit(cfg, torch.device("cpu")),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = export_summary(reports, Path(tmp), {"quick": True})
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("aggregate", data)
        self.assertIn("overall_decision", data)
        self.assertIn("equations", data)


if __name__ == "__main__":
    unittest.main()
