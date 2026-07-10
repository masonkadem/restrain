"""Tests for the shortcut-learner demo (right for the wrong reason)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from ptt_gate import WIN, generate  # noqa: E402
from ptt_shortcut import _shift_distal, amplitude_normalize  # noqa: E402


class ShortcutDataTests(unittest.TestCase):
    def test_amp_confound_makes_amplitude_predict_bp(self) -> None:
        # With the confound on, waveform amplitude correlates with BP; with it
        # decorrelated, that correlation vanishes -- the whole basis of the demo.
        corr = generate(2000, "clean", seed=0, amp_gain=0.8)
        amp = corr["x"][:, :WIN].std(axis=1)
        self.assertGreater(abs(np.corrcoef(amp, corr["y"])[0, 1]), 0.6)

        dec = generate(2000, "clean", seed=0, amp_gain=0.8, amp_decorrelate=True)
        amp_d = dec["x"][:, :WIN].std(axis=1)
        self.assertLess(abs(np.corrcoef(amp_d, dec["y"])[0, 1]), 0.2)

    def test_amplitude_normalize_removes_the_cue(self) -> None:
        d = generate(500, "clean", seed=1, amp_gain=0.8)
        n = amplitude_normalize(d)
        amp = n["x"][:, :WIN].std(axis=1)
        # after per-stream std normalization every waveform has unit amplitude,
        # so amplitude no longer varies and cannot carry BP (a residual
        # correlation may survive at the 1e-6 level but is functionally zero).
        self.assertLess(float(amp.std()), 1e-3)

    def test_shift_distal_changes_only_the_distal_stream(self) -> None:
        d = generate(50, "clean", seed=2, amp_gain=0.8)
        s = _shift_distal(d, 3)
        np.testing.assert_allclose(d["x"][:, :WIN], s["x"][:, :WIN])          # proximal unchanged
        np.testing.assert_allclose(d["x"][:, 2 * WIN:], s["x"][:, 2 * WIN:])  # calibration unchanged
        self.assertFalse(np.allclose(d["x"][:, WIN:2 * WIN], s["x"][:, WIN:2 * WIN]))

    def test_default_generate_unchanged_without_confound(self) -> None:
        # amp_gain=0 must leave the base task identical (no silent side effects).
        a = generate(100, "clean", seed=3)
        b = generate(100, "clean", seed=3, amp_gain=0.0)
        np.testing.assert_allclose(a["x"], b["x"])


if __name__ == "__main__":
    unittest.main()
