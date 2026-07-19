"""Unit tests for the tube-load simulator (Part 1). Run: python -m unittest -v."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bpf.simulator import (apply_tube_load, foot_to_foot_delay, group_delay_dc,
                           make_proximal_waveform, sanity_check_gamma0, transfer_function)

FS, HR, NB, T = 500.0, 60.0, 12, 0.15


class TestSimulator(unittest.TestCase):
    def test_dc_gain_unity(self):
        for g in (0.0, 0.3, 0.6, 0.8):
            self.assertAlmostEqual(abs(transfer_function(np.array([0.0]), T, g)[0]), 1.0, places=9)

    def test_group_delay_matches_closed_form(self):
        f = np.linspace(0, 2.0, 200001)
        for g in (0.0, 0.2, 0.4, 0.6, 0.8):
            phase = np.unwrap(np.angle(transfer_function(f, T, g)))
            tau = -np.gradient(phase, f)[1] / (2 * np.pi)     # near DC
            self.assertAlmostEqual(tau, group_delay_dc(T, g), places=4)

    def test_resonance_frequency_and_gain(self):
        fq = 1.0 / (4 * T)
        fgrid = np.linspace(0.05, 2 * fq, 60000)
        for g in (0.4, 0.6, 0.8):
            mag = np.abs(transfer_function(fgrid, T, g))
            self.assertAlmostEqual(fgrid[int(np.argmax(mag))], fq, delta=0.02)
            self.assertAlmostEqual(mag.max(), (1 + g) / (1 - g), delta=0.02)

    def test_gamma0_recovers_T(self):
        # loud sanity check must pass; and distal is a pure delay by T
        sanity_check_gamma0(FS, HR, NB, T)
        _, p = make_proximal_waveform(FS, HR, NB)
        d = apply_tube_load(p, FS, T, 0.0)
        shift = int(round(T * FS))
        self.assertLess(np.max(np.abs(d - np.roll(p, shift))), 1e-9)
        self.assertAlmostEqual(foot_to_foot_delay(p, d, FS, HR, NB), T, delta=0.003)

    def test_reflection_bounded(self):
        # gamma < 1 keeps H bounded (denominator never zero)
        f = np.linspace(0, 50, 5000)
        self.assertTrue(np.all(np.isfinite(np.abs(transfer_function(f, T, 0.8)))))


if __name__ == "__main__":
    unittest.main()
