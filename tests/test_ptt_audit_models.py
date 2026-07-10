"""Tests for the two-architecture PTT audit (cross-attention vs self-attention)."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import torch

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from ptt_audit_models import (  # noqa: E402
    CrossAttentionBP,
    SelfAttentionPoolBP,
    aggregate,
)
from ptt_gate import WIN, generate  # noqa: E402


class ArchitectureTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        prox = torch.randn(4, WIN); dist = torch.randn(4, WIN); cal = torch.randn(4, 3)
        for M in (CrossAttentionBP, SelfAttentionPoolBP):
            out, feat = M()(prox, dist, cal, return_features=True)
            self.assertEqual(out.shape, (4,))
            self.assertEqual(feat.shape[0], 4)

    def test_self_attention_pool_is_shift_invariant(self) -> None:
        # The pooled per-stream representation should barely change when the
        # distal stream is rolled (a lag) -- that is exactly why it cannot
        # encode PTT.
        torch.manual_seed(0)
        m = SelfAttentionPoolBP().eval()
        prox = torch.randn(1, WIN); dist = torch.randn(1, WIN); cal = torch.randn(1, 3)
        with torch.no_grad():
            _, f0 = m(prox, dist, cal, return_features=True)
            _, f1 = m(prox, torch.roll(dist, 7, dims=1), cal, return_features=True)
        # distal half of the concatenated feature is ~unchanged by the roll
        w = f0.shape[1] // 2
        rel = (f1[:, w:] - f0[:, w:]).abs().mean() / (f0[:, w:].abs().mean() + 1e-6)
        self.assertLess(float(rel), 0.15)

    def test_audit_separates_the_two_architectures(self) -> None:
        cfg = argparse.Namespace(n_train=1000, n_probe=300, n_test=400, epochs=120, seeds=[0])
        agg = aggregate(cfg)
        self.assertGreater(agg["cross_attention"]["ptt_r2_mean"], 0.6)
        self.assertLess(agg["self_attention_pool"]["ptt_r2_mean"], 0.4)


if __name__ == "__main__":
    unittest.main()
