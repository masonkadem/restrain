"""
Fast architecture sanity check — no data, no GPU, ~10s on CPU.
Tests every model does a forward pass and produces (B, 2) output.
Tests feature extraction produces the right shape.
"""
import sys, math
import numpy as np
import torch
import torch.nn as nn

# ── tiny dims so CPU finishes fast ────────────────────────────────────────────
SEQ   = 64
B     = 2
D     = 32
HEADS = 4
LAYERS= 2
FS    = 125

sys.path.insert(0, ".")
from train import (
    BPTransformer, BPDualStreamTransformer, BPTriStreamTransformer,
    BPS4, BPS4CrossChannel, extract_features, _feature_names,
)

PASS = []; FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")

# ── models ────────────────────────────────────────────────────────────────────
def fwd(model, x):
    out = model(x)
    assert out.shape == (B, 2), f"expected ({B},2) got {out.shape}"

check("BPTransformer(ppg)",
      lambda: fwd(BPTransformer(1, D, HEADS, LAYERS, SEQ), torch.randn(B, SEQ, 1)))
check("BPTransformer(ppg+ecg+resp)",
      lambda: fwd(BPTransformer(3, D, HEADS, LAYERS, SEQ), torch.randn(B, SEQ, 3)))
check("BPDualStreamTransformer",
      lambda: fwd(BPDualStreamTransformer(D, HEADS, LAYERS, SEQ), torch.randn(B, SEQ, 2)))
check("BPTriStreamTransformer",
      lambda: fwd(BPTriStreamTransformer(D, HEADS, LAYERS, SEQ), torch.randn(B, SEQ, 3)))
check("BPS4(ppg)",
      lambda: fwd(BPS4(1, D, 32, LAYERS), torch.randn(B, SEQ, 1)))
check("BPS4(ppg+ecg+resp)",
      lambda: fwd(BPS4(3, D, 32, LAYERS), torch.randn(B, SEQ, 3)))
check("BPS4CrossChannel(ppg+ecg)",
      lambda: fwd(BPS4CrossChannel(2, D, 16, LAYERS, HEADS), torch.randn(B, SEQ, 2)))
check("BPS4CrossChannel(ppg+ecg+resp)",
      lambda: fwd(BPS4CrossChannel(3, D, 16, LAYERS, HEADS), torch.randn(B, SEQ, 3)))

# ── feature extraction ────────────────────────────────────────────────────────
for chs in [["ppg"], ["ppg","ecg"], ["ppg","resp"], ["ppg","ecg","resp"]]:
    def _check(chs=chs):
        X = np.random.randn(B, 1000, len(chs)).astype(np.float32)
        F = extract_features(X, fs=FS, channels=chs)
        names = _feature_names(chs)
        assert F.shape == (B, len(names)), f"expected ({B},{len(names)}) got {F.shape}"
    check(f"features({'+'.join(chs)})", _check)

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
