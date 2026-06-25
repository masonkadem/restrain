"""
Fast architecture sanity check — no data, no GPU, ~10s on CPU.
Tests every model does a forward pass and produces (B, 2) output.
Tests feature extraction produces the right shape.
"""
import sys, math
import numpy as np
import torch

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
    BPNoiseRobustTransformer, BPS4, BPS4CrossChannel,
    extract_features, _feature_names,
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

def fwd(model, x):
    out = model(x)
    assert out.shape == (B, 2), f"expected ({B},2) got {out.shape}"

# ── transformer ───────────────────────────────────────────────────────────────
check("BPTransformer(1ch)",
      lambda: fwd(BPTransformer(1, D, HEADS, LAYERS), torch.randn(B, SEQ, 1)))
check("BPTransformer(3ch)",
      lambda: fwd(BPTransformer(3, D, HEADS, LAYERS), torch.randn(B, SEQ, 3)))

# ── dual / tri stream (sinusoidal PE, no seq_len arg) ─────────────────────────
check("BPDualStreamTransformer",
      lambda: fwd(BPDualStreamTransformer(D, HEADS, LAYERS), torch.randn(B, SEQ, 2)))
check("BPTriStreamTransformer",
      lambda: fwd(BPTriStreamTransformer(D, HEADS, LAYERS), torch.randn(B, SEQ, 3)))

# ── noise-robust (train mode → noisy path active; eval mode → no noise) ───────
def _noise_robust_train():
    m = BPNoiseRobustTransformer(2, D, HEADS, LAYERS); m.train()
    fwd(m, torch.randn(B, SEQ, 2))
def _noise_robust_eval():
    m = BPNoiseRobustTransformer(2, D, HEADS, LAYERS); m.eval()
    fwd(m, torch.randn(B, SEQ, 2))
check("BPNoiseRobustTransformer(train)", _noise_robust_train)
check("BPNoiseRobustTransformer(eval)",  _noise_robust_eval)

# ── S4 variants ───────────────────────────────────────────────────────────────
check("BPS4(1ch)",
      lambda: fwd(BPS4(1, D, 16, LAYERS), torch.randn(B, SEQ, 1)))
check("BPS4(3ch)",
      lambda: fwd(BPS4(3, D, 16, LAYERS), torch.randn(B, SEQ, 3)))
check("BPS4CrossChannel(2ch)",
      lambda: fwd(BPS4CrossChannel(2, D, 16, LAYERS, HEADS), torch.randn(B, SEQ, 2)))
check("BPS4CrossChannel(3ch)",
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
