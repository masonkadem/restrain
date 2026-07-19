"""Numerical confirmation of the transfer-function implementation (run this first).

Four independent checks, each with a hand-derived expected value:
  1. DC gain is unity.
  2. Numerical group delay at low f matches the closed form T(1-gamma)/(1+gamma).
  3. |H| resonance peak sits at the quarter-wave frequency f = 1/(4T).
  4. gamma=0 recovers T (foot-to-foot AND cross-correlation), and foot-to-foot
     decreases monotonically as gamma rises (the identifiability collapse).
"""

import numpy as np

from bpf.simulator import (
    apply_tube_load, foot_to_foot_delay, group_delay_dc, make_proximal_waveform,
    transfer_function, xcorr_delay,
)

fs, hr, n_beats, T = 500.0, 60.0, 12, 0.15
gammas = [0.0, 0.2, 0.4, 0.6, 0.8]

print("=== Check 1: DC gain H(0) = 1 ===")
for g in gammas:
    h0 = transfer_function(np.array([0.0]), T, g)[0]
    print(f"  gamma={g:.1f}  |H(0)|={abs(h0):.6f}")

print("\n=== Check 2: low-f group delay vs closed form T(1-g)/(1+g) ===")
f = np.linspace(0, 2.0, 200001)                     # fine grid, DC to 2 Hz
print(f"  {'gamma':>6}{'numeric (ms)':>14}{'closed form (ms)':>18}{'abs err (ms)':>14}")
for g in gammas:
    H = transfer_function(f, T, g)
    phase = np.unwrap(np.angle(H))
    tau = -np.gradient(phase, f) / (2 * np.pi)      # group delay tau_g(f)
    num = tau[1]                                     # near f=0 (skip exact DC)
    cf = group_delay_dc(T, g)
    print(f"  {g:>6.1f}{num*1e3:>14.3f}{cf*1e3:>18.3f}{abs(num-cf)*1e3:>14.4f}")

print("\n=== Check 3: |H| resonance at quarter-wave f = 1/(4T) ===")
fq = 1.0 / (4 * T)                                   # first resonance (odd multiples repeat)
fgrid = np.linspace(0.05, 2 * fq, 60000)            # window contains only the first peak
for g in [0.4, 0.6, 0.8]:
    mag = np.abs(transfer_function(fgrid, T, g))
    fpk = fgrid[int(np.argmax(mag))]
    peak_pred = (1 + g) / (1 - g)                   # analytic peak gain at resonance
    print(f"  gamma={g:.1f}  peak f={fpk:.3f} Hz (expected {fq:.3f})  "
          f"peak|H|={mag.max():.3f} (expected {peak_pred:.3f})")

print("\n=== Check 4: gamma=0 recovery + monotone collapse ===")
_, p = make_proximal_waveform(fs, hr, n_beats)
print(f"  {'gamma':>6}{'foot-to-foot (ms)':>20}{'xcorr (ms)':>13}{'T(1-g)/(1+g) (ms)':>20}")
prev = np.inf
for g in gammas:
    d = apply_tube_load(p, fs, T, g)
    ftf = foot_to_foot_delay(p, d, fs, hr, n_beats) * 1e3
    xc = xcorr_delay(p, d, fs, hr) * 1e3
    cf = group_delay_dc(T, g) * 1e3
    flag = "" if ftf <= prev + 1e-6 else "  <-- NOT monotone!"
    print(f"  {g:>6.1f}{ftf:>20.3f}{xc:>13.3f}{cf:>20.3f}{flag}")
    prev = ftf

print("\n=== loud sanity check ===")
from bpf.simulator import sanity_check_gamma0
sanity_check_gamma0(fs, hr, n_beats, T)
print("  PASS: gamma=0 foot-to-foot recovers T within tolerance.")
