"""Tube-load arterial pulse simulator (Part 1).

Lossless uniform tube with a terminal reflection, implemented as a discrete
frequency-domain filter. No PDE solver, no symbolic math. References:
Mukkamala et al., IEEE TBME 2015 (Fig 5b); Rashedi et al., J Biomech Eng 2013.

------------------------------------------------------------------------------
The model (frequency domain), with a REAL scalar reflection coefficient Gamma:

        1 + Gamma
H(f) = ----------- * exp(-j 2 pi f T)     -- forward wave, delayed by one-way T
        1 + Gamma * exp(-j 4 pi f T)      -- denominator: infinite re-reflection
                                             sum (round trip = 2T -> phase 4 pi f T)

  T     : true one-way transit time [s].  GROUND TRUTH; never shown to the model.
  Gamma : reflection coefficient in [0, 0.8].  THE IDENTIFIABILITY KNOB.
          Gamma = 0  -> H(f) = exp(-j 2 pi f T): a pure delay by T.
                        Foot-to-foot delay recovers T exactly. PTT recoverable.
          Gamma -> 1 -> the reflected wave dominates; the apparent (low-frequency)
                        delay collapses toward 0 and PTT is unrecoverable.

Two facts we verify numerically (see verify_physics.py), both derived by hand
in the comments so nothing is a black box:

  (1) DC gain is unity:            H(0) = (1+Gamma)/(1+Gamma) = 1.
  (2) Low-frequency group delay:   tau_g(0) = T * (1 - Gamma) / (1 + Gamma).
      Derivation in group_delay_dc() below. This closed form IS the
      identifiability statement: the observable delay shrinks with Gamma.

A full three-element Windkessel load would make Gamma frequency dependent,
Gamma(f) = (Z_L(f) - Z_c)/(Z_L(f) + Z_c). We use a real scalar Gamma as the spec
directs. The single place to swap in Gamma(f) is marked TODO in
transfer_function().
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Proximal (input) waveform
# --------------------------------------------------------------------------- #
def make_proximal_waveform(fs: float, hr_bpm: float, n_beats: int,
                           dbp: float = 80.0, pulse_pressure: float = 40.0):
    """A physiologically plausible periodic proximal pressure waveform.

    Design choice (documented): each beat is a *systolic* Gaussian followed by a
    smaller *dicrotic* Gaussian, riding on a diastolic exponential decay. This
    gives (a) a sharp upstroke whose onset ("foot") is well defined via the peak
    of the second derivative, and (b) smoothness, so the fractional delay applied
    in the frequency domain does not ring. Amplitude is scaled to a realistic
    [dbp, dbp + pulse_pressure] range purely for display; the filter is linear so
    absolute scale does not matter.

    Returns (t, p): time vector [s] and pressure [mmHg], length n_beats*Tb*fs.
    """
    Tb = 60.0 / hr_bpm                      # beat period [s]
    N = int(round(n_beats * Tb * fs))
    t = np.arange(N) / fs
    ph = (t % Tb) / Tb                      # phase within beat, in [0, 1)

    # Two Gaussians only (systolic + dicrotic). Both decay to ~0 well before the
    # beat boundary, so the waveform is smooth AND periodic (value ~equal at
    # phase 0 and 1). No baseline discontinuity -> no spurious curvature at the
    # wrap, and no ringing when we apply a fractional delay in the frequency
    # domain. The foot is then the systolic upstroke onset.
    systolic = np.exp(-((ph - 0.15) ** 2) / (2 * 0.045 ** 2))
    dicrotic = 0.30 * np.exp(-((ph - 0.35) ** 2) / (2 * 0.040 ** 2))
    shape = systolic + dicrotic

    shape = shape / shape.max()             # normalize to ~[0, 1]
    p = dbp + pulse_pressure * shape
    return t, p


# --------------------------------------------------------------------------- #
# 2. Transfer function and its application
# --------------------------------------------------------------------------- #
def transfer_function(f: np.ndarray, T: float, gamma: float) -> np.ndarray:
    """H(f) for the tube-load model. f in Hz (array), T in s, gamma scalar in [0,1)."""
    num = (1.0 + gamma) * np.exp(-1j * 2 * np.pi * f * T)
    den = 1.0 + gamma * np.exp(-1j * 4 * np.pi * f * T)
    # TODO(frequency-dependent load): replace the scalar `gamma` with
    #   gamma_f = (Z_L(f) - Z_c) / (Z_L(f) + Z_c)
    # where Z_L(f) is a 3-element Windkessel. Everything else stays the same.
    return num / den


def apply_tube_load(p: np.ndarray, fs: float, T: float, gamma: float) -> np.ndarray:
    """Apply H(f) to a periodic proximal waveform via the real FFT.

    For a periodic steady-state waveform, multiplying the spectrum by H(f) is the
    exact discrete realization of the tube-load filter: the circular convolution
    reproduces the periodic train of reflections. gamma < 1 keeps the denominator
    bounded away from zero (|gamma * e^{...}| = gamma < 1), so H is stable.
    """
    N = len(p)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)          # nonneg frequencies of the rFFT
    H = transfer_function(freqs, T, gamma)
    return np.fft.irfft(np.fft.rfft(p) * H, n=N)


def group_delay_dc(T: float, gamma: float) -> float:
    """Closed-form low-frequency group delay tau_g(0) = T (1 - gamma)/(1 + gamma).

    Derivation (all by hand, no symbolic math):
      phase(f) = -2 pi f T - arg(1 + gamma e^{-j theta}),   theta = 4 pi f T
      Let g(theta) = arg(1 + gamma e^{-j theta})
                   = atan2(-gamma sin theta, 1 + gamma cos theta).
      dg/dtheta = (-gamma cos theta - gamma^2) / (1 + 2 gamma cos theta + gamma^2).
      At theta = 0: dg/dtheta = -gamma(1+gamma)/(1+gamma)^2 = -gamma/(1+gamma).
      dphase/df   = -2 pi T - (dg/dtheta)(dtheta/df)
                  = -2 pi T + [gamma/(1+gamma)] * 4 pi T.
      tau_g = -1/(2 pi) dphase/df = T - 2 T gamma/(1+gamma)
                                  = T (1 - gamma)/(1 + gamma).
    """
    return T * (1.0 - gamma) / (1.0 + gamma)


# --------------------------------------------------------------------------- #
# 3. Foot detection and foot-to-foot delay (the clinical PTT observable)
# --------------------------------------------------------------------------- #
def detect_feet(sig: np.ndarray, fs: float, hr_bpm: float, n_beats: int,
                search_frac: float = 0.6) -> np.ndarray:
    """Foot = onset of the systolic upstroke, one per beat, times in [s].

    Intersecting-tangents method (the standard clinical foot detector): within
    each beat, (1) find the steepest upstroke u = argmax of the first derivative,
    (2) take the diastolic baseline as the minimum before u, then (3) the foot is
    where the tangent line at u crosses that baseline:

        foot = u + (baseline - sig[u]) / slope(u).

    This is exactly translation-equivariant -- shift the whole signal and every
    term (u, sig[u], slope, baseline) shifts with it -- so at gamma=0 the distal
    foot lands exactly one delay T after the proximal foot. It also degrades
    gracefully as the distal shape changes with gamma, which is the point.
    """
    Tb = 60.0 / hr_bpm
    d1 = np.gradient(sig) * fs                              # slope in units/second
    feet = []
    for k in range(n_beats):
        i0 = int(round(k * Tb * fs))
        i1 = min(int(round((k + search_frac) * Tb * fs)), len(sig) - 1)
        if i1 <= i0 + 4:
            continue
        u = i0 + int(np.argmax(d1[i0:i1]))                 # steepest upstroke
        if u <= i0 + 1:
            continue
        baseline = float(np.min(sig[i0:u + 1]))            # end-diastolic level
        slope = float(d1[u])
        if slope <= 1e-9:
            continue
        foot_samp = u + (baseline - sig[u]) / slope * fs   # tangent crosses baseline
        feet.append(foot_samp / fs)
    return np.asarray(feet)


def foot_to_foot_delay(prox: np.ndarray, dist: np.ndarray, fs: float,
                       hr_bpm: float, n_beats: int) -> float:
    """Median distal-minus-proximal foot delay over interior beats [s].

    This is the quantity a clinician measures as "PTT". At gamma = 0 it equals T;
    as gamma grows it shrinks below T -- the divergence the UI displays.
    """
    fp = detect_feet(prox, fs, hr_bpm, n_beats)
    fd = detect_feet(dist, fs, hr_bpm, n_beats)
    m = min(len(fp), len(fd))
    if m < 3:
        return float("nan")
    d = fd[1:m - 1] - fp[1:m - 1]                          # drop first/last (edge)
    Tb = 60.0 / hr_bpm
    d = np.mod(d, Tb)                                       # wrap into one beat
    return float(np.median(d))


def xcorr_delay(prox: np.ndarray, dist: np.ndarray, fs: float, hr_bpm: float) -> float:
    """Global lag by cross-correlation over lags [0, one beat], sub-sample [s].

    dist[n] ~ prox[n - lag], so correlation sum_n prox[n]*dist[n+lag] peaks at
    lag = delay. Shape-independent; used only to validate the gamma=0 recovery
    (there the whole waveform is a pure shift), not as the clinical observable.
    """
    a = prox - prox.mean()
    b = dist - dist.mean()
    maxlag = int(round((60.0 / hr_bpm) * fs))
    corr = np.array([np.dot(a[: len(a) - L], b[L:]) for L in range(maxlag)])
    k = int(np.argmax(corr))
    if 1 <= k < len(corr) - 1:                                  # parabolic sub-sample
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = (y0 - 2 * y1 + y2)
        k = k + (0.5 * (y0 - y2) / denom if denom != 0 else 0.0)
    return k / fs


# --------------------------------------------------------------------------- #
# 4. Transit time -> blood pressure (Mukkamala eq. 11)
# --------------------------------------------------------------------------- #
def bp_from_T(T: float, K1: float, K2: float) -> float:
    """BP = K1 / T + K2, with subject-specific K1, K2 (never exposed)."""
    return K1 / T + K2


# --------------------------------------------------------------------------- #
# 5. Loud sanity check
# --------------------------------------------------------------------------- #
def sanity_check_gamma0(fs: float = 500.0, hr_bpm: float = 60.0, n_beats: int = 12,
                        T: float = 0.15, tol_ms: float = 3.0) -> None:
    """At gamma = 0 and PEP = 0 the foot-to-foot delay MUST recover T. Fail loudly."""
    _, p = make_proximal_waveform(fs, hr_bpm, n_beats)
    d = apply_tube_load(p, fs, T, gamma=0.0)
    ftf = foot_to_foot_delay(p, d, fs, hr_bpm, n_beats)
    err_ms = abs(ftf - T) * 1e3
    if not (err_ms < tol_ms):
        raise AssertionError(
            f"gamma=0 recovery FAILED: foot-to-foot={ftf*1e3:.2f} ms vs "
            f"T={T*1e3:.2f} ms (err {err_ms:.2f} ms > tol {tol_ms} ms)")
