"""Physics-law credibility benchmark: Beer-Lambert and Moens-Korteweg.

Controlled simulators where examples are either law-identifiable or deliberately
incomplete.  A frozen activation monitor predicts identifiability and supports
selective abstention without seeing test labels.

Run from ``supervised_baselines/``:

    python analysis/physics_law_credibility.py --device cuda
    python analysis/physics_law_credibility.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import matplotlib
from scipy import stats as scipy_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch import nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from abstention_utils import (  # noqa: E402
    area_under_risk_coverage,
    binary_detection_metrics,
    binary_ranking_metrics,
    fit_answerability_probe,
    fit_bootstrap_ridge,
    risk_coverage_curve,
    risks_at_coverages,
)

FS = 25.0
WIN_LEN = 128
# Moens-Korteweg needs its own, much higher sampling rate: at FS=25Hz (40ms
# period) the entire physiological PTT range (~5-43ms) rounds to exactly 1
# sample for every example -- the distal waveform's time-shift carries zero
# BP information regardless of the true value (verified: shift_samples was
# constant across 2000 generated examples). 400Hz gives ~2-17 sample range,
# resolvable by cross-attention. Beer-Lambert doesn't need this (its signal
# is a per-stream AC/DC ratio, not cross-stream timing) so it keeps FS=25.
MK_FS = 400.0


# ── utilities ─────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empirical_clean_confidence(
    reference: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(reference, scores, side="right") / max(len(reference), 1)


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


# ── Beer-Lambert simulator ────────────────────────────────────────────────────


@dataclass
class BeerLambertSample:
    red: np.ndarray
    ir: np.ndarray
    spo2: float
    ratio_r: float
    ac_red: float
    ac_ir: float
    answerable: bool
    scenario: str
    meta: dict = field(default_factory=dict)


def _pulse_wave(n: int, hr_bpm: float, fs: float = FS) -> np.ndarray:
    t = np.arange(n) / fs
    f = hr_bpm / 60.0
    base = np.sin(2 * np.pi * f * t)
    harmonic = 0.25 * np.sin(4 * np.pi * f * t + 0.3)
    return base + harmonic


def _beer_lambert_intensity(
    spo2: float,
    c_total: float,
    path_length: float,
    eps_red_hbo2: float,
    eps_red_hb: float,
    eps_ir_hbo2: float,
    eps_ir_hb: float,
    pulse: np.ndarray,
    baseline_atten: float,
    scatter: float,
    gain: float,
    wavelength: str,
) -> np.ndarray:
    """Modified Beer-Lambert: I = I0 * exp(-mu * L) with pulsatile L.

    mu uses c_total (total hemoglobin concentration, previously threaded
    through unused) so absorbance actually depends on subject hemoglobin
    level, matching real pulse oximetry. The extinction-coefficient
    magnitudes are scaled so mu * path_length lands in a realistic optical
    density range (~0.3-2.5) instead of ~20-30: at the un-scaled literature
    magnitudes, exp(-(dc+ac)) underflows toward the additive noise floor and
    the AC/DC ratio carries essentially no SpO2 information (verified:
    |corr(ratio_R, SpO2)| ~ 0.03 before this fix).
    """
    s = spo2 / 100.0
    if wavelength == "red":
        mu = c_total * (s * eps_red_hbo2 + (1 - s) * eps_red_hb)
    else:
        mu = c_total * (s * eps_ir_hbo2 + (1 - s) * eps_ir_hb)
    mu *= 1.0 + scatter
    dc = baseline_atten + mu * path_length
    ac = mu * path_length * 0.08 * pulse
    intensity = gain * np.exp(-(dc + ac))
    return intensity.astype(np.float32)


def _ac_dc_ratio(signal: np.ndarray) -> tuple[float, float, float]:
    sig = np.asarray(signal, dtype=float)
    dc = float(np.mean(sig))
    ac = float(np.std(sig))
    return ac, dc, ac / (abs(dc) + 1e-8)


def generate_beer_lambert_sample(
    rng: np.random.Generator,
    scenario: str = "clean",
    win_len: int = WIN_LEN,
) -> BeerLambertSample:
    spo2 = float(rng.uniform(85, 99))
    c_total = float(rng.uniform(0.8, 1.2))
    path_length = float(rng.uniform(0.015, 0.025))
    # Extinction-coefficient *ratios* follow real HbO2/Hb spectra (Hb absorbs
    # red much more strongly than HbO2; both absorb IR comparably); the
    # absolute magnitude is scaled down 25x from raw literature values so
    # mu * path_length lands in a realistic transmission optical-density
    # range instead of saturating exp() to the noise floor (see docstring).
    eps_red_hbo2, eps_red_hb = 12.8, 340.0
    eps_ir_hbo2, eps_ir_hb = 48.0, 160.0
    hr = float(rng.uniform(55, 95))
    pulse = _pulse_wave(win_len, hr)
    baseline = float(rng.uniform(0.5, 1.5))
    scatter = float(rng.uniform(0.0, 0.15))
    melanin = float(rng.uniform(0.0, 0.3))
    gain = float(rng.uniform(0.8, 1.2))
    motion = rng.normal(0, 0.02 * (1 + melanin), win_len)

    red = _beer_lambert_intensity(
        spo2, c_total, path_length,
        eps_red_hbo2, eps_red_hb, eps_ir_hbo2, eps_ir_hb,
        pulse, baseline + melanin, scatter, gain, "red",
    )
    ir = _beer_lambert_intensity(
        spo2, c_total, path_length,
        eps_red_hbo2, eps_red_hb, eps_ir_hbo2, eps_ir_hb,
        pulse, baseline + 0.5 * melanin, scatter * 0.8, gain, "ir",
    )
    red += motion.astype(np.float32)
    ir += motion.astype(np.float32)

    answerable = True
    meta = {"path_length": path_length, "scatter": scatter, "melanin": melanin}

    if scenario == "missing_red":
        red = np.zeros_like(red)
        answerable = False
    elif scenario == "missing_ir":
        ir = np.zeros_like(ir)
        answerable = False
    elif scenario == "wavelength_mismatch":
        ir = _beer_lambert_intensity(
            float(rng.uniform(85, 99)), c_total, path_length * 1.2,
            eps_red_hbo2, eps_red_hb, eps_ir_hbo2, eps_ir_hb,
            pulse, baseline, scatter, gain, "ir",
        )
        answerable = False
    elif scenario == "calibration_shift":
        gain *= float(rng.uniform(1.4, 1.8))
        red *= gain
        ir *= gain
    elif scenario == "pigmentation_shift":
        melanin = float(rng.uniform(0.5, 0.9))
        red *= np.exp(-melanin * 0.5)
        ir *= np.exp(-melanin * 0.3)
    elif scenario == "motion_noise":
        red += rng.normal(0, 0.15, win_len).astype(np.float32)
        ir += rng.normal(0, 0.15, win_len).astype(np.float32)
    elif scenario == "model_discrepancy":
        red = np.sqrt(np.maximum(red, 0))  # breaks ideal log-linear law
        ir = np.sqrt(np.maximum(ir, 0))

    ac_r, dc_r, r_red = _ac_dc_ratio(red)
    ac_i, dc_i, r_ir = _ac_dc_ratio(ir)
    ratio_r = r_red / (r_ir + 1e-8)

    return BeerLambertSample(
        red=red, ir=ir, spo2=spo2, ratio_r=float(ratio_r),
        ac_red=r_red, ac_ir=r_ir, answerable=answerable,
        scenario=scenario, meta=meta,
    )


BEER_SCENARIOS_TRAIN = ("clean",)
BEER_SCENARIOS_PROBE = ("clean", "missing_red", "missing_ir")
BEER_SCENARIOS_VAL = ("clean", "calibration_shift", "pigmentation_shift")
BEER_SCENARIOS_TEST = (
    "clean", "missing_red", "missing_ir", "wavelength_mismatch",
    "calibration_shift", "pigmentation_shift", "motion_noise", "model_discrepancy",
)
BEER_SCENARIOS_HELD_OUT = ("wavelength_mismatch", "motion_noise", "model_discrepancy")


# ── Moens-Korteweg simulator ──────────────────────────────────────────────────


@dataclass
class MoensKortewegSample:
    proximal: np.ndarray
    distal: np.ndarray
    bp: float
    ptt_ms: float
    pwv: float
    stiffness: float
    answerable: bool
    scenario: str
    # [path_m, K/100, alpha, availability] where K = E0*h/(rho*D) is the
    # subject's lumped stiffness/geometry constant. PWV^2 = K*exp(alpha*BP/100),
    # so (path_m, K, alpha) is exactly the calibration a real cuffless-BP
    # device needs (typically from an initial cuff-based calibration) to
    # invert PTT into BP -- without it BP and the unknown geometry/stiffness
    # trade off against each other with no unique solution. Zeroed with
    # availability=0 when scenario represents "no prior calibration."
    calibration: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    meta: dict = field(default_factory=dict)


def _moens_korteweg_pwv(
    e_pa: float, h_m: float, rho: float, d_m: float
) -> float:
    return float(math.sqrt(e_pa * h_m / (rho * d_m)))


def _pressure_stiffness(p_mmhg: float, e0: float, alpha: float) -> float:
    return float(e0 * math.exp(alpha * p_mmhg / 100.0))


def generate_moens_korteweg_sample(
    rng: np.random.Generator,
    scenario: str = "clean",
    win_len: int = WIN_LEN,
    fs: float = MK_FS,
) -> MoensKortewegSample:
    bp = float(rng.uniform(90, 150))
    path_m = float(rng.uniform(0.25, 0.45))
    diameter = float(rng.uniform(0.003, 0.006))
    thickness = float(rng.uniform(0.0003, 0.0008))
    rho = 1060.0
    e0 = float(rng.uniform(0.5e6, 2.0e6))
    alpha = float(rng.uniform(0.8, 1.5))
    subject_cal = float(rng.uniform(0.9, 1.1))
    hr = float(rng.uniform(55, 95))

    stiffness = _pressure_stiffness(bp, e0, alpha)
    pwv = _moens_korteweg_pwv(stiffness, thickness, rho, diameter)
    ptt_ms = 1000.0 * path_m / pwv

    shift_samples = int(round(ptt_ms * fs / 1000.0))
    shift_samples = min(max(shift_samples, 1), win_len // 4)
    pulse = _pulse_wave(win_len, hr)
    damping = float(rng.uniform(0.85, 0.95))
    noise = rng.normal(0, 0.03, win_len).astype(np.float32)

    proximal = pulse.astype(np.float32) + noise
    distal = np.zeros(win_len, dtype=np.float32)
    distal[shift_samples:] = damping * pulse[:-shift_samples].astype(np.float32)
    distal += noise * 0.5

    answerable = True
    calibration_available = True
    meta = {
        "path_m": path_m, "subject_cal": subject_cal,
        "e0": e0, "alpha": alpha,
    }

    if scenario == "missing_path_length":
        path_m = float("nan")
        answerable = False
        calibration_available = False
    elif scenario == "missing_stiffness_cal":
        subject_cal = float("nan")
        answerable = False
        calibration_available = False
    elif scenario == "missing_distal":
        distal = np.zeros_like(distal)
        answerable = False
    elif scenario == "temporal_shift":
        shift_samples = int(rng.integers(win_len // 8, win_len // 3))
        distal = np.zeros(win_len, dtype=np.float32)
        distal[shift_samples:] = damping * pulse[:-shift_samples].astype(np.float32)
    elif scenario == "anatomy_shift":
        path_m *= float(rng.uniform(1.3, 1.6))
        ptt_ms = 1000.0 * path_m / pwv
    elif scenario == "viscoelastic_shift":
        distal = np.zeros(win_len, dtype=np.float32)
        visco = 0.7 * pulse + 0.3 * np.roll(pulse, 3)
        distal[shift_samples:] = damping * visco[:-shift_samples].astype(np.float32)
    elif scenario == "model_discrepancy":
        pwv *= float(rng.uniform(1.2, 1.5))
        ptt_ms = 1000.0 * path_m / pwv

    k_lumped = e0 * thickness / (rho * diameter)
    if calibration_available:
        calibration = np.array(
            [path_m, k_lumped / 100.0, alpha, 1.0], dtype=np.float32
        )
    else:
        calibration = np.zeros(4, dtype=np.float32)

    return MoensKortewegSample(
        proximal=proximal, distal=distal, bp=bp,
        ptt_ms=ptt_ms, pwv=pwv, stiffness=stiffness,
        calibration=calibration,
        answerable=answerable, scenario=scenario, meta=meta,
    )


MK_SCENARIOS_TRAIN = ("clean",)
MK_SCENARIOS_PROBE = ("clean", "missing_distal", "missing_stiffness_cal")
MK_SCENARIOS_VAL = ("clean", "temporal_shift", "anatomy_shift")
MK_SCENARIOS_TEST = (
    "clean", "missing_path_length", "missing_stiffness_cal", "missing_distal",
    "temporal_shift", "anatomy_shift", "viscoelastic_shift", "model_discrepancy",
)
MK_SCENARIOS_HELD_OUT = ("viscoelastic_shift", "model_discrepancy", "missing_path_length")


# ── dataset builders ──────────────────────────────────────────────────────────


def build_beer_dataset(
    n_per_scenario: int, scenarios: tuple[str, ...], seed: int
) -> list[BeerLambertSample]:
    rng = np.random.default_rng(seed)
    out = []
    for scenario in scenarios:
        for _ in range(n_per_scenario):
            out.append(generate_beer_lambert_sample(rng, scenario))
    return out


def build_mk_dataset(
    n_per_scenario: int, scenarios: tuple[str, ...], seed: int
) -> list[MoensKortewegSample]:
    rng = np.random.default_rng(seed)
    out = []
    for scenario in scenarios:
        for _ in range(n_per_scenario):
            out.append(generate_moens_korteweg_sample(rng, scenario))
    return out


def beer_to_tensors(samples: list[BeerLambertSample]) -> dict:
    return {
        "stream_a": np.stack([s.red for s in samples]),
        "stream_b": np.stack([s.ir for s in samples]),
        "target": np.array([s.spo2 for s in samples], dtype=np.float32),
        "ratio_r": np.array([s.ratio_r for s in samples], dtype=np.float32),
        "answerable": np.array([s.answerable for s in samples], dtype=int),
        "scenario": [s.scenario for s in samples],
    }


def mk_to_tensors(samples: list[MoensKortewegSample]) -> dict:
    return {
        "stream_a": np.stack([s.proximal for s in samples]),
        "stream_b": np.stack([s.distal for s in samples]),
        "calibration": np.stack([s.calibration for s in samples]),
        "target": np.array([s.bp for s in samples], dtype=np.float32),
        "ptt_ms": np.array([s.ptt_ms for s in samples], dtype=np.float32),
        "pwv": np.array([s.pwv for s in samples], dtype=np.float32),
        "answerable": np.array([s.answerable for s in samples], dtype=int),
        "scenario": [s.scenario for s in samples],
    }


# ── models ────────────────────────────────────────────────────────────────────


def _positional_encoding(n_positions: int, dim: int) -> torch.Tensor:
    """Deterministic sinusoidal encoding giving each timestep an identity.

    Without this, a per-timestep encoder that only sees a raw amplitude
    scalar has no notion of sequence position, so cross-attention can only
    match by amplitude similarity -- it cannot express anything like a
    cross-correlation / time-lag estimate, which is exactly what recovering
    PTT from a phase-shifted periodic wave requires (verified: MK stayed
    stuck at the predict-mean baseline even after fixing normalization and
    exposing calibration, until this was added).
    """
    positions = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    freqs = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim)
    )
    encoding = torch.zeros(n_positions, dim)
    encoding[:, 0::2] = torch.sin(positions * freqs)
    encoding[:, 1::2] = torch.cos(positions * freqs[: encoding[:, 1::2].shape[1]])
    return encoding


class TwoStreamCrossAttention(nn.Module):
    """Cross-attention between two 1D waveform streams, plus a symmetric
    per-stream pooling branch.

    Beer-Lambert's ratio-R and Moens-Korteweg's PTT are different kinds of
    statistics: ratio-R is a symmetric per-stream (mean, std) computation
    (verified: a Ridge fit on [a.mean, a.std, b.mean, b.std] alone recovers
    SpO2 well), while PTT is a cross-stream *timing* relationship that
    mean/std cannot see (verified: the same pooled features carry ~zero BP
    signal for Moens-Korteweg, since mean/std of a periodic wave is
    shift-invariant). An "adequate" architecture needs both pathways to be a
    genuine superset of the pooling-only control -- a pure cross-attention
    retriever without the pooling branch loses to TwoStreamGlobalPool on
    Beer-Lambert despite being more complex, because it has no efficient way
    to express a simple mean/std computation through an attention bottleneck.
    """

    def __init__(
        self, win_len: int, width: int = 64, hidden: int = 64, calibration_dim: int = 0,
        pos_dim: int = 8,
    ):
        super().__init__()
        self.width = width
        self.calibration_dim = calibration_dim
        self.register_buffer("pos_encoding", _positional_encoding(win_len, pos_dim))
        self.enc_a = nn.Sequential(
            nn.Linear(1 + pos_dim, hidden), nn.ReLU(), nn.Linear(hidden, width)
        )
        self.enc_b = nn.Sequential(
            nn.Linear(1 + pos_dim, hidden), nn.ReLU(), nn.Linear(hidden, width)
        )
        self.pool_proj = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, width)
        )
        n_branches = 2
        if calibration_dim > 0:
            self.cal_proj = nn.Sequential(
                nn.Linear(calibration_dim, hidden), nn.ReLU(), nn.Linear(hidden, width)
            )
            n_branches = 3
        self.output = nn.Sequential(
            nn.Linear(n_branches * width, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        calibration: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        # a, b: (B, L)
        pos = self.pos_encoding.unsqueeze(0).expand(a.shape[0], -1, -1)
        qa = self.enc_a(torch.cat([a.unsqueeze(-1), pos], dim=-1))
        kb = self.enc_b(torch.cat([b.unsqueeze(-1), pos], dim=-1))
        vb = kb
        scores = torch.bmm(qa, kb.transpose(1, 2)) / math.sqrt(self.width)
        attn = scores.softmax(dim=-1)
        ctx_attn = torch.bmm(attn, vb).mean(dim=1)
        pooled = torch.stack([a.mean(1), a.std(1), b.mean(1), b.std(1)], dim=-1)
        ctx_pool = self.pool_proj(pooled)
        ctx_parts = [ctx_attn, ctx_pool]
        if self.calibration_dim > 0:
            if calibration is None:
                raise ValueError("Model configured with calibration_dim>0 but no calibration passed.")
            ctx_parts.append(self.cal_proj(calibration))
        ctx = torch.cat(ctx_parts, dim=-1)
        pred = self.output(ctx).squeeze(-1)
        if return_features:
            return pred, ctx, attn
        return pred


class TwoStreamGlobalPool(nn.Module):
    """Control: global pool both streams, no indexed cross-stream retrieval."""

    def __init__(
        self, win_len: int, width: int = 64, hidden: int = 64, calibration_dim: int = 0
    ):
        super().__init__()
        self.calibration_dim = calibration_dim
        self.net = nn.Sequential(
            nn.Linear(4 + calibration_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        calibration: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        feat = torch.stack([
            a.mean(1), a.std(1), b.mean(1), b.std(1),
        ], dim=-1)
        if self.calibration_dim > 0:
            if calibration is None:
                raise ValueError("Model configured with calibration_dim>0 but no calibration passed.")
            feat = torch.cat([feat, calibration], dim=-1)
        pred = self.net(feat).squeeze(-1)
        ctx = feat
        if return_features:
            return pred, ctx, None
        return pred


@dataclass
class Normalizer:
    """Fixed affine rescaling fit on the training split only.

    Streams get one shared global mean/std (not per-example) so the
    per-example DC-offset differences that ratio_R depends on are preserved
    -- this only fixes gradient conditioning, it does not touch the signal.
    Target standardization matters because raw MSE loss on targets ~O(90-150)
    with small-magnitude inputs gives Adam a poorly-conditioned problem: the
    net converges instantly to a near-constant predictor and then never
    escapes it, regardless of training budget (verified empirically).
    """

    stream_mean: float
    stream_std: float
    target_mean: float
    target_std: float
    # Per-column stats for the optional calibration vector (Moens-Korteweg
    # only). Unlike streams, calibration entries are heterogeneous scalars
    # (path length, lumped stiffness, alpha, availability flag) on very
    # different natural scales, so each column is standardized independently.
    calibration_mean: np.ndarray | None = None
    calibration_std: np.ndarray | None = None

    def transform_streams(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (a - self.stream_mean) / self.stream_std, (b - self.stream_mean) / self.stream_std

    def transform_target(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.target_mean) / self.target_std

    def inverse_target(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.target_std + self.target_mean

    def transform_calibration(self, c: np.ndarray) -> np.ndarray:
        if self.calibration_mean is None:
            return c
        return (c - self.calibration_mean) / self.calibration_std


def fit_normalizer(data: dict) -> Normalizer:
    streams = np.concatenate([data["stream_a"].reshape(-1), data["stream_b"].reshape(-1)])
    target = np.asarray(data["target"], dtype=float)
    calibration_mean = calibration_std = None
    if "calibration" in data:
        cal = np.asarray(data["calibration"], dtype=float)
        calibration_mean = cal.mean(axis=0)
        calibration_std = cal.std(axis=0) + 1e-8
    return Normalizer(
        stream_mean=float(streams.mean()),
        stream_std=float(streams.std() + 1e-8),
        target_mean=float(target.mean()),
        target_std=float(target.std() + 1e-8),
        calibration_mean=calibration_mean,
        calibration_std=calibration_std,
    )


def train_regressor(
    model: nn.Module,
    data: dict,
    device: torch.device,
    epochs: int,
    batch_size: int,
    normalizer: Normalizer,
    lr: float = 1e-3,
) -> list[float]:
    model.to(device)
    sa, sb = normalizer.transform_streams(data["stream_a"], data["stream_b"])
    a = torch.from_numpy(sa).float()
    b = torch.from_numpy(sb).float()
    cal = None
    if "calibration" in data:
        cal = torch.from_numpy(normalizer.transform_calibration(data["calibration"])).float()
    y = normalizer.transform_target(torch.from_numpy(data["target"]).float())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(a))
        total = 0.0
        for start in range(0, len(a), batch_size):
            idx = order[start : start + batch_size]
            ba, bb, by = a[idx].to(device), b[idx].to(device), y[idx].to(device)
            bc = cal[idx].to(device) if cal is not None else None
            pred = model(ba, bb, calibration=bc)
            loss = ((pred - by) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            # Guards against occasional gradient blow-ups (observed: without
            # this, held-out extrapolation on shifted-but-answerable
            # scenarios like anatomy_shift could produce predictions in the
            # billions on a fraction of test examples, dominating every
            # downstream AURC computation with numerically meaningless
            # values rather than genuine selective-risk signal).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        losses.append(total / len(a))
    return losses


def evaluate_regressor(
    model: nn.Module, data: dict, device: torch.device, batch_size: int, normalizer: Normalizer
) -> dict:
    model.eval()
    sa, sb = normalizer.transform_streams(data["stream_a"], data["stream_b"])
    a = torch.from_numpy(sa).float()
    b = torch.from_numpy(sb).float()
    cal = None
    if "calibration" in data:
        cal = torch.from_numpy(normalizer.transform_calibration(data["calibration"])).float()
    y = torch.from_numpy(data["target"]).float()
    preds, acts = [], []
    with torch.no_grad():
        for start in range(0, len(a), batch_size):
            ba = a[start : start + batch_size].to(device)
            bb = b[start : start + batch_size].to(device)
            bc = cal[start : start + batch_size].to(device) if cal is not None else None
            pred, ctx, _ = model(ba, bb, calibration=bc, return_features=True)
            # Extrapolating far outside the (clean-only) training manifold
            # -- e.g. held-out scenarios like anatomy_shift that scale a
            # calibration feature well beyond its training range -- can
            # drive an unbounded ReLU network's raw output to diverge
            # wildly (observed: standardized outputs in the hundreds of
            # millions on a subset of test examples, which would dominate
            # every downstream AURC computation with numerically
            # meaningless values). +-15 std is a wide margin -- true targets
            # are within a few std of the training mean by construction --
            # so this only catches genuine blow-ups, not real predictions.
            pred = torch.clamp(pred, -15.0, 15.0)
            pred = normalizer.inverse_target(pred)
            preds.append(pred.cpu().numpy())
            acts.append(ctx.cpu().numpy())
    pred = np.concatenate(preds)
    act = np.concatenate(acts)
    return {
        "prediction": pred,
        "activation": act,
        "loss": np.abs(pred - y.numpy()),
        "target": y.numpy(),
        "answerable": data["answerable"],
        "scenario": data["scenario"],
    }


# ── intervention ──────────────────────────────────────────────────────────────


def probe_direction(probe: Ridge, scaler: StandardScaler, act_dim: int) -> np.ndarray:
    w = probe.coef_.reshape(-1)
    w = w / (np.linalg.norm(w) + 1e-8)
    if w.shape[0] < act_dim:
        w = np.pad(w, (0, act_dim - w.shape[0]))
    return w[:act_dim].astype(np.float32)


def intervene_activation(
    activations: np.ndarray, direction: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    return activations - scale * direction[None, :]


def random_direction(act_dim: int, norm: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=act_dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v * norm


# ── law audit pipeline ────────────────────────────────────────────────────────


@dataclass
class LawAuditConfig:
    n_train: int = 400
    n_probe: int = 120
    n_val: int = 120
    n_test: int = 80
    epochs: int = 120
    lr: float = 1e-3
    batch_size: int = 64
    width: int = 64
    low_width: int = 8
    n_bootstrap: int = 15
    seed: int = 0


def run_law_audit(
    law_name: str,
    build_fn: Callable,
    to_tensors: Callable,
    probe_targets: dict[str, Callable[[dict], np.ndarray]],
    train_scenarios: tuple[str, ...],
    probe_scenarios: tuple[str, ...],
    val_scenarios: tuple[str, ...],
    test_scenarios: tuple[str, ...],
    held_out: tuple[str, ...],
    cfg: LawAuditConfig,
    device: torch.device,
) -> dict:
    seed = cfg.seed
    train = to_tensors(build_fn(cfg.n_train, train_scenarios, seed * 100 + 1))
    probe_data = to_tensors(build_fn(cfg.n_probe, probe_scenarios, seed * 100 + 2))
    val_data = to_tensors(build_fn(cfg.n_val, val_scenarios, seed * 100 + 3))
    test_parts = []
    for i, sc in enumerate(test_scenarios):
        test_parts.append(to_tensors(build_fn(cfg.n_test, (sc,), seed * 100 + 10 + i)))
    stack_keys = set(test_parts[0].keys()) - {"scenario"}
    test = {k: np.concatenate([p[k] for p in test_parts]) for k in stack_keys}
    test["scenario"] = [s for p in test_parts for s in p["scenario"]]

    normalizer = fit_normalizer(train)
    calibration_dim = train["calibration"].shape[-1] if "calibration" in train else 0

    factories = {
        "cross_attention": lambda: TwoStreamCrossAttention(
            WIN_LEN, cfg.width, calibration_dim=calibration_dim
        ),
        "low_capacity": lambda: TwoStreamCrossAttention(
            WIN_LEN, cfg.low_width, calibration_dim=calibration_dim
        ),
        "no_retrieval": lambda: TwoStreamGlobalPool(WIN_LEN, calibration_dim=calibration_dim),
    }
    results = {}
    intervention_payload = None

    for model_name, factory in factories.items():
        model = factory()
        train_regressor(model, train, device, cfg.epochs, cfg.batch_size, normalizer, lr=cfg.lr)
        probe_eval = evaluate_regressor(model, probe_data, device, cfg.batch_size, normalizer)
        val_eval = evaluate_regressor(model, val_data, device, cfg.batch_size, normalizer)
        test_eval = evaluate_regressor(model, test, device, cfg.batch_size, normalizer)

        ans_probe = fit_answerability_probe(
            probe_eval["activation"], probe_eval["answerable"], seed
        )
        val_clean = val_eval["answerable"].astype(bool)
        val_scores = ans_probe.predict_proba(val_eval["activation"])
        test_probs = ans_probe.predict_proba(test_eval["activation"])

        probe_r2 = {}
        for pname, getter in probe_targets.items():
            if pname not in probe_data:
                continue
            tgt = getter(probe_data)
            finite = np.isfinite(tgt)
            if finite.sum() < 4:
                continue
            rp = fit_bootstrap_ridge(
                probe_eval["activation"][finite], tgt[finite],
                n_bootstrap=cfg.n_bootstrap, seed=seed,
            )
            rp.calibrate_confidence(probe_eval["activation"][finite])
            pred = rp.predict(test_eval["activation"])
            tgt_test = getter(test)
            ft = np.isfinite(tgt_test)
            if ft.sum() > 1:
                probe_r2[pname] = float(r2_score(tgt_test[ft], pred[ft]))

        clean_mask = np.array([s == "clean" for s in test_eval["scenario"]])
        held_out_mask = np.array([s in held_out for s in test_eval["scenario"]])
        unanswerable_mask = test_eval["answerable"] == 0

        losses = test_eval["loss"]
        random_conf = np.random.default_rng(seed).random(len(losses))
        val_ref = val_scores[val_clean] if val_clean.any() else val_scores
        emp_conf = empirical_clean_confidence(val_ref, test_probs)
        oracle_conf = test_eval["answerable"].astype(float)

        # Non-probing UQ baseline: a bootstrap-ensemble regressor fit only on
        # examples known to be answerable, gated by ensemble
        # disagreement + distance from the training manifold -- unlike the
        # linear probe, this never sees the true answerable label during
        # fitting. Isolates what supervised activation probing specifically
        # buys over a standard, label-free uncertainty-quantification method
        # applied to the same frozen activations.
        answerable_probe_mask = probe_eval["answerable"].astype(bool)
        ensemble_conf = None
        if answerable_probe_mask.sum() >= 4:
            ensemble_probe = fit_bootstrap_ridge(
                probe_eval["activation"][answerable_probe_mask],
                probe_data["target"][answerable_probe_mask],
                n_bootstrap=cfg.n_bootstrap, seed=seed,
            )
            ensemble_probe.calibrate_confidence(probe_eval["activation"][answerable_probe_mask])
            ensemble_conf = ensemble_probe.confidence(test_eval["activation"])

        scenario_mae = {}
        for sc in test_scenarios:
            m = np.array([s == sc for s in test_eval["scenario"]])
            if m.any():
                scenario_mae[sc] = float(losses[m].mean())

        results[model_name] = {
            "scenario_mae": scenario_mae,
            "probe_r2": probe_r2,
            "answerability_all": binary_detection_metrics(
                test_eval["answerable"], test_probs
            ),
            "answerability_held_out": binary_ranking_metrics(
                test_eval["answerable"][held_out_mask | clean_mask],
                test_probs[held_out_mask | clean_mask],
            ) if (held_out_mask | clean_mask).sum() and np.unique(
                test_eval["answerable"][held_out_mask | clean_mask]
            ).size > 1 else {},
            "selective": {
                "probe": {
                    "aurc": area_under_risk_coverage(losses, test_probs),
                    "risk": risks_at_coverages(losses, test_probs),
                },
                "random": {
                    "aurc": area_under_risk_coverage(losses, random_conf),
                    "risk": risks_at_coverages(losses, random_conf),
                },
                "empirical": {
                    "aurc": area_under_risk_coverage(losses, emp_conf),
                    "risk": risks_at_coverages(losses, emp_conf),
                },
                "oracle": {
                    "aurc": area_under_risk_coverage(losses, oracle_conf),
                    "risk": risks_at_coverages(losses, oracle_conf),
                },
                **(
                    {
                        "ensemble_disagreement": {
                            "aurc": area_under_risk_coverage(losses, ensemble_conf),
                            "risk": risks_at_coverages(losses, ensemble_conf),
                        }
                    }
                    if ensemble_conf is not None
                    else {}
                ),
            },
        }

        if model_name == "cross_attention":
            clean_mask = np.array([s == "clean" for s in test_eval["scenario"]])
            clean_act = test_eval["activation"][clean_mask]
            clean_tgt = test_eval["target"][clean_mask]
            if len(clean_act) > 2:
                scaler = StandardScaler().fit(clean_act)
                ridge = Ridge(alpha=10.0).fit(
                    scaler.transform(clean_act), clean_tgt
                )
                direction = probe_direction(ridge, scaler, clean_act.shape[1])
                base_pred = ridge.predict(scaler.transform(clean_act))
                base_mae = float(np.mean(np.abs(base_pred - clean_tgt)))
                ablated = intervene_activation(
                    scaler.transform(clean_act), direction, scale=1.0
                )
                abl_mae = float(
                    np.mean(np.abs(ridge.predict(ablated) - clean_tgt))
                )
                rand_dir = random_direction(
                    clean_act.shape[1], float(np.linalg.norm(direction)), seed + 99
                )
                rand_abl = intervene_activation(
                    scaler.transform(clean_act), rand_dir, scale=1.0
                )
                rand_mae = float(
                    np.mean(np.abs(ridge.predict(rand_abl) - clean_tgt))
                )
                intervention_payload = {
                    "baseline_readout_mae": base_mae,
                    "probe_direction_ablated_mae": abl_mae,
                    "random_direction_ablated_mae": rand_mae,
                    "probe_beats_random_intervention": bool(abl_mae > rand_mae),
                }
        del model

    decision = _law_decision(results, intervention_payload)
    return {
        "law": law_name,
        "seed": seed,
        "models": results,
        "intervention": intervention_payload,
        "decision": decision,
        "held_out_scenarios": list(held_out),
    }


def _law_decision(results: dict, intervention: dict | None) -> dict:
    cross = results.get("cross_attention", {})
    low = results.get("low_capacity", {})
    nr = results.get("no_retrieval", {})
    clean_cross = cross.get("scenario_mae", {}).get("clean", float("inf"))
    clean_low = low.get("scenario_mae", {}).get("clean", float("inf"))
    clean_nr = nr.get("scenario_mae", {}).get("clean", float("inf"))
    sel = cross.get("selective", {})
    probe_aurc = sel.get("probe", {}).get("aurc", float("inf"))
    random_aurc = sel.get("random", {}).get("aurc", float("inf"))
    ans = cross.get("answerability_all", {}).get("auroc", 0.0)
    held = cross.get("answerability_held_out", {}).get("auroc", 0.0)
    probe_r2 = cross.get("probe_r2", {})
    decodable = any(v > 0 for v in probe_r2.values()) if probe_r2 else False
    return {
        "cross_beats_controls": bool(
            clean_cross < clean_low and clean_cross < clean_nr
        ),
        "law_statistic_decodable": decodable,
        "answerability_above_chance": bool(ans > 0.5),
        "held_out_detection_above_chance": bool(held > 0.5) if held else False,
        "probe_beats_random_aurc": bool(probe_aurc < random_aurc),
        "probe_intervention_beats_random": bool(
            (intervention or {}).get("probe_beats_random_intervention", False)
        ),
    }


def run_beer_audit(cfg: LawAuditConfig, device: torch.device) -> dict:
    return run_law_audit(
        "beer_lambert",
        build_beer_dataset,
        beer_to_tensors,
        {"ratio_r": lambda d: d["ratio_r"], "spo2": lambda d: d["target"]},
        BEER_SCENARIOS_TRAIN,
        BEER_SCENARIOS_PROBE,
        BEER_SCENARIOS_VAL,
        BEER_SCENARIOS_TEST,
        BEER_SCENARIOS_HELD_OUT,
        cfg,
        device,
    )


def run_mk_audit(cfg: LawAuditConfig, device: torch.device) -> dict:
    return run_law_audit(
        "moens_korteweg",
        build_mk_dataset,
        mk_to_tensors,
        {
            "ptt_ms": lambda d: d["ptt_ms"],
            "pwv": lambda d: d["pwv"],
            "bp": lambda d: d["target"],
        },
        MK_SCENARIOS_TRAIN,
        MK_SCENARIOS_PROBE,
        MK_SCENARIOS_VAL,
        MK_SCENARIOS_TEST,
        MK_SCENARIOS_HELD_OUT,
        cfg,
        device,
    )


def paired_significance(
    a: np.ndarray, b: np.ndarray, seed: int = 0, n_boot: int = 4000
) -> dict:
    """Paired comparison of a vs b (one pair per seed), lower-is-better.

    Reports the mean difference (a-b), a percentile-bootstrap 95% CI on that
    difference, and a Wilcoxon signed-rank p-value. This is the
    statistically appropriate bar with few seeds -- requiring "a beats b on
    literally every seed" is a much stricter and noisier criterion than
    "a is significantly lower than b in aggregate," and can fail by chance
    even when the underlying effect is real and consistent (see
    Beer-Lambert's cross-attention vs low-capacity gap, which is small
    per-seed because the task has a low-dimensional sufficient statistic,
    but consistent in direction).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = len(diff)
    if n < 2:
        return {
            "n": n, "mean_diff": float(diff.mean()) if n else float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "wilcoxon_p": float("nan"), "significant": False,
        }
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    try:
        wilcoxon_p = float(scipy_stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        wilcoxon_p = float("nan")
    return {
        "n": n,
        "mean_diff": float(diff.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "wilcoxon_p": wilcoxon_p,
        "significant": bool(ci_high < 0),
    }


def summarize_across_seeds(reports: list[dict]) -> dict:
    def mean_nested(reps: list[dict], *keys: str) -> float:
        vals = []
        for r in reps:
            obj = r
            for k in keys:
                if not isinstance(obj, dict):
                    break
                obj = obj.get(k, {})
            if isinstance(obj, (int, float)) and np.isfinite(obj):
                vals.append(float(obj))
        return float(np.mean(vals)) if vals else float("nan")

    def collect_clean_mae(reps: list[dict], model_name: str) -> np.ndarray:
        return np.array([
            r["models"][model_name]["scenario_mae"]["clean"]
            for r in reps
            if "clean" in r.get("models", {}).get(model_name, {}).get("scenario_mae", {})
        ])

    def mean_clean_mae(reps: list[dict]) -> float:
        vals = collect_clean_mae(reps, "cross_attention")
        return float(vals.mean()) if len(vals) else float("nan")

    def collect_aurc(reps: list[dict], key: str) -> np.ndarray:
        return np.array([
            r["models"]["cross_attention"]["selective"][key]["aurc"]
            for r in reps
            if key in r.get("models", {}).get("cross_attention", {}).get("selective", {})
        ])

    def significance_block(reps: list[dict], seed: int) -> dict:
        cross = collect_clean_mae(reps, "cross_attention")
        low = collect_clean_mae(reps, "low_capacity")
        nr = collect_clean_mae(reps, "no_retrieval")
        probe_aurc = collect_aurc(reps, "probe")
        random_aurc = collect_aurc(reps, "random")
        result = {
            "cross_vs_low_capacity_mae": paired_significance(cross, low, seed=seed),
            "cross_vs_no_retrieval_mae": paired_significance(cross, nr, seed=seed),
            "probe_vs_random_aurc": paired_significance(probe_aurc, random_aurc, seed=seed),
        }
        # Isolates what supervised activation probing buys over a
        # label-free, non-probing UQ baseline (bootstrap-ensemble
        # disagreement) on the exact same frozen activations. Exploratory,
        # not part of the core success gate.
        ensemble_aurc = collect_aurc(reps, "ensemble_disagreement")
        if len(ensemble_aurc) == len(probe_aurc) and len(ensemble_aurc) > 0:
            result["probe_vs_ensemble_disagreement_aurc"] = paired_significance(
                probe_aurc, ensemble_aurc, seed=seed
            )
        return result

    beer = [r for r in reports if r["law"] == "beer_lambert"]
    mk = [r for r in reports if r["law"] == "moens_korteweg"]
    return {
        "beer_lambert": {
            "n_seeds": len(beer),
            "mean_clean_mae": mean_clean_mae(beer),
            "mean_answerability_auroc": mean_nested(
                beer, "models", "cross_attention", "answerability_all", "auroc"
            ),
            "mean_probe_aurc": mean_nested(
                beer, "models", "cross_attention", "selective", "probe", "aurc"
            ),
            "mean_random_aurc": mean_nested(
                beer, "models", "cross_attention", "selective", "random", "aurc"
            ),
            "decisions": [b["decision"] for b in beer],
            "significance": significance_block(beer, seed=0),
        },
        "moens_korteweg": {
            "n_seeds": len(mk),
            "mean_clean_mae": mean_clean_mae(mk),
            "mean_answerability_auroc": mean_nested(
                mk, "models", "cross_attention", "answerability_all", "auroc"
            ),
            "mean_probe_aurc": mean_nested(
                mk, "models", "cross_attention", "selective", "probe", "aurc"
            ),
            "mean_random_aurc": mean_nested(
                mk, "models", "cross_attention", "selective", "random", "aurc"
            ),
            "decisions": [m["decision"] for m in mk],
            "significance": significance_block(mk, seed=1),
        },
    }


def plot_summary(reports: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for law in ("beer_lambert", "moens_korteweg"):
        subset = [r for r in reports if r["law"] == law]
        if not subset:
            continue
        scenarios = list(subset[0]["models"]["cross_attention"]["scenario_mae"].keys())
        maes = [
            np.mean([r["models"]["cross_attention"]["scenario_mae"].get(sc, np.nan) for r in subset])
            for sc in scenarios
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(scenarios, maes)
        ax.set_ylabel("MAE")
        ax.set_title(f"{law}: cross-attention MAE by scenario (mean over seeds)")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / f"{law}_scenario_mae.png", dpi=180)
        plt.close(fig)


EQUATIONS = {
    "beer_lambert": (
        "I_lambda(t) = I0 * exp(-(eps_HbO2 * c_HbO2 + eps_Hb * c_Hb) * L(t)); "
        "R = (AC_red/DC_red) / (AC_ir/DC_ir); SpO2 = f(R, calibration)"
    ),
    "moens_korteweg": (
        "PWV = sqrt(E * h / (rho * D)); PTT = L / PWV; "
        "E(P) = E0 * exp(alpha * P); BP requires subject calibration"
    ),
}

LIMITATIONS = [
    "Simulators use simplified modified Beer-Lambert and Moens-Korteweg approximations.",
    "Not validated against clinical pulse oximetry or cuffless BP devices.",
    "Activation probes are external monitors, not intrinsic model uncertainty.",
    "Held-out corruptions test controlled law violations, not all real-world OOD.",
    "Do not use for clinical decision-making.",
]


# Mirrors the five-part "Success rule" documented in README.md: cross-attention
# beats both controls, the law statistic is decodable, held-out answerability
# detection beats chance, probe abstention beats random abstention, and the
# probe direction is the one causally responsible (ablating it hurts more than
# ablating a random direction of the same norm).
_SUCCESS_CRITERIA = (
    "cross_beats_controls",
    "law_statistic_decodable",
    "held_out_detection_above_chance",
    "probe_beats_random_aurc",
    "probe_intervention_beats_random",
)


def _law_success(decisions: list[dict]) -> bool:
    return bool(decisions) and all(
        d.get(key, False) for d in decisions for key in _SUCCESS_CRITERIA
    )


# Non-comparative criteria (a per-seed AUROC/R2 threshold, not a head-to-head
# MAE/AURC race) stay per-seed booleans -- they are much less noisy than a
# capacity comparison on a handful of seeds. The two head-to-head
# comparisons ("beats controls", "probe beats random abstention") instead
# require aggregate statistical significance (paired_significance), which is
# the appropriate bar with only a few seeds: literally beating a control on
# every single seed is a stricter and noisier requirement than being
# significantly better on average, and can fail by chance even when the
# underlying effect is real (see Beer-Lambert, whose cross-attention vs
# low-capacity gap is small per-seed because the task has a
# low-dimensional sufficient statistic, but is consistent in direction).
_NON_COMPARATIVE_CRITERIA = (
    "law_statistic_decodable",
    "answerability_above_chance",
    "held_out_detection_above_chance",
    "probe_intervention_beats_random",
)


def _law_success_statistical(decisions: list[dict], significance: dict) -> bool:
    if not decisions:
        return False
    non_comparative_pass = all(
        d.get(key, False) for d in decisions for key in _NON_COMPARATIVE_CRITERIA
    )
    comparative_pass = all(
        significance.get(key, {}).get("significant", False)
        for key in ("cross_vs_low_capacity_mae", "cross_vs_no_retrieval_mae", "probe_vs_random_aurc")
    )
    return bool(non_comparative_pass and comparative_pass)


def export_summary(
    reports: list[dict], output_dir: Path, config: dict
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = summarize_across_seeds(reports)
    payload = {
        "claim": (
            "Frozen activation probes can detect when a physiological law is "
            "incomplete and support selective abstention; this is an audit "
            "protocol, not intrinsic self-knowledge."
        ),
        "equations": EQUATIONS,
        "limitations": LIMITATIONS,
        "config": config,
        "aggregate": aggregate,
        "reports": reports,
        "overall_decision": {
            "beer_success": _law_success(aggregate["beer_lambert"]["decisions"]),
            "mk_success": _law_success(aggregate["moens_korteweg"]["decisions"]),
            "beer_success_statistical": _law_success_statistical(
                aggregate["beer_lambert"]["decisions"], aggregate["beer_lambert"]["significance"]
            ),
            "mk_success_statistical": _law_success_statistical(
                aggregate["moens_korteweg"]["decisions"], aggregate["moens_korteweg"]["significance"]
            ),
        },
    }
    path = output_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2)
    plot_summary(reports, output_dir)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/physics_credibility"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument(
        "--mk-epochs", type=int, default=400,
        help=(
            "Moens-Korteweg needs substantially more training than "
            "Beer-Lambert: extracting a phase-shift (PTT) via cross-attention "
            "is a harder optimization problem than Beer-Lambert's per-stream "
            "AC/DC ratio (verified: loss was still improving at 120 epochs "
            "and had not plateaued until ~400-600)."
        ),
    )
    p.add_argument("--mk-lr", type=float, default=3e-3)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg_kwargs = {"epochs": args.epochs}
    mk_epochs, mk_lr = args.mk_epochs, args.mk_lr
    if args.quick:
        args.seeds = [0]
        cfg_kwargs.update(
            n_train=64, n_probe=24, n_val=24, n_test=16,
            epochs=min(args.epochs, 5), n_bootstrap=3,
        )
        mk_epochs = min(mk_epochs, 5)
    reports = []
    for seed in args.seeds:
        print(f"[physics] seed={seed} device={device}", flush=True)
        cfg = LawAuditConfig(seed=seed, **cfg_kwargs)
        mk_cfg = replace(cfg, epochs=mk_epochs, lr=mk_lr)
        reports.append(run_beer_audit(cfg, device))
        reports.append(run_mk_audit(mk_cfg, device))
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    path = export_summary(reports, args.output_dir, config)
    print(f"[physics] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
