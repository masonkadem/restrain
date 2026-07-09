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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

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
    """Modified Beer-Lambert: I = I0 * exp(-mu * L) with pulsatile L."""
    s = spo2 / 100.0
    if wavelength == "red":
        mu = s * eps_red_hbo2 + (1 - s) * eps_red_hb
    else:
        mu = s * eps_ir_hbo2 + (1 - s) * eps_ir_hb
    mu *= 1.0 + scatter
    l_t = path_length * (1.0 + 0.08 * pulse)
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
    eps_red_hbo2, eps_red_hb = 320.0, 8500.0
    eps_ir_hbo2, eps_ir_hb = 1200.0, 4000.0
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
    fs: float = FS,
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
    meta = {
        "path_m": path_m, "subject_cal": subject_cal,
        "e0": e0, "alpha": alpha,
    }

    if scenario == "missing_path_length":
        path_m = float("nan")
        answerable = False
    elif scenario == "missing_stiffness_cal":
        subject_cal = float("nan")
        answerable = False
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

    return MoensKortewegSample(
        proximal=proximal, distal=distal, bp=bp,
        ptt_ms=ptt_ms, pwv=pwv, stiffness=stiffness,
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
        "target": np.array([s.bp for s in samples], dtype=np.float32),
        "ptt_ms": np.array([s.ptt_ms for s in samples], dtype=np.float32),
        "pwv": np.array([s.pwv for s in samples], dtype=np.float32),
        "answerable": np.array([s.answerable for s in samples], dtype=int),
        "scenario": [s.scenario for s in samples],
    }


# ── models ────────────────────────────────────────────────────────────────────


class TwoStreamCrossAttention(nn.Module):
    """Cross-attention between two 1D waveform streams."""

    def __init__(self, win_len: int, width: int = 64, hidden: int = 64):
        super().__init__()
        self.width = width
        self.enc_a = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, width)
        )
        self.enc_b = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, width)
        )
        self.output = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor, return_features: bool = False):
        # a, b: (B, L)
        qa = self.enc_a(a.unsqueeze(-1))
        kb = self.enc_b(b.unsqueeze(-1))
        vb = kb
        scores = torch.bmm(qa, kb.transpose(1, 2)) / math.sqrt(self.width)
        attn = scores.softmax(dim=-1)
        ctx = torch.bmm(attn, vb).mean(dim=1)
        pred = self.output(ctx).squeeze(-1)
        if return_features:
            return pred, ctx, attn
        return pred


class TwoStreamGlobalPool(nn.Module):
    """Control: global pool both streams, no indexed cross-stream retrieval."""

    def __init__(self, win_len: int, width: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor, return_features: bool = False):
        feat = torch.stack([
            a.mean(1), a.std(1), b.mean(1), b.std(1),
        ], dim=-1)
        pred = self.net(feat).squeeze(-1)
        ctx = feat
        if return_features:
            return pred, ctx, None
        return pred


def train_regressor(
    model: nn.Module,
    data: dict,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float = 1e-3,
) -> list[float]:
    model.to(device)
    a = torch.from_numpy(data["stream_a"]).float()
    b = torch.from_numpy(data["stream_b"]).float()
    y = torch.from_numpy(data["target"]).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(a))
        total = 0.0
        for start in range(0, len(a), batch_size):
            idx = order[start : start + batch_size]
            ba, bb, by = a[idx].to(device), b[idx].to(device), y[idx].to(device)
            pred = model(ba, bb)
            loss = ((pred - by) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        losses.append(total / len(a))
    return losses


def evaluate_regressor(
    model: nn.Module, data: dict, device: torch.device, batch_size: int
) -> dict:
    model.eval()
    a = torch.from_numpy(data["stream_a"]).float()
    b = torch.from_numpy(data["stream_b"]).float()
    y = torch.from_numpy(data["target"]).float()
    preds, acts = [], []
    with torch.no_grad():
        for start in range(0, len(a), batch_size):
            ba = a[start : start + batch_size].to(device)
            bb = b[start : start + batch_size].to(device)
            pred, ctx, _ = model(ba, bb, return_features=True)
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

    factories = {
        "cross_attention": lambda: TwoStreamCrossAttention(WIN_LEN, cfg.width),
        "low_capacity": lambda: TwoStreamCrossAttention(WIN_LEN, cfg.low_width),
        "no_retrieval": lambda: TwoStreamGlobalPool(WIN_LEN),
    }
    results = {}
    intervention_payload = None

    for model_name, factory in factories.items():
        model = factory()
        train_regressor(model, train, device, cfg.epochs, cfg.batch_size)
        probe_eval = evaluate_regressor(model, probe_data, device, cfg.batch_size)
        val_eval = evaluate_regressor(model, val_data, device, cfg.batch_size)
        test_eval = evaluate_regressor(model, test, device, cfg.batch_size)

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

    def mean_clean_mae(reps: list[dict]) -> float:
        vals = []
        for r in reps:
            mae = r.get("models", {}).get("cross_attention", {}).get("scenario_mae", {})
            if "clean" in mae:
                vals.append(float(mae["clean"]))
        return float(np.mean(vals)) if vals else float("nan")

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
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg_kwargs = {"epochs": args.epochs}
    if args.quick:
        args.seeds = [0]
        cfg_kwargs.update(
            n_train=64, n_probe=24, n_val=24, n_test=16,
            epochs=min(args.epochs, 5), n_bootstrap=3,
        )
    reports = []
    for seed in args.seeds:
        print(f"[physics] seed={seed} device={device}", flush=True)
        cfg = LawAuditConfig(seed=seed, **cfg_kwargs)
        reports.append(run_beer_audit(cfg, device))
        reports.append(run_mk_audit(cfg, device))
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    path = export_summary(reports, args.output_dir, config)
    print(f"[physics] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
