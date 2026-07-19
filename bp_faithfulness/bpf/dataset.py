"""Multi-subject dataset generation (Part 1 -> data) with the option-A design.

Generative story (documented so nothing is hidden):

  * Per sample the latent is the transit time T (in T_range). It drives BP:
        SBP = K1_sys/T + K2_sys ,  DBP = K1_dia/T + K2_dia     (Mukkamala eq. 11)
    with subject-specific K's, PLUS a NON-T "vascular tone" driver z ~ N(0,1):
        SBP += confound_bp * z ,   DBP += 0.6 * confound_bp * z
  * The proximal waveform's foot is jittered per beat by PEP (recoverability knob):
    the observed proximal marker is PAT = PTT + PEP, so timing no longer pins T.
  * The distal waveform is the tube-load output at (T, gamma). We ALSO inject the
    confound z into distal morphology with amplitude proportional to gamma:
        distal += (confound_gain * gamma * z) * slow_shape
    So at gamma=0 the z-part of BP is irreducible noise (a MAE floor); as gamma
    rises, z becomes readable from morphology -> the model can predict BP without
    using T (the null branch). gamma is the shortcut knob, PEP/noise are the
    recoverability knobs.

Each sample stores the true T (probe target) and (K's, z) which are NEVER shown
to the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulator import apply_tube_load, make_proximal_waveform


@dataclass
class Dataset:
    prox: np.ndarray     # (N, L) observed proximal (with PEP jitter + noise)
    dist: np.ndarray     # (N, L) observed distal (reflections + confound + noise)
    T: np.ndarray        # (N,) true transit time  -- PROBE TARGET, hidden from model
    bp: np.ndarray       # (N, 2) [SBP, DBP]        -- model target
    subject: np.ndarray  # (N,) subject id
    z: np.ndarray        # (N,) confound value      -- hidden


def _slow_shape(L: int) -> np.ndarray:
    """A smooth low-frequency template the confound rides on (one half-cosine)."""
    x = np.linspace(0, np.pi, L)
    return np.sin(x)


def _subject_params(rng, cfg) -> dict:
    s = cfg["subjects"]
    return dict(
        K1_sys=rng.uniform(*s["K1_sys"]), K2_sys=rng.uniform(*s["K2_sys"]),
        K1_dia=rng.uniform(*s["K1_dia"]), K2_dia=rng.uniform(*s["K2_dia"]),
        amp=1.0 + rng.uniform(-s["amp_jitter"], s["amp_jitter"]),   # from l, A
        hr=cfg["simulator"]["hr_bpm"] * (1.0 + rng.uniform(-0.1, 0.1)),
    )


def generate_dataset(cfg, gamma=None, pep_sd_ms=None, seed=0,
                     t_fixed=False, wrong_sign=False) -> Dataset:
    """Build a full multi-subject dataset at the given (gamma, pep) setting.

    Three-model regimes (aligned with the abstract Act-1 task):
      * faithful   : defaults (T varies, true BP).
      * shortcut   : t_fixed=True + gamma>0 -> T does not vary in training, so BP
                     variation comes only from the reflection-morphology confound;
                     the model has no reason to encode PTT.
      * unfaithful : wrong_sign=True -> the T->BP relation is inverted, so the
                     model uses PTT but the WRONG way.
    """
    sim = cfg["simulator"]
    fs = sim["fs"]
    gamma = sim["gamma"] if gamma is None else gamma
    pep_sd = (sim["pep_sd_ms"] if pep_sd_ms is None else pep_sd_ms) / 1e3
    noise_sd, cbp, cgain = sim["noise_sd"], sim["confound_bp"], sim["confound_gain"]
    n_sub, per = cfg["subjects"]["n"], cfg["subjects"]["samples_per_subject"]
    T_lo, T_hi = sim["T_range_s"]
    T_mid = 0.5 * (T_lo + T_hi)
    rng = np.random.default_rng(seed)

    prox_all, dist_all, T_all, bp_all, sub_all, z_all = [], [], [], [], [], []
    for sid in range(n_sub):
        sp = _subject_params(rng, cfg)
        Tb = 60.0 / sp["hr"]
        L = int(round(sim["seg_beats"] * Tb * fs))
        slow = _slow_shape(L)
        for _ in range(per):
            T = T_mid if t_fixed else rng.uniform(T_lo, T_hi)   # shortcut: T does not vary
            z = rng.normal()
            sbp = sp["K1_sys"] / T + sp["K2_sys"] + cbp * z
            dbp = sp["K1_dia"] / T + sp["K2_dia"] + 0.6 * cbp * z

            _, p = make_proximal_waveform(fs, sp["hr"], sim["seg_beats"])
            p = p[:L] * sp["amp"]
            amp = p.max() - p.min()

            d = apply_tube_load(p, fs, T, gamma)[:L]
            d = d + (cgain * gamma * z) * slow * (0.15 * amp)   # confound morphology

            # Random ABSOLUTE phase applied to BOTH channels: removes any absolute-
            # timing leak, so T is recoverable ONLY from the RELATIVE prox->distal
            # delay -- which is exactly what PEP then corrupts (real PTT is relative).
            phi = int(rng.integers(0, L))
            p_g, d = np.roll(p, phi), np.roll(d, phi)
            # PEP jitter: extra relative shift of the OBSERVED proximal only, so the
            # measured marker is PAT = PTT + PEP, decoupled from true T.
            pep = rng.normal(scale=pep_sd)
            p_obs = np.roll(p_g, int(round(pep * fs)))

            p_obs = p_obs + rng.normal(scale=noise_sd * amp, size=L)
            d = d + rng.normal(scale=noise_sd * amp, size=L)

            prox_all.append(p_obs.astype("float32")); dist_all.append(d.astype("float32"))
            T_all.append(T); bp_all.append([sbp, dbp]); sub_all.append(sid); z_all.append(z)

    # pad/truncate to common length (subjects differ in HR -> L)
    Lmax = max(len(x) for x in prox_all)
    def stack(list_):
        return np.stack([np.pad(x, (0, Lmax - len(x))) for x in list_]).astype("float32")
    bp = np.array(bp_all, dtype="float32")
    if wrong_sign:                                # unfaithful: invert the T->BP relation
        bp = (2 * bp.mean(0) - bp).astype("float32")
    return Dataset(stack(prox_all), stack(dist_all), np.array(T_all),
                   bp, np.array(sub_all), np.array(z_all))


def subject_split(ds: Dataset, train_subjects: int, seed=0):
    """Subject-level split: first `train_subjects` train, rest test (no overlap)."""
    ids = np.unique(ds.subject)
    tr = set(ids[:train_subjects].tolist())
    train = np.array([i for i in range(len(ds.T)) if ds.subject[i] in tr])
    test = np.array([i for i in range(len(ds.T)) if ds.subject[i] not in tr])
    return train, test
