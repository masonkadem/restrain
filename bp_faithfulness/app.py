"""Interactive UI (Part 2). Streamlit chosen over Gradio: sliders + live
matplotlib plots + a cached dataset button are all one-liners in Streamlit, and
it keeps the whole app in one readable script with no callback wiring.

    streamlit run app.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from bpf.config import load_config
from bpf.dataset import generate_dataset
from bpf.simulator import (apply_tube_load, detect_feet, foot_to_foot_delay,
                           group_delay_dc, make_proximal_waveform)

st.set_page_config(page_title="Cuffless-BP faithfulness sandbox", layout="wide")
st.title("Tube-load simulator + PTT identifiability sandbox")

cfg = load_config()
fs = cfg["simulator"]["fs"]
hr = cfg["simulator"]["hr_bpm"]
n_beats = 8

s = st.sidebar
s.header("Simulator parameters")
T = s.slider("True one-way transit time T (ms)", 100, 220, 150) / 1e3
gamma = s.slider("Reflection Γ (shortcut knob)", 0.0, 0.8, 0.0, 0.05)
pep = s.slider("PEP jitter sd (ms) — recoverability knob", 0, 60, 0)
noise = s.slider("Noise sd (fraction of pulse amp)", 0.0, 0.1, 0.02, 0.01)
motion = s.checkbox("Motion artifact")

# build waveforms at the current settings
_, p = make_proximal_waveform(fs, hr, n_beats)
d = apply_tube_load(p, fs, T, gamma)
rng = np.random.default_rng(0)
p_obs = np.roll(p, int(round(rng.normal(scale=pep / 1e3) * fs)))
amp = p.max() - p.min()
if motion:
    d = d + 0.4 * amp * np.sin(2 * np.pi * 0.7 * np.arange(len(d)) / fs)
p_obs = p_obs + rng.normal(scale=noise * amp, size=len(p))
d = d + rng.normal(scale=noise * amp, size=len(d))
t = np.arange(len(p)) / fs

col1, col2 = st.columns([2, 1])
with col1:
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t, p_obs, color="#1a1a1a"); ax[0].set_ylabel("proximal")
    ax[1].plot(t, d, color="#7b7b7b"); ax[1].set_ylabel("distal")
    ax[2].plot(t, p_obs, color="#1a1a1a", label="proximal")
    ax[2].plot(t, d, color="#7b7b7b", label="distal")
    for sig, c in ((p_obs, "#1a1a1a"), (d, "#b04a3a")):
        feet = detect_feet(sig, fs, hr, n_beats)
        ax[2].plot(feet, np.interp(feet, t, sig), "v", color=c, ms=7)
    ax[2].set_ylabel("overlay + feet"); ax[2].set_xlabel("time (s)"); ax[2].legend()
    st.pyplot(fig)

with col2:
    ftf = foot_to_foot_delay(p_obs, d, fs, hr, n_beats)
    st.metric("True T", f"{T*1e3:.1f} ms")
    st.metric("Measured foot-to-foot", f"{ftf*1e3:.1f} ms", f"{(ftf-T)*1e3:+.1f} ms vs T")
    st.metric("Low-freq group delay T(1−Γ)/(1+Γ)", f"{group_delay_dc(T, gamma)*1e3:.1f} ms")
    st.caption("Note: the foot marks first arrival, so foot-to-foot stays ≈ T even "
               "at high Γ (reflection-robust). PEP jitter and noise are what make T "
               "unrecoverable; Γ mainly changes morphology (the shortcut route).")

st.divider()
st.subheader("Generate a multi-subject dataset at these settings")
if st.button("Generate + cache dataset"):
    ds = generate_dataset(cfg, gamma=gamma, pep_sd_ms=float(pep), seed=cfg.seed)
    st.session_state["ds"] = ds
    st.success(f"Cached {len(ds.T)} samples across {len(np.unique(ds.subject))} subjects "
               f"(shape {ds.prox.shape}). BP range "
               f"{ds.bp.min(0).round(0)}–{ds.bp.max(0).round(0)} mmHg.")
