# Cuffless-BP faithfulness sandbox

An interactive research tool for validating a **faithfulness probe**: given a
trained blood-pressure model, does it *internally encode pulse transit time
(PTT)*, or is it accurate for some other reason? Because we generate the data
from a known physics model, we control ground truth and can demonstrate both
**true positives** (the probe fires when PTT is genuinely used) and **true
negatives** (it stops firing when PTT is unrecoverable). *A verification method
that always fires is not a verification method.*

## The physics, in plain language

A pressure pulse travels down an artery and partially **reflects** off the
periphery. We model this with a lossless uniform tube plus a terminal reflection
(Mukkamala et al., IEEE TBME 2015, Fig 5b; Rashedi et al., JBME 2013). We do not
solve a PDE — we use the transfer function directly:

```
        (1 + Γ) · exp(-j 2π f T)
H(f) = ----------------------------
        1 + Γ · exp(-j 4π f T)
```

- **T** = true one-way transit time (ground truth, never shown to the model).
- **Γ** = reflection coefficient (0 → 0.8).
- Blood pressure comes from T via `BP = K1/T + K2` (Mukkamala eq. 11).

`H` is applied as a discrete FFT filter to a synthetic proximal pulse. Two facts
are derived by hand and checked numerically (`verify_physics.py`,
`tests/test_simulator.py`):

1. **Unit DC gain**, `H(0) = 1`.
2. **Low-frequency group delay** `τ_g(0) = T·(1−Γ)/(1+Γ)` — plus a quarter-wave
   resonance at `f = 1/(4T)` with peak gain `(1+Γ)/(1−Γ)`.

## The identifiability argument (important, and corrected from first intuition)

The reflection series is
`distal(t) = (1+Γ)[prox(t−T) − Γ·prox(t−3T) + Γ²·prox(t−5T) − …]`.
The **first arrival is always at T**; reflections arrive later (3T, 5T, …). So:

- The **foot** (upstroke onset) marks first arrival and stays ≈ T for *all* Γ —
  foot-based PTT is reflection-robust. **Γ alone does not make PTT
  unrecoverable.** (T is even over-determined: it also sets the resonance
  frequency `1/(4T)`.)
- What genuinely makes PTT unrecoverable is **PEP** (`PAT = PTT + PEP`, a per-beat
  proximal-marker jitter that decouples the measured timing from T) and **noise**.
- **Γ is best cast as the *shortcut* knob**: it grows a reflection/augmentation
  morphology that lets a model predict BP *without* using T (the null branch).

So the sweeps below use **PEP + noise as the recoverability knobs** and **Γ as the
shortcut knob** — and the tool is built to detect the "accurate without encoding
PTT" outcome, never assuming the probe should succeed.

## The faithfulness probe battery (Part 4)

- **Linear probe** of true T from frozen activations (ridge), with bootstrap CIs,
  a ridge–lasso gap, and bootstrap **stability selection** over units.
- **Shuffle control**: refit against shuffled T; the probe is meaningful only if
  it beats this floor.
- **Causal ablation**: remove the probe direction from activations; if BP error
  does not rise, the model was not using it.
- **Donor-swap activation patching**: inject a donor subject's PTT component and
  test the BP prediction shifts with the **sign and bounded magnitude** the
  physics predicts (not just correlation).
- **Abstention** via split conformal, with the probe's faithfulness readout as the
  nonconformity score, **calibrated per subject** with a burn-in. Cross-subject
  shift breaks exchangeability — this is handled per subject, and the spot for a
  conformal-beyond-exchangeability variant is marked in `bpf/analysis.py`.

## Reproducibility

- All thresholds (probe significance, conformal target coverage) live in
  `config.yaml`, read by the sweep — pre-registered, not tuned after seeing
  results.
- Everything is seeded from the config.
- Sweep results are saved to `results/` so figures regenerate without retraining.

## Run it

```bash
pip install -r requirements.txt

python verify_physics.py                      # confirm the transfer function
python -m unittest -v tests.test_simulator    # unit tests
python run_demo.py                            # baseline probe battery + sweeps + figure
python run_demo.py --quick                    # faster
streamlit run app.py                          # interactive sandbox
```

`run_demo.py` writes `results/comprehensive.png` (8-panel figure: waveforms,
foot-to-foot vs T, probe-fires, ablation, donor-swap, Γ sweep, PEP sweep,
per-subject MAE), plus `demo_results.npz` and `summary.json`.

## Modeling note (a finding, kept honest)

We first tried a small **from-scratch transformer** — the natural mech-interp
substrate. At this data scale (~500 training samples) it did not fit the task
(probe R² for the true intermediate stayed at chance, BP MAE stayed at
mean-prediction), while a compact **1D-CNN** learns it cleanly (probe R² ≈ 0.9).
Transformers are data-hungry; a physiological-scale toy is not where they shine.
Crucially the faithfulness battery is **architecture-agnostic** — probing, causal
ablation, donor-swap patching, and input saliency all operate on `represent()` /
`from_h()` and the raw inputs — so we use the CNN and keep the audit general.

## PAT vs PTT (which intermediate to probe)

- **ECG + PPG** model → the observable interval is **PAT = PEP + PTT** (ECG R-peak
  to distal foot). PEP is a confound, so PAT is a *noisy* proxy of the BP-relevant
  PTT. The proximal marker in the simulator, jittered by PEP, plays exactly this
  ECG-R-wave role — so the **PEP knob is the PAT→PTT confound**.
- **PPG + PPG** (two optical sites) → the observable is **PTT** directly (no PEP).

## The self-explaining figure (`results/comprehensive.png`)

Six panels; each carries a plain-language point and a one-line pseudocode of what
it computes:

- **A** the two-channel input (proximal marker + distal PPG).
- **B** the probe **fires** at Γ=0, PEP=0 (R² ≈ 0.9 vs a near-zero shuffle floor).
- **C** the **causal** donor-swap: patching the PTT direction shifts BP with the
  sign physics predicts (~90%).
- **D** the **benchmark** — the PEP recoverability sweep: as PEP hides PTT, the
  probe correctly stops firing (discriminant validity; *a method that always
  fires is not a verification method*).
- **E** BP MAE over the same sweep (rises here → accuracy depends on PTT; a *flat*
  MAE while D collapses would flag the null branch).
- **F** input saliency concentrating at the upstroke feet (the PTT landmarks).

`config.yaml` also exposes Γ (the reflection/shortcut knob) and a `--`sweep over
it; Γ is deliberately **not** the recoverability knob (foot-based PTT is
reflection-robust, see the identifiability argument above), so it serves as a
controlled negative.
