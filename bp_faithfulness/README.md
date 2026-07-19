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

## What the headline figure shows

- **True positive** (Γ=0, PEP=0): probe R² ≈ 0.9, well above the shuffle floor;
  donor-swap sign accuracy ≈ 90%.
- **PEP sweep** (Γ=0): probe R² collapses toward 0 and abstention rises as PEP
  grows — the recoverability collapse.
- **Γ sweep** (PEP=0): probe R² stays high (Γ is *not* a recoverability knob) while
  BP MAE falls as the model exploits the reflection shortcut — a controlled
  negative for the recoverability hypothesis and a window on the null branch.
