# Physics-aware abstention proof of concept

This experiment asks a narrow question: can a frozen linear readout of model
activations identify inputs on which the prediction should not be trusted?
“Knows what it does not know” means lower selective risk after abstention. It
does not imply awareness, and linear decodability alone is not enough.

## 1. Synthetic cross-retrieval

Run from the repository root:

```bash
python analysis/synthetic_knowledge_boundary.py --device cuda
```

The clean task is the CrossSA retrieval-and-averaging task from
`analysis/representational_capacity.ipynb`. Missing selected targets create genuinely
underdetermined queries. The experiment compares adequate cross-attention, a
small attention bottleneck, and a control without indexed cross-stream
retrieval over three seeds.

Outputs under `results/physics_abstention/synthetic/`:

- `metrics.json`: clean/shifted/missing MSE, probe R², unknown-detection
  AUROC/AUPRC/calibration, within-missing-condition detection, and selective
  risk. Every missing-condition example has the same number of absent target
  slots, preventing a global missingness shortcut.
- `curves.npz`: reusable ROC and risk–coverage arrays.
- `synthetic_capacity.png`, `synthetic_unknown_detection.png`, and
  `synthetic_risk_coverage.png`.

The result supports the proof of concept only when adequate cross-attention
solves the clean task, its probe detects unanswerable queries above chance, and
probe-ranked abstention beats both full coverage and random abstention on the
pre-registered clean-plus-missing comparison. Shifted-but-answerable inputs are
reported separately as a harder epistemic stress test.

## 2. Physics-law credibility benchmark (Beer–Lambert + Moens–Korteweg)

```bash
python analysis/physics_law_credibility.py --device cuda
python analysis/physics_law_credibility.py --quick
```

Tutorial notebook: `analysis/physics_law_credibility.ipynb`

Interactive report (serve over HTTP):

```bash
python -m http.server 8000
# open http://localhost:8000/analysis/credibility_report/index.html
```

### Equations (simplified simulators)

**Beer–Lambert:** \(I_\lambda = I_0 \exp(-\mu_\lambda c L(t))\);  
\(R = (AC_{red}/DC_{red}) / (AC_{ir}/DC_{ir})\); SpO₂ from calibration.

**Moens–Korteweg:** \(PWV = \sqrt{Eh/(\rho D)}\); \(PTT = L/PWV\);  
\(E(P) = E_0 \exp(\alpha P/100)\). Uncalibrated PTT does not uniquely determine BP.

### Scenarios

| Law | Identifiable | Held-out violations |
|-----|--------------|---------------------|
| Beer–Lambert | both wavelengths present | missing channel, wavelength mismatch, motion |
| Moens–Korteweg | path + calibration present | missing distal, stiffness/path missing |

Outputs under `results/physics_credibility/`:

- `summary.json`: per-seed reports, aggregate metrics, pass/fail decisions
- `beer_lambert_scenario_mae.png`, `moens_korteweg_scenario_mae.png`

### Literature positioning

Prior work covers synthetic pulse oximetry (Monte Carlo), simulated PTT→BP,
physics-informed networks, and uncertainty-aware hemodynamic models. This
benchmark’s contribution is the **cross-law activation audit protocol** with
explicit identifiability boundaries and abstention—not claiming novelty for
the physics simulators alone.

### Success rule (per law, across seeds)

- Cross-attention beats low-capacity and no-retrieval controls on clean data
- Law statistics decodable (probe R² > 0 on held-out clean)
- Answerability AUROC > 0.5 on held-out corruption families
- Probe AURC < random AURC
- Probe-direction activation ablation hurts readout more than random direction

Report negative/mixed results directly.

### Cross-law transfer

```bash
python analysis/cross_law_transfer.py
```

Trains a compact answerability signature on one law and evaluates it
zero-shot on the other, testing whether the "I cannot answer" signal is
domain-general. Result: a sharp, reproducible asymmetry (Moens–Korteweg →
Beer–Lambert transfers, Beer–Lambert → Moens–Korteweg does not) — evidence
for a domain-general *mechanism*, not a domain-general *direction*.

### Identifiability toy model (γ sweep)

```bash
python analysis/toy_identifiability.py
python analysis/toy_identifiability.py --quick
```

A minimal linear inverse problem where the identifiability gap γ is dialed
exactly. Its headline finding: at *matched* γ, an input-visible ("zeroed")
corruption is detected and gating captures ~85–90% of the oracle benefit,
while an in-distribution ("resampled") corruption of identical error
magnitude is undetectable and gating gains nothing — even at γ = 171.
**Detectability tracks input-visibility, not error size.** Output:
`results/toy_identifiability/toy_gating.png` (four panels, including the
canonical risk–coverage "the model restrains itself" curve).

**Reproduce it in one notebook:** `analysis/identifiability_toy_model.ipynb`
runs the whole thing top to bottom (~2 min, CPU) with a plain-language
explanation of each step.

Geometry of the answerability signal:

```bash
python analysis/toy_geometry.py
```

Asks *how* "unanswerable" is arranged in activation space: it is
approximately low-rank — one dominant "answerability axis" carries ~88% of
the detection, with a tail smeared over roughly the corrupted-block
dimensionality (`results/toy_identifiability/toy_geometry.png`).

Can an ensemble break the single-pass ceiling?

```bash
python analysis/toy_ensemble.py
```

Pits three detectors (single-pass probe, label-free ensemble disagreement,
label-free Mahalanobis OOD) against three corruptions. Ensembles/OOD **do**
catch a large off-manifold corruption the probe misses (label-free), but
every detector has a blind spot and **nothing** catches an on-manifold
corruption — the boundary is data-manifold membership, not the number of
forward passes (`results/toy_identifiability/toy_ensemble.png`).

### Physics-grounded gating proof of concept

```bash
python analysis/physio_gating_poc.py
```

The clean toy-task design instantiated on two *real governing laws*
(Beer–Lambert → SpO₂, Moens–Korteweg → BP). Trains a model on the law,
freezes it, and gates on a linear probe of its activations. Produces, per
law, a risk–coverage curve (the model restraining itself: probe hugs the
oracle, far below random) and a "who catches which failure" panel across
three clinical failure modes — missing channel, saturated/distorted channel,
and a plausible-but-mismatched channel. Outputs under `results/physio_poc/`.
**Reproduce and read it in one notebook:**
`analysis/physio_gating_proposal.ipynb` (proposal-ready, ~3 min, CPU).

Unit tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

### Further reading

- `analysis/SUMMARY.md` — structured, honest write-up of method, theory,
  and frozen results (including the missingness-shortcut limitation).
- `analysis/THEORY.md` — the identifiability-gap (γ) mechanism and its
  empirical checks.
- `analysis/dashboard.html` — interactive multi-panel results dashboard.
- `analysis/PHYSICS_ABSTENTION.md` — extended notes.
