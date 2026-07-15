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

### Model auditing: decoding PTT from a model's activations

```bash
python analysis/ptt_gate.py
```

A model-certification demo. A model maps pulse waveforms (+ calibration) to
blood pressure; we check whether it encodes the physically-required
pulse-transit-time (PTT) — **using only synthetic waveforms, no access to the
model's original training data**. Two per-input gates on the frozen model:

- **threshold** — is the decoded PTT in physiological range? (catches gross
  failures; here it catches *nothing* — the model always decodes a plausible
  in-range PTT even when wrong)
- **consistency** — does the decoded PTT (from activations) match the PTT
  *measured directly from the raw signal* (cross-correlation)? This catches
  the failures the threshold misses.

Result: PTT decodes at R² ≈ 1.0 on clean data; the consistency gate catches a
missing distal channel (AUROC 1.00) and a novel pulse morphology the model
misreads (0.86), but **not** a wrong-subject calibration (0.51) — because the
waveform PTT is genuinely correct there, so the failure is a non-observable
identifiability problem, not a PTT-competence one. Output:
`results/ptt_gate/ptt_gate.png`.

## 3. Causal-use audit toy (encoding vs. causal use)

```bash
python analysis/causal_mediation_toy.py --law blood_pressure --quick
python analysis/causal_mediation_toy.py --law saturation --quick
```

A minimal, self-contained demonstration that *decodability is not trust*. Two
governing laws are supported, each with a two-component structure where the
answer is only determined when **both** components are used:

- `blood_pressure` — `BP = P₀ + S·[2 ln(L/PTT) − ln(E₀/E_ref)]` (Moens–Korteweg:
  pulse transit time **PTT** plus a per-subject arterial-stiffness calibration
  **E₀**). Uncalibrated PTT does not determine BP — the textbook cuffless-BP
  failure mode.
- `saturation` — `y = v_i / (v_i + v_j)` (Beer–Lambert-flavored ratio).

A two-query cross-attention retriever names slots `i` and `j`, retrieves their
values into separate subspaces, and predicts the answer. Two models share the
same architecture and task: the **law** model is trained where component `j`
(the calibration) varies; the **shortcut** model is trained where `j` is
effectively constant, so reading `i` alone predicts the answer — the everyday
failure of a confounded or narrow training set (a cuffless-BP model fit on a
cohort with similar arterial stiffness).

**Three models** share the architecture and task (see the table below). On the
validation distribution a practitioner actually has (`j` barely varies) all three
are indistinguishable — and *this is the point* (blood-pressure law, `--quick`,
3 seeds, mean):

| check | law | unfaithful | shortcut | separates law? |
|-------|-----|-----------|----------|----------------|
| validation MSE (`j`≈const) | ~0 | ~0 | ~0 | no |
| decodability probe R² | ~0.99 | ~0.9 | ~0.9 | no |
| uses `j`? gradient magnitude `[0-1]` | ~1.0 | ~1.0 | ~0.0 | **no** (misses unfaithful) |
| **interchange-intervention accuracy** | **~0.99** | **~−0.55** | **~0.18** | **yes** |
| OOD MSE, revealed after (`j` varies) | ~0 | ~0.70 | ~0.17 | **yes** |

- **law** — trained with `j` varying → uses both components (true law).
- **unfaithful** — trained on a *wrong-form* law of `j` (for BP, the calibration
  term with the wrong sign) → it *does* use `j` (nonzero, correct-magnitude
  gradient) but implements the wrong equation.
- **shortcut** — trained with `j` fixed → ignores `j`.

Predictive error, a decodability probe, and even a gradient-*magnitude*
sensitivity check all fail to separate the faithful model from the unfaithful
one. Only the **interchange-intervention audit** (a DAS-style alignment search:
is there a subspace of the internal state that, swapped between two inputs,
transfers the answer as the equation predicts?) isolates the law model — and the
held-out `j`-varying error, revealed only afterward, confirms it. The equation
is the oracle (evaluated at known component values); no measured OOD label is
used to reach the verdict.

Outputs under `results/causal_mediation_toy/<law>/`:

- `causal_mediation.png` — six panels: task schematic, indistinguishable
  in-distribution fit, OOD reveal, counterfactual response curve, interchange
  accuracy vs. subspace dimension (with seed error bars), and a scorecard
  showing the conventional checks agreeing while the audits separate.
- `metrics.json` — per-metric mean ± s.d. over seeds.
- `../audit_map.png` (from `python analysis/audit_map.py`) — precise diagram of
  the forward computation and exactly where each audit reads (`h`) or intervenes.

Precise formulation of the task, architecture, models, and each audit:
`analysis/CAUSAL_AUDIT_MATH.md`. Concise walkthrough notebook (abstract BP →
BP simulator → SpO₂): `analysis/causal_audit_walkthrough.ipynb`. **Remaining
work and the real-data plan:** `analysis/NEXT_STEPS.md`.

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
