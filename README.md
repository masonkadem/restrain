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

## 3. Causal-mediation BP audit (BP = a·PTT + confound)

The audits above detect *contextual* unanswerability: a channel is missing, so
the input is incomplete and any input-only detector could flag it. This
experiment asks a stronger, distribution-internal question: does the model
compute BP *through* the governing mediator PTT, or does it exploit a spurious
shortcut that merely predicts BP in the training distribution? A model can be
accurate in-distribution for the wrong reason.

```bash
python analysis/physics_law_credibility.py --experiment causal_bp --device cuda
python analysis/physics_law_credibility.py --experiment causal_bp --quick
```

Two pulse streams are generated where the value stream carries PTT (the
legitimate mediator, `BP = a·PTT + b`) alongside a spurious confound on a
cleaner, easier channel. A sweep varies how strongly the confound tracks PTT in
training; at test time an **out-of-distribution split decorrelates the
confound**, breaking the shortcut while leaving the causal PTT path intact.

The trustworthiness signal is **causal fidelity**: a Ridge probe locates the
PTT subspace (on decorrelated data, so it is identifiable), the model's own
context is patched so its *decoded* PTT shifts by Δ, and its BP head is read.
The recovered slope divided by the true constant `a` is the causal fidelity —
1.0 means the model routes BP through PTT as the physics requires, ~0 means it
does not. Crucially this needs **no test labels**.

Outputs under `results/physics_credibility/causal_bp/`:

- `causal_bp_summary.json`: per-model records (in-distribution MAE, OOD MAE, PTT
  causal fidelity, confound causal slope, random-direction control) and the
  aggregate decision.
- `causal_fidelity_vs_ood.png`: the headline figure — causal fidelity vs OOD
  error, coloured by train confound correlation.
- `causal_bp_accuracy_gap.png`: in-distribution accuracy stays flat while OOD
  error climbs as the shortcut takes over.

### Success rule

- Random-direction interventions are a null (near-zero fidelity)
- Higher confound reliance raises OOD error
- In-distribution accuracy does not reveal the shortcut (low ID MAE, high OOD MAE)
- **Causal fidelity, computed without test labels, negatively predicts OOD error**

This is a pre-deployment mechanism check for any frozen model that outputs BP:
low causal fidelity flags a model that will fail when the training confound
breaks. It is an external audit protocol, not intrinsic self-knowledge.

## 4. Abstract parametric-knowledge boundary

A second synthetic angle (`analysis/abstract_knowledge_boundary.py`, results in
`results/physics_abstention/abstract/`): a codebook of random key→value
associations is split into a memorized "known" half and a never-trained
"unknown" half drawn from the *identical* input distribution, so no surface cue
marks a query as unanswerable — the abstract analogue of asking about a fact
that was never in training. A frozen activation probe (fit and evaluated on
disjoint keys) detects whether the queried association was trained; a
"no-boundary" control trained on every key checks the signal is model knowledge,
not a protocol artifact.

```bash
python analysis/abstract_knowledge_boundary.py --device cuda
python analysis/abstract_knowledge_boundary.py --quick
```

Unit tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

See also `analysis/PHYSICS_ABSTENTION.md` for extended notes.
