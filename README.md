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

## 3. Causal-use audit toy (encoding vs. causal use)

```bash
python analysis/causal_mediation_toy.py
python analysis/causal_mediation_toy.py --quick
```

A minimal, self-contained demonstration that decodability is not trust. The
governing law is a two-component saturation `y = v_i / (v_i + v_j)` (a
Beer–Lambert-flavored ratio). A two-query cross-attention retriever names slots
`i` and `j`, retrieves their values, and predicts `y`.

Two models share the same architecture and task. The **law** model is trained
where component `j` varies; the **shortcut** model is trained where `j` is
effectively constant, so reading `i` alone predicts `y` — the everyday failure
of a confounded or narrow training set.

On the validation distribution a practitioner actually has (where `j` barely
varies) the two are indistinguishable, and *this is the point*:

| metric | law | shortcut |
|--------|-----|----------|
| validation MSE (`j`≈const) | ~0 | ~0 |
| decodability probe R² for the ratio | ~1.0 | ~1.0 |
| **interchange-intervention accuracy (the audit)** | **~0.99** | **~0.51** |
| held-out MSE, revealed after the audit (`j` varies) | ~0 | ~0.015 |
| `\|dy/dv_j\|` model vs. equation | 0.26 / 0.26 | 0.00 / 0.26 |

Predictive error and a linear probe cannot tell the models apart. Two
post-hoc, label-free, frozen-model audits derived from the equation do:

- **Counterfactual sensitivity** to component `j` — the equation requires
  `dy/dv_j != 0`; the shortcut model is insensitive.
- **Interchange-intervention accuracy** (a DAS-style alignment search): is there
  a subspace of the internal state that, swapped between two inputs, transfers
  `y` as the equation predicts? A model that merely *encodes* the ratio fails;
  one that *causally routes* through it passes.

The held-out `j`-varying error is revealed only afterward and confirms the
audit's verdict — the audit substitutes for OOD labels the practitioner does
not have. Outputs under `results/causal_mediation_toy/`:
`metrics.json`, `causal_mediation.png`.

Unit tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

See also `analysis/PHYSICS_ABSTENTION.md` for extended notes.
