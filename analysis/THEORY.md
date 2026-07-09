# When is an identifiability-violation indicator linearly decodable?

This is a theory sketch, not a peer-reviewed proof — the goal is to state a
falsifiable mechanism for *why* frozen activations should encode
answerability at all, precise enough that the empirical results in this repo
either support or contradict it, rather than leaving "probes just seem to
work" unexplained.

## Setup

Let `x` be an input, `y` the regression target, and `a(x) ∈ {0,1}` the
*true* answerability label, defined by construction from the governing
equation (see the main README's "manufacturing provably unanswerable
inputs" methodology) — `a(x)=1` iff every variable the equation's inverse
requires is present and consistent in `x`.

Let `A = {x : a(x)=1}` (answerable) and `A^c` (unanswerable). On `A`, assume
`y = f(x) + ε` with `ε` zero-mean noise of variance `σ²_A` (the governing
equation determines `y` up to measurement noise). On `A^c`, by definition of
non-identifiability, `y` is **not** a function of the observed `x` alone —
some required variable is missing or corrupted, so the true conditional
distribution `y | x` has strictly higher variance/entropy than on `A`.
Write `σ²_{A^c} ≥ γ · σ²_A` for some `γ > 1` (the "identifiability gap").

Let `z = φ(x) ∈ R^d` be the model's frozen bottleneck activation and
`ŷ = g(z)` the readout, trained under a proper scoring rule (MSE) to
convergence with enough capacity and data that `ŷ(x) ≈ E[y | x]`
(Bayes-consistent on the training distribution).

## Proposition (informal)

Define the pointwise expected loss `ℓ(x) = E[(y - ŷ(x))² | x]`. Because a
Bayes-consistent predictor cannot reduce loss below the *irreducible*
conditional variance, `ℓ(x) ≥ σ²_A` on `A` and `ℓ(x) ≥ γσ²_A` on `A^c` in
expectation. **If any linear functional `w·z` of the bottleneck is
monotonically related to `ℓ(x)`** — which holds whenever `z` retains enough
information for `g` to express its own confidence (e.g. an
attention-entropy feature, an activation-norm feature, or, for an ensemble
readout, prediction disagreement) — **then `a(x)` is linearly separable
from `z` up to the separability of `ℓ(x)` between `A` and `A^c`**, governed
by the identifiability gap `γ`: by a standard signal-detection argument, if
`ℓ(x)|A` and `ℓ(x)|A^c` are (approximately) Gaussian with means `σ²_A` and
`γσ²_A` and comparable spread, `AUROC(a(x); w·z) → 1` as `γ → ∞` and
`→ 0.5` as `γ → 1`.

**The mechanism does not require the model to be told which inputs are
unanswerable.** It only requires that unanswerable inputs genuinely have
higher irreducible label variance given what the model observes, and that
training pushes the model to (implicitly) represent its own uncertainty
about `y` somewhere linearly readable in `z`. Both are structural
consequences of training on the forward regression task, not of any
epistemic-awareness supervision — this is the sense in which "the model was
never told which inputs are unanswerable, and its representations carry the
information anyway."

## Corollary: why transfer should be partial and law-specific, not universal

The identifiability gap `γ` is a property of the *task*, not of the
representation. A law whose unanswerable-condition irreducible variance is
only marginally larger than its answerable-condition noise floor (small
`γ`) will produce a **weak, low-margin** separating direction — one more
easily dominated by task-specific noise in the direction a linear probe
happens to fit. A law with a large `γ` produces a **wide-margin**
direction. Nothing in the mechanism implies these directions point the same
way in any shared space across two structurally unrelated models — they are
each fit independently to each task's own noise structure. The theory
therefore predicts:

1. Within a single law, the probe should separate answerable from
   unanswerable well above chance, with the margin scaling with that law's
   `γ`.
2. Cross-law transfer of a *compact, comparable* signature (not raw
   activations, which aren't even the same dimensionality) should work
   better in the direction *from* the law with large, clean `γ` *to* the
   law with small, noisier `γ` than the reverse — the noisy-`γ` law's
   fitted direction has less true signal to transfer in the first place.

## Empirical check 1: the synthetic slot-retrieval task (cleanest case)

`generate_cross_retrieval_data` constructs `missing`-condition queries where
the true label depends on slot values the network's input has literally
been zeroed for (`observed_values = values * availability[...,None]`, but
`labels` is computed from the un-masked `values`) — an unanswerable query
by exactly the construction above, since the true label is not a function
of what the model observes. From the committed results
(`results/physics_abstention/synthetic/metrics.json`, seed 0):

| condition | MSE |
|---|---|
| clean | 0.0485 |
| missing | 0.0593 (`γ ≈ 1.22`) |
| shifted (answerable, distribution-shifted) | 0.0654 |

`answerability_within_missing_condition` AUROC = **0.757** — well above
chance, consistent with a real but moderate `γ ≈ 1.22`. Note "shifted"
(answerable but distributionally unusual) has *higher* MSE than "missing"
here, yet the probe is trained to target genuine non-identifiability, not
generic difficulty — the theory's mechanism is about `ℓ(x)` separating `A`
from `A^c` specifically, not about difficulty in general, which is exactly
why the benchmark's held-out "shifted" condition exists as a stress test
for whether the probe over-fires on merely-unusual-but-answerable inputs.

## Empirical check 2: cross-law transfer (`analysis/cross_law_transfer.py`)

Mean over 3 seeds, using the compact 3-feature signature (residual
z-score, activation-norm z-score, probe-confidence percentile):

| | AUROC (mean of 3 seeds) |
|---|---|
| within-domain, Beer-Lambert | 0.86 |
| within-domain, Moens-Korteweg | 0.997 |
| Moens-Korteweg → Beer-Lambert (transfer) | 0.75 |
| Beer-Lambert → Moens-Korteweg (transfer) | **0.31** (below chance) |

This is exactly the predicted asymmetry, sharp and reproducible (per-seed
beer→mk: 0.302 / 0.312 / 0.319; mk→beer: 0.770 / 0.717 / 0.750). MK's
identifiability gap is large and clean (missing calibration makes BP
*undecidable*, not just noisier — the Moens-Korteweg simulator's
"missing_stiffness_cal"/"missing_path_length" conditions remove information
the closed-form equation strictly requires) — its within-domain ceiling is
near-perfect (0.997), and its fitted direction carries enough real signal to
transfer moderately even into an unrelated domain (0.75). Beer-Lambert's
gap is smaller (a missing channel still leaves a well-defined, bounded
signal, just wrong) and its within-domain ceiling is correspondingly lower
(0.86); the direction fit to it is noisier and — per the theory — carries
comparatively little transferable signal. That it transfers *below* chance
rather than merely near it suggests the sign of at least one feature's
relationship to `a(x)` is task-specific in a way this simple linear
signature doesn't capture; resolving that (e.g. per-law feature calibration
before combining) is a natural next step rather than a refutation of the
mechanism.

**Bottom line:** the theory predicts a domain-general *mechanism*
(uncertainty-correlated activations, driven by the task's own
identifiability gap), not a domain-general *direction*. The data are
consistent with that weaker, more precise claim, and inconsistent with the
stronger "universal semantic direction" claim the "cross-law audit
protocol" name could be read as implying — which is itself a useful,
falsifiable distinction to have sharpened.

## Empirical check 3: the toy model, and a correction to the naive γ story

The minimal instrument in `analysis/toy_identifiability.py` dials the gap γ
exactly and reveals a subtlety the physiological simulators could only hint
at. The naive reading of the proposition above — "detection AUROC → 1 as
γ → ∞" — is **wrong as stated**, and the toy model shows why.

The proposition is about `ℓ(x) = E[loss | x]` separating `A` from `A^c`. But
a probe fit to the *answerability label* `a(x)` can only succeed when `a(x)`
is (statistically) a function of the observable input `x`. Two corruptions at
*identical* γ pull these apart:

- **Zeroed** (a channel set to 0): unanswerability is written into `x` — the
  peripheral coordinates are exactly zero. `a(x)` is a function of `x`, so a
  probe detects it (AUROC ≈ 0.9) and gating captures ~85–90% of the oracle
  benefit — **at every γ, essentially flat in γ**.
- **Resampled** (a channel replaced by an in-distribution draw): the marginal
  input distribution is *unchanged*, so `a(x)` is independent of `x` even
  though the error is huge. No single deterministic forward pass can
  distinguish it: detection AUROC stays at chance (0.49–0.51) and gating
  benefit stays at zero — **even at γ = 171** (`results/toy_identifiability/sweep.json`).

So the corrected statement: **γ governs the *loss* separation, but
single-pass detectability is governed by whether unanswerability is a
function of the observable input, not by γ.** The identifiability gap tells
you how much there is to gain from abstaining; input-visibility tells you
whether a frozen probe can realize any of it.

This exactly explains the physiological results. Missing channels /
calibration are zeroed-type (input-visible) → detected. Held-out *non-missing*
Beer–Lambert corruptions that barely move the input distribution are
resampled-type → detection collapses to chance. It also sets a hard ceiling
on the whole approach: **a purely value-level corruption that preserves the
input distribution is undetectable by single-pass activation probing, no
matter how wrong the answer becomes.** Escaping that ceiling requires signals
a single forward pass does not have — ensembles, test-time perturbations, or
access to the corrupted variable itself.
