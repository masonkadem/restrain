# Physics-aware abstention: a proof-of-concept summary

This document ties together the method, theory, and frozen experimental
results in this repository. It is written to be honest about what holds up
and what does not — the mixed and negative results are the point, not a
failure to paper over. An interactive version of the results is in
`analysis/dashboard.html`.

## 1. Motivation and the one novel idea

The question: **can a frozen linear read-out of a network's activations
identify inputs on which its prediction should not be trusted, and can
gating on that read-out let the model restrain itself when it cannot know?**

The individual ingredients — selective prediction, "does the model know what
it knows" probing, directional-ablation causal tests — are each established.
The one genuinely novel contribution is the **labeling methodology**:

> Manufacture provably-unanswerable inputs from a governing equation, giving
> *verifiable ground-truth epistemic labels*.

Almost all "does the model know what it doesn't know" work labels an input
"unknown" by a proxy (distributional distance, ensemble disagreement, human
judgment of difficulty). Here the label is a *mathematical fact*: a governing
equation `y = f(x)` is identifiable only when every variable its inverse
requires is present. Remove one — zero a wavelength channel, withhold a
calibration constant — and the inverse problem is underdetermined by
construction. Because we build the simulator, the answerable/unanswerable
label is exact, not estimated. Physiology (pulse oximetry, cuffless BP) is
the demonstration domain, not the claim.

## 2. Method: the cross-law activation audit protocol

Two governing laws, chosen for two *different* ways to become unanswerable:

| Law | Target | Identifiable when | Unanswerable (held-out families) |
|-----|--------|-------------------|----------------------------------|
| Beer–Lambert | SpO₂ | both wavelength channels present | missing channel, wavelength mismatch, motion |
| Moens–Korteweg | BP | transit path + subject stiffness calibration known | missing calibration, missing path, missing distal |

Protocol (identical per law, no epistemic supervision anywhere):

1. **Train** a two-stream cross-attention regressor on clean/identifiable data only.
2. **Freeze** all weights.
3. **Probe** answerability with a linear read-out of frozen activations.
4. **Gate**: rank by probe confidence, abstain on the low-confidence tail,
   measure selective risk (area under the risk–coverage curve).
5. **Ablate**: remove the probe direction from the activations and confirm
   the readout degrades more than under a random direction of equal norm
   (a causal, not merely correlational, test).

Controls: a low-capacity bottleneck and a no-retrieval global-pool model
establish whether the "adequate" architecture is actually needed. Held-out
corruption families (never seen when fitting the probe) test whether
detection generalizes beyond memorized patterns. Shifted-but-answerable
scenarios test whether the probe over-fires on merely-unusual inputs.

## 3. Theory: the identifiability gap (see `THEORY.md`)

A Bayes-consistent readout cannot drive its loss below the *irreducible*
variance of `y` given what it observes. On unanswerable inputs that variance
is strictly higher; call the ratio the identifiability gap **γ > 1**. Then
the model's own pointwise loss — and anything linearly correlated with it in
the activations — *must* separate answerable from unanswerable inputs, with a
margin that grows with γ. Prediction: detection AUROC → 1 as γ → ∞, → 0.5 as
γ → 1. No epistemic label is ever required; the signal is a structural
by-product of training on the forward task. The theory predicts a
domain-general *mechanism*, not a domain-general *direction*.

## 4. Results (3 seeds, CPU, frozen)

Every number below is read from the committed
`results/physics_credibility/summary.json` and `cross_law_transfer.json`.

**What works, robustly, on both laws:**
- Clean statistics are linearly decodable (Beer ratio_R R² = 0.98; MK
  PTT/PWV R² = 0.95/0.96 on held-out clean).
- Probe-ranked gating beats random abstention (Beer AURC 3.5 < 7.2; MK
  33.6 < 87.8, all seeds).
- The probe direction is *causal*: ablating it degrades the readout more
  than a random direction, all seeds, both laws.

**What is weaker or fails:**
- **Cross-attention does not reliably beat the controls.** Beer's task is
  near-1D (a single ratio), so capacity is not needed and the controls tie
  (0/3 seeds). MK is marginal (1/3) and its training is unstable across
  seeds (clean MAE 10.7 / 17.9 / 14.8; one seed's "adequate" model is worse
  than its own control).
- **Detection leans heavily on trivial missingness.** All-scenario
  answerability AUROC looks excellent (Beer 0.71, MK 1.00), but that is
  dominated by the easy case: a zeroed channel is trivial to spot. On
  **held-out non-missing** corruptions, Beer-Lambert detection collapses to
  chance (0.47 / 0.52 / 0.30). MK's perfect held-out score is itself inflated
  because its held-out set still contains one zeroed-channel scenario.
- **Supervised probing's margin over cheap UQ is modest.** A label-free
  bootstrap-ensemble-disagreement baseline on the same activations is
  competitive with the supervised probe on selective risk (MK: 29.3 vs 33.6).

**Cross-law transfer (the phenomenon test):** sharp, reproducible asymmetry.
MK → Beer transfers moderately (0.75), Beer → MK is *below chance* (0.31),
against within-domain ceilings of 0.86 / 0.997. Exactly the γ-dependence the
theory predicts: MK's large clean gap yields a transferable direction; Beer's
small noisy gap does not.

**Pre-registered verdict:** neither law clears the full five-criterion bar.
The honest one-line reading: *the abstention machinery works, but current
detection largely reflects missing information, not subtly-wrong-but-complete
information — the case that matters clinically.*

## 5. Honest limitations

- Synthetic simulators, toy 2-layer models, CPU-only. Not validated against
  any real device or dataset.
- The missingness shortcut is the central weakness — see §4.
- MK training is unstable across seeds; needs more seeds and tuning.
- "Mechanistic" is not yet earned: this is linear probing plus one causal
  ablation (representation-level), not circuit-level analysis.
- 3 seeds is too few for the significance tests to have real power.

## 6. What would make this a stronger contribution

Ranked by leverage:

1. **Isolate held-out detection on non-missing corruptions** as its own
   headline metric. This directly measures the claim that matters and is the
   single most important next number.
2. **The "toy model" move.** Replace the messy physiological laws with a
   minimal `y = Ax` instrument where removing rows of `A` tunes γ exactly.
   Sweep γ and look for a *phase transition* in decodability or clean
   geometric structure in the answerability representation — in the spirit of
   *Toy Models of Superposition*. This is the phenomenon study with real
   main-track potential; the physiological laws become the "and it appears in
   realistic settings too" section.
3. **More laws** (Fick's principle, Windkessel) as out-of-sample tests that
   the protocol generalizes without per-law tuning.
4. **Real data** (BIDMC PPG/SpO₂, MIMIC waveform-matched PPG+ABP) at least once.
5. **Stronger baselines** already partially present (ensemble, oracle);
   add conformal / evidential / Mahalanobis and the clinical signal-quality-
   index heuristic.

## 7. Reproducing

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'      # 21 tests
python analysis/physics_law_credibility.py               # main benchmark
python analysis/cross_law_transfer.py                    # transfer experiment
python analysis/plot_data_preview.py                     # data-preview figure
# open analysis/dashboard.html for the interactive results
```
