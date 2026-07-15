# Next steps — real-data demonstration (thesis chapter 3)

**If you are an agent picking this up: read this file first, then
`analysis/CAUSAL_AUDIT_MATH.md` for the precise method. The goal below is scoped
to FINISH a thesis chapter, not to write a top-venue paper — a modest, honest
result is the target.**

## What this project is (one paragraph)

We audit whether a *frozen* neural network **causally uses** a known governing
equation, or merely lands on the right answer via a shortcut. A model retrieves
two components (e.g. pulse-transit-time PTT and a per-subject calibration E0) and
must combine them via the physical law (Moens–Korteweg for BP; Beer–Lambert ratio
for SpO2). The claim: **a model can be accurate and have the answer linearly
decodable inside it, yet not compute the answer through the law** — and the
equation, used as a label-free oracle, lets us detect this on a frozen model.

## What is already done (on branch `claude/governing-law-audit`)

- `analysis/causal_mediation_toy.py` — the main synthetic experiment. Three
  models (law / unfaithful / shortcut), two laws (BP, SpO2), and the audits
  (accuracy, linear probe, gradient sensitivity, interchange/DAS). Multi-seed,
  monochrome publication figure. Run: `python analysis/causal_mediation_toy.py
  --law blood_pressure --quick`.
- `analysis/causal_audit_walkthrough.ipynb` — concise, executed teaching
  notebook: abstract BP → simple BP simulator → SpO2, with human comments.
- `analysis/CAUSAL_AUDIT_MATH.md` — precise formulation of task, architecture,
  the three models, and each audit.
- `analysis/audit_map.py` — diagram of the forward pass and where each audit
  reads/intervenes.

This synthetic + necessity result IS the bulk of the chapter. It is a workshop /
thesis-section-strength contribution on its own.

## The one finishing piece: a modest real-data demonstration

Train ONE standard cuffless-BP model on real data, run the audit on it, and
report the verdict — whatever it is. **A negative or "faithful" result is a fine
thesis result.** Do NOT try to beat uncertainty baselines or prove a theorem;
those are out of scope for finishing the chapter.

### Task list

1. **Data (kept OUT of git).** Put the BP dataset under `data/` (git-ignored).
   Use a manageable subset — a few hundred subjects is plenty. Recommended public
   options if needed: PulseDB (MIMIC-III + VitalDB), VitalDB, UCI/Kachuee.
   - **Split BY SUBJECT** (no subject in both train and test). This is critical:
     record-level splits leak the per-subject baseline, which is the exact
     failure we study.

2. **Train one model, standard recipe.** A small 1D-CNN or transformer on
   PPG/ECG windows → SBP/DBP. Do NOT engineer a shortcut. Report standard
   accuracy (MAE, and AAMI/BHS grade if easy) so the audited model is credible.

3. **Define the oracle.** Fit a per-subject Moens–Korteweg calibration on a
   held-out calibration segment: `BP = a_s - b * ln(PTT)`. Where the per-subject
   fit is unreliable, fall back to a DIRECTIONAL constraint (BP must decrease as
   PTT increases). State which oracle each result uses.

4. **Faithfulness floor (do this BEFORE trusting the audit).** On clean windows,
   verify that setting the PTT subspace to a known value reproduces the oracle
   BP. If the floor fails, the audit is not trustworthy for this model — report
   that and stop.

5. **Run the audit.** Locate the PTT subspace of a chosen hidden layer (learn a
   DAS rotation `R`; component j is no longer axis-aligned as in the toy). Between
   two windows from the SAME subject with DIFFERENT PTT, swap the PTT subspace and
   check the output moves as the oracle predicts (interchange accuracy). Also run
   the cheaper baselines for contrast: linear probe R^2 for PTT, and
   probe-direction ablation/steering with a sign check.

6. **Report.** One table (accuracy, probe R^2, gradient sign, interchange
   accuracy) + one figure mirroring the toy layout. State the verdict: does the
   model route through PTT, or is it mediated by the subject baseline?

### Reuse from the existing code

- The interchange logic lives in `interchange_accuracy` in
  `analysis/causal_mediation_toy.py`. For a real model it must operate on a chosen
  layer's hidden state via a forward hook + a learned rotation `R` (the toy
  version swaps an axis-aligned half; the real version learns the subspace). See
  `CAUSAL_AUDIT_MATH.md` §6 and the "audit map" for the exact operation.
- The probe / gradient / MSE helpers transfer directly.

### Sketch of the real-data audit (pseudocode)

```python
LAYER, POS = 6, -1                      # residual stream at a mid layer, answer token
h_base   = activation(model, base_window,   LAYER, POS)   # same subject, PTT state A
h_source = activation(model, source_window, LAYER, POS)   # same subject, PTT state B
z_b, z_s = R.T @ h_base, R.T @ h_source                   # R learned by DAS (model frozen)
z_b[:k]  = z_s[:k]                                          # swap the PTT subspace
h_patched = R @ z_b
out = run_with_patch(model, base_window, LAYER, POS, h_patched)   # forward hook
target = mk_oracle(base.ptt_after_swap = source.ptt, subject_calibration)  # physics counterfactual
# interchange accuracy = R^2(out, target) over many (base, source) pairs, same subject
```

## Data handling

- Big data does NOT go in git. Add it under `data/` (already git-ignored) and
  point the code at it with a `--data-dir` flag or an env var.
- Commit only a tiny sample (a few subjects) under `data/sample/` so the pipeline
  is runnable/testable from the repo.
- The real-data run is best done LOCALLY where the data lives (remote sessions are
  ephemeral).

## Go / no-go and honesty

- Subject-level split is mandatory; a record-level split invalidates the result.
- Establish the faithfulness floor before reporting any audit number.
- If the model turns out faithful, report it — that is a valid certification
  result, not a failure of the project.
- Report negative/mixed results directly (consistent with the rest of the repo).

## Chapter skeleton (the finish line)

1. Motivation — silent shortcut failure in physics-based sensing (BP calibration).
2. Method — the equation-as-oracle causal audit (`CAUSAL_AUDIT_MATH.md` + audit map).
3. Understanding experiment — the synthetic toy + necessity result (done).
4. Real-data demonstration — the task list above (this is the remaining work).
5. Limitations & future work — estimated oracle, partial-law extension, scaling.

Fill section 4 and the chapter is complete.
