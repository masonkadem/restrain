"""Cross-law probe transfer: does an answerability signature learned on one
physical law generalize zero-shot to a structurally different law?

Beer-Lambert (missing-channel identifiability: a whole wavelength is absent)
and Moens-Korteweg (missing-calibration identifiability: subject-specific
geometry/stiffness is unknown) fail for different physical reasons, and
their models have different, incompatible activation dimensionalities --
so raw activation vectors from one law's model cannot be fed to a probe
fit on the other. Instead this experiment tests transfer of a small,
architecture-agnostic signature computed identically in both domains:

  1. z_residual: the model's own prediction loss, z-scored against that
     law's clean-scenario loss distribution.
  2. z_activation_norm: the L2 norm of the frozen activation vector,
     z-scored against that law's clean-scenario norm distribution.
  3. confidence_percentile: the law's own within-domain answerability
     probe's score, expressed as an empirical percentile against a clean
     reference distribution (already in [0, 1], comparable across laws by
     construction).

A logistic probe is fit on one law's 3-feature signature and evaluated
zero-shot (no refitting) on the other law's signature. If cross-law AUROC
is well above chance, that's evidence for a domain-general "I am
under-determined" signature rather than a law-specific artifact -- the
central claim the "cross-law audit protocol" framing needs to actually
earn its name.

    python analysis/cross_law_transfer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _logistic():
    # Standardize before fitting: the three signature features are on very
    # different scales (z-scores vs a [0,1] percentile), which otherwise
    # leaves LogisticRegression's lbfgs unconverged. For cross-law transfer
    # the pipeline correctly applies the *source* law's scaler to the
    # *target* law's features -- standardizing the target with the source's
    # statistics is exactly what zero-shot transfer means.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

from physics_law_credibility import (  # noqa: E402
    BEER_SCENARIOS_PROBE,
    BEER_SCENARIOS_TEST,
    BEER_SCENARIOS_TRAIN,
    BEER_SCENARIOS_VAL,
    MK_SCENARIOS_PROBE,
    MK_SCENARIOS_TEST,
    MK_SCENARIOS_TRAIN,
    MK_SCENARIOS_VAL,
    WIN_LEN,
    LawAuditConfig,
    TwoStreamCrossAttention,
    beer_to_tensors,
    build_beer_dataset,
    build_mk_dataset,
    empirical_clean_confidence,
    evaluate_regressor,
    fit_answerability_probe,
    fit_normalizer,
    mk_to_tensors,
    train_regressor,
)


def build_signature(
    law_name: str,
    build_fn,
    to_tensors,
    train_scenarios: tuple[str, ...],
    probe_scenarios: tuple[str, ...],
    val_scenarios: tuple[str, ...],
    test_scenarios: tuple[str, ...],
    cfg: LawAuditConfig,
    device: torch.device,
) -> dict:
    seed = cfg.seed
    train = to_tensors(build_fn(cfg.n_train, train_scenarios, seed * 100 + 1))
    probe_data = to_tensors(build_fn(cfg.n_probe, probe_scenarios, seed * 100 + 2))
    val_data = to_tensors(build_fn(cfg.n_val, val_scenarios, seed * 100 + 3))
    test_parts = [
        to_tensors(build_fn(cfg.n_test, (sc,), seed * 100 + 10 + i))
        for i, sc in enumerate(test_scenarios)
    ]
    stack_keys = set(test_parts[0].keys()) - {"scenario"}
    test = {k: np.concatenate([p[k] for p in test_parts]) for k in stack_keys}
    test["scenario"] = [s for p in test_parts for s in p["scenario"]]

    normalizer = fit_normalizer(train)
    calibration_dim = train["calibration"].shape[-1] if "calibration" in train else 0
    model = TwoStreamCrossAttention(WIN_LEN, cfg.width, calibration_dim=calibration_dim)
    train_regressor(model, train, device, cfg.epochs, cfg.batch_size, normalizer, lr=cfg.lr)

    probe_eval = evaluate_regressor(model, probe_data, device, cfg.batch_size, normalizer)
    val_eval = evaluate_regressor(model, val_data, device, cfg.batch_size, normalizer)
    test_eval = evaluate_regressor(model, test, device, cfg.batch_size, normalizer)

    ans_probe = fit_answerability_probe(probe_eval["activation"], probe_eval["answerable"], seed)
    val_clean_mask = val_eval["answerable"].astype(bool)
    val_scores = ans_probe.predict_proba(val_eval["activation"])
    test_probs = ans_probe.predict_proba(test_eval["activation"])
    val_ref = val_scores[val_clean_mask] if val_clean_mask.any() else val_scores
    confidence_percentile = empirical_clean_confidence(val_ref, test_probs)

    probe_clean_mask = probe_eval["answerable"].astype(bool)
    clean_loss = probe_eval["loss"][probe_clean_mask]
    clean_norm = np.linalg.norm(probe_eval["activation"][probe_clean_mask], axis=1)
    loss_mean, loss_std = float(clean_loss.mean()), float(clean_loss.std() + 1e-8)
    norm_mean, norm_std = float(clean_norm.mean()), float(clean_norm.std() + 1e-8)

    z_residual = (test_eval["loss"] - loss_mean) / loss_std
    z_norm = (np.linalg.norm(test_eval["activation"], axis=1) - norm_mean) / norm_std
    features = np.stack([z_residual, z_norm, confidence_percentile], axis=1)

    return {
        "law": law_name,
        "features": features,
        "answerable": test_eval["answerable"],
        "scenario": test_eval["scenario"],
    }


def within_domain_reference(features: np.ndarray, answerable: np.ndarray, seed: int) -> float:
    """Same 3-feature signature, split within one law -- the honest ceiling
    to compare cross-law transfer against (transfer AUROC alone is
    uninformative without knowing what the same compact signature achieves
    in-domain)."""
    n = len(answerable)
    idx = np.random.default_rng(seed).permutation(n)
    half = n // 2
    if np.unique(answerable[idx[:half]]).size < 2 or np.unique(answerable[idx[half:]]).size < 2:
        return float("nan")
    clf = _logistic().fit(features[idx[:half]], answerable[idx[:half]])
    return float(roc_auc_score(answerable[idx[half:]], clf.predict_proba(features[idx[half:]])[:, 1]))


def run(seeds: tuple[int, ...] = (0, 1, 2), beer_epochs: int = 120, mk_epochs: int = 400, mk_lr: float = 3e-3) -> dict:
    device = torch.device("cpu")
    per_seed = []
    for seed in seeds:
        print(f"[cross-law] seed={seed}", flush=True)
        beer_cfg = LawAuditConfig(seed=seed, epochs=beer_epochs)
        mk_cfg = LawAuditConfig(seed=seed, epochs=mk_epochs, lr=mk_lr)
        beer_sig = build_signature(
            "beer_lambert", build_beer_dataset, beer_to_tensors,
            BEER_SCENARIOS_TRAIN, BEER_SCENARIOS_PROBE, BEER_SCENARIOS_VAL, BEER_SCENARIOS_TEST,
            beer_cfg, device,
        )
        mk_sig = build_signature(
            "moens_korteweg", build_mk_dataset, mk_to_tensors,
            MK_SCENARIOS_TRAIN, MK_SCENARIOS_PROBE, MK_SCENARIOS_VAL, MK_SCENARIOS_TEST,
            mk_cfg, device,
        )

        clf_beer = _logistic().fit(beer_sig["features"], beer_sig["answerable"])
        beer_to_mk = float(roc_auc_score(mk_sig["answerable"], clf_beer.predict_proba(mk_sig["features"])[:, 1]))

        clf_mk = _logistic().fit(mk_sig["features"], mk_sig["answerable"])
        mk_to_beer = float(roc_auc_score(beer_sig["answerable"], clf_mk.predict_proba(beer_sig["features"])[:, 1]))

        within_beer = within_domain_reference(beer_sig["features"], beer_sig["answerable"], seed)
        within_mk = within_domain_reference(mk_sig["features"], mk_sig["answerable"], seed)

        print(
            f"  beer->mk AUROC={beer_to_mk:.3f}  mk->beer AUROC={mk_to_beer:.3f}  "
            f"within_beer={within_beer:.3f}  within_mk={within_mk:.3f}",
            flush=True,
        )
        per_seed.append({
            "seed": seed,
            "beer_to_mk_auroc": beer_to_mk,
            "mk_to_beer_auroc": mk_to_beer,
            "within_beer_auroc": within_beer,
            "within_mk_auroc": within_mk,
        })

    def mean_key(key: str) -> float:
        vals = [p[key] for p in per_seed if np.isfinite(p[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "claim": (
            "Tests whether a compact, architecture-agnostic answerability "
            "signature (residual z-score, activation-norm z-score, probe "
            "confidence percentile) transfers zero-shot between two "
            "structurally different physical laws, as evidence for a "
            "domain-general 'under-determined' signature rather than a "
            "law-specific artifact."
        ),
        "per_seed": per_seed,
        "mean_beer_to_mk_auroc": mean_key("beer_to_mk_auroc"),
        "mean_mk_to_beer_auroc": mean_key("mk_to_beer_auroc"),
        "mean_within_beer_auroc": mean_key("within_beer_auroc"),
        "mean_within_mk_auroc": mean_key("within_mk_auroc"),
        "chance": 0.5,
    }


def main() -> None:
    output_dir = Path("results/physics_credibility")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run()
    path = output_dir / "cross_law_transfer.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[cross-law] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
