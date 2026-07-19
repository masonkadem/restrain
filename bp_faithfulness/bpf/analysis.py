"""Training, MAE, the faithfulness probe battery (Part 4), conformal (Part 4e),
and the sweep driver (Part 5). Kept explicit and commented."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score

from .dataset import generate_dataset, subject_split
from .model import BPModel


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train_model(ds, train_idx, cfg, seed=0):
    torch.manual_seed(seed)
    m = cfg["model"]
    model = BPModel(m["d_model"], m["heads"], m["conv_stride"])
    prox = torch.tensor(ds.prox[train_idx]); dist = torch.tensor(ds.dist[train_idx])
    bp = torch.tensor(ds.bp[train_idx])
    mu, sd = bp.mean(0), bp.std(0) + 1e-6                 # normalize target for training
    bpn = (bp - mu) / sd
    opt = torch.optim.Adam(model.parameters(), lr=m["lr"])
    n = len(train_idx)
    for _ in range(m["epochs"]):
        perm = torch.randperm(n)
        for s in range(0, n, m["batch"]):
            i = perm[s:s + m["batch"]]
            opt.zero_grad()
            loss = ((model(prox[i], dist[i]) - bpn[i]) ** 2).mean()
            loss.backward(); opt.step()
    model.eval()
    model._mu, model._sd = mu, sd                          # stash for un-normalization
    return model


@torch.no_grad()
def representations(model, ds, idx):
    h = model.represent(torch.tensor(ds.prox[idx]), torch.tensor(ds.dist[idx]))
    return h.numpy()


@torch.no_grad()
def predict_bp(model, ds, idx):
    pred = model(torch.tensor(ds.prox[idx]), torch.tensor(ds.dist[idx]))
    return (pred * model._sd + model._mu).numpy()          # back to mmHg


def mae_per_subject(model, ds, idx):
    pred = predict_bp(model, ds, idx)
    err = np.abs(pred - ds.bp[idx])                        # (n, 2)
    subs = ds.subject[idx]
    per = {int(s): float(err[subs == s].mean()) for s in np.unique(subs)}
    return float(err.mean()), per


# --------------------------------------------------------------------------- #
# Part 4a-b: linear probe for T, shuffle control, ridge-lasso gap, stability
# --------------------------------------------------------------------------- #
def probe_T(model, ds, train_idx, test_idx, cfg):
    Htr, Hte = representations(model, ds, train_idx), representations(model, ds, test_idx)
    Ttr, Tte = ds.T[train_idx], ds.T[test_idx]
    tmu, tsd = Ttr.mean(), Ttr.std() + 1e-9
    ridge = Ridge(alpha=cfg["probe"]["ridge_alpha"]).fit(Htr, (Ttr - tmu) / tsd)
    pred = ridge.predict(Hte)
    r2 = r2_score((Tte - tmu) / tsd, pred)

    # bootstrap CI over test resamples
    rng = np.random.default_rng(0)
    boots = []
    yte = (Tte - tmu) / tsd
    for _ in range(cfg["probe"]["n_bootstrap"]):
        j = rng.integers(0, len(yte), len(yte))
        boots.append(r2_score(yte[j], pred[j]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # shuffle control: refit against shuffled labels
    ysh = rng.permutation((Ttr - tmu) / tsd)
    r2_shuf = r2_score(yte, Ridge(alpha=cfg["probe"]["ridge_alpha"]).fit(Htr, ysh).predict(Hte))

    # ridge-lasso gap + stability selection (which units carry the signal)
    lasso = Lasso(alpha=0.01, max_iter=5000).fit(Htr, (Ttr - tmu) / tsd)
    r2_lasso = r2_score(yte, lasso.predict(Hte))
    sel = np.zeros(Htr.shape[1])
    for _ in range(50):
        j = rng.integers(0, len(Htr), len(Htr))
        sel += np.abs(Lasso(alpha=0.02, max_iter=2000).fit(Htr[j], ((Ttr - tmu) / tsd)[j]).coef_) > 1e-6
    stability = sel / 50.0

    w = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-9)     # unit probe direction
    return dict(r2=float(r2), ci=(float(lo), float(hi)), r2_shuffle=float(r2_shuf),
                r2_lasso=float(r2_lasso), ridge_lasso_gap=float(r2 - r2_lasso),
                stability=stability, direction=w, tmu=float(tmu), tsd=float(tsd))


# --------------------------------------------------------------------------- #
# Part 4c: causal ablation of the probe direction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ablation(model, ds, test_idx, direction):
    H = torch.tensor(representations(model, ds, test_idx))
    w = torch.tensor(direction, dtype=torch.float32)
    H_abl = H - (H @ w).unsqueeze(1) * w                       # remove PTT direction
    def mae(hmat):
        pred = model.from_h(hmat) * model._sd + model._mu
        return float(np.abs(pred.numpy() - ds.bp[test_idx]).mean())
    return dict(mae_intact=mae(H), mae_ablated=mae(H_abl))


# --------------------------------------------------------------------------- #
# Part 4d: donor-swap activation patching (sign + bounded magnitude)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def donor_swap(model, ds, test_idx, direction, cfg):
    H = torch.tensor(representations(model, ds, test_idx))
    w = torch.tensor(direction, dtype=torch.float32)
    T = ds.T[test_idx]
    rng = np.random.default_rng(1)
    donor = rng.permutation(len(test_idx))
    comp_base = (H @ w)
    comp_donor = comp_base[donor]
    H_patched = H + (comp_donor - comp_base).unsqueeze(1) * w  # inject donor's PTT component
    base_pred = (model.from_h(H) * model._sd + model._mu).numpy()[:, 0]      # SBP
    patch_pred = (model.from_h(H_patched) * model._sd + model._mu).numpy()[:, 0]
    shift = patch_pred - base_pred
    x = (1.0 / T[donor] - 1.0 / T)                            # physics: SBP ~ K1/T -> sign of shift
    sign_acc = float(np.mean(np.sign(shift) == np.sign(x)))
    slope = float(np.polyfit(x, shift, 1)[0])                 # ~ effective K1 (mmHg * s)
    k1_ref = float(np.mean(cfg["subjects"]["K1_sys"]))
    bounded = bool(0.2 * k1_ref < slope < 5 * k1_ref)         # sign-and-bounded-magnitude test
    return dict(sign_acc=sign_acc, slope=slope, k1_ref=k1_ref, bounded_ok=bounded,
                x=x.tolist(), shift=shift.tolist())


# --------------------------------------------------------------------------- #
# Part 4e: split conformal abstention with the faithfulness readout as score
# --------------------------------------------------------------------------- #
@torch.no_grad()
def conformal_abstention(model, ds, test_idx, direction, cfg):
    """Per-subject split conformal. Nonconformity = how LITTLE the PTT direction
    contributes to this sample's BP prediction (an unfaithful prediction scores
    high and is abstained on). Calibrated on a per-subject labelled burn-in.

    NOTE ON EXCHANGEABILITY: cross-subject distribution shift breaks
    exchangeability, so a single pooled calibration is not valid. We calibrate
    PER SUBJECT on a burn-in. A conformal-beyond-exchangeability variant
    (weighted/Mondrian, or an adaptive shift model) would go HERE.
    """
    H = torch.tensor(representations(model, ds, test_idx))
    w = torch.tensor(direction, dtype=torch.float32)
    H_abl = H - (H @ w).unsqueeze(1) * w
    contrib = np.abs((model.from_h(H) - model.from_h(H_abl)).numpy()[:, 0])   # PTT contribution
    score = -contrib                                          # nonconformity: low contribution -> high
    subs = ds.subject[test_idx]
    burn, cov = cfg["conformal"]["burn_in"], cfg["conformal"]["target_coverage"]
    abstained = 0; total = 0
    for s in np.unique(subs):
        si = np.where(subs == s)[0]
        if len(si) <= burn:
            continue
        cal, evl = si[:burn], si[burn:]
        q = np.quantile(score[cal], cov)                     # coverage quantile from burn-in
        abstained += int(np.sum(score[evl] > q)); total += len(evl)
    return float(abstained / max(total, 1))


# --------------------------------------------------------------------------- #
# One setting = one point in the sweep
# --------------------------------------------------------------------------- #
def run_setting(cfg, gamma, pep_sd_ms, seed):
    ds = generate_dataset(cfg, gamma=gamma, pep_sd_ms=pep_sd_ms, seed=seed)
    tr, te = subject_split(ds, cfg["sweep"]["train_subjects"], seed=seed)
    model = train_model(ds, tr, cfg, seed=seed)
    mae, _ = mae_per_subject(model, ds, te)
    pr = probe_T(model, ds, tr, te, cfg)
    abst = conformal_abstention(model, ds, te, pr["direction"], cfg)
    return dict(gamma=gamma, pep=pep_sd_ms, seed=seed, mae=mae,
                r2=pr["r2"], r2_shuffle=pr["r2_shuffle"], abstention=abst)


def run_sweep(cfg, axis):
    """axis in {'gamma','pep'}: vary one knob, hold the other at 0, average seeds."""
    grid = cfg["sweep"]["gamma_grid"] if axis == "gamma" else cfg["sweep"]["pep_sd_grid_ms"]
    out = []
    for v in grid:
        rows = [run_setting(cfg, v if axis == "gamma" else 0.0,
                            v if axis == "pep" else 0.0, s) for s in cfg["sweep"]["seeds"]]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in ("mae", "r2", "r2_shuffle", "abstention")}
        agg["std_r2"] = float(np.std([r["r2"] for r in rows]))
        agg["x"] = v
        out.append(agg)
    return out
