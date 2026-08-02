"""
================================================================================
metrics.py -- Regression metric suite for the MS-CA pipeline
================================================================================

WHY MORE THAN MAE
-----------------
The label distribution is extreme (skew 40.03, max/median 3532). A single
aggregate number hides everything that matters:

  * 99% of samples sit below 1206, so overall MAE is dominated by small cells
    and a model that ignores every power plant still scores well.
  * R^2 on the raw scale is dominated by the handful of giant cells, so it can
    look excellent while the bulk of the map is wrong.
  * L1 training learns the conditional MEDIAN, which biases predictions low.
    Bias is invisible in MAE/RMSE/R^2 but is fatal if you want total emissions.

So this module reports MAE, RMSE, R^2, Pearson r, Spearman rho, bias, median
AE, sMAPE, top-decile recall, and MAE stratified by target quantile.

EFFECTIVE SAMPLE SIZE
---------------------
Do not bootstrap over samples. The 36 months of a given cell are near
replicates (within-cell variance is only 0.85% of the total), so the ~618k
samples carry roughly 17k cells of independent information. Resampling samples
would shrink the confidence interval by about sqrt(36) = 6x. Use
``bootstrap_ci_by_cell`` and resample CELLS.
================================================================================
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.stats import spearmanr
    _SCIPY = True
except ImportError:  # scipy is present on Kaggle, but degrade gracefully
    _SCIPY = False


# ==============================================================================
# CORE METRICS
# ==============================================================================
def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    """1 - SS_res / SS_tot. Zero means 'as good as predicting the mean'."""
    pred = np.asarray(pred, np.float64).ravel()
    true = np.asarray(true, np.float64).ravel()
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def mae_score(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64).ravel()
                                - np.asarray(true, np.float64).ravel())))


def full_metrics(pred: np.ndarray, true: np.ndarray,
                 top_frac: float = 0.10) -> dict:
    """Complete metric dict for one (pred, true) pair in a single space."""
    pred = np.asarray(pred, np.float64).ravel()
    true = np.asarray(true, np.float64).ravel()
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")

    errors = pred - true
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))

    out = {
        "n": int(pred.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(errors)),
        "medae": float(np.median(np.abs(errors))),
    }

    # Correlations (guard against zero variance).
    if pred.std() > 0 and true.std() > 0:
        out["pearson"] = float(np.corrcoef(pred, true)[0, 1])
        if _SCIPY:
            out["spearman"] = float(spearmanr(pred, true).statistic)
        else:
            pr = np.argsort(np.argsort(pred)).astype(np.float64)
            tr = np.argsort(np.argsort(true)).astype(np.float64)
            out["spearman"] = float(np.corrcoef(pr, tr)[0, 1])
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")

    # Symmetric MAPE over strictly positive targets only.
    positive = true > 0
    if positive.any():
        denom = np.abs(pred[positive]) + np.abs(true[positive])
        safe = denom > 0
        out["smape"] = float(np.mean(2.0 * np.abs(errors[positive][safe])
                                     / denom[safe])) if safe.any() else float("nan")
    else:
        out["smape"] = float("nan")

    # Hotspot detection: how many of the true top-decile cells are recovered
    # by ranking on the prediction. No arbitrary threshold required.
    k = max(1, int(top_frac * true.size))
    top_true = set(np.argsort(-true)[:k].tolist())
    top_pred = set(np.argsort(-pred)[:k].tolist())
    out["top10_recall"] = len(top_true & top_pred) / k

    # Stratified MAE -- where does the model fall apart?
    quantiles = np.quantile(true, [0.0, 0.5, 0.9, 0.99, 1.0])
    for i, name in enumerate(["p0_50", "p50_90", "p90_99", "p99_100"]):
        lo, hi = quantiles[i], quantiles[i + 1]
        sel = (true >= lo) & (true <= hi)
        out[f"mae_{name}"] = float(np.mean(np.abs(errors[sel]))) if sel.any() \
            else float("nan")

    return out


# ==============================================================================
# BOTH SPACES (log + original)
# ==============================================================================
def metrics_both_spaces(pred: np.ndarray, true: np.ndarray,
                        target_transform: str = "log1p",
                        smearing: float | None = None) -> dict:
    """Metrics in the training space AND the original label space.

    The headline claim ("the proxies reproduce the carbon map") is about the
    ORIGINAL scale, so the original-space numbers are the ones to report --
    log-space R^2 always looks better.

    Because E[expm1(y_hat)] != expm1(E[y_hat]) (Jensen), back-transforming a
    model trained on log targets under-predicts systematically. Pass a Duan
    smearing factor (see ``duan_smearing``) to correct it, and always check
    ``bias`` afterwards.
    """
    result = {"transform": target_transform,
              "train_space": full_metrics(pred, true)}

    if target_transform == "none":
        result["orig_space"] = result["train_space"]
        return result

    if target_transform == "log1p":
        true_orig = np.expm1(np.asarray(true, np.float64))
        if smearing is not None:
            pred_orig = apply_smearing(pred, smearing)
            result["smearing"] = float(smearing)
        else:
            pred_orig = np.expm1(np.asarray(pred, np.float64))
        result["orig_space"] = full_metrics(pred_orig, true_orig)
        return result

    raise ValueError(f"Unknown target transform: {target_transform!r}")


def duan_smearing(train_pred_log: np.ndarray, train_true_log: np.ndarray) -> float:
    """Duan's smearing estimator S = mean(exp(residual)), TRAIN residuals only.

    For y = log1p(x) with y_true = y_hat + eps:
        E[x | y_hat] = E[exp(y_hat + eps)] - 1 = exp(y_hat) * E[exp(eps)] - 1
    so the factor multiplies exp(y_hat), and 1 is subtracted AFTER scaling
    (see apply_smearing). A healthy factor sits near exp(sigma_eps^2 / 2);
    with residual std ~0.25 that is ~1.03.

    BUG HISTORY: an earlier version returned mean(expm1(residual)) =
    E[exp(eps)] - 1 (~0.06 instead of ~1.06). The 0.5-5.0 sanity gate
    rejected it at runtime, so no published number was affected, but the
    correction silently degraded to a no-op.
    """
    residual = np.asarray(train_true_log, np.float64) - np.asarray(train_pred_log, np.float64)
    return float(np.mean(np.exp(residual)))


def apply_smearing(pred_log: np.ndarray, smearing: float) -> np.ndarray:
    """Back-transform log1p-space predictions with Duan's correction.

    exp(y_hat) * S - 1, equivalently (expm1(y_hat) + 1) * S - 1.
    With S = 1.0 this reduces to plain expm1. Results are clipped at 0:
    emissions cannot be negative, and predictions slightly below
    -ln(S) in log space would otherwise map below zero.
    """
    out = np.exp(np.asarray(pred_log, np.float64)) * smearing - 1.0
    return np.clip(out, 0.0, None)


# ==============================================================================
# UNCERTAINTY -- resample cells, not samples
# ==============================================================================
def bootstrap_ci_by_cell(pred: np.ndarray, true: np.ndarray,
                         cell_ids: np.ndarray, stat_fn,
                         n_boot: int = 1000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    """Cluster bootstrap over cells. Returns (point_estimate, lo, hi).

    ``cell_ids`` must give the cell each prediction belongs to (e.g.
    ``row * WIDTH + col``). All 36 monthly samples of a cell are resampled
    together, which is what keeps the interval honest.
    """
    pred = np.asarray(pred, np.float64).ravel()
    true = np.asarray(true, np.float64).ravel()
    cell_ids = np.asarray(cell_ids).ravel()

    unique = np.unique(cell_ids)
    lookup = {cid: np.where(cell_ids == cid)[0] for cid in unique}

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, np.float64)
    for b in range(n_boot):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([lookup[c] for c in drawn])
        stats[b] = stat_fn(pred[idx], true[idx])

    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return stat_fn(pred, true), float(lo), float(hi)


# ==============================================================================
# PRINTING
# ==============================================================================
def format_metrics(m: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"--- {title} ---")
    lines.append(f"  n                 : {m['n']}")
    lines.append(f"  R^2               : {m['r2']:.6f}")
    lines.append(f"  MAE               : {m['mae']:.6f}")
    lines.append(f"  RMSE              : {m['rmse']:.6f}")
    lines.append(f"  MedAE             : {m['medae']:.6f}")
    lines.append(f"  Bias (pred-true)  : {m['bias']:+.6f}")
    lines.append(f"  Pearson r         : {m['pearson']:.6f}")
    lines.append(f"  Spearman rho      : {m['spearman']:.6f}")
    lines.append(f"  sMAPE             : {m['smape']:.6f}")
    lines.append(f"  Top-10% recall    : {m['top10_recall']:.6f}")
    lines.append("  MAE by true quantile:")
    for name, label in [("p0_50", "p0-50  "), ("p50_90", "p50-90 "),
                        ("p90_99", "p90-99 "), ("p99_100", "p99-100")]:
        lines.append(f"    {label}         : {m[f'mae_{name}']:.6f}")
    return "\n".join(lines)


def print_metrics(m: dict, title: str = "") -> None:
    print(format_metrics(m, title))
