"""
Multiple-testing and researcher-menu analysis for panel trading audits.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from sddm_bootstrap import Exposure, PanelData, sddm_inference, _sharpe, optimal_block_length


def holm_bonferroni(p_values: list[float] | np.ndarray) -> list[float]:
    """Holm step-down adjusted p-values controlling FWER."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = np.zeros(m)
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(float(p) * (m - rank), 1.0)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max
    return adjusted.tolist()


def benjamini_hochberg(p_values: list[float] | np.ndarray) -> list[float]:
    """Benjamini-Hochberg FDR adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1], reverse=True)
    adjusted = np.zeros(m)
    running_min = 1.0
    for rank_desc, (orig_idx, p) in enumerate(indexed):
        rank_asc = m - rank_desc
        adj_p = min(float(p) * m / rank_asc, 1.0)
        running_min = min(running_min, adj_p)
        adjusted[orig_idx] = running_min
    return adjusted.tolist()


def benjamini_yekutieli(p_values: list[float] | np.ndarray) -> list[float]:
    """Benjamini-Yekutieli FDR adjusted p-values for arbitrary dependence."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1))) if m else 1.0
    bh = np.asarray(benjamini_hochberg(p_values))
    return np.minimum(bh * harmonic, 1.0).tolist()


def storey_q_values(p_values: list[float] | np.ndarray, lam: float = 0.5) -> list[float]:
    """Storey-style q-values with a fixed pi0 tuning parameter."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return []
    pi0 = min(1.0, float(np.mean(p_values > lam) / max(1.0 - lam, 1e-12)))
    indexed = sorted(enumerate(p_values), key=lambda x: x[1], reverse=True)
    q = np.zeros(m)
    running_min = 1.0
    for rank_desc, (orig_idx, p) in enumerate(indexed):
        rank_asc = m - rank_desc
        val = min(pi0 * m * float(p) / rank_asc, 1.0)
        running_min = min(running_min, val)
        q[orig_idx] = running_min
    return q.tolist()


def _joint_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    block_size = max(1, min(int(block_size), n))
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]


def _t_stats(returns: np.ndarray) -> np.ndarray:
    means = np.nanmean(returns, axis=0)
    sds = np.nanstd(returns, axis=0, ddof=1)
    ns = np.sum(np.isfinite(returns), axis=0)
    se = np.where((sds > 0) & (ns > 1), sds / np.sqrt(ns), np.inf)
    return means / se


def romano_wolf_stepdown(
    returns_by_strategy: np.ndarray,
    n_boot: int = 2000,
    block_size: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Stepdown max-statistic adjusted p-values using joint date resampling.

    Input is T x M date-level returns for the researcher menu.  Columns are
    centered under the complete null before bootstrap resampling.
    """
    return romano_wolf_menu_test(
        returns_by_strategy,
        n_boot=n_boot,
        block_size=block_size,
        seed=seed,
    )["p_adjusted"].to_numpy(dtype=float)


def romano_wolf_menu_test(
    returns_by_strategy: np.ndarray,
    n_boot: int = 2000,
    block_size: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Raw and stepdown Romano-Wolf p-values from one joint bootstrap.

    The raw p-values are the marginal studentized mean-return p-values
    from the same centered bootstrap statistics used by the stepdown
    max-statistic.  Reporting these side by side makes the invariant
    p_adjusted >= p_raw mechanically checkable.
    """
    rng = np.random.default_rng(seed)
    R = np.asarray(returns_by_strategy, dtype=float)
    original_m = R.shape[1] if R.ndim == 2 else 1
    if R.ndim == 1:
        R = R[:, None]
    valid_rows = np.any(np.isfinite(R), axis=1)
    R = R[valid_rows]
    if R.shape[0] < 10:
        return pd.DataFrame({
            "strategy": np.arange(original_m),
            "t_stat": np.zeros(original_m),
            "p_raw": np.ones(original_m),
            "p_adjusted": np.ones(original_m),
            "block_size": np.nan,
            "n_boot": n_boot,
        })
    col_means = np.nanmean(R, axis=0)
    centered = R - col_means
    centered = np.where(np.isfinite(centered), centered, 0.0)
    obs = _t_stats(R)
    m = R.shape[1]
    if block_size is None:
        block_size = optimal_block_length(np.nanmean(R, axis=1), objective="distribution")

    boot_stats = np.empty((n_boot, m))
    for b in range(n_boot):
        idx = _joint_block_indices(R.shape[0], block_size, rng)
        boot_stats[b] = _t_stats(centered[idx])

    raw = np.empty(m)
    for j in range(m):
        raw[j] = (1.0 + np.sum(boot_stats[:, j] >= obs[j])) / (n_boot + 1.0)

    order = np.argsort(obs)[::-1]
    adjusted = np.ones(m)
    running = 0.0
    for step, j in enumerate(order):
        remaining = order[step:]
        max_boot = np.nanmax(boot_stats[:, remaining], axis=1)
        p = (1.0 + np.sum(max_boot >= obs[j])) / (n_boot + 1.0)
        running = max(running, float(p))
        adjusted[j] = running
    adjusted = np.maximum(np.minimum(adjusted, 1.0), raw)
    return pd.DataFrame({
        "strategy": np.arange(m),
        "t_stat": obs,
        "p_raw": raw,
        "p_adjusted": adjusted,
        "block_size": int(block_size),
        "n_boot": int(n_boot),
    })


def white_reality_check(
    returns_by_strategy: np.ndarray,
    n_boot: int = 2000,
    block_size: int | None = None,
    seed: int = 42,
) -> dict:
    """White Reality Check p-value for max mean performance over a menu."""
    rng = np.random.default_rng(seed)
    R = np.asarray(returns_by_strategy, dtype=float)
    valid_rows = np.any(np.isfinite(R), axis=1)
    R = R[valid_rows]
    if R.shape[0] < 10:
        return {"observed_max_mean": np.nan, "p_value": 1.0, "block_size": np.nan}
    means = np.nanmean(R, axis=0)
    obs = float(np.sqrt(R.shape[0]) * np.nanmax(means))
    centered = np.where(np.isfinite(R - means), R - means, 0.0)
    if block_size is None:
        block_size = optimal_block_length(np.nanmean(R, axis=1), objective="distribution")
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = _joint_block_indices(R.shape[0], block_size, rng)
        boot[b] = np.sqrt(R.shape[0]) * np.max(np.mean(centered[idx], axis=0))
    p = (1.0 + np.sum(boot >= obs)) / (n_boot + 1.0)
    return {"observed_max_mean": float(np.nanmax(means)), "p_value": float(p), "block_size": int(block_size)}


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    annualise: float = 252.0,
) -> dict:
    """
    Bailey-Lopez de Prado style Deflated Sharpe Ratio diagnostic.

    The calculation uses the non-annualised Sharpe internally, as in the
    published formula, and reports the probability that the observed Sharpe
    exceeds the expected maximum null Sharpe after skew/kurtosis adjustment.
    """
    from scipy import stats

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return {"dsr": np.nan, "sr_star": np.nan, "skew": np.nan, "kurtosis": np.nan}
    sr_daily = _sharpe(r, annualise=1.0)
    n_trials = max(int(n_trials), 1)
    if n_trials == 1:
        sr_star = 0.0
    else:
        euler_gamma = 0.5772156649015329
        sr_star = (
            (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + euler_gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        ) / np.sqrt(len(r) - 1.0)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    denom = 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * sr_daily * sr_daily
    if denom <= 0:
        dsr = np.nan
    else:
        z = (sr_daily - sr_star) * np.sqrt(len(r) - 1.0) / np.sqrt(denom)
        dsr = float(stats.norm.cdf(z))
    return {
        "dsr": dsr,
        "sr_star": float(sr_star * np.sqrt(annualise)),
        "skew": skew,
        "kurtosis": kurt,
    }


def returns_matrix(
    panel: PanelData,
    thresholds: list[float],
    exposure: Exposure = "as_selected",
) -> np.ndarray:
    """Build a T x M matrix of date-level returns for a threshold menu."""
    return np.column_stack([panel.date_returns(thr, exposure=exposure) for thr in thresholds])


def multi_threshold_analysis(
    panel: PanelData,
    thresholds: list[float] | None = None,
    method: str = "blocked",
    n_boot: int = 10_000,
    correction: Literal["holm", "bh", "by", "storey", "romano_wolf", "bonferroni", "none"] = "holm",
    seed: int = 42,
    exposure: Exposure = "as_selected",
) -> pd.DataFrame:
    """Run inference across thresholds with marginal and joint corrections."""
    if thresholds is None:
        thresholds = [0.0, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    rows = []
    raw_pvals = []
    for thr in thresholds:
        returns = panel.date_returns(thr, exposure=exposure)
        n_dates = int(np.sum(np.isfinite(returns)))
        n_preds = int(np.sum((panel.confidence >= thr) & np.isfinite(panel.confidence)))
        avg_names = float(np.nanmean(panel.selected_counts(thr))) if n_dates else np.nan
        if n_dates < 30:
            raw_pvals.append(1.0)
            rows.append({
                "threshold": thr,
                "sharpe": np.nan,
                "se": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "p_raw": 1.0,
                "n_eff": np.nan,
                "n_dates": n_dates,
                "n_predictions": n_preds,
                "avg_names": avg_names,
                "block_size": np.nan,
            })
            continue
        res = sddm_inference(panel, threshold=thr, method=method, n_boot=n_boot, seed=seed, exposure=exposure)
        raw_pvals.append(res.positive_p_value)
        rows.append({
            "threshold": thr,
            "sharpe": res.sharpe_point,
            "se": res.sharpe_se,
            "ci_lo": res.sharpe_ci_lo,
            "ci_hi": res.sharpe_ci_hi,
            "p_raw": res.positive_p_value,
            "n_eff": res.n_effective,
            "n_dates": n_dates,
            "n_predictions": n_preds,
            "avg_names": avg_names,
            "block_size": res.block_size,
        })

    if correction == "holm":
        adjusted = holm_bonferroni(raw_pvals)
    elif correction == "bh":
        adjusted = benjamini_hochberg(raw_pvals)
    elif correction == "by":
        adjusted = benjamini_yekutieli(raw_pvals)
    elif correction == "storey":
        adjusted = storey_q_values(raw_pvals)
    elif correction == "bonferroni":
        adjusted = [min(float(p) * len(raw_pvals), 1.0) for p in raw_pvals]
    elif correction == "romano_wolf":
        rw = romano_wolf_menu_test(returns_matrix(panel, thresholds, exposure=exposure), n_boot=n_boot, seed=seed)
        adjusted = rw["p_adjusted"].tolist()
    else:
        adjusted = list(raw_pvals)

    df = pd.DataFrame(rows)
    df["p_adjusted"] = adjusted
    df["significant_raw"] = df["p_raw"] < 0.05
    df["significant_adjusted"] = df["p_adjusted"] < 0.05
    return df


def nested_cv_threshold_selection(
    panel: PanelData,
    thresholds: list[float] | None = None,
    n_outer_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Expanding-window threshold selection without overlapping test periods."""
    if thresholds is None:
        thresholds = [0.0, 0.30, 0.50, 0.60, 0.70, 0.80]
    returns_all = {thr: panel.date_returns(thr) for thr in thresholds}

    T = panel.T
    min_train = T // 3
    test_size = max(1, (T - min_train) // n_outer_folds)
    rows = []
    for fold in range(n_outer_folds):
        train_end = min_train + fold * test_size
        test_start = train_end
        test_end = min(test_start + test_size, T)
        if test_end <= test_start:
            break

        inner_split = int(train_end * 0.7)
        best_thr = thresholds[0]
        best_sr = -np.inf
        for thr in thresholds:
            val = returns_all[thr][inner_split:train_end]
            sr = _sharpe(val)
            if sr > best_sr:
                best_sr = sr
                best_thr = thr

        is_returns = returns_all[best_thr][:train_end]
        oos_returns = returns_all[best_thr][test_start:test_end]
        rows.append({
            "fold": fold + 1,
            "train_start": str(panel.dates[0]),
            "train_end": str(panel.dates[train_end - 1]),
            "test_start": str(panel.dates[test_start]),
            "test_end": str(panel.dates[test_end - 1]),
            "selected_threshold": best_thr,
            "inner_val_sharpe": best_sr,
            "is_sharpe": _sharpe(is_returns),
            "oos_sharpe": _sharpe(oos_returns),
            "oos_n_valid": int(np.sum(np.isfinite(oos_returns))),
        })
    return pd.DataFrame(rows)


def main():
    from simulation_study import DGPConfig, generate_panel

    cfg = DGPConfig(
        name="Threshold Analysis Demo",
        T=800,
        N=40,
        true_mu=0.0003,
        ar1_serial=0.15,
        rho_cross=0.25,
        conf_quality=0.65,
    )
    panel = generate_panel(cfg, seed=42)
    for correction in ["holm", "by", "storey", "romano_wolf"]:
        df = multi_threshold_analysis(panel, correction=correction, n_boot=300)
        print(f"\n{correction}")
        print(df[["threshold", "sharpe", "p_raw", "p_adjusted", "significant_adjusted"]].to_string(index=False))


if __name__ == "__main__":
    main()
