"""
Public Kenneth French experiment for the panel trading audit.

The public experiment is deliberately reproducible.  It builds a
momentum-like illustrative panel and a shuffled placebo from Kenneth French
portfolio returns, then evaluates both with dependence-aware inference,
factor-alpha adjustment, joint multiple-testing corrections, stationarity
diagnostics, and turnover-scaled costs.
"""

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sddm_bootstrap import (
    PanelData,
    _sharpe,
    andrew_bartlett_bandwidth,
    compare_methods,
    dependence_summary,
    format_comparison_table,
    fixed_b_hac_sensitivity,
    hac_sharpe_delta_prewhite,
    hac_sharpe_delta,
    percentile_rank_confidence,
    safe_sign,
    sharpe_effective_sample_size,
    skipped_compounded_signal,
)
from threshold_analysis import (
    deflated_sharpe_ratio,
    multi_threshold_analysis,
    returns_matrix,
    romano_wolf_stepdown,
    white_reality_check,
)
from walk_forward import compute_turnover, turnover_decomposition


OUTDIR = os.environ.get("AUDIT_OUTPUT_DIR", "output_prod")
N_BOOT = int(os.environ.get("SDDM_N_BOOT", 1000))
N_PERMS = int(os.environ.get("SDDM_N_PERMS", 1000))
os.makedirs(OUTDIR, exist_ok=True)


@dataclass
class PublicPanels:
    positive: PanelData
    placebo: PanelData
    factors: pd.DataFrame
    portfolios: pd.DataFrame
    lookback: int
    skip: int


def _download_zip_text(url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    return zf.read(name).decode("utf-8", errors="replace")


def _parse_french_csv(raw: str) -> pd.DataFrame:
    lines = raw.splitlines()
    start = None
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit() and len(parts[0]) == 8:
            start = i
            break
    if start is None:
        raise ValueError("could not locate French daily data start")
    rows = []
    for line in lines[start:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts)
    df = pd.DataFrame(rows)
    df.columns = ["date"] + [f"C{i}" for i in range(1, df.shape[1])]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([-99.99, -999.0], np.nan) / 100.0


def download_ff_portfolios() -> pd.DataFrame:
    """Download 25 Size-B/M daily value-weighted portfolios."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_5x5_Daily_CSV.zip"
    try:
        df = _parse_french_csv(_download_zip_text(url))
        df.columns = [f"P{i:02d}" for i in range(df.shape[1])]
        return df.dropna(how="all")
    except Exception as exc:
        print(f"French portfolio download failed; using synthetic fallback: {exc}")
        return _synthetic_portfolios()


def download_ff_factors() -> pd.DataFrame:
    """Download FF3 daily factors plus momentum when available."""
    factor_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
    try:
        ff3 = _parse_french_csv(_download_zip_text(factor_url))
        ff3 = ff3.iloc[:, :4]
        ff3.columns = ["Mkt-RF", "SMB", "HML", "RF"]
        mom = _parse_french_csv(_download_zip_text(mom_url))
        mom = mom.iloc[:, :1]
        mom.columns = ["MOM"]
        return ff3.join(mom, how="left").dropna(how="all")
    except Exception as exc:
        print(f"French factor download failed; using synthetic fallback: {exc}")
        return _synthetic_factors()


def _synthetic_portfolios(T: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2005-01-03", periods=T, freq="B")
    factors = _synthetic_factors(T)
    f = factors[["Mkt-RF", "SMB", "HML", "MOM"]].to_numpy()
    betas = rng.normal(0, 0.4, (25, 4))
    betas[:, 0] += 1.0
    eps = rng.normal(0, 0.012, (T, 25))
    data = f @ betas.T + eps
    return pd.DataFrame(data, index=dates, columns=[f"P{i:02d}" for i in range(25)])


def _synthetic_factors(T: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2005-01-03", periods=T, freq="B")
    data = {
        "Mkt-RF": rng.normal(0.00025, 0.011, T),
        "SMB": rng.normal(0.00005, 0.006, T),
        "HML": rng.normal(0.00005, 0.006, T),
        "RF": np.full(T, 0.00001),
        "MOM": rng.normal(0.00012, 0.008, T),
    }
    return pd.DataFrame(data, index=dates)


def build_momentum_panel(
    portfolio_returns: pd.DataFrame,
    lookback: int = 21,
    skip: int = 2,
    placebo: bool = False,
    seed: int = 42,
) -> PanelData:
    """
    Build signed next-day returns from skipped compounded momentum.

    sign(0)=0.  Zero and missing signals are excluded by setting confidence
    and signed realised returns to NaN.
    """
    ret = portfolio_returns.to_numpy(dtype=float)
    signal = skipped_compounded_signal(ret, lookback=lookback, skip=skip)
    if placebo:
        rng = np.random.default_rng(seed)
        shuffled = signal.copy()
        for t in range(shuffled.shape[0]):
            valid = np.isfinite(shuffled[t])
            shuffled[t, valid] = rng.permutation(shuffled[t, valid])
        signal = shuffled
    return panel_from_signal(portfolio_returns, signal)


def panel_from_signal(
    portfolio_returns: pd.DataFrame,
    signal: np.ndarray,
    confidence: np.ndarray | None = None,
) -> PanelData:
    """Build a signed-return panel from an already computed signal matrix."""
    ret = portfolio_returns.to_numpy(dtype=float)
    sign = safe_sign(signal)
    realised = np.full_like(signal, np.nan, dtype=float)
    realised[:-1] = sign[:-1] * ret[1:]
    active = (sign != 0.0) & np.isfinite(signal) & np.isfinite(realised)
    realised[~active] = np.nan

    if confidence is None:
        confidence = percentile_rank_confidence(signal, use_abs=True)
    else:
        confidence = np.asarray(confidence, dtype=float).copy()
    confidence[~active] = np.nan

    dates = np.array([np.datetime64(str(d.date())) for d in portfolio_returns.index])
    tickers = np.array(portfolio_returns.columns.astype(str))
    return PanelData(
        dates=dates,
        tickers=tickers,
        predictions=signal,
        realised=realised,
        confidence=confidence,
    )


def build_public_panels(lookback: int = 21, skip: int = 2) -> PublicPanels:
    portfolios = download_ff_portfolios().iloc[-3500:].dropna(how="all")
    factors = download_ff_factors()
    positive = build_momentum_panel(portfolios, lookback=lookback, skip=skip, placebo=False)
    placebo = build_momentum_panel(portfolios, lookback=lookback, skip=skip, placebo=True)
    return PublicPanels(
        positive=positive,
        placebo=placebo,
        factors=factors,
        portfolios=portfolios,
        lookback=lookback,
        skip=skip,
    )


def factor_alpha_test(
    returns: np.ndarray,
    dates: np.ndarray,
    factors: pd.DataFrame,
    annualise: float = 252.0,
) -> dict:
    """HAC alpha test against FF factors available on matching dates."""
    import statsmodels.api as sm
    from scipy import stats

    y = pd.Series(returns, index=pd.to_datetime(dates)).dropna()
    X = factors.reindex(y.index)[["Mkt-RF", "SMB", "HML", "MOM"]].dropna(how="all")
    joined = pd.concat([y.rename("ret"), X], axis=1).dropna()
    if len(joined) < 30:
        return {"alpha_period": np.nan, "alpha_ann": np.nan, "t_alpha": np.nan, "p_positive": 1.0, "n": len(joined)}
    Xmat = sm.add_constant(joined[["Mkt-RF", "SMB", "HML", "MOM"]].fillna(0.0))
    model = sm.OLS(joined["ret"], Xmat)
    maxlags = andrew_bartlett_bandwidth(joined["ret"].to_numpy())
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    alpha = float(res.params["const"])
    se = float(res.bse["const"])
    t_alpha = alpha / se if se > 0 else np.inf
    p_positive = float(1.0 - stats.norm.cdf(t_alpha))
    return {
        "alpha_period": alpha,
        "alpha_ann": alpha * annualise,
        "alpha_se_period": se,
        "t_alpha": float(t_alpha),
        "p_positive": p_positive,
        "n": int(len(joined)),
        "hac_lag": int(maxlags),
    }


def stationarity_diagnostics(returns: np.ndarray, annualise: float = 252.0) -> dict:
    """ADF/KPSS plus rolling Sharpe instability diagnostics."""
    from statsmodels.tsa.stattools import adfuller, kpss

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    out = {"n": int(len(r))}
    if len(r) < 30:
        return out | {"adf_p": np.nan, "kpss_p": np.nan, "rolling_sr_min": np.nan, "rolling_sr_max": np.nan}
    try:
        out["adf_p"] = float(adfuller(r, autolag="AIC")[1])
    except Exception:
        out["adf_p"] = np.nan
    try:
        out["kpss_p"] = float(kpss(r, regression="c", nlags="auto")[1])
    except Exception:
        out["kpss_p"] = np.nan
    window = min(252, max(30, len(r) // 5))
    roll = pd.Series(r).rolling(window).apply(lambda x: _sharpe(x.to_numpy(), annualise=annualise), raw=False).dropna()
    out["rolling_window"] = int(window)
    out["rolling_sr_min"] = float(roll.min()) if len(roll) else np.nan
    out["rolling_sr_max"] = float(roll.max()) if len(roll) else np.nan
    return out


def cost_sensitivity(
    panel: PanelData,
    threshold: float = 0.5,
    exposure: str = "as_selected",
    annualise: float = 252.0,
) -> pd.DataFrame:
    """Turnover-scaled cost stress in bps per full rebalance."""
    returns = panel.date_returns(threshold, exposure=exposure)
    valid = returns[np.isfinite(returns)]
    turnover_parts = turnover_decomposition(panel, threshold, 0, panel.T, exposure=exposure)
    turnover = turnover_parts["daily_turnover"]
    mean_return = float(np.mean(valid)) if len(valid) else np.nan
    break_even = float(mean_return * 10_000.0 / turnover) if turnover > 0 and np.isfinite(mean_return) else np.nan
    rows = []
    for bps in [0, 1, 5, 10, 25]:
        daily_cost = turnover * bps / 10_000.0
        net = valid - daily_cost
        rows.append({
            "threshold": threshold,
            "exposure": exposure,
            "cost_bps_per_rebalance": bps,
            "daily_turnover": turnover,
            "turnover_rescale": turnover_parts["turnover_rescale"],
            "turnover_entry_exit_flip": turnover_parts["turnover_entry_exit_flip"],
            "gross_sharpe": _sharpe(valid, annualise=annualise),
            "net_sharpe": _sharpe(net, annualise=annualise),
            "annual_cost_drag": daily_cost * annualise,
            "break_even_cost_bps": break_even,
        })
    return pd.DataFrame(rows)


def selected_count_summary(panel: PanelData, thresholds: list[float]) -> pd.DataFrame:
    """Summarise the date-level selected-count distribution by threshold."""
    rows = []
    for threshold in thresholds:
        counts = panel.selected_counts(threshold)
        positive = counts[counts > 0]
        rows.append({
            "threshold": threshold,
            "n_dates": int(len(counts)),
            "n_dates_selected": int(len(positive)),
            "n_predictions": int(np.sum(counts)),
            "avg_names": float(np.mean(positive)) if len(positive) else np.nan,
            "min_names": int(np.min(positive)) if len(positive) else 0,
            "p25_names": float(np.quantile(positive, 0.25)) if len(positive) else np.nan,
            "median_names": float(np.quantile(positive, 0.50)) if len(positive) else np.nan,
            "p75_names": float(np.quantile(positive, 0.75)) if len(positive) else np.nan,
            "max_names": int(np.max(positive)) if len(positive) else 0,
        })
    return pd.DataFrame(rows)


def same_date_permutation_test(
    portfolio_returns: pd.DataFrame,
    thresholds: list[float],
    lookback: int,
    skip: int,
    n_perms: int,
    seed: int = 123,
    groups: np.ndarray | None = None,
    design: str = "unrestricted",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Same-date permutation null for the public momentum signal.

    Each draw permutes signals within each date, optionally within
    pre-specified structural groups, recomputes signed next-day returns and
    confidence ranks, and preserves the same date-level return panel shape.
    The one-sided p-value is the share of permuted Sharpe ratios at least as
    large as the observed Sharpe, with one-draw smoothing.
    """
    ret = portfolio_returns.to_numpy(dtype=float)
    base_signal = skipped_compounded_signal(ret, lookback=lookback, skip=skip)
    base_confidence = percentile_rank_confidence(base_signal, use_abs=True)
    observed_panel = panel_from_signal(portfolio_returns, base_signal, base_confidence)
    observed = {thr: _sharpe(observed_panel.date_returns(thr)) for thr in thresholds}
    rng = np.random.default_rng(seed)
    draws = {thr: [] for thr in thresholds}
    null_rows = []
    group_values = np.unique(groups) if groups is not None and len(groups) == base_signal.shape[1] else None
    for perm in range(n_perms):
        perm_seed = int(rng.integers(0, 2**31 - 1))
        perm_rng = np.random.default_rng(perm_seed)
        if group_values is None:
            order = np.argsort(perm_rng.random(base_signal.shape), axis=1)
            shuffled_signal = np.take_along_axis(base_signal, order, axis=1)
            shuffled_confidence = np.take_along_axis(base_confidence, order, axis=1)
        else:
            shuffled_signal = base_signal.copy()
            shuffled_confidence = base_confidence.copy()
            for group in group_values:
                idx = np.where(groups == group)[0]
                if len(idx) < 2:
                    continue
                order = np.argsort(perm_rng.random((base_signal.shape[0], len(idx))), axis=1)
                shuffled_signal[:, idx] = np.take_along_axis(base_signal[:, idx], order, axis=1)
                shuffled_confidence[:, idx] = np.take_along_axis(base_confidence[:, idx], order, axis=1)
        sign = safe_sign(shuffled_signal)
        realised = np.full_like(shuffled_signal, np.nan, dtype=float)
        realised[:-1] = sign[:-1] * ret[1:]
        active = (sign != 0.0) & np.isfinite(shuffled_signal) & np.isfinite(realised)
        for thr in thresholds:
            mask = (shuffled_confidence >= thr) & active
            denom = mask.sum(axis=1)
            numer = np.nansum(np.where(mask, realised, np.nan), axis=1)
            date_returns = np.full_like(numer, np.nan, dtype=float)
            np.divide(numer, denom, out=date_returns, where=denom > 0)
            sharpe = _sharpe(date_returns)
            draws[thr].append(sharpe)
            null_rows.append({
                "perm": perm,
                "design": design,
                "threshold": thr,
                "sharpe": sharpe,
                "seed": perm_seed,
            })

    summary_rows = []
    for thr in thresholds:
        values = np.asarray(draws[thr], dtype=float)
        values = values[np.isfinite(values)]
        obs = observed[thr]
        p_positive = float((1.0 + np.sum(values >= obs)) / (len(values) + 1.0)) if len(values) else 1.0
        summary_rows.append({
            "design": design,
            "threshold": thr,
            "observed_sharpe": obs,
            "null_mean": float(np.mean(values)) if len(values) else np.nan,
            "null_sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "null_q025": float(np.quantile(values, 0.025)) if len(values) else np.nan,
            "null_q500": float(np.quantile(values, 0.500)) if len(values) else np.nan,
            "null_q975": float(np.quantile(values, 0.975)) if len(values) else np.nan,
            "p_positive": p_positive,
            "n_perms": int(len(values)),
            "seed": seed,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(null_rows)


def ff_25_structural_groups(n_cols: int, design: str) -> np.ndarray:
    """Structural groups for the French 25 Size-B/M portfolio ordering."""
    if n_cols != 25:
        raise ValueError("French 25 structural groups require exactly 25 portfolios")
    idx = np.arange(n_cols)
    if design == "within_size":
        return idx // 5
    if design == "within_bm":
        return idx % 5
    raise ValueError(f"unknown structural permutation design: {design}")


def hac_bandwidth_sensitivity(returns: np.ndarray, annualise: float = 252.0) -> pd.DataFrame:
    """Sharpe HAC sensitivity across fixed Bartlett bandwidths plus default."""
    rows = []
    for label, lag in [("auto", None), ("0", 0), ("1", 1), ("5", 5), ("10", 10), ("21", 21), ("63", 63), ("126", 126)]:
        try:
            res = hac_sharpe_delta(returns, max_lag=lag, annualise=annualise)
            rows.append({
                "lag_label": label,
                "requested_lag": np.nan if lag is None else lag,
                "used_bandwidth": res.bandwidth,
                "sharpe": res.sharpe,
                "se": res.se,
                "ci_lo": res.ci_lo,
                "ci_hi": res.ci_hi,
                "positive_p_value": res.positive_p_value,
            })
        except Exception as exc:
            rows.append({
                "lag_label": label,
                "requested_lag": np.nan if lag is None else lag,
                "status": f"failed: {exc}",
            })
    return pd.DataFrame(rows)


def _common_metadata(
    panel: PanelData,
    lookback: int,
    skip: int,
    thresholds: list[float],
) -> dict:
    finite_dates = panel.dates[np.isfinite(panel.date_returns(0.0))]
    return {
        "sample_start": str(finite_dates[0]) if len(finite_dates) else "",
        "sample_end": str(finite_dates[-1]) if len(finite_dates) else "",
        "lookback": lookback,
        "skip": skip,
        "n_portfolios": panel.N,
        "threshold_menu": ";".join(str(x) for x in thresholds),
        "n_boot": N_BOOT,
        "n_perms": N_PERMS,
        "bootstrap_seed": 42,
        "permutation_seed": 123,
        "data_source": "Kenneth French 25 Size-B/M daily portfolios; FF3 daily factors; momentum factor",
    }


def audit_one_panel(
    name: str,
    panel: PanelData,
    factors: pd.DataFrame,
    lookback: int,
    skip: int,
    thresholds: list[float],
) -> None:
    returns = panel.date_returns(0.5)
    common = _common_metadata(panel, lookback, skip, thresholds)

    dep = dependence_summary(returns, panel)
    dep.update(sharpe_effective_sample_size(returns))
    pd.DataFrame([{"panel": name, **common, **dep}]).to_csv(
        f"{OUTDIR}/public_{name}_diagnostics.csv", index=False
    )

    selected_count_summary(panel, thresholds).assign(panel=name).to_csv(
        f"{OUTDIR}/public_{name}_selected_counts.csv", index=False
    )

    results = compare_methods(panel, threshold=0.5, n_boot=N_BOOT, seed=42)
    print(f"\n{name}: bootstrap methods at threshold 0.5")
    print(format_comparison_table(results))
    pd.DataFrame([{
        "panel": name,
        **common,
        "method": r.method,
        "sharpe": r.sharpe_point,
        "se": r.sharpe_se,
        "ci_lo": r.sharpe_ci_lo,
        "ci_hi": r.sharpe_ci_hi,
        "p_positive": r.positive_p_value,
        "n_eff": r.n_effective,
        "n": r.n_nominal,
        "block_size": r.block_size,
    } for r in results]).to_csv(f"{OUTDIR}/public_{name}_methods.csv", index=False)

    hac = hac_sharpe_delta(returns)
    pd.DataFrame([{"panel": name, **common, **hac.__dict__}]).to_csv(
        f"{OUTDIR}/public_{name}_hac_delta.csv", index=False
    )
    try:
        prewhite = hac_sharpe_delta_prewhite(returns)
        pd.DataFrame([{"panel": name, **common, **prewhite.__dict__}]).to_csv(
            f"{OUTDIR}/public_{name}_hac_prewhite.csv", index=False
        )
    except Exception as exc:
        pd.DataFrame([{"panel": name, **common, "status": f"failed: {exc}"}]).to_csv(
            f"{OUTDIR}/public_{name}_hac_prewhite.csv", index=False
        )
    try:
        fixedb = fixed_b_hac_sensitivity(returns, n_sim=int(os.environ.get("SDDM_FIXEDB_SIM", 1000)))
        fixedb.assign(panel=name, **common).to_csv(f"{OUTDIR}/public_{name}_fixed_b_hac.csv", index=False)
    except Exception as exc:
        pd.DataFrame([{"panel": name, **common, "status": f"failed: {exc}"}]).to_csv(
            f"{OUTDIR}/public_{name}_fixed_b_hac.csv", index=False
        )
    hac_bandwidth_sensitivity(returns).assign(panel=name, **common).to_csv(
        f"{OUTDIR}/public_{name}_hac_bandwidth.csv", index=False
    )

    alpha = factor_alpha_test(returns, panel.dates, factors)
    pd.DataFrame([{"panel": name, **common, **alpha}]).to_csv(
        f"{OUTDIR}/public_{name}_factor_alpha.csv", index=False
    )

    stat = stationarity_diagnostics(returns)
    pd.DataFrame([{"panel": name, **common, **stat}]).to_csv(
        f"{OUTDIR}/public_{name}_stationarity.csv", index=False
    )

    all_corr = []
    for correction in ["holm", "by", "storey", "romano_wolf"]:
        df = multi_threshold_analysis(panel, thresholds=thresholds, correction=correction, n_boot=N_BOOT)
        df["panel"] = name
        df["correction"] = correction
        all_corr.append(df)
    pd.concat(all_corr, ignore_index=True).to_csv(f"{OUTDIR}/public_{name}_threshold_corrections.csv", index=False)

    R = returns_matrix(panel, thresholds)
    rw = romano_wolf_stepdown(R, n_boot=N_BOOT, seed=42)
    wrc = white_reality_check(R, n_boot=N_BOOT, seed=42)
    dsr_rows = []
    for i, thr in enumerate(thresholds):
        r = panel.date_returns(thr)
        dsr = deflated_sharpe_ratio(r, n_trials=len(thresholds))
        dsr_rows.append({"panel": name, "threshold": thr, "romano_wolf_p": rw[i], **dsr})
    pd.DataFrame(dsr_rows).to_csv(f"{OUTDIR}/public_{name}_dsr_romano_wolf.csv", index=False)
    pd.DataFrame([{"panel": name, **common, **wrc}]).to_csv(
        f"{OUTDIR}/public_{name}_white_reality_check.csv", index=False
    )

    cost_sensitivity(panel, threshold=0.5).assign(panel=name).to_csv(
        f"{OUTDIR}/public_{name}_costs.csv", index=False
    )


def run_public_data():
    t0 = time.time()
    print("=" * 72)
    print("PUBLIC KENNETH FRENCH PANEL AUDIT")
    print("=" * 72)
    lookback = 21
    skip = 2
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    panels = build_public_panels(lookback=lookback, skip=skip)

    metadata = _common_metadata(panels.positive, lookback, skip, thresholds)
    pd.DataFrame([
        {"key": key, "value": value}
        for key, value in {
            **metadata,
            "output_dir": OUTDIR,
            "run_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }.items()
    ]).to_csv(f"{OUTDIR}/public_run_metadata.csv", index=False)

    audit_one_panel("momentum", panels.positive, panels.factors, lookback, skip, thresholds)
    audit_one_panel("placebo", panels.placebo, panels.factors, lookback, skip, thresholds)

    perm_summary, perm_null = same_date_permutation_test(
        panels.portfolios,
        thresholds=thresholds,
        lookback=lookback,
        skip=skip,
        n_perms=N_PERMS,
        seed=123,
    )
    perm_summary.to_csv(f"{OUTDIR}/public_permutation.csv", index=False)
    perm_null.to_csv(f"{OUTDIR}/public_permutation_null.csv", index=False)
    grouped_summaries = []
    for design in ["within_size", "within_bm"]:
        grouped, _ = same_date_permutation_test(
            panels.portfolios,
            thresholds=thresholds,
            lookback=lookback,
            skip=skip,
            n_perms=N_PERMS,
            seed=123,
            groups=ff_25_structural_groups(panels.portfolios.shape[1], design),
            design=design,
        )
        grouped_summaries.append(grouped)
    pd.concat(grouped_summaries, ignore_index=True).to_csv(
        f"{OUTDIR}/public_grouped_permutation.csv", index=False
    )
    print(f"\nPublic audit complete in {time.time() - t0:.1f}s. Output: {OUTDIR}/")


if __name__ == "__main__":
    run_public_data()
