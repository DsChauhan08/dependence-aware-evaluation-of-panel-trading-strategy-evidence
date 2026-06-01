"""
Panel audit bootstrap and HAC inference.

The public manuscript uses these routines for date-level strategy
evaluation.  The file name is retained for compatibility with earlier
experiments, but the manuscript-facing method is a dependence-aware
panel trading audit rather than a branded framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
from numpy.typing import NDArray


Annualisation = float
BootstrapMethod = Literal["iid", "blocked", "stationary", "cluster_date", "cluster_ticker"]
Weighting = Literal["equal", "confidence", "signal_abs", "custom"]
Exposure = Literal["as_selected", "dollar_neutral"]


@dataclass
class PanelData:
    """A prediction panel indexed by dates and assets."""

    dates: NDArray[np.datetime64]
    tickers: NDArray[np.str_]
    predictions: NDArray[np.float64]
    realised: NDArray[np.float64]
    confidence: NDArray[np.float64]

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.tickers)

    def date_returns(
        self,
        threshold: float = 0.0,
        weighting: Weighting = "equal",
        weights: NDArray[np.float64] | None = None,
        exposure: Exposure = "as_selected",
    ) -> NDArray[np.float64]:
        """
        Aggregate selected row-level signed returns to one return per date.

        The baseline audit portfolio is equal weighted.  Confidence and
        signal-magnitude weighting are sensitivity analyses.  Rows with a
        zero signal should have zero signed realised return or NaN upstream;
        this function then gives them zero economic weight.
        """
        mask = (
            (self.confidence >= threshold)
            & np.isfinite(self.confidence)
            & np.isfinite(self.realised)
            & np.isfinite(self.predictions)
        )

        if weighting == "equal":
            w = mask.astype(float)
        elif weighting == "confidence":
            w = np.where(mask, np.maximum(self.confidence, 0.0), 0.0)
        elif weighting == "signal_abs":
            w = np.where(mask, np.abs(self.predictions), 0.0)
        elif weighting == "custom":
            if weights is None:
                raise ValueError("custom weighting requires a weights array")
            if weights.shape != self.realised.shape:
                raise ValueError("weights must have the same shape as realised")
            w = np.where(mask, np.maximum(weights, 0.0), 0.0)
        else:
            raise ValueError(f"unknown weighting: {weighting}")

        if exposure == "as_selected":
            denom = np.nansum(w, axis=1)
            numer = np.nansum(self.realised * w, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.where(denom > 0, numer / denom, np.nan)

        if exposure != "dollar_neutral":
            raise ValueError(f"unknown exposure mode: {exposure}")

        long_mask = mask & (self.predictions > 0)
        short_mask = mask & (self.predictions < 0)
        long_w = np.where(long_mask, w, 0.0)
        short_w = np.where(short_mask, w, 0.0)
        long_denom = np.nansum(long_w, axis=1)
        short_denom = np.nansum(short_w, axis=1)
        long_numer = np.nansum(self.realised * long_w, axis=1)
        short_numer = np.nansum(self.realised * short_w, axis=1)
        out = np.full(self.T, np.nan, dtype=float)
        valid = (long_denom > 0) & (short_denom > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[valid] = 0.5 * (long_numer[valid] / long_denom[valid]) + 0.5 * (
                short_numer[valid] / short_denom[valid]
            )
        return out

    def selected_counts(self, threshold: float = 0.0) -> NDArray[np.int64]:
        """Number of finite selected rows per date."""
        mask = (
            (self.confidence >= threshold)
            & np.isfinite(self.confidence)
            & np.isfinite(self.realised)
        )
        return mask.sum(axis=1)


@dataclass
class BootstrapResult:
    """Results from a single date-level bootstrap run."""

    sharpe_point: float
    sharpe_se: float
    sharpe_ci_lo: float
    sharpe_ci_hi: float
    mean_return: float
    mean_return_se: float
    t_statistic: float
    p_value: float
    positive_p_value: float
    two_sided_p_value: float
    n_effective: float
    n_nominal: int
    bootstrap_sharpes: NDArray[np.float64]
    method: str
    block_size: int
    ci_method: str = "percentile"


@dataclass
class HACSharpeResult:
    """Delta-method HAC inference for the annualised Sharpe ratio."""

    sharpe: float
    se: float
    ci_lo: float
    ci_hi: float
    z_statistic: float
    positive_p_value: float
    two_sided_p_value: float
    mean_return: float
    mean_return_se: float
    bandwidth: int


def safe_sign(x: NDArray[np.float64] | float, tol: float = 0.0):
    """Sign convention used by the audit: sign(0)=0."""
    arr = np.asarray(x)
    signed = np.where(arr > tol, 1.0, np.where(arr < -tol, -1.0, 0.0))
    if np.isscalar(x):
        return float(signed)
    return signed


def skipped_compounded_signal(
    returns: NDArray[np.float64],
    lookback: int,
    skip: int = 2,
) -> NDArray[np.float64]:
    """
    Compute s[t,i] = prod_{j=t-L-q+1}^{t-q} (1+r[j,i]) - 1.

    A signal at date t therefore excludes the most recent `skip` returns.
    Windows containing missing values are returned as NaN.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if skip < 0:
        raise ValueError("skip must be non-negative")

    ret = np.asarray(returns, dtype=float)
    if ret.ndim != 2:
        raise ValueError("returns must be a T x N matrix")

    out = np.full_like(ret, np.nan, dtype=float)
    for t in range(ret.shape[0]):
        start = t - lookback - skip + 1
        end = t - skip + 1
        if start < 0 or end <= start:
            continue
        window = ret[start:end]
        finite = np.isfinite(window).all(axis=0)
        if finite.any():
            out[t, finite] = np.prod(1.0 + window[:, finite], axis=0) - 1.0
    return out


def percentile_rank_confidence(
    scores: NDArray[np.float64],
    use_abs: bool = True,
) -> NDArray[np.float64]:
    """
    Date-wise percentile ranks scaled to [0, 1] with midranks for ties.

    For each date with N_t > 1 finite scores, confidence is
    (rank - 1) / (N_t - 1).  A single finite score is assigned confidence
    1 by convention, representing a single-asset conviction.  Dates with
    no finite scores remain NaN.
    """
    from scipy.stats import rankdata

    values = np.abs(scores) if use_abs else np.asarray(scores, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    for t in range(values.shape[0]):
        valid = np.isfinite(values[t])
        n = int(valid.sum())
        if n == 0:
            continue
        if n == 1:
            out[t, valid] = 1.0
            continue
        ranks = rankdata(values[t, valid], method="average")
        out[t, valid] = (ranks - 1.0) / (n - 1.0)
    return out


def acf(x: NDArray[np.float64], max_lag: int = 20) -> NDArray[np.float64]:
    """Sample autocorrelation function up to max_lag."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.ones(max_lag + 1)
    max_lag = min(max_lag, max(0, n - 1))
    xm = x - x.mean()
    c0 = float(np.dot(xm, xm) / n)
    acf_vals = np.ones(max_lag + 1)
    if c0 <= 0:
        acf_vals[1:] = 0.0
        return acf_vals
    for k in range(1, max_lag + 1):
        acf_vals[k] = np.dot(xm[: n - k], xm[k:]) / (n * c0)
    return acf_vals


def effective_sample_size(x: NDArray[np.float64], max_lag: int = 50) -> float:
    """
    Positive-sequence effective sample size.

    N_eff = n / (1 + 2 sum_{k=1}^K rho_k), truncated at the first
    negative sample autocorrelation.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= 1:
        return float(max(n, 1))
    rho = acf(x, max_lag=min(max_lag, max(1, n // 3)))
    tau = 0.0
    for k in range(1, len(rho)):
        if rho[k] < 0:
            break
        tau += float(rho[k])
    return max(1.0, n / (1.0 + 2.0 * tau))


def cross_sectional_correlation(panel: PanelData) -> float:
    """Average pairwise cross-sectional correlation of realised returns."""
    R = np.asarray(panel.realised, dtype=float)
    valid_cols = np.isfinite(R).sum(axis=0) >= 10
    R = R[:, valid_cols]
    if R.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(R, rowvar=False)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = corr[mask]
    finite = vals[np.isfinite(vals)]
    return float(np.mean(finite)) if len(finite) else 0.0


def dependence_summary(returns: NDArray[np.float64], panel: PanelData) -> dict:
    """Dependence diagnostics for a date-level return series."""
    valid = np.asarray(returns, dtype=float)
    valid = valid[np.isfinite(valid)]
    rho = acf(valid, max_lag=20)
    n_eff = effective_sample_size(valid)
    return {
        "n_nominal": len(valid),
        "n_effective": n_eff,
        "n_eff_ratio": n_eff / len(valid) if len(valid) else np.nan,
        "inflation_ratio": len(valid) / n_eff if n_eff > 0 else np.nan,
        "acf_lag1": rho[1] if len(rho) > 1 else 0.0,
        "acf_lag5": rho[5] if len(rho) > 5 else 0.0,
        "acf_lag10": rho[10] if len(rho) > 10 else 0.0,
        "cross_sectional_corr": cross_sectional_correlation(panel),
    }


def _sharpe(returns: NDArray[np.float64], annualise: Annualisation = 252.0) -> float:
    """Annualised plug-in Sharpe ratio."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    sd = np.std(r, ddof=1)
    if sd <= 0:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(annualise))


def _sharpe_matrix(samples: NDArray[np.float64], annualise: Annualisation = 252.0) -> NDArray[np.float64]:
    """Vectorized annualized Sharpe for a bootstrap sample matrix."""
    means = np.mean(samples, axis=1)
    sds = np.std(samples, axis=1, ddof=1)
    out = np.zeros(samples.shape[0], dtype=float)
    valid = sds > 0
    out[valid] = means[valid] / sds[valid] * np.sqrt(annualise)
    return out


def _bootstrap_chunk_size(n: int, n_boot: int, target_cells: int = 5_000_000) -> int:
    """Limit temporary bootstrap matrices to a bounded number of cells."""
    return max(1, min(int(n_boot), max(1, target_cells // max(1, int(n)))))


def _iid_bootstrap(
    returns: NDArray[np.float64], n_boot: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    valid = returns[np.isfinite(returns)]
    n = len(valid)
    sharpes = np.empty(n_boot)
    chunk = _bootstrap_chunk_size(n, n_boot)
    pos = 0
    while pos < n_boot:
        size = min(chunk, n_boot - pos)
        idx = rng.integers(0, n, size=(size, n))
        sharpes[pos : pos + size] = _sharpe_matrix(valid[idx])
        pos += size
    return sharpes


def _blocked_bootstrap(
    returns: NDArray[np.float64],
    n_boot: int,
    block_size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    valid = returns[np.isfinite(returns)]
    n = len(valid)
    block_size = max(1, min(int(block_size), n))
    n_blocks = int(np.ceil(n / block_size))
    sharpes = np.empty(n_boot)
    chunk = _bootstrap_chunk_size(n_blocks * block_size, n_boot)
    offsets = np.arange(block_size)
    pos = 0
    while pos < n_boot:
        size = min(chunk, n_boot - pos)
        starts = rng.integers(0, n - block_size + 1, size=(size, n_blocks))
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(size, n_blocks * block_size)[:, :n]
        sharpes[pos : pos + size] = _sharpe_matrix(valid[idx])
        pos += size
    return sharpes


def _stationary_bootstrap(
    returns: NDArray[np.float64],
    n_boot: int,
    mean_block: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    valid = returns[np.isfinite(returns)]
    n = len(valid)
    mean_block = max(1.0, min(float(mean_block), float(n)))
    p = 1.0 / mean_block
    sharpes = np.empty(n_boot)
    chunk = _bootstrap_chunk_size(n, n_boot)
    pos = 0
    while pos < n_boot:
        size = min(chunk, n_boot - pos)
        starts = rng.integers(0, n, size=(size, n))
        switches = rng.random((size, n)) < p
        switches[:, 0] = True
        idx = np.empty((size, n), dtype=np.int64)
        idx[:, 0] = starts[:, 0]
        for i in range(1, n):
            idx[:, i] = np.where(switches[:, i], starts[:, i], idx[:, i - 1] + 1)
        idx %= n
        sharpes[pos : pos + size] = _sharpe_matrix(valid[idx])
        pos += size
    return sharpes


def _cluster_bootstrap(
    panel: PanelData,
    threshold: float,
    n_boot: int,
    cluster_by: Literal["date", "ticker"] = "date",
    rng: np.random.Generator | None = None,
    weighting: Weighting = "equal",
    exposure: Exposure = "as_selected",
) -> NDArray[np.float64]:
    if rng is None:
        rng = np.random.default_rng()

    if cluster_by == "date":
        base_returns = panel.date_returns(threshold, weighting=weighting, exposure=exposure)
        valid_idx = np.where(np.isfinite(base_returns))[0]
        n_valid = len(valid_idx)
        sharpes = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.choice(valid_idx, size=n_valid, replace=True)
            sharpes[b] = _sharpe(base_returns[idx])
        return sharpes

    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        ticker_idx = rng.integers(0, panel.N, size=panel.N)
        sub_panel = PanelData(
            dates=panel.dates,
            tickers=panel.tickers[ticker_idx],
            predictions=panel.predictions[:, ticker_idx],
            realised=panel.realised[:, ticker_idx],
            confidence=panel.confidence[:, ticker_idx],
        )
        sharpes[b] = _sharpe(sub_panel.date_returns(threshold, weighting=weighting, exposure=exposure))
    return sharpes


def optimal_block_length(
    x: NDArray[np.float64],
    objective: Literal["distribution", "variance"] = "distribution",
) -> int:
    """
    Rule-of-thumb block length with explicit objective.

    Distribution estimation for percentile CIs uses an n^(1/5) rate in the
    manuscript.  Variance-estimation sensitivity runs may request n^(1/3).
    This is not a full Politis-White plug-in selector; the manuscript treats
    it as a documented default and reports broad sensitivity.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return max(1, n)
    exponent = 1.0 / 5.0 if objective == "distribution" else 1.0 / 3.0
    rho1 = float(acf(x, max_lag=1)[1]) if n > 2 else 0.0
    rho1 = float(np.clip(rho1, 0.0, 0.95))
    dep_factor = (2.0 * rho1 / max(1.0 - rho1, 1e-6)) ** (2.0 / 3.0) if rho1 > 0 else 1.0
    b = (n**exponent) * max(dep_factor, 1.0)
    return max(1, min(int(np.ceil(b)), max(1, n // 3)))


def andrew_bartlett_bandwidth(x: NDArray[np.float64]) -> int:
    """
    Andrews (1991)-style automatic Bartlett bandwidth under AR(1) working model.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return 0
    xm = x - x.mean()
    denom = float(np.dot(xm[:-1], xm[:-1]))
    rho = float(np.dot(xm[:-1], xm[1:]) / denom) if denom > 0 else 0.0
    rho = float(np.clip(rho, -0.95, 0.95))
    alpha = 4.0 * rho * rho / max((1.0 - rho) ** 4, 1e-8)
    bw = 1.1447 * max(alpha * n, 1.0) ** (1.0 / 3.0)
    return max(1, min(int(np.ceil(bw)), max(1, n - 1)))


def long_run_covariance(
    values: NDArray[np.float64],
    max_lag: int | None = None,
) -> tuple[NDArray[np.float64], int]:
    """Bartlett-kernel HAC long-run covariance of a vector process."""
    z = np.asarray(values, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    z = z[np.all(np.isfinite(z), axis=1)]
    n = len(z)
    if n < 2:
        raise ValueError("at least two finite observations required")
    if max_lag is None:
        max_lag = andrew_bartlett_bandwidth(z[:, 0])
    max_lag = min(int(max_lag), n - 1)

    centered = z - z.mean(axis=0)
    S = centered.T @ centered / n
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = centered[:-lag].T @ centered[lag:] / n
        S += weight * (gamma + gamma.T)
    return S, max_lag


def newey_west_se(x: NDArray[np.float64], max_lag: int | None = None) -> float:
    """Newey-West HAC standard error of the sample mean."""
    valid = np.asarray(x, dtype=float)
    valid = valid[np.isfinite(valid)]
    S, _ = long_run_covariance(valid, max_lag=max_lag)
    return float(np.sqrt(max(S[0, 0], 0.0) / len(valid)))


def hac_sharpe_delta(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
    max_lag: int | None = None,
    annualise: Annualisation = 252.0,
) -> HACSharpeResult:
    """Delta-method HAC inference for annualised Sharpe."""
    from scipy import stats

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        raise ValueError("at least five returns required for HAC Sharpe inference")

    mu = float(np.mean(r))
    m2 = float(np.mean(r * r))
    var = max(m2 - mu * mu, 1e-16)
    sr = float(np.sqrt(annualise) * mu / np.sqrt(var))

    moments = np.column_stack([r, r * r])
    lrv, bw = long_run_covariance(moments, max_lag=max_lag)
    cov_mean = lrv / len(r)
    grad = np.array([
        np.sqrt(annualise) * m2 / (var ** 1.5),
        -0.5 * np.sqrt(annualise) * mu / (var ** 1.5),
    ])
    se = float(np.sqrt(max(grad @ cov_mean @ grad, 0.0)))
    z = sr / se if se > 0 else np.inf
    alpha = 1.0 - confidence
    crit = stats.norm.ppf(1.0 - alpha / 2.0)
    pos_p = float(1.0 - stats.norm.cdf(z))
    two_sided = float(2.0 * min(pos_p, 1.0 - pos_p))
    mean_se = newey_west_se(r, max_lag=bw)
    return HACSharpeResult(
        sharpe=sr,
        se=se,
        ci_lo=float(sr - crit * se),
        ci_hi=float(sr + crit * se),
        z_statistic=float(z),
        positive_p_value=pos_p,
        two_sided_p_value=two_sided,
        mean_return=mu,
        mean_return_se=mean_se,
        bandwidth=bw,
    )


def _sharpe_delta_from_lrv(
    returns: NDArray[np.float64],
    lrv: NDArray[np.float64],
    bandwidth: int,
    confidence: float = 0.95,
    annualise: Annualisation = 252.0,
) -> HACSharpeResult:
    """Build a HACSharpeResult from a long-run covariance for (r, r^2)."""
    from scipy import stats

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        raise ValueError("at least five returns required for HAC Sharpe inference")

    mu = float(np.mean(r))
    m2 = float(np.mean(r * r))
    var = max(m2 - mu * mu, 1e-16)
    sr = float(np.sqrt(annualise) * mu / np.sqrt(var))
    grad = np.array([
        np.sqrt(annualise) * m2 / (var ** 1.5),
        -0.5 * np.sqrt(annualise) * mu / (var ** 1.5),
    ])
    cov_mean = lrv / len(r)
    se = float(np.sqrt(max(grad @ cov_mean @ grad, 0.0)))
    z = sr / se if se > 0 else np.inf
    alpha = 1.0 - confidence
    crit = stats.norm.ppf(1.0 - alpha / 2.0)
    pos_p = float(1.0 - stats.norm.cdf(z))
    two_sided = float(2.0 * min(pos_p, 1.0 - pos_p))
    mean_se = float(np.sqrt(max(lrv[0, 0], 0.0) / len(r)))
    return HACSharpeResult(
        sharpe=sr,
        se=se,
        ci_lo=float(sr - crit * se),
        ci_hi=float(sr + crit * se),
        z_statistic=float(z),
        positive_p_value=pos_p,
        two_sided_p_value=two_sided,
        mean_return=mu,
        mean_return_se=mean_se,
        bandwidth=int(bandwidth),
    )


def hac_sharpe_delta_prewhite(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
    max_lag: int | None = None,
    annualise: Annualisation = 252.0,
) -> HACSharpeResult:
    """
    Prewhitened delta-method HAC inference for the annualised Sharpe ratio.

    A VAR(1) is fitted to centered moments (r_t, r_t^2).  HAC is applied to
    the residuals and then recolored by (I - Phi)^{-1}.  This is a
    robustness estimator, not the manuscript's primary gate.
    """
    z = np.asarray(returns, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 8:
        raise ValueError("at least eight returns required for prewhitened HAC")

    moments = np.column_stack([z, z * z])
    centered = moments - moments.mean(axis=0)
    x_lag = centered[:-1]
    y_now = centered[1:]
    phi_t, *_ = np.linalg.lstsq(x_lag, y_now, rcond=None)
    phi = phi_t.T

    eig = np.linalg.eigvals(phi)
    radius = float(np.max(np.abs(eig))) if eig.size else 0.0
    if np.isfinite(radius) and radius >= 0.98:
        phi *= 0.98 / radius

    resid = y_now - x_lag @ phi.T
    if max_lag is None:
        max_lag = andrew_bartlett_bandwidth(resid[:, 0])
    eps_lrv, bw = long_run_covariance(resid, max_lag=max_lag)
    recolor = np.linalg.pinv(np.eye(phi.shape[0]) - phi)
    lrv = recolor @ eps_lrv @ recolor.T
    return _sharpe_delta_from_lrv(
        z,
        lrv,
        bandwidth=bw,
        confidence=confidence,
        annualise=annualise,
    )


def fixed_b_hac_sensitivity(
    returns: NDArray[np.float64],
    b_grid: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20),
    n_sim: int = 1000,
    sim_length: int = 1000,
    seed: int = 42,
    annualise: Annualisation = 252.0,
) -> "pd.DataFrame":
    """
    Fixed-fraction HAC sensitivity for Sharpe positive-edge inference.

    For each bandwidth fraction b=K/T, this reports the usual normal HAC
    result on the observed sample and a simulation-calibrated p-value under
    an iid zero-mean null using the same bandwidth fraction on a bounded
    simulation grid.  This is a practical fixed-b-style robustness diagnostic
    rather than a replacement for the primary small-b HAC gate.
    """
    import pandas as pd

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        raise ValueError("at least twenty returns required for fixed-b sensitivity")

    rng = np.random.default_rng(seed)
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        sd = 1.0
    rows = []
    n = len(r)
    sim_n = max(50, min(int(sim_length), n))
    for b in b_grid:
        b = float(b)
        bw = max(0, min(n - 1, int(round(b * n))))
        sim_bw = max(0, min(sim_n - 1, int(round(b * sim_n))))
        res = hac_sharpe_delta(r, max_lag=bw, annualise=annualise)
        null_stats = np.empty(int(n_sim), dtype=float)
        for i in range(int(n_sim)):
            draw = rng.normal(0.0, sd, sim_n)
            try:
                null_stats[i] = hac_sharpe_delta(draw, max_lag=sim_bw, annualise=annualise).z_statistic
            except Exception:
                null_stats[i] = np.nan
        null_stats = null_stats[np.isfinite(null_stats)]
        if len(null_stats):
            fixed_p = float((1.0 + np.sum(null_stats >= res.z_statistic)) / (len(null_stats) + 1.0))
            fixed_two = float((1.0 + np.sum(np.abs(null_stats) >= abs(res.z_statistic))) / (len(null_stats) + 1.0))
            crit_95 = float(np.quantile(null_stats, 0.95))
            abs_crit_95 = float(np.quantile(np.abs(null_stats), 0.95))
        else:
            fixed_p = np.nan
            fixed_two = np.nan
            crit_95 = np.nan
            abs_crit_95 = np.nan
        rows.append({
            "b": b,
            "bandwidth": bw,
            "simulation_length": sim_n,
            "simulation_bandwidth": sim_bw,
            "sharpe": res.sharpe,
            "se": res.se,
            "z_statistic": res.z_statistic,
            "normal_positive_p": res.positive_p_value,
            "fixedb_positive_p": fixed_p,
            "fixedb_two_sided_p": fixed_two,
            "fixedb_crit_95": crit_95,
            "fixedb_abs_crit_95": abs_crit_95,
            "n_sim": int(len(null_stats)),
        })
    return pd.DataFrame(rows)


def sharpe_effective_sample_size(
    returns: NDArray[np.float64],
    annualise: Annualisation = 252.0,
) -> dict:
    """
    Sharpe-specific effective sample size implied by HAC-delta inflation.

    The old ACF-only diagnostic can flatline at T when the first return
    autocorrelation is negative.  This diagnostic scales T by the ratio of
    IID delta variance to HAC delta variance for the Sharpe functional.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        return {
            "n_eff_sr": float(max(n, 1)),
            "se_iid_delta": np.nan,
            "se_hac_delta": np.nan,
            "se_inflation": np.nan,
            "hac_bandwidth": np.nan,
        }
    iid = hac_sharpe_delta(r, max_lag=0, annualise=annualise)
    hac = hac_sharpe_delta(r, max_lag=None, annualise=annualise)
    if hac.se <= 0 or not np.isfinite(hac.se):
        n_eff = float(n)
        inflation = np.nan
    else:
        inflation = hac.se / iid.se if iid.se > 0 else np.nan
        n_eff = float(n * (iid.se * iid.se) / (hac.se * hac.se)) if iid.se > 0 else float(n)
    return {
        "n_eff_sr": max(1.0, min(float(n), n_eff)),
        "se_iid_delta": float(iid.se),
        "se_hac_delta": float(hac.se),
        "se_inflation": float(inflation) if np.isfinite(inflation) else np.nan,
        "hac_bandwidth": int(hac.bandwidth),
    }


def bootstrap_positive_p_value(bootstrap_sharpes: NDArray[np.float64]) -> float:
    """One-sided positive-edge bootstrap p-value with one-draw smoothing."""
    b = np.asarray(bootstrap_sharpes, dtype=float)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return 1.0
    return float((1.0 + np.sum(b <= 0.0)) / (len(b) + 1.0))


def sddm_inference(
    panel: PanelData,
    threshold: float = 0.0,
    method: BootstrapMethod = "iid",
    n_boot: int = 10_000,
    confidence: float = 0.95,
    block_size: int | None = None,
    seed: int = 42,
    weighting: Weighting = "equal",
    exposure: Exposure = "as_selected",
) -> BootstrapResult:
    """
    Run date-level bootstrap inference on a prediction panel.

    The interval is a two-sided percentile CI for estimation.  The primary
    decision p-value is one-sided for a positive edge and is computed from
    the bootstrap Sharpe distribution with one-draw smoothing.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    returns = panel.date_returns(threshold, weighting=weighting, exposure=exposure)
    valid = returns[np.isfinite(returns)]
    n = len(valid)
    if n < 10:
        raise ValueError(f"Only {n} valid dates after filtering; insufficient for inference.")

    sharpe_hat = _sharpe(valid)
    mean_ret = float(np.mean(valid))
    n_eff = effective_sample_size(valid)

    if block_size is None:
        block_size = optimal_block_length(valid, objective="distribution")

    if method == "iid":
        boot_sharpes = _iid_bootstrap(valid, n_boot, rng)
    elif method == "blocked":
        boot_sharpes = _blocked_bootstrap(valid, n_boot, block_size, rng)
    elif method == "stationary":
        boot_sharpes = _stationary_bootstrap(valid, n_boot, float(block_size), rng)
    elif method == "cluster_date":
        boot_sharpes = _cluster_bootstrap(panel, threshold, n_boot, "date", rng, weighting, exposure)
    elif method == "cluster_ticker":
        boot_sharpes = _cluster_bootstrap(panel, threshold, n_boot, "ticker", rng, weighting, exposure)
    else:
        raise ValueError(f"Unknown method: {method}")

    boot_se = float(np.std(boot_sharpes, ddof=1))
    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(boot_sharpes, 100.0 * alpha / 2.0))
    ci_hi = float(np.percentile(boot_sharpes, 100.0 * (1.0 - alpha / 2.0)))

    se_mean = float(np.std(valid, ddof=1) / np.sqrt(n_eff))
    t_stat = mean_ret / se_mean if se_mean > 0 else np.inf
    pos_t_p = float(1.0 - stats.t.cdf(t_stat, df=max(1, n_eff - 1)))
    two_sided_t_p = float(2.0 * min(pos_t_p, 1.0 - pos_t_p))
    positive_p = bootstrap_positive_p_value(boot_sharpes)

    return BootstrapResult(
        sharpe_point=sharpe_hat,
        sharpe_se=boot_se,
        sharpe_ci_lo=ci_lo,
        sharpe_ci_hi=ci_hi,
        mean_return=mean_ret,
        mean_return_se=se_mean,
        t_statistic=float(t_stat),
        p_value=positive_p,
        positive_p_value=positive_p,
        two_sided_p_value=two_sided_t_p,
        n_effective=n_eff,
        n_nominal=n,
        bootstrap_sharpes=boot_sharpes,
        method=method,
        block_size=int(block_size),
    )


def compare_methods(
    panel: PanelData,
    threshold: float = 0.0,
    n_boot: int = 10_000,
    seed: int = 42,
    weighting: Weighting = "equal",
    exposure: Exposure = "as_selected",
) -> list[BootstrapResult]:
    """Run common bootstrap methods for side-by-side comparison."""
    methods: list[BootstrapMethod] = ["iid", "blocked", "stationary", "cluster_date", "cluster_ticker"]
    return [
        sddm_inference(
            panel,
            threshold=threshold,
            method=m,
            n_boot=n_boot,
            seed=seed,
            weighting=weighting,
            exposure=exposure,
        )
        for m in methods
    ]


def format_comparison_table(results: list[BootstrapResult]) -> str:
    """Format comparison results as a text table."""
    header = (
        f"{'Method':<16} {'Sharpe':>8} {'SE':>8} {'95% CI':>18} "
        f"{'n_eff/T':>8} {'p+':>8} {'block':>6}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        ci = f"[{r.sharpe_ci_lo:.2f}, {r.sharpe_ci_hi:.2f}]"
        ratio = r.n_effective / r.n_nominal if r.n_nominal else np.nan
        lines.append(
            f"{r.method:<16} {r.sharpe_point:>8.3f} {r.sharpe_se:>8.3f} "
            f"{ci:>18} {ratio:>8.3f} {r.positive_p_value:>8.4f} {r.block_size:>6d}"
        )
    return "\n".join(lines)
