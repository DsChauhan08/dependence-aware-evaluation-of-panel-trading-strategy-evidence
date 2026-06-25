"""
Synthetic positive-control audit.

The public candidate campaign can validly end with no public audit-gate pass.
This script supplies a reproducible counterexample to the stronger criticism
that the audit gates are impossible to pass.  It builds a panel with a known
directional edge, cross-sectional dependence, serially persistent common
shocks, and persistent signals so turnover costs are finite.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from public_data import cost_sensitivity, selected_count_summary
from sddm_bootstrap import PanelData, _sharpe, cross_sectional_correlation, hac_sharpe_delta, safe_sign, sddm_inference
from threshold_analysis import returns_matrix, romano_wolf_menu_test


@dataclass(frozen=True)
class SyntheticPositiveControlConfig:
    name: str = "synthetic_positive_control"
    T: int = 5000
    N: int = 100
    annual_alpha: float = 0.12
    idiosyncratic_sigma: float = 0.01
    common_factor_loading: float = 0.50
    common_factor_ar1: float = 0.10
    signal_stay_probability: float = 0.98
    seed: int = 777

    @property
    def daily_alpha(self) -> float:
        return self.annual_alpha / 252.0


def generate_positive_control(cfg: SyntheticPositiveControlConfig) -> PanelData:
    """Generate a signed-return panel with a known positive edge."""
    rng = np.random.default_rng(cfg.seed)
    phi = float(np.clip(cfg.common_factor_ar1, -0.99, 0.99))

    factor_state = np.empty(cfg.T, dtype=float)
    factor_state[0] = rng.normal(0.0, cfg.idiosyncratic_sigma)
    innov_sd = cfg.idiosyncratic_sigma * np.sqrt(max(1.0 - phi * phi, 1e-12))
    for t in range(1, cfg.T):
        factor_state[t] = phi * factor_state[t - 1] + rng.normal(0.0, innov_sd)
    common = cfg.common_factor_loading * factor_state

    idiosyncratic = rng.normal(0.0, cfg.idiosyncratic_sigma, size=(cfg.T, cfg.N))
    signed_returns = cfg.daily_alpha + common[:, None] + idiosyncratic

    signs = np.empty((cfg.T, cfg.N), dtype=float)
    signs[0] = rng.choice(np.array([-1.0, 1.0]), size=cfg.N)
    flip_probability = 1.0 - cfg.signal_stay_probability
    for t in range(1, cfg.T):
        flips = rng.random(cfg.N) < flip_probability
        signs[t] = np.where(flips, -signs[t - 1], signs[t - 1])

    raw_returns = signs * signed_returns
    realised = signs * raw_returns
    confidence = np.ones_like(realised)
    dates = np.array([np.datetime64("2000-01-03") + np.timedelta64(i, "D") for i in range(cfg.T)])
    tickers = np.array([f"SYN{i:03d}" for i in range(cfg.N)])
    return PanelData(dates=dates, tickers=tickers, predictions=signs, realised=realised, confidence=confidence)


def _raw_returns_from_panel(panel: PanelData) -> np.ndarray:
    signs = safe_sign(panel.predictions)
    out = np.full_like(panel.realised, np.nan, dtype=float)
    ok = (signs != 0.0) & np.isfinite(signs) & np.isfinite(panel.realised)
    out[ok] = panel.realised[ok] / signs[ok]
    return out


def _date_returns(predictions: np.ndarray, confidence: np.ndarray, realised: np.ndarray, threshold: float) -> np.ndarray:
    mask = (
        (confidence >= threshold)
        & np.isfinite(confidence)
        & np.isfinite(realised)
        & np.isfinite(predictions)
    )
    denom = mask.sum(axis=1).astype(float)
    numer = np.nansum(np.where(mask, realised, np.nan), axis=1)
    out = np.full(realised.shape[0], np.nan, dtype=float)
    np.divide(numer, denom, out=out, where=denom > 0)
    return out


def same_date_permutation(panel: PanelData, thresholds: list[float], n_perms: int, seed: int) -> pd.DataFrame:
    """Permute signals within each date and recompute signed date returns."""
    raw_returns = _raw_returns_from_panel(panel)
    observed = {thr: _sharpe(panel.date_returns(thr)) for thr in thresholds}
    rng = np.random.default_rng(seed)
    draws = {thr: [] for thr in thresholds}

    for _ in range(n_perms):
        order = np.argsort(rng.random(panel.predictions.shape), axis=1)
        shuffled_pred = np.take_along_axis(panel.predictions, order, axis=1)
        shuffled_conf = np.take_along_axis(panel.confidence, order, axis=1)
        signed = safe_sign(shuffled_pred) * raw_returns
        signed[~np.isfinite(raw_returns)] = np.nan
        for thr in thresholds:
            draws[thr].append(_sharpe(_date_returns(shuffled_pred, shuffled_conf, signed, thr)))

    rows = []
    for thr in thresholds:
        values = np.asarray(draws[thr], dtype=float)
        values = values[np.isfinite(values)]
        obs = observed[thr]
        rows.append({
            "threshold": thr,
            "observed_sharpe": obs,
            "null_mean": float(np.mean(values)) if len(values) else np.nan,
            "null_sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "null_q025": float(np.quantile(values, 0.025)) if len(values) else np.nan,
            "null_q500": float(np.quantile(values, 0.500)) if len(values) else np.nan,
            "null_q975": float(np.quantile(values, 0.975)) if len(values) else np.nan,
            "p_positive": float((1.0 + np.sum(values >= obs)) / (len(values) + 1.0)) if len(values) else np.nan,
            "n_perms": int(len(values)),
            "seed": seed,
        })
    return pd.DataFrame(rows)


def render_latex(summary: pd.DataFrame, path: Path) -> None:
    row = summary.iloc[0]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Synthetic positive-control evaluation.}",
        r"\label{tab:synthetic-positive-control}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Panel & $\bar\rho$ & Adj. & SR & HAC $p^+$ & RW $p$ & Perm. $p^+$ \\",
        r"\midrule",
        (
            "Synthetic control"
            f" & {row['avg_pairwise_corr']:.3f}"
            f" & {row['proposition_adjustment_factor']:.2f}"
            f" & {row['gross_sharpe']:.3f}"
            f" & {row['hac_p_positive']:.2e}"
            f" & {row['romano_wolf_p']:.2e}"
            f" & {row['permutation_p']:.2e}"
            r" \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    cfg: SyntheticPositiveControlConfig,
    output_dir: Path,
    paper_dir: Path | None,
    n_boot: int,
    n_perms: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [0.0]
    primary = 0.0
    panel = generate_positive_control(cfg)
    returns = panel.date_returns(primary)

    methods = []
    for method in ["iid", "blocked", "stationary", "cluster_date"]:
        res = sddm_inference(panel, threshold=primary, method=method, n_boot=n_boot, seed=cfg.seed)
        methods.append({
            "method": method,
            "sharpe": res.sharpe_point,
            "se": res.sharpe_se,
            "ci_lo": res.sharpe_ci_lo,
            "ci_hi": res.sharpe_ci_hi,
            "p_positive": res.positive_p_value,
            "n": res.n_nominal,
            "block_size": res.block_size,
        })
    pd.DataFrame(methods).to_csv(output_dir / "positive_control_methods.csv", index=False)

    hac = hac_sharpe_delta(returns)
    pd.DataFrame([hac.__dict__]).to_csv(output_dir / "positive_control_hac_delta.csv", index=False)
    rw = romano_wolf_menu_test(returns_matrix(panel, thresholds), n_boot=n_boot, seed=cfg.seed)
    rw["threshold"] = thresholds
    rw.to_csv(output_dir / "positive_control_romano_wolf.csv", index=False)
    perm = same_date_permutation(panel, thresholds, n_perms=n_perms, seed=cfg.seed + 81)
    perm.to_csv(output_dir / "positive_control_permutation.csv", index=False)
    selected_count_summary(panel, thresholds).to_csv(output_dir / "positive_control_selected_counts.csv", index=False)
    costs = cost_sensitivity(panel, threshold=primary)
    costs.to_csv(output_dir / "positive_control_costs.csv", index=False)

    rho = cross_sectional_correlation(panel)
    adj = float(np.sqrt(1.0 + (cfg.N - 1.0) * max(rho, 0.0)))
    net_5 = float(costs.loc[costs["cost_bps_per_rebalance"].eq(5), "net_sharpe"].iloc[0])
    daily_turnover = float(costs.loc[costs["cost_bps_per_rebalance"].eq(5), "daily_turnover"].iloc[0])
    rw_p = float(rw["p_adjusted"].iloc[0])
    perm_p = float(perm["p_positive"].iloc[0])
    row = {
        **asdict(cfg),
        "daily_alpha": cfg.daily_alpha,
        "avg_pairwise_corr": rho,
        "proposition_adjustment_factor": adj,
        "gross_sharpe": _sharpe(returns),
        "hac_p_positive": float(hac.positive_p_value),
        "romano_wolf_p": rw_p,
        "permutation_p": perm_p,
        "net_sharpe_5bps": net_5,
        "daily_turnover": daily_turnover,
        "n_boot": int(n_boot),
        "n_perms": int(n_perms),
        "passes_hac": bool(hac.positive_p_value <= 0.05),
        "passes_romano_wolf": bool(rw_p <= 0.05),
        "passes_permutation": bool(perm_p <= 0.05),
        "passes_cost": bool(net_5 > 0.0),
    }
    row["passes_all_gates"] = bool(row["passes_hac"] and row["passes_romano_wolf"] and row["passes_permutation"] and row["passes_cost"])
    summary = pd.DataFrame([row])
    summary.to_csv(output_dir / "positive_control_summary.csv", index=False)
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        render_latex(summary, paper_dir / "generated_positive_control_artifacts.tex")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("output_synthetic_positive_control")))
    parser.add_argument("--paper-dir", default=str(Path(__file__).resolve().parents[1] / "paper"))
    parser.add_argument("--n-boot", type=int, default=int(os.environ.get("SDDM_N_BOOT", 2000)))
    parser.add_argument("--n-perms", type=int, default=int(os.environ.get("SDDM_N_PERMS", 1000)))
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SyntheticPositiveControlConfig(seed=args.seed)
    summary = run_audit(
        cfg,
        output_dir=Path(args.output_dir),
        paper_dir=Path(args.paper_dir) if args.paper_dir else None,
        n_boot=args.n_boot,
        n_perms=args.n_perms,
    )
    print(summary.to_string(index=False, float_format="%.6g"))


if __name__ == "__main__":
    main()
