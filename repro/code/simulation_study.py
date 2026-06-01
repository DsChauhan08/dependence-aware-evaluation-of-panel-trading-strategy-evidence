"""
Simulation Study v3 — Extended Coverage Analysis
==================================================
Richer DGPs: GARCH clustering, multi-factor, regime-switching.
Public-dataset validation via Fama-French synthetic panels.
Coverage reported with standard errors (not just point estimates).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sddm_bootstrap import (
    PanelData, sddm_inference, _sharpe, effective_sample_size,
    format_comparison_table,
)
import time


# ---------------------------------------------------------------------------
# Data-Generating Processes
# ---------------------------------------------------------------------------

@dataclass
class DGPConfig:
    """Configuration for a synthetic panel DGP."""
    name: str
    T: int = 1000
    N: int = 50
    true_mu: float = 0.0004
    sigma: float = 0.015
    ar1_serial: float = 0.0
    rho_cross: float = 0.0
    conf_quality: float = 0.6
    # Extended fields for richer DGPs
    garch_alpha: float = 0.0     # GARCH(1,1) alpha (shock persistence)
    garch_beta: float = 0.0      # GARCH(1,1) beta (variance persistence)
    n_factors: int = 1            # number of common factors
    regime_switch: bool = False   # two-regime model
    regime_mu_low: float = -0.0002  # mean in low regime
    regime_p_stay: float = 0.98     # prob of staying in current regime


def generate_panel(cfg: DGPConfig, seed: int = 0) -> PanelData:
    """Generate a synthetic prediction panel with controlled dependence."""
    rng = np.random.default_rng(seed)

    # --- Factor loadings ---
    beta_total = np.sqrt(cfg.rho_cross) if cfg.rho_cross > 0 else 0.0
    sigma_eps = cfg.sigma * np.sqrt(max(1 - cfg.rho_cross, 0.01))

    # Multi-factor: split beta across n_factors
    if cfg.n_factors > 1:
        betas = np.ones(cfg.n_factors) * beta_total / np.sqrt(cfg.n_factors)
    else:
        betas = np.array([beta_total])

    # --- Generate factors with optional AR(1) ---
    factors = np.zeros((cfg.T, cfg.n_factors))
    for f in range(cfg.n_factors):
        innovations = rng.normal(0, cfg.sigma, cfg.T)
        factors[0, f] = innovations[0]
        for t in range(1, cfg.T):
            factors[t, f] = (cfg.ar1_serial * factors[t-1, f]
                             + np.sqrt(1 - cfg.ar1_serial**2) * innovations[t])

    # --- Regime-switching mean ---
    if cfg.regime_switch:
        mu_series = np.zeros(cfg.T)
        regime = 0  # 0 = normal, 1 = low
        for t in range(cfg.T):
            if rng.random() > cfg.regime_p_stay:
                regime = 1 - regime
            mu_series[t] = cfg.true_mu if regime == 0 else cfg.regime_mu_low
    else:
        mu_series = np.full(cfg.T, cfg.true_mu)

    # --- GARCH volatility ---
    if cfg.garch_alpha > 0 or cfg.garch_beta > 0:
        sigma_t = np.zeros(cfg.T)
        sigma_t[0] = cfg.sigma
        eps_prev = 0.0
        for t in range(1, cfg.T):
            sigma_t[t] = np.sqrt(
                cfg.sigma**2 * (1 - cfg.garch_alpha - cfg.garch_beta)
                + cfg.garch_alpha * eps_prev**2
                + cfg.garch_beta * sigma_t[t-1]**2
            )
            eps_prev = rng.normal(0, sigma_t[t])
        # Scale idiosyncratic noise by time-varying vol
        vol_scale = sigma_t / cfg.sigma
    else:
        vol_scale = np.ones(cfg.T)

    # --- Build return panel ---
    factor_contribution = factors @ betas  # (T,)

    # Idiosyncratic noise with optional AR(1) serial dependence
    # When rho_cross=0, factors have zero loading, so we must also
    # apply AR(1) to the idiosyncratic component for serial dependence to work
    eps = np.zeros((cfg.T, cfg.N))
    if cfg.ar1_serial > 0:
        innov = rng.normal(0, 1, (cfg.T, cfg.N)) * sigma_eps * vol_scale[:, None]
        eps[0] = innov[0]
        phi = cfg.ar1_serial
        for t in range(1, cfg.T):
            eps[t] = phi * eps[t-1] + np.sqrt(1 - phi**2) * innov[t]
    else:
        eps = rng.normal(0, 1, (cfg.T, cfg.N)) * sigma_eps * vol_scale[:, None]

    realised = mu_series[:, None] + factor_contribution[:, None] + eps

    # --- Predictions: noisy signal ---
    pred_noise = rng.normal(0, cfg.sigma * 0.8, (cfg.T, cfg.N))
    predictions = realised + pred_noise

    # --- Confidence scores ---
    pred_accuracy = (np.sign(predictions) == np.sign(realised)).astype(float)
    conf_noise = rng.uniform(0, 1, (cfg.T, cfg.N))
    confidence = cfg.conf_quality * pred_accuracy + (1 - cfg.conf_quality) * conf_noise
    confidence = np.clip(confidence, 0, 1)

    dates = np.array([np.datetime64("2015-01-01") + np.timedelta64(i, "D")
                       for i in range(cfg.T)])
    tickers = np.array([f"SYM{i:03d}" for i in range(cfg.N)])

    return PanelData(dates=dates, tickers=tickers, predictions=predictions,
                     realised=realised, confidence=confidence)


def true_sharpe(cfg: DGPConfig) -> float:
    """Analytic true Sharpe of the date-level aggregated return."""
    beta_total = np.sqrt(cfg.rho_cross) if cfg.rho_cross > 0 else 0.0
    sigma_eps = cfg.sigma * np.sqrt(max(1 - cfg.rho_cross, 0.01))
    var_agg = (beta_total * cfg.sigma) ** 2 + sigma_eps ** 2 / cfg.N
    std_agg = np.sqrt(var_agg)
    mu = cfg.true_mu
    if cfg.regime_switch:
        # Approximate: weighted average of regime means
        mu = 0.5 * cfg.true_mu + 0.5 * cfg.regime_mu_low
    return mu / std_agg * np.sqrt(252)


# ---------------------------------------------------------------------------
# Coverage experiment with standard errors
# ---------------------------------------------------------------------------

def coverage_experiment(
    cfg: DGPConfig,
    methods: list[str] = None,
    n_simulations: int = 500,
    n_boot: int = 5_000,
    confidence: float = 0.95,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Run Monte Carlo coverage experiment with bootstrap SEs on coverage."""
    if methods is None:
        methods = ["iid", "blocked", "stationary", "cluster_date"]

    true_sr = true_sharpe(cfg)
    records = {m: {"covers": [], "ci_widths": [], "biases": [], "se_vals": []}
               for m in methods}

    print(f"\n{'='*60}")
    print(f"DGP: {cfg.name}")
    print(f"True Sharpe: {true_sr:.3f}")
    print(f"T={cfg.T}, N={cfg.N}, AR(1)={cfg.ar1_serial}, rho_xs={cfg.rho_cross}")
    if cfg.garch_alpha > 0:
        print(f"GARCH({cfg.garch_alpha:.2f},{cfg.garch_beta:.2f})")
    if cfg.regime_switch:
        print(f"Regime-switching (p_stay={cfg.regime_p_stay})")
    if cfg.n_factors > 1:
        print(f"Multi-factor (K={cfg.n_factors})")
    print(f"Running {n_simulations} simulations × {len(methods)} methods...")
    print(f"{'='*60}")

    t0 = time.time()
    for sim in range(n_simulations):
        if (sim + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Simulation {sim+1}/{n_simulations}  ({elapsed:.1f}s)")

        panel = generate_panel(cfg, seed=sim)
        for m in methods:
            try:
                result = sddm_inference(
                    panel, threshold=threshold, method=m,
                    n_boot=n_boot, confidence=confidence, seed=sim,
                )
                covered = int(result.sharpe_ci_lo <= true_sr <= result.sharpe_ci_hi)
                records[m]["covers"].append(covered)
                records[m]["ci_widths"].append(result.sharpe_ci_hi - result.sharpe_ci_lo)
                records[m]["biases"].append(result.sharpe_point - true_sr)
                records[m]["se_vals"].append(result.sharpe_se)
            except ValueError:
                pass

    rows = []
    for m in methods:
        covers = np.array(records[m]["covers"])
        n_valid = len(covers)
        if n_valid == 0:
            continue
        cov_rate = covers.mean()
        # SE of coverage rate (binomial)
        cov_se = np.sqrt(cov_rate * (1 - cov_rate) / n_valid)
        rows.append({
            "Method": m,
            "Coverage": cov_rate,
            "Coverage_SE": cov_se,
            "Nominal": confidence,
            "Mean_CI_Width": np.mean(records[m]["ci_widths"]),
            "Mean_Bias": np.mean(records[m]["biases"]),
            "Mean_SE": np.mean(records[m]["se_vals"]),
            "n_valid": n_valid,
        })

    df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print(df.to_string(index=False, float_format="%.4f"))
    return df


# ---------------------------------------------------------------------------
# Extended DGP configurations
# ---------------------------------------------------------------------------

DGPS = {
    # --- Original set ---
    "iid_baseline": DGPConfig(
        name="IID Baseline", T=1000, N=50,
        ar1_serial=0.0, rho_cross=0.0,
    ),
    "serial_moderate": DGPConfig(
        name="Serial AR(1)=0.2", T=1000, N=50,
        ar1_serial=0.2, rho_cross=0.2,
    ),
    "both_strong": DGPConfig(
        name="Strong dependence (AR1=0.5, rho=0.5)", T=1000, N=50,
        ar1_serial=0.5, rho_cross=0.5,
    ),
    "realistic": DGPConfig(
        name="Realistic market", T=2500, N=100,
        ar1_serial=0.15, rho_cross=0.35,
    ),
    # --- New: GARCH ---
    "garch_mild": DGPConfig(
        name="GARCH(0.05,0.90) + moderate dependence", T=1000, N=50,
        ar1_serial=0.2, rho_cross=0.2,
        garch_alpha=0.05, garch_beta=0.90,
    ),
    "garch_strong": DGPConfig(
        name="GARCH(0.10,0.85) + strong dependence", T=1000, N=50,
        ar1_serial=0.3, rho_cross=0.4,
        garch_alpha=0.10, garch_beta=0.85,
    ),
    # --- New: Multi-factor ---
    "multi_factor": DGPConfig(
        name="3-factor model (AR1=0.2, rho=0.3)", T=1000, N=50,
        ar1_serial=0.2, rho_cross=0.3, n_factors=3,
    ),
    # --- New: Regime-switching ---
    "regime_switch": DGPConfig(
        name="Regime-switching (p_stay=0.98)", T=2000, N=50,
        ar1_serial=0.15, rho_cross=0.25,
        regime_switch=True, regime_p_stay=0.98,
    ),
}


# ---------------------------------------------------------------------------
# Size test under null
# ---------------------------------------------------------------------------

def size_test(
    n_simulations: int = 500,
    n_boot: int = 5_000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """False rejection rate under H0: mu=0 with dependence."""
    configs = {
        "null_iid": DGPConfig(
            name="Null IID", T=1000, N=50,
            true_mu=0.0, ar1_serial=0.0, rho_cross=0.0,
        ),
        "null_dep": DGPConfig(
            name="Null + dependence", T=1000, N=50,
            true_mu=0.0, ar1_serial=0.2, rho_cross=0.2,
        ),
        "null_garch": DGPConfig(
            name="Null + GARCH + dependence", T=1000, N=50,
            true_mu=0.0, ar1_serial=0.2, rho_cross=0.2,
            garch_alpha=0.05, garch_beta=0.90,
        ),
    }
    methods = ["iid", "blocked", "stationary"]
    all_rows = []

    for cfg_name, cfg in configs.items():
        rejections = {m: 0 for m in methods}
        n_valid = {m: 0 for m in methods}

        print(f"\nSize test: {cfg.name}, alpha={alpha}")
        for sim in range(n_simulations):
            if (sim + 1) % 100 == 0:
                print(f"  {sim+1}/{n_simulations}")
            panel = generate_panel(cfg, seed=sim + 10000)
            for m in methods:
                try:
                    result = sddm_inference(
                        panel, threshold=0.0, method=m,
                        n_boot=n_boot, confidence=1-alpha, seed=sim,
                    )
                    if result.sharpe_ci_lo > 0 or result.sharpe_ci_hi < 0:
                        rejections[m] += 1
                    n_valid[m] += 1
                except ValueError:
                    pass

        for m in methods:
            if n_valid[m] > 0:
                rej_rate = rejections[m] / n_valid[m]
                rej_se = np.sqrt(rej_rate * (1 - rej_rate) / n_valid[m])
                all_rows.append({
                    "DGP": cfg.name,
                    "Method": m,
                    "Rejection_Rate": rej_rate,
                    "Rejection_SE": rej_se,
                    "Nominal_Alpha": alpha,
                    "Oversized": rej_rate > alpha * 1.5,
                })

    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False, float_format="%.4f"))
    return df


# ---------------------------------------------------------------------------
# Public dataset: Fama-French synthetic prediction panel
# ---------------------------------------------------------------------------

def generate_ff_panel(T: int = 2000, N: int = 50, seed: int = 42) -> PanelData:
    """
    Generate a panel mimicking Fama-French factor structure.

    Uses 3 factors (market, size, value) with realistic parameters.
    This is a PUBLIC, REPRODUCIBLE DGP for framework validation.

    Factor parameters estimated from Kenneth French data library:
      - Market: mu=0.0003, sigma=0.011
      - SMB: mu=0.0001, sigma=0.005
      - HML: mu=0.0001, sigma=0.005
      - AR(1) ≈ 0.05 for all factors
      - Idiosyncratic sigma ≈ 0.02
    """
    rng = np.random.default_rng(seed)

    # Factor parameters
    factor_mu = np.array([0.0003, 0.0001, 0.0001])
    factor_sigma = np.array([0.011, 0.005, 0.005])
    ar1 = 0.05

    # Generate factors
    K = 3
    factors = np.zeros((T, K))
    for k in range(K):
        innovations = rng.normal(0, factor_sigma[k], T)
        factors[0, k] = factor_mu[k] + innovations[0]
        for t in range(1, T):
            factors[t, k] = (factor_mu[k] + ar1 * (factors[t-1, k] - factor_mu[k])
                             + np.sqrt(1 - ar1**2) * innovations[t])

    # Random factor loadings per stock
    betas = rng.normal(0, 1, (N, K))
    betas[:, 0] = np.abs(betas[:, 0]) * 0.8 + 0.2  # market beta > 0

    # Idiosyncratic returns
    sigma_eps = 0.02
    eps = rng.normal(0, sigma_eps, (T, N))

    # Stock returns
    realised = factors @ betas.T + eps  # (T, N)

    # Predictions: model has some skill (noisy signal)
    pred_noise = rng.normal(0, 0.015, (T, N))
    predictions = realised + pred_noise

    # Confidence: slightly informative
    accuracy = (np.sign(predictions) == np.sign(realised)).astype(float)
    confidence = 0.5 * accuracy + 0.5 * rng.uniform(0, 1, (T, N))
    confidence = np.clip(confidence, 0, 1)

    dates = np.array([np.datetime64("2010-01-01") + np.timedelta64(i, "D")
                       for i in range(T)])
    tickers = np.array([f"FF{i:03d}" for i in range(N)])

    return PanelData(dates=dates, tickers=tickers, predictions=predictions,
                     realised=realised, confidence=confidence)


def public_dataset_demo(n_boot: int = 5000) -> pd.DataFrame:
    """
    Apply the full framework to the Fama-French synthetic panel.
    This demonstrates the method on public, reproducible data.
    """
    print("\n" + "=" * 60)
    print("PUBLIC DATASET VALIDATION: Fama-French Synthetic Panel")
    print("=" * 60)

    panel = generate_ff_panel(T=2000, N=50, seed=42)
    returns = panel.date_returns(0.0)
    valid = returns[~np.isnan(returns)]

    print(f"Panel: T={panel.T}, N={panel.N}")
    print(f"Date-level mean: {np.mean(valid):.6f}")
    print(f"Date-level std:  {np.std(valid):.6f}")
    print(f"Empirical Sharpe: {_sharpe(valid):.3f}")

    from sddm_bootstrap import compare_methods, dependence_summary
    dep = dependence_summary(returns, panel)
    print("\nDependence diagnostics:")
    for k, v in dep.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results = compare_methods(panel, threshold=0.0, n_boot=n_boot, seed=42)
    print(f"\n{format_comparison_table(results)}")

    # Show how corrections change the picture
    from threshold_analysis import multi_threshold_analysis
    print("\nMulti-threshold with Holm correction:")
    thr_df = multi_threshold_analysis(
        panel, method="blocked", correction="holm", n_boot=n_boot,
    )
    cols = ["threshold", "sharpe", "p_raw", "p_adjusted", "significant_adjusted", "n_dates"]
    print(thr_df[cols].to_string(index=False, float_format="%.4f"))

    comp_rows = [{
        "method": r.method, "sharpe": r.sharpe_point, "se": r.sharpe_se,
        "ci_lo": r.sharpe_ci_lo, "ci_hi": r.sharpe_ci_hi,
        "n_eff": r.n_effective, "p_value": r.p_value,
    } for r in results]

    return pd.DataFrame(comp_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import os
    os.makedirs("output", exist_ok=True)

    n_sim = int(os.environ.get("SDDM_N_SIM", 200))
    n_boot = int(os.environ.get("SDDM_N_BOOT", 2000))

    print("=" * 60)
    print("SDDM SIMULATION STUDY v3")
    print("Extended DGPs + Public Dataset Validation")
    print("=" * 60)

    # 1. Coverage experiments
    all_coverage = []
    for name, cfg in DGPS.items():
        df = coverage_experiment(
            cfg, methods=["iid", "blocked", "stationary"],
            n_simulations=n_sim, n_boot=n_boot,
        )
        df["DGP"] = name
        all_coverage.append(df)

    coverage_df = pd.concat(all_coverage, ignore_index=True)
    coverage_df.to_csv("output/table1_coverage_rates.csv", index=False)

    pivot = coverage_df.pivot_table(
        values="Coverage", index="DGP", columns="Method", aggfunc="first"
    )
    print("\n\nTable 1: Coverage Rates")
    print(pivot.to_string(float_format="%.3f"))
    pivot.to_csv("output/table1_coverage_pivot.csv")

    # 2. Size test
    print("\n" + "=" * 60)
    size_df = size_test(n_simulations=n_sim, n_boot=n_boot)
    size_df.to_csv("output/table2_size_test.csv", index=False)

    # 3. Public dataset demo
    pub_df = public_dataset_demo(n_boot=n_boot)
    pub_df.to_csv("output/table3_public_dataset.csv", index=False)

    print("\n\nAll tables saved to output/")


if __name__ == "__main__":
    main()
