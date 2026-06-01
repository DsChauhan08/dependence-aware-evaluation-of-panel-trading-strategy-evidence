"""
Size and power curves for the audit methods.

This script keeps the production path compact but includes every method
reported in the revised manuscript: row-level naive, date-IID bootstrap,
HAC-delta, moving-block bootstrap, stationary bootstrap, and a one-column
Romano-Wolf joint-resampling check.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sddm_bootstrap import _sharpe, hac_sharpe_delta, sddm_inference
from simulation_study import DGPConfig, generate_panel, true_sharpe
from threshold_analysis import returns_matrix, romano_wolf_stepdown


OUTDIR = os.environ.get("AUDIT_OUTPUT_DIR", "output_prod")
N_SIM = int(os.environ.get("SDDM_N_SIM", 200))
N_BOOT = int(os.environ.get("SDDM_N_BOOT", 1000))
N_JOBS = int(os.environ.get("SDDM_N_JOBS", min(os.cpu_count() or 1, 8)))
os.makedirs(OUTDIR, exist_ok=True)


def row_level_positive_p(panel) -> float:
    y = panel.realised[np.isfinite(panel.realised)]
    if len(y) < 3:
        return 1.0
    se = np.std(y, ddof=1) / np.sqrt(len(y))
    t = np.mean(y) / se if se > 0 else np.inf
    return float(1.0 - stats.t.cdf(t, df=len(y) - 1))


def method_rejections(panel, alpha: float = 0.05) -> dict[str, bool]:
    returns = panel.date_returns(0.0)
    out = {"row_naive": row_level_positive_p(panel) < alpha}

    for method, label in [("iid", "date_iid"), ("blocked", "moving_block"), ("stationary", "stationary")]:
        try:
            res = sddm_inference(panel, threshold=0.0, method=method, n_boot=N_BOOT, seed=17)
            out[label] = res.positive_p_value < alpha
        except ValueError:
            out[label] = False

    try:
        out["hac_delta"] = hac_sharpe_delta(returns).positive_p_value < alpha
    except ValueError:
        out["hac_delta"] = False

    try:
        rw = romano_wolf_stepdown(returns_matrix(panel, [0.0]), n_boot=N_BOOT, seed=17)
        out["romano_wolf"] = bool(rw[0] < alpha)
    except Exception:
        out["romano_wolf"] = False
    return out


def _simulate_rejections(args) -> dict[str, bool]:
    cfg, seed = args
    panel = generate_panel(cfg, seed=seed)
    return method_rejections(panel)


def _count_rejections(cfg: DGPConfig, seeds: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    jobs = [(cfg, seed) for seed in seeds]
    if N_JOBS <= 1:
        iterator = map(_simulate_rejections, jobs)
    else:
        chunksize = max(1, len(jobs) // (N_JOBS * 4))
        pool = ProcessPoolExecutor(max_workers=N_JOBS)
        iterator = pool.map(_simulate_rejections, jobs, chunksize=chunksize)
    try:
        for rej in iterator:
            for method, flag in rej.items():
                counts[method] = counts.get(method, 0) + int(flag)
    finally:
        if N_JOBS > 1:
            pool.shutdown(wait=True)
    return counts


def run_size_test() -> pd.DataFrame:
    null_cfgs = [
        ("null_iid", DGPConfig("Null IID", T=1000, N=50, true_mu=0.0)),
        ("null_dep", DGPConfig("Null dependence", T=1000, N=50, true_mu=0.0, ar1_serial=0.2, rho_cross=0.2)),
        ("null_garch", DGPConfig(
            "Null GARCH dependence",
            T=1000,
            N=50,
            true_mu=0.0,
            ar1_serial=0.2,
            rho_cross=0.2,
            garch_alpha=0.05,
            garch_beta=0.90,
        )),
    ]
    rows = []
    for key, cfg in null_cfgs:
        print(f"Size DGP {key}: {N_SIM} sims, {N_BOOT} boot, {N_JOBS} jobs", flush=True)
        counts = _count_rejections(cfg, [50_000 + sim for sim in range(N_SIM)])
        for method, count in sorted(counts.items()):
            rate = count / N_SIM
            rows.append({
                "dgp": key,
                "name": cfg.name,
                "method": method,
                "rejection_rate": rate,
                "se": np.sqrt(rate * (1.0 - rate) / N_SIM),
                "n": N_SIM,
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/size_test_audit.csv", index=False)
    return df


def run_power() -> pd.DataFrame:
    mus = [0.0, 0.00005, 0.0001, 0.0002, 0.0004, 0.0008, 0.0015]
    rows = []
    for mu in mus:
        cfg = DGPConfig(
            f"Power(mu={mu})",
            T=1000,
            N=50,
            true_mu=mu,
            ar1_serial=0.2,
            rho_cross=0.3,
        )
        print(f"Power mu={mu}: {N_SIM} sims, {N_BOOT} boot, {N_JOBS} jobs", flush=True)
        counts = _count_rejections(cfg, [80_000 + sim for sim in range(N_SIM)])
        for method, count in sorted(counts.items()):
            rows.append({
                "true_mu": mu,
                "true_sharpe": true_sharpe(cfg),
                "method": method,
                "power": count / N_SIM,
                "n": N_SIM,
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/power_audit.csv", index=False)
    df.pivot_table(values="power", index="true_sharpe", columns="method").to_csv(
        f"{OUTDIR}/power_audit_pivot.csv"
    )
    return df


if __name__ == "__main__":
    print(f"Size/power audit: {N_SIM} sims, {N_BOOT} bootstrap draws, {N_JOBS} jobs")
    print(run_size_test().to_string(index=False, float_format="%.4f"))
    print(run_power().to_string(index=False, float_format="%.4f"))
