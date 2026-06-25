"""
Hard-gate validator for full campaign artifacts.

This script validates research outputs before any reviewer memo should be used
as the basis for a paper rewrite.  It intentionally rejects smoke-scale runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FULL_N_SIM = 1000
FULL_N_BOOT = 5000
FULL_N_PERMS = 10000


class ValidationError(RuntimeError):
    pass


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def check_no_smoke(outdir: Path) -> None:
    lowered = str(outdir).lower()
    assert_true("smoke" not in lowered, f"output path looks like smoke artifacts: {outdir}")
    for path in outdir.rglob("*"):
        if path.is_file() and "smoke" in path.name.lower():
            raise ValidationError(f"smoke artifact found: {path}")


def check_metadata(outdir: Path) -> dict:
    meta_path = outdir / "campaign_metadata.json"
    assert_true(meta_path.exists(), "missing campaign_metadata.json")
    meta = read_json(meta_path)
    assert_true(int(meta.get("n_boot", 0)) >= FULL_N_BOOT, f"n_boot below full scale: {meta.get('n_boot')}")
    assert_true(int(meta.get("n_perms", 0)) >= FULL_N_PERMS, f"n_perms below full scale: {meta.get('n_perms')}")
    assert_true(int(meta.get("n_sim", 0)) >= FULL_N_SIM, f"n_sim below full scale: {meta.get('n_sim')}")
    return meta


def check_finite_required_csv(path: Path, allow_nan_columns: set[str] | None = None) -> None:
    allow_nan_columns = allow_nan_columns or set()
    df = pd.read_csv(path)
    assert_true(len(df) > 0, f"empty required CSV: {path}")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return
    for col in numeric.columns:
        vals = numeric[col].to_numpy(dtype=float)
        if np.isinf(vals).any():
            raise ValidationError(f"infinite values in {path}:{col}")
        if col not in allow_nan_columns and np.isnan(vals).any():
            raise ValidationError(f"NaN values in required column {path}:{col}")


def check_candidate(candidate_dir: Path) -> None:
    failure = candidate_dir / "failure.json"
    if failure.exists():
        return
    required = [
        "metadata.json",
        "methods.csv",
        "hac_delta.csv",
        "threshold_menu.csv",
        "romano_wolf.csv",
        "permutation.csv",
        "costs.csv",
        "holdout_subperiods.csv",
        "row_boundary.csv",
        "audit_gate.json",
    ]
    for name in required:
        assert_true((candidate_dir / name).exists(), f"missing {name} in {candidate_dir}")
    for name in ["methods.csv", "hac_delta.csv", "threshold_menu.csv", "romano_wolf.csv", "holdout_subperiods.csv"]:
        check_finite_required_csv(candidate_dir / name, allow_nan_columns={"block_size"})
    check_finite_required_csv(candidate_dir / "costs.csv", allow_nan_columns={"break_even_cost_bps"})

    hac = pd.read_csv(candidate_dir / "hac_delta.csv")
    for col in ["n_eff_sr", "se_iid_delta", "se_hac_delta", "se_inflation", "hac_bandwidth"]:
        assert_true(col in hac.columns, f"missing Sharpe-specific HAC diagnostic {col} in {candidate_dir}")

    rw = pd.read_csv(candidate_dir / "romano_wolf.csv")
    assert_true({"p_raw", "p_adjusted"}.issubset(rw.columns), f"Romano-Wolf raw/adjusted columns missing in {candidate_dir}")
    bad = rw["p_adjusted"].to_numpy(dtype=float) + 1e-12 < rw["p_raw"].to_numpy(dtype=float)
    assert_true(not bool(bad.any()), f"Romano-Wolf adjusted p-value below raw p-value in {candidate_dir}")

    perm = pd.read_csv(candidate_dir / "permutation.csv")
    if "status" in perm.columns and (perm["status"] == "ok").any():
        ok = perm[perm["status"] == "ok"]
        assert_true((ok["n_perms"] >= FULL_N_PERMS).all(), f"permutation count below full scale in {candidate_dir}")

    if candidate_dir.name == "french_momentum_deciles_daily_wml":
        benchmark = candidate_dir / "momentum_benchmark.csv"
        assert_true(benchmark.exists(), f"missing momentum_benchmark.csv in {candidate_dir}")
        check_finite_required_csv(benchmark)
        bench = pd.read_csv(benchmark)
        assert_true((bench["sharpe_abs_diff"] <= 1e-10).all(), f"WML Sharpe benchmark mismatch in {candidate_dir}")
        assert_true((bench["panel_to_direct_median_scale"] - 0.5).abs().le(1e-10).all(), f"WML panel/direct scale mismatch in {candidate_dir}")


def check_empirical(outdir: Path) -> None:
    attempts_path = outdir / "campaign_attempts.csv"
    assert_true(attempts_path.exists(), "missing campaign_attempts.csv")
    attempts = pd.read_csv(attempts_path)
    assert_true(len(attempts) > 0, "campaign_attempts.csv is empty")
    emp_dir = outdir / "empirical"
    assert_true(emp_dir.exists(), "missing empirical output directory")
    for candidate_dir in sorted(p for p in emp_dir.iterdir() if p.is_dir()):
        check_candidate(candidate_dir)


def check_simulation(outdir: Path) -> None:
    sim_dir = outdir / "simulation"
    assert_true(sim_dir.exists(), "missing simulation output directory")
    coverage = sim_dir / "coverage_all_merged.csv"
    size = sim_dir / "size_test_audit.csv"
    power = sim_dir / "power_audit.csv"
    for path in [coverage, size, power]:
        assert_true(path.exists(), f"missing simulation artifact: {path}")
        check_finite_required_csv(path)
    cov = pd.read_csv(coverage)
    assert_true((cov["n_sim"] >= FULL_N_SIM).all(), "coverage n_sim below full scale")
    assert_true((cov["n_boot"] >= FULL_N_BOOT).all(), "coverage n_boot below full scale")
    size_df = pd.read_csv(size)
    power_df = pd.read_csv(power)
    assert_true((size_df["n"] >= FULL_N_SIM).all(), "size-test n below full scale")
    assert_true((power_df["n"] >= FULL_N_SIM).all(), "power-test n below full scale")


def check_provenance(outdir: Path) -> None:
    path = outdir / "artifact_provenance.csv"
    assert_true(path.exists(), "missing artifact_provenance.csv")
    prov = pd.read_csv(path)
    assert_true(len(prov) > 0, "artifact_provenance.csv is empty")
    for col in ["path", "bytes", "sha256"]:
        assert_true(col in prov.columns, f"missing provenance column: {col}")
    assert_true((prov["bytes"] > 0).all(), "zero-byte file in provenance")
    assert_true(prov["sha256"].astype(str).str.len().eq(64).all(), "invalid sha256 in provenance")


def validate(outdir: Path) -> list[str]:
    check_no_smoke(outdir)
    check_metadata(outdir)
    check_empirical(outdir)
    check_simulation(outdir)
    check_provenance(outdir)
    return ["full campaign validation passed"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    args = parser.parse_args()
    try:
        messages = validate(Path(args.output_root))
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
    for message in messages:
        print(message)


if __name__ == "__main__":
    main()
