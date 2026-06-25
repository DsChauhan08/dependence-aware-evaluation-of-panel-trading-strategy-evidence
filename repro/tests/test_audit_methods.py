import os
import sys

import numpy as np
import pandas as pd
import pytest

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, CODE_DIR)

from sddm_bootstrap import (
    PanelData,
    bootstrap_positive_p_value,
    cross_sectional_correlation,
    fixed_b_hac_sensitivity,
    hac_sharpe_delta_prewhite,
    hac_sharpe_delta,
    percentile_rank_confidence,
    safe_sign,
    sddm_inference,
    _sharpe,
    sharpe_effective_sample_size,
    skipped_compounded_signal,
)
from synthetic_positive_control import SyntheticPositiveControlConfig, generate_positive_control
from threshold_analysis import (
    benjamini_hochberg,
    benjamini_yekutieli,
    holm_bonferroni,
    romano_wolf_menu_test,
    romano_wolf_stepdown,
)
from public_data import cost_sensitivity, hac_bandwidth_sensitivity, same_date_permutation_test
from render_manuscript_artifacts import reject_smoke_path
from campaign_sources import LoadedCandidate, _ranked_long_short_panel, load_registry
from run_full_campaign import french_wml_benchmark, row_boundary_diagnostics
from make_release_bundle import FORBIDDEN_PARTS, guarded_copy
from walk_forward import compute_turnover, target_weights, turnover_decomposition


def test_skipped_compounded_signal_bounds():
    r = np.array(
        [
            [0.10, 0.01],
            [0.20, 0.02],
            [0.30, 0.03],
            [0.40, 0.04],
            [0.50, 0.05],
        ]
    )
    sig = skipped_compounded_signal(r, lookback=2, skip=1)
    # At t=2, use j=0..1.
    assert np.isclose(sig[2, 0], (1.10 * 1.20) - 1.0)
    # At t=4, use j=2..3, not the current return.
    assert np.isclose(sig[4, 1], (1.03 * 1.04) - 1.0)
    assert np.isnan(sig[0, 0])


def test_safe_sign_zero_convention():
    x = np.array([-2.0, 0.0, 3.0])
    assert safe_sign(0.0) == 0.0
    assert np.array_equal(safe_sign(x), np.array([-1.0, 0.0, 1.0]))


def test_percentile_rank_confidence_midrank_and_missing():
    scores = np.array([[1.0, 2.0, 2.0, np.nan], [np.nan, 5.0, np.nan, np.nan]])
    conf = percentile_rank_confidence(scores)
    assert np.isclose(conf[0, 0], 0.0)
    assert np.isclose(conf[0, 1], 0.75)
    assert np.isclose(conf[0, 2], 0.75)
    assert np.isclose(conf[1, 1], 1.0)


def test_confidence_two_name_edge_and_ties():
    scores = np.array([[1.0, 2.0], [3.0, 3.0]])
    conf = percentile_rank_confidence(scores)
    assert np.allclose(conf[0], np.array([0.0, 1.0]))
    assert np.allclose(conf[1], np.array([0.5, 0.5]))


def test_date_returns_weighting_variants():
    panel = PanelData(
        dates=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        tickers=np.array(["A", "B"]),
        predictions=np.array([[1.0, 2.0], [1.0, -1.0]]),
        realised=np.array([[0.01, 0.03], [0.02, -0.01]]),
        confidence=np.array([[0.5, 1.0], [0.5, 0.9]]),
    )
    assert np.allclose(panel.date_returns(0.0), np.array([0.02, 0.005]))
    conf_weighted = panel.date_returns(0.0, weighting="confidence")
    assert np.isclose(conf_weighted[0], (0.01 * 0.5 + 0.03 * 1.0) / 1.5)
    signal_weighted = panel.date_returns(0.0, weighting="signal_abs")
    assert np.isclose(signal_weighted[0], (0.01 * 1.0 + 0.03 * 2.0) / 3.0)


def test_date_returns_dollar_neutral_requires_both_sides():
    panel = PanelData(
        dates=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        tickers=np.array(["A", "B", "C"]),
        predictions=np.array([[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]),
        realised=np.array([[0.02, 0.04, 0.06], [0.01, 0.03, 0.05]]),
        confidence=np.ones((2, 3)),
    )
    neutral = panel.date_returns(0.0, exposure="dollar_neutral")
    assert np.isclose(neutral[0], 0.5 * ((0.02 + 0.06) / 2.0) + 0.5 * 0.04)
    assert np.isnan(neutral[1])

    weights = target_weights(panel, threshold=0.0, start=0, end=2, exposure="dollar_neutral")
    assert np.isclose(weights[0, weights[0] > 0].sum(), 0.5)
    assert np.isclose(weights[0, weights[0] < 0].sum(), -0.5)


def test_selected_counts_by_threshold():
    panel = PanelData(
        dates=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        tickers=np.array(["A", "B", "C"]),
        predictions=np.ones((2, 3)),
        realised=np.ones((2, 3)) * 0.01,
        confidence=np.array([[0.2, 0.5, 0.9], [0.1, np.nan, 0.95]]),
    )
    assert np.array_equal(panel.selected_counts(0.5), np.array([2, 1]))
    assert np.array_equal(panel.selected_counts(0.9), np.array([1, 1]))


def test_french_wml_registry_candidate_uses_extreme_deciles_only():
    registry = load_registry()
    spec = next(c for c in registry["candidates"] if c["id"] == "french_momentum_deciles_daily_wml")
    assert spec["primary_threshold"] == 0.9
    assert spec["thresholds"] == [0.9]
    assert "groups" not in spec

    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    low = np.array([-0.020, -0.005, 0.004, -0.003, 0.006, -0.004, 0.003, 0.002])
    spread = np.array([0.010, 0.018, -0.006, 0.012, 0.020, -0.010, 0.014, 0.016])
    returns = np.column_stack([low + spread * j / 9.0 for j in range(10)])
    df = pd.DataFrame(returns, index=dates, columns=[f"MOM{i:02d}" for i in range(1, 11)])
    panel = _ranked_long_short_panel(df)

    selected = (
        (panel.confidence >= 0.9)
        & np.isfinite(panel.confidence)
        & np.isfinite(panel.realised)
        & np.isfinite(panel.predictions)
    )
    assert np.array_equal(selected.sum(axis=1), np.array([2, 2, 2, 2, 2, 2, 2, 0]))
    chosen_predictions = panel.predictions[selected].reshape(7, 2)
    assert np.all((chosen_predictions < 0).sum(axis=1) == 1)
    assert np.all((chosen_predictions > 0).sum(axis=1) == 1)

    loaded = LoadedCandidate(spec=spec, panel=panel, returns=df, factors=None, groups=None, notes=[])
    bench = french_wml_benchmark(loaded, threshold=0.9).iloc[0]
    assert bench["n"] == 7
    assert bench["sharpe_abs_diff"] < 1e-12
    assert np.isclose(bench["panel_to_direct_median_scale"], 0.5)


def test_rank_profile_threshold_07_is_not_canonical_wml():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        np.tile(np.linspace(-0.01, 0.01, 10), (5, 1)),
        index=dates,
        columns=[f"MOM{i:02d}" for i in range(1, 11)],
    )
    panel = _ranked_long_short_panel(df)
    counts = panel.selected_counts(0.7)
    assert np.array_equal(counts, np.array([4, 4, 4, 4, 0]))


def test_bootstrap_positive_p_value_smoothing():
    assert np.isclose(bootstrap_positive_p_value(np.array([1.0, 2.0, 3.0])), 1.0 / 4.0)
    assert np.isclose(bootstrap_positive_p_value(np.array([-1.0, 0.0, 1.0])), 3.0 / 4.0)


def test_hac_delta_matches_iid_lo_order():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.02, 800)
    hac = hac_sharpe_delta(r, max_lag=0)
    sr_daily = np.mean(r) / np.std(r, ddof=1)
    lo_se = np.sqrt(252.0 * (1.0 + 0.5 * sr_daily * sr_daily) / len(r))
    assert hac.se > 0
    assert abs(hac.se - lo_se) / lo_se < 0.20


def test_sharpe_annualization_factor_scales_monthly_vs_daily():
    r = np.array([0.01, -0.002, 0.006, 0.004, -0.001, 0.008])
    daily = _sharpe(r, annualise=252.0)
    monthly = _sharpe(r, annualise=12.0)
    assert np.isclose(monthly, daily * np.sqrt(12.0 / 252.0))


def test_sddm_inference_threads_annualization_factor():
    panel = PanelData(
        dates=np.arange(12).astype("datetime64[D]"),
        tickers=np.array(["A"]),
        predictions=np.ones((12, 1)),
        realised=np.array([[0.01], [-0.002], [0.006], [0.004], [-0.001], [0.008], [0.003], [0.005], [-0.004], [0.007], [0.002], [0.006]]),
        confidence=np.ones((12, 1)),
    )
    daily = sddm_inference(panel, method="iid", n_boot=49, seed=4, annualise=252.0)
    monthly = sddm_inference(panel, method="iid", n_boot=49, seed=4, annualise=12.0)
    assert np.isclose(monthly.sharpe_point, daily.sharpe_point * np.sqrt(12.0 / 252.0))
    assert monthly.sharpe_se < daily.sharpe_se


def test_registry_declares_monthly_aqr_annualization_and_daily_defaults():
    registry = load_registry()
    by_id = {c["id"]: c for c in registry["candidates"]}
    assert by_id["aqr_value_momentum_everywhere_monthly"]["periodicity"] == "monthly"
    assert by_id["aqr_value_momentum_everywhere_monthly"]["annualization_factor"] == 12
    assert by_id["french_momentum_deciles_daily_wml"]["periodicity"] == "daily"
    assert by_id["french_momentum_deciles_daily_wml"]["annualization_factor"] == 252


def test_hac_bandwidth_sensitivity_reports_fixed_and_auto_lags():
    rng = np.random.default_rng(11)
    r = rng.normal(0.001, 0.02, 200)
    table = hac_bandwidth_sensitivity(r)
    assert {"auto", "0", "5", "126"}.issubset(set(table["lag_label"]))
    assert np.all(table["positive_p_value"].dropna().between(0.0, 1.0))


def test_prewhitened_hac_returns_finite_inference():
    rng = np.random.default_rng(21)
    e = rng.normal(0.0, 0.01, 250)
    r = np.empty_like(e)
    r[0] = e[0]
    for t in range(1, len(r)):
        r[t] = 0.35 * r[t - 1] + e[t] + 0.0002
    res = hac_sharpe_delta_prewhite(r, max_lag=5)
    assert np.isfinite(res.sharpe)
    assert res.se > 0
    assert 0.0 <= res.positive_p_value <= 1.0


def test_fixed_b_hac_sensitivity_shape_and_values():
    rng = np.random.default_rng(22)
    r = rng.normal(0.0001, 0.01, 180)
    table = fixed_b_hac_sensitivity(r, b_grid=(0.05, 0.10), n_sim=25, sim_length=120, seed=1)
    assert list(table["b"]) == [0.05, 0.10]
    assert {"fixedb_positive_p", "fixedb_crit_95", "simulation_length"}.issubset(table.columns)
    assert np.all(table["fixedb_positive_p"].between(0.0, 1.0))


def test_sharpe_effective_sample_size_uses_hac_inflation():
    rng = np.random.default_rng(4)
    e = rng.normal(0.0, 0.01, 1000)
    r = np.empty_like(e)
    r[0] = e[0]
    for t in range(1, len(r)):
        r[t] = 0.65 * r[t - 1] + e[t]
    diag = sharpe_effective_sample_size(r)
    assert diag["se_hac_delta"] > diag["se_iid_delta"]
    assert diag["se_inflation"] > 1.0
    assert diag["n_eff_sr"] < len(r)


def test_row_boundary_diagnostics_estimates_trading_moulton_factor():
    rng = np.random.default_rng(31)
    t, n, rho = 1500, 6, 0.30
    common = rng.normal(0.0, np.sqrt(rho), size=(t, 1))
    idio = rng.normal(0.0, np.sqrt(1.0 - rho), size=(t, n))
    realised = 0.001 + 0.01 * (common + idio)
    panel = PanelData(
        dates=np.arange(t).astype("datetime64[D]"),
        tickers=np.array([f"A{i}" for i in range(n)]),
        predictions=np.ones((t, n)),
        realised=realised,
        confidence=np.ones((t, n)),
    )
    diag = row_boundary_diagnostics(panel, threshold=0.0, candidate_id="synthetic", romano_wolf_p=0.04)
    row = diag.iloc[0]
    expected = np.sqrt(1.0 + (n - 1.0) * rho)
    assert row["status"] == "ok"
    assert abs(row["rho_same_date"] - rho) < 0.05
    assert abs(row["trading_moulton_factor"] - expected) < 0.15


def test_row_boundary_diagnostics_marks_single_series_not_applicable():
    panel = PanelData(
        dates=np.arange(20).astype("datetime64[D]"),
        tickers=np.array(["A"]),
        predictions=np.ones((20, 1)),
        realised=np.ones((20, 1)) * 0.01,
        confidence=np.ones((20, 1)),
    )
    diag = row_boundary_diagnostics(panel, threshold=0.0, candidate_id="single", romano_wolf_p=0.05)
    assert diag.iloc[0]["status"] == "not_applicable_single_series"


def test_row_boundary_diagnostics_reports_finite_p_values():
    rng = np.random.default_rng(32)
    panel = PanelData(
        dates=np.arange(80).astype("datetime64[D]"),
        tickers=np.array(["A", "B", "C"]),
        predictions=np.ones((80, 3)),
        realised=rng.normal(0.001, 0.01, size=(80, 3)),
        confidence=np.ones((80, 3)),
    )
    row = row_boundary_diagnostics(panel, threshold=0.0, candidate_id="panel", romano_wolf_p=0.07).iloc[0]
    for col in ["row_p_positive", "hac_p_positive", "romano_wolf_p"]:
        assert np.isfinite(row[col])
        assert 0.0 <= row[col] <= 1.0


def test_multiple_testing_adjustments_shape_and_order():
    p = [0.01, 0.02, 0.20]
    holm = holm_bonferroni(p)
    bh = benjamini_hochberg(p)
    by = benjamini_yekutieli(p)
    assert len(holm) == len(bh) == len(by) == 3
    assert holm[0] <= holm[2]
    assert by[0] >= bh[0]


def test_romano_wolf_returns_adjusted_p_values():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.01, (80, 3))
    r[:, 0] += 0.003
    adj = romano_wolf_stepdown(r, n_boot=99, block_size=5, seed=3)
    assert adj.shape == (3,)
    assert np.all((adj >= 0.0) & (adj <= 1.0))


def test_romano_wolf_menu_reports_matching_raw_and_adjusted():
    rng = np.random.default_rng(12)
    r = rng.normal(0.0, 0.01, (120, 4))
    r[:, 0] += 0.002
    rw = romano_wolf_menu_test(r, n_boot=99, block_size=6, seed=7)
    assert {"p_raw", "p_adjusted", "t_stat"}.issubset(rw.columns)
    assert np.all(rw["p_adjusted"].to_numpy() + 1e-12 >= rw["p_raw"].to_numpy())


def test_turnover_scaled_costs_reduce_sharpe_monotonically():
    panel = PanelData(
        dates=np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        tickers=np.array(["A", "B"]),
        predictions=np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]]),
        realised=np.array([[0.01, 0.02], [0.01, 0.02], [0.01, 0.02]]),
        confidence=np.ones((3, 2)),
    )
    costs = cost_sensitivity(panel, threshold=0.0)
    assert costs.loc[costs["cost_bps_per_rebalance"].eq(0), "annual_cost_drag"].iloc[0] == 0
    assert costs["annual_cost_drag"].is_monotonic_increasing


def test_turnover_uses_half_l1_target_weights():
    panel = PanelData(
        dates=np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        tickers=np.array(["A", "B"]),
        predictions=np.array([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]),
        realised=np.ones((3, 2)) * 0.01,
        confidence=np.ones((3, 2)),
    )
    # t0 and t1 have identical weights, turnover 0.
    # t2 flips from [0.5, -0.5] to [-0.5, 0.5], turnover 1.
    assert np.isclose(compute_turnover(panel, threshold=0.0, start=0, end=3), 0.5)


def test_turnover_decomposition_separates_rescaling():
    panel = PanelData(
        dates=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        tickers=np.array(["A", "B", "C"]),
        predictions=np.array([[1.0, -1.0, 0.0], [1.0, -1.0, 1.0]]),
        realised=np.ones((2, 3)) * 0.01,
        confidence=np.array([[1.0, 1.0, np.nan], [1.0, 1.0, 1.0]]),
    )
    parts = turnover_decomposition(panel, threshold=0.0, start=0, end=2)
    assert parts["daily_turnover"] > 0
    assert parts["turnover_rescale"] > 0
    assert parts["turnover_entry_exit_flip"] > 0
    assert np.isclose(
        parts["daily_turnover"],
        parts["turnover_rescale"] + parts["turnover_entry_exit_flip"],
    )


def test_synthetic_positive_control_contains_edge_and_dependence():
    cfg = SyntheticPositiveControlConfig(
        T=800,
        N=30,
        annual_alpha=0.30,
        idiosyncratic_sigma=0.01,
        common_factor_loading=0.40,
        common_factor_ar1=0.10,
        signal_stay_probability=0.97,
        seed=19,
    )
    panel = generate_positive_control(cfg)
    assert _sharpe(panel.date_returns(0.0)) > 1.0
    assert cross_sectional_correlation(panel) > 0.05


def test_same_date_permutation_supports_structural_groups():
    rng = np.random.default_rng(13)
    import pandas as pd

    data = pd.DataFrame(
        rng.normal(0.0, 0.01, (40, 4)),
        index=pd.date_range("2020-01-01", periods=40, freq="B"),
        columns=["A", "B", "C", "D"],
    )
    groups = np.array([0, 0, 1, 1])
    summary, null = same_date_permutation_test(
        data,
        thresholds=[0.0, 0.5],
        lookback=3,
        skip=1,
        n_perms=5,
        seed=5,
        groups=groups,
        design="within_pair",
    )
    assert set(summary["design"]) == {"within_pair"}
    assert len(null) == 10


def test_renderer_refuses_smoke_paths():
    with pytest.raises(ValueError):
        reject_smoke_path(os.path.join(CODE_DIR, "output_smoke_v2"))


def test_release_bundle_redaction_guard(tmp_path):
    assert "artifacts" in FORBIDDEN_PARTS
    forbidden = os.path.abspath(os.path.join(CODE_DIR, "..", "venv", "private.txt"))
    with pytest.raises(ValueError):
        guarded_copy(__import__("pathlib").Path(forbidden), tmp_path / "private.txt")
