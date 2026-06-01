"""
Walk-Forward Validation — Rigorous Protocol
=============================================
Proper expanding-window walk-forward with:
  1. Strictly non-overlapping train/test splits
  2. Transaction cost sensitivity analysis
  3. Capacity estimation
  4. Turnover computation
"""

import numpy as np
import pandas as pd
from sddm_bootstrap import Exposure, PanelData, sddm_inference, newey_west_se, safe_sign
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    """Simple transaction cost model."""
    spread_bps: float = 10.0      # half-spread in basis points
    commission_bps: float = 2.0   # commission per side in bps
    impact_bps: float = 5.0       # market impact per side in bps

    @property
    def round_trip_bps(self) -> float:
        return 2 * (self.spread_bps + self.commission_bps + self.impact_bps)

    def cost_per_trade(self) -> float:
        """Round-trip cost as a decimal fraction."""
        return self.round_trip_bps / 10_000


def target_weights(
    panel: PanelData,
    threshold: float,
    start: int,
    end: int,
    exposure: Exposure = "as_selected",
) -> np.ndarray:
    """Target portfolio weights for the equal-weighted audit portfolio."""
    conf = panel.confidence[start:end]
    realised = panel.realised[start:end]
    predictions = panel.predictions[start:end]
    mask = (
        (conf >= threshold)
        & np.isfinite(conf)
        & np.isfinite(realised)
        & np.isfinite(predictions)
    )
    positions = np.zeros_like(predictions, dtype=float)
    signs = safe_sign(predictions)

    if exposure == "as_selected":
        counts = mask.sum(axis=1)
        active_dates = counts > 0
        positions[active_dates] = (
            signs[active_dates] * mask[active_dates].astype(float) / counts[active_dates, None]
        )
        return positions

    if exposure != "dollar_neutral":
        raise ValueError(f"unknown exposure mode: {exposure}")

    long_mask = mask & (predictions > 0)
    short_mask = mask & (predictions < 0)
    long_counts = long_mask.sum(axis=1)
    short_counts = short_mask.sum(axis=1)
    active_dates = (long_counts > 0) & (short_counts > 0)
    positions[active_dates] = (
        0.5 * long_mask[active_dates].astype(float) / long_counts[active_dates, None]
        - 0.5 * short_mask[active_dates].astype(float) / short_counts[active_dates, None]
    )
    return positions


def turnover_decomposition(
    panel: PanelData,
    threshold: float,
    start: int,
    end: int,
    exposure: Exposure = "as_selected",
) -> dict:
    """Split half-L1 turnover into total, rescaling, and entry/exit/flip parts."""
    positions = target_weights(panel, threshold, start, end, exposure=exposure)
    if positions.shape[0] < 2:
        return {
            "daily_turnover": 0.0,
            "turnover_rescale": 0.0,
            "turnover_entry_exit_flip": 0.0,
            "n_turnover_pairs": 0,
        }

    active = np.abs(positions).sum(axis=1) > 0
    has_active_pair = active[1:] | active[:-1]
    if not np.any(has_active_pair):
        return {
            "daily_turnover": 0.0,
            "turnover_rescale": 0.0,
            "turnover_entry_exit_flip": 0.0,
            "n_turnover_pairs": 0,
        }

    old = positions[:-1]
    new = positions[1:]
    daily_turnover = 0.5 * np.sum(np.abs(new - old), axis=1)
    same_side = (old * new) > 0
    rescale = 0.5 * np.sum(np.where(same_side, np.abs(new - old), 0.0), axis=1)
    entry_exit_flip = np.maximum(daily_turnover - rescale, 0.0)
    return {
        "daily_turnover": float(np.mean(daily_turnover[has_active_pair])),
        "turnover_rescale": float(np.mean(rescale[has_active_pair])),
        "turnover_entry_exit_flip": float(np.mean(entry_exit_flip[has_active_pair])),
        "n_turnover_pairs": int(np.sum(has_active_pair)),
    }


def compute_turnover(
    panel: PanelData,
    threshold: float,
    start: int,
    end: int,
    exposure: Exposure = "as_selected",
) -> float:
    """
    Estimate daily turnover from target portfolio weights.

    Daily turnover is 0.5 * sum_i |w[t, i] - w[t-1, i]|, averaged over
    adjacent dates with at least one active side.
    """
    return turnover_decomposition(panel, threshold, start, end, exposure=exposure)["daily_turnover"]


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward_validation(
    panel: PanelData,
    threshold: float,
    n_folds: int = 5,
    min_train_frac: float = 0.4,
    method: str = "blocked",
    n_boot: int = 5_000,
    cost_scenarios: list[CostModel] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward validation.

    Each fold:
      - Training: dates[0 : train_end]
      - Test: dates[train_end : test_end]
      - No overlap between train and test
      - Train window expands with each fold

    Reports gross and net-of-cost returns for each fold.
    """
    if cost_scenarios is None:
        cost_scenarios = [
            CostModel(spread_bps=5, commission_bps=1, impact_bps=2),   # optimistic
            CostModel(spread_bps=10, commission_bps=2, impact_bps=5),  # moderate
            CostModel(spread_bps=20, commission_bps=5, impact_bps=10), # pessimistic
        ]

    T = panel.T
    min_train = int(T * min_train_frac)
    remaining = T - min_train
    test_size = remaining // n_folds

    results = []

    for fold in range(n_folds):
        train_end = min_train + fold * test_size
        test_start = train_end
        test_end = min(test_start + test_size, T)

        if test_end <= test_start + 10:
            break

        # Date ranges for reporting
        train_dates = (panel.dates[0], panel.dates[train_end - 1])
        test_dates = (panel.dates[test_start], panel.dates[test_end - 1])

        # Gross returns
        returns = panel.date_returns(threshold)
        oos_returns = returns[test_start:test_end]
        valid_oos = oos_returns[~np.isnan(oos_returns)]
        is_returns = returns[:train_end]
        valid_is = is_returns[~np.isnan(is_returns)]

        if len(valid_oos) < 10:
            continue

        # Compute metrics
        oos_mean = float(np.mean(valid_oos))
        oos_std = float(np.std(valid_oos, ddof=1))
        oos_sharpe = oos_mean / oos_std * np.sqrt(252) if oos_std > 0 else 0
        is_sharpe = float(np.mean(valid_is) / np.std(valid_is, ddof=1) * np.sqrt(252)) if len(valid_is) > 1 and np.std(valid_is, ddof=1) > 0 else 0

        oos_ann_ret = oos_mean * 252
        oos_ann_vol = oos_std * np.sqrt(252)

        # Newey-West SE
        nw_se = newey_west_se(valid_oos)

        # Turnover
        turnover = compute_turnover(panel, threshold, test_start, test_end)

        # Net returns under each cost scenario
        net_sharpes = []
        net_returns = []
        for cm in cost_scenarios:
            daily_cost = turnover * cm.cost_per_trade()
            net_daily = valid_oos - daily_cost
            net_mean = float(np.mean(net_daily))
            net_std = float(np.std(net_daily, ddof=1))
            net_sr = net_mean / net_std * np.sqrt(252) if net_std > 0 else 0
            net_sharpes.append(net_sr)
            net_returns.append(net_mean * 252)

        # Bootstrap inference on OOS
        try:
            # Create a mini-panel for the test period
            test_panel = PanelData(
                dates=panel.dates[test_start:test_end],
                tickers=panel.tickers,
                predictions=panel.predictions[test_start:test_end],
                realised=panel.realised[test_start:test_end],
                confidence=panel.confidence[test_start:test_end],
            )
            boot_res = sddm_inference(
                test_panel, threshold=threshold, method=method,
                n_boot=n_boot, seed=seed + fold,
            )
            oos_ci_lo = boot_res.sharpe_ci_lo
            oos_ci_hi = boot_res.sharpe_ci_hi
            oos_p = boot_res.p_value
            oos_n_eff = boot_res.n_effective
        except ValueError:
            oos_ci_lo = oos_ci_hi = oos_p = oos_n_eff = np.nan

        row = {
            "fold": fold + 1,
            "train_start": str(train_dates[0]),
            "train_end": str(train_dates[1]),
            "test_start": str(test_dates[0]),
            "test_end": str(test_dates[1]),
            "train_days": train_end,
            "test_days": test_end - test_start,
            "is_sharpe": is_sharpe,
            "oos_sharpe_gross": oos_sharpe,
            "oos_ann_ret_gross": oos_ann_ret,
            "oos_ann_vol": oos_ann_vol,
            "oos_ci_lo": oos_ci_lo,
            "oos_ci_hi": oos_ci_hi,
            "oos_p_value": oos_p,
            "oos_n_effective": oos_n_eff,
            "nw_se_daily": nw_se,
            "daily_turnover": turnover,
        }

        for i, cm in enumerate(cost_scenarios):
            label = f"{cm.round_trip_bps:.0f}bps"
            row[f"oos_sharpe_net_{label}"] = net_sharpes[i]
            row[f"oos_ann_ret_net_{label}"] = net_returns[i]

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Capacity analysis
# ---------------------------------------------------------------------------

def capacity_analysis(
    panel: PanelData,
    threshold: float,
    aum_levels_mm: list[float] = None,
) -> pd.DataFrame:
    """
    Strategy capacity estimation with sensitivity analysis.

    Tests across:
      - Multiple AUM levels
      - Three liquidity tiers (large-cap, mid-cap, small-cap)
      - Square-root impact model with realistic parameters

    This is an ORDER-OF-MAGNITUDE estimate, not a deployment-ready
    market-impact study. The paper should say so explicitly.
    """
    if aum_levels_mm is None:
        aum_levels_mm = [1, 5, 10, 25, 50, 100, 250, 500]

    mask = panel.confidence >= threshold
    avg_positions = float(mask.sum(axis=1).mean())

    # Turnover estimate
    positions = safe_sign(panel.predictions) * mask.astype(float)
    if positions.shape[0] >= 2:
        changes = np.abs(positions[1:] - positions[:-1])
        active = np.maximum(mask[1:].sum(axis=1), 1)
        daily_turnover = float(np.mean(changes.sum(axis=1) / active))
    else:
        daily_turnover = 1.0

    # Liquidity tiers: (ADV in USD, daily vol, label)
    liquidity_tiers = [
        ("Large-cap ($200M ADV)", 200_000_000, 0.018),
        ("Mid-cap ($50M ADV)", 50_000_000, 0.025),
        ("Small-cap ($10M ADV)", 10_000_000, 0.035),
    ]

    rows = []
    for aum in aum_levels_mm:
        aum_usd = aum * 1_000_000
        for tier_name, adv, daily_vol in liquidity_tiers:
            per_position = aum_usd / max(avg_positions, 1)
            participation_rate = per_position / adv
            # Square-root impact (Almgren et al. 2005)
            impact_bps = daily_vol * np.sqrt(participation_rate) * 10_000
            # Total round-trip cost including spread + impact
            spread_bps = 5.0 if "Large" in tier_name else (10.0 if "Mid" in tier_name else 20.0)
            total_cost_bps = 2 * (spread_bps + impact_bps)
            # Annual cost drag = turnover * cost * 252
            annual_drag_pct = daily_turnover * total_cost_bps / 10_000 * 252 * 100
            rows.append({
                "AUM_MM": aum,
                "Liquidity_Tier": tier_name,
                "Positions": avg_positions,
                "Participation_Rate": participation_rate,
                "Impact_BPS": impact_bps,
                "Total_RT_Cost_BPS": total_cost_bps,
                "Daily_Turnover": daily_turnover,
                "Annual_Cost_Drag_Pct": annual_drag_pct,
                "Viable": impact_bps < 50 and participation_rate < 0.05,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from simulation_study import generate_panel, DGPConfig

    cfg = DGPConfig(
        name="Walk-Forward Demo",
        T=2500, N=100, true_mu=0.0004,
        ar1_serial=0.15, rho_cross=0.35, conf_quality=0.65,
    )
    panel = generate_panel(cfg, seed=42)

    print("=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)

    wf_df = walk_forward_validation(
        panel, threshold=0.50, n_folds=5, method="blocked",
    )

    # Display key columns
    display_cols = [
        "fold", "train_days", "test_days",
        "is_sharpe", "oos_sharpe_gross", "oos_ann_ret_gross",
        "oos_p_value", "daily_turnover",
    ]
    # Add net sharpe columns
    for c in wf_df.columns:
        if "sharpe_net" in c:
            display_cols.append(c)

    print(wf_df[display_cols].to_string(index=False, float_format="%.4f"))
    wf_df.to_csv("walk_forward_results.csv", index=False)

    print("\n" + "=" * 70)
    print("CAPACITY ANALYSIS")
    print("=" * 70)
    cap_df = capacity_analysis(panel, threshold=0.50)
    print(cap_df.to_string(index=False, float_format="%.4f"))
    cap_df.to_csv("capacity_analysis.csv", index=False)


if __name__ == "__main__":
    main()
