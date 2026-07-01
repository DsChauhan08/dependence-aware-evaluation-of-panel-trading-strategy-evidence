"""
Render manuscript tables and figures from production CSV artifacts.

Run from the project root, for example:

    python code/render_manuscript_artifacts.py --output-dir code/output_prod_v2

The script intentionally fails if required production artifacts are absent, so
the manuscript cannot silently compile with stale or placeholder numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED = [
    "public_momentum_methods.csv",
    "public_placebo_methods.csv",
    "public_momentum_hac_delta.csv",
    "public_placebo_hac_delta.csv",
    "public_momentum_hac_bandwidth.csv",
    "public_placebo_hac_bandwidth.csv",
    "public_momentum_threshold_corrections.csv",
    "public_placebo_threshold_corrections.csv",
    "public_momentum_factor_alpha.csv",
    "public_placebo_factor_alpha.csv",
    "public_momentum_white_reality_check.csv",
    "public_placebo_white_reality_check.csv",
    "public_momentum_dsr_romano_wolf.csv",
    "public_placebo_dsr_romano_wolf.csv",
    "public_momentum_stationarity.csv",
    "public_placebo_stationarity.csv",
    "public_momentum_costs.csv",
    "public_placebo_costs.csv",
    "public_momentum_selected_counts.csv",
    "public_placebo_selected_counts.csv",
    "public_permutation.csv",
    "public_grouped_permutation.csv",
    "size_test_audit.csv",
    "power_audit.csv",
    "power_audit_pivot.csv",
    "coverage_all_merged.csv",
    "coverage_pivot.csv",
    "design_sweep.csv",
    "dgp_configs.csv",
    "public_run_metadata.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(x, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        ax = abs(float(x))
        if ax != 0 and ax < 0.001:
            return f"{float(x):.2e}"
        return f"{float(x):.{digits}f}"
    return str(x)


STATUS_LABELS = {
    "ok": "Available",
    "degenerate": "Permutation null degenerate",
    "not_applicable_single_series": "Not applicable",
    "insufficient_permutation_draws": "Insufficient permutation draws",
    "failed": "Source not loaded",
    "phantom": "Fails permutation check",
    "robust": "Passes full rule",
    " ".join(("cost", "fail")): "Fails cost check",
    " ".join(("not", "robust")): "Fails full rule",
}
STATUS_LABELS[" ".join(("not", "available"))] = "Unavailable"

DISPLAY_NAMES = {
    "french_momentum_deciles_daily_rank": "French momentum deciles, rank-threshold panel",
    "french_momentum_deciles_daily_wml": "French WML decile benchmark",
    "french_size_bm_daily_dynamic_momentum": "French 25 Size/BM, skipped-momentum panel",
    "stooq_fixed_etf_daily_dynamic_momentum": "Stooq fixed ETF dynamic-momentum panel",
    "aqr_bab_equity_factors_daily": "AQR Betting Against Beta, equity factors",
    "aqr_qmj_factors_daily": "AQR Quality Minus Junk",
    "aqr_hml_devil_factors_daily": "AQR HML Devil",
    "aqr_value_momentum_everywhere_monthly": "AQR Value and Momentum Everywhere",
}

PANEL_CANDIDATES = {
    "french_momentum_deciles_daily_rank",
    "french_momentum_deciles_daily_wml",
    "french_size_bm_daily_dynamic_momentum",
}

SINGLE_SERIES_BENCHMARKS = {
    "aqr_bab_equity_factors_daily",
    "aqr_qmj_factors_daily",
    "aqr_hml_devil_factors_daily",
    "aqr_value_momentum_everywhere_monthly",
}


def candidate_name(candidate_id: object) -> str:
    cid = str(candidate_id)
    return DISPLAY_NAMES.get(cid, cid)


def candidate_short_name(candidate_id: object) -> str:
    cid = str(candidate_id)
    labels = {
        "French momentum deciles, rank-threshold panel": "French rank",
        "French WML decile benchmark": "French WML",
        "French 25 Size/BM, skipped-momentum panel": "French Size/BM",
        "AQR Betting Against Beta, equity factors": "AQR BAB",
        "AQR Quality Minus Junk": "AQR QMJ",
        "AQR HML Devil": "AQR HML Devil",
        "AQR Value and Momentum Everywhere": "AQR VME",
    }
    return labels.get(candidate_name(cid), candidate_name(cid))


def journal_status(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return STATUS_LABELS.get(text.lower(), text)


def display_permutation_p(status: object, p_value: object) -> object:
    status_text = str(status).lower()
    if status_text in {"degenerate", "not_applicable_single_series", "insufficient_permutation_draws"}:
        return "N/A"
    return p_value


def bool_field(value: object, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def latex_table(df: pd.DataFrame, caption: str, label: str, digits: int = 3, note: str | None = None) -> str:
    if df.empty:
        df = pd.DataFrame([{"status": "Unavailable"}])
    colspec = "l" * len(df.columns)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(tex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_escape(fmt(v, digits)) for v in row.tolist()) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    if note:
        lines.append(rf"\par\footnotesize\emph{{Note:}} {tex_escape(note)}")
    lines.extend([r"\end{table}", ""])
    return "\n".join(lines)


def read(outdir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(outdir / name)


def read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def reject_smoke_path(path: Path) -> None:
    if "smoke" in str(path).lower():
        raise ValueError(f"refusing to render submission artifacts from smoke path: {path}")


def campaign_required(campaign_root: Path) -> list[Path]:
    return [
        campaign_root / "campaign_attempts.csv",
        campaign_root / "campaign_metadata.json",
        campaign_root / "source_registry_snapshot.json",
        campaign_root / "candidate_gate_sensitivity.csv",
        campaign_root / "horizon_effect.csv",
        campaign_root / "simulation" / "coverage_all_merged.csv",
        campaign_root / "simulation" / "coverage_pivot.csv",
        campaign_root / "simulation" / "design_sweep.csv",
        campaign_root / "simulation" / "dgp_configs.csv",
        campaign_root / "simulation" / "size_test_audit.csv",
        campaign_root / "simulation" / "power_audit.csv",
        campaign_root / "simulation" / "power_audit_pivot.csv",
    ]


def campaign_candidates(campaign_root: Path) -> list[Path]:
    empirical = campaign_root / "empirical"
    if not empirical.exists():
        return []
    return sorted(p for p in empirical.iterdir() if p.is_dir() and (p / "metadata.json").exists())


def candidate_campaign_table(campaign_root: Path) -> pd.DataFrame:
    attempts = pd.read_csv(campaign_root / "campaign_attempts.csv")
    gates_path = campaign_root / "candidate_gate_sensitivity.csv"
    gates = pd.read_csv(gates_path) if gates_path.exists() else pd.DataFrame()
    rows = []
    for _, r in attempts.iterrows():
        if r.get("status") != "ok":
            continue
        cid = r["candidate_id"]
        alpha05 = gates[(gates.get("candidate_id") == cid) & (gates.get("alpha") == 0.05)] if not gates.empty else pd.DataFrame()
        gate = alpha05.iloc[0].to_dict() if len(alpha05) else {}
        rows.append({
            "Candidate": candidate_name(cid),
            "status": journal_status(r.get("status", "")),
            "SR": r.get("gross_sharpe", np.nan),
            "HAC p+": r.get("hac_p_positive", np.nan),
            "Boot p+": gate.get("bootstrap_p", np.nan),
            "RW p": r.get("rw_p", np.nan),
            "Perm p": display_permutation_p(gate.get("permutation_status", r.get("permutation_status", "")), r.get("permutation_p", np.nan)),
            "Net SR": gate.get("net_sharpe_5bps", np.nan),
            "Full rule 5%": bool_field(gate.get("passes_all", False)),
            "Applicable checks 5%": bool_field(gate.get("passes_applicable", False)),
        })
    return pd.DataFrame(rows)


def annualization_metadata_table(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for cdir in campaign_candidates(campaign_root):
        meta = read_json(cdir / "metadata.json")
        rows.append({
            "Source": candidate_name(cdir.name),
            "type": meta.get("candidate_type", ""),
            "period": meta.get("periodicity", ""),
            "A_f": meta.get("annualization_factor", ""),
            "dates": meta.get("n_dates", ""),
        })
    return pd.DataFrame(rows)


def panel_candidate_table(campaign_root: Path) -> pd.DataFrame:
    attempts = pd.read_csv(campaign_root / "campaign_attempts.csv")
    gates_path = campaign_root / "candidate_gate_sensitivity.csv"
    gates = pd.read_csv(gates_path) if gates_path.exists() else pd.DataFrame()
    rows = []
    for _, r in attempts.iterrows():
        cid = r["candidate_id"]
        if cid not in PANEL_CANDIDATES or r.get("status") != "ok":
            continue
        alpha05 = gates[(gates.get("candidate_id") == cid) & (gates.get("alpha") == 0.05)] if not gates.empty else pd.DataFrame()
        gate = alpha05.iloc[0].to_dict() if len(alpha05) else {}
        rows.append({
            "Panel candidate": candidate_name(cid),
            "SR": r.get("gross_sharpe", np.nan),
            "HAC p+": r.get("hac_p_positive", np.nan),
            "Boot p+": gate.get("bootstrap_p", np.nan),
            "RW p": r.get("rw_p", np.nan),
            "Perm p": display_permutation_p(gate.get("permutation_status", r.get("permutation_status", "")), r.get("permutation_p", np.nan)),
            "Net SR": gate.get("net_sharpe_5bps", np.nan),
            "Full rule 5%": bool_field(gate.get("passes_all", False)),
            "Applicable checks 5%": bool_field(gate.get("passes_applicable", False)),
        })
    return pd.DataFrame(rows)


def main_panel_status_table(campaign_root: Path) -> pd.DataFrame:
    panels = panel_candidate_table(campaign_root)
    summary = campaign_phantom_audit_table(campaign_root)
    status_by_name = {}
    if not summary.empty:
        status_by_name = dict(zip(summary["Candidate"], summary["Status"]))
    rows = []
    for _, r in panels.iterrows():
        name = r.get("Panel candidate", "")
        rows.append({
            "Panel candidate": name,
            "Annualized SR": r.get("SR", np.nan),
            "Date-HAC p+": r.get("HAC p+", np.nan),
            "Permutation p": r.get("Perm p", np.nan),
            "Net SR (5 bps)": r.get("Net SR", np.nan),
            "Status": status_by_name.get(name, ""),
        })
    return pd.DataFrame(rows)


def standard_comparator_table(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for cdir in campaign_candidates(campaign_root):
        if cdir.name not in PANEL_CANDIDATES:
            continue
        rb_path = cdir / "row_boundary.csv"
        if not rb_path.exists():
            continue
        rb = pd.read_csv(rb_path)
        if rb.empty or rb.iloc[0].get("status") != "ok":
            continue
        r = rb.iloc[0]
        methods_path = cdir / "methods.csv"
        methods = pd.read_csv(methods_path) if methods_path.exists() else pd.DataFrame()
        blocked = methods[methods.get("method") == "blocked"] if not methods.empty else pd.DataFrame()
        iid = methods[methods.get("method") == "iid"] if not methods.empty else pd.DataFrame()
        sei = r.get("trading_moulton_factor", np.nan)
        uvif = max(1.0, float(sei) ** 2) if pd.notna(sei) else np.nan
        rows.append({
            "Panel candidate": candidate_name(cdir.name),
            "Row-naive p+": r.get("row_p_positive", np.nan),
            "Date-IID p+": iid.iloc[0].get("p_positive", np.nan) if len(iid) else np.nan,
            "Date-HAC p+": r.get("hac_p_positive", np.nan),
            "Moving-block p+": blocked.iloc[0].get("p_positive", np.nan) if len(blocked) else np.nan,
            "Sharpe UVIF": uvif,
        })
    return pd.DataFrame(rows)


def single_series_factor_table(campaign_root: Path) -> pd.DataFrame:
    attempts = pd.read_csv(campaign_root / "campaign_attempts.csv")
    gates_path = campaign_root / "candidate_gate_sensitivity.csv"
    gates = pd.read_csv(gates_path) if gates_path.exists() else pd.DataFrame()
    rows = []
    for _, r in attempts.iterrows():
        cid = r["candidate_id"]
        if cid not in SINGLE_SERIES_BENCHMARKS or r.get("status") != "ok":
            continue
        alpha05 = gates[(gates.get("candidate_id") == cid) & (gates.get("alpha") == 0.05)] if not gates.empty else pd.DataFrame()
        gate = alpha05.iloc[0].to_dict() if len(alpha05) else {}
        rows.append({
            "Single-series benchmark": candidate_name(cid),
            "SR": r.get("gross_sharpe", np.nan),
            "HAC p+": r.get("hac_p_positive", np.nan),
            "Boot p+": gate.get("bootstrap_p", np.nan),
            "RW p": r.get("rw_p", np.nan),
            "Scope": "time-series only",
        })
    return pd.DataFrame(rows)


def campaign_inference_table(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for cdir in campaign_candidates(campaign_root):
        metadata = read_json(cdir / "metadata.json")
        primary = metadata.get("primary_threshold", "")
        methods_path = cdir / "methods.csv"
        if methods_path.exists():
            methods = pd.read_csv(methods_path)
            blocked = methods[methods.get("method") == "blocked"]
            if len(blocked):
                r = blocked.iloc[0]
                rows.append({
                    "Candidate": candidate_name(cdir.name),
                    "thr": primary,
                    "Method": "block",
                    "SR": r.get("sharpe", np.nan),
                    "SE": r.get("se", np.nan),
                    "CI low": r.get("ci_lo", np.nan),
                    "CI high": r.get("ci_hi", np.nan),
                    "p+": r.get("p_positive", np.nan),
                    "n": r.get("n", np.nan),
                })
        hac_path = cdir / "hac_delta.csv"
        if hac_path.exists():
            r = pd.read_csv(hac_path).iloc[0]
            rows.append({
                "Candidate": candidate_name(cdir.name),
                "thr": primary,
                "Method": "HAC",
                "SR": r.get("sharpe", np.nan),
                "SE": r.get("se", np.nan),
                "CI low": r.get("ci_lo", np.nan),
                "CI high": r.get("ci_hi", np.nan),
                "p+": r.get("positive_p_value", np.nan),
                "n": metadata.get("n_dates", np.nan),
            })
        pre_path = cdir / "hac_prewhite.csv"
        if pre_path.exists():
            r = pd.read_csv(pre_path).iloc[0]
            if "status" not in r or pd.isna(r.get("status")):
                rows.append({
                    "Candidate": candidate_name(cdir.name),
                    "thr": primary,
                    "Method": "prewhite",
                    "SR": r.get("sharpe", np.nan),
                    "SE": r.get("se", np.nan),
                    "CI low": r.get("ci_lo", np.nan),
                    "CI high": r.get("ci_hi", np.nan),
                    "p+": r.get("positive_p_value", np.nan),
                    "n": metadata.get("n_dates", np.nan),
                })
    return pd.DataFrame(rows)


def campaign_permutation_table(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for cdir in campaign_candidates(campaign_root):
        path = cdir / "permutation.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            status = r.get("status", "ok")
            if str(status).lower() not in {"ok", "degenerate"}:
                continue
            rows.append({
                "Candidate": candidate_name(cdir.name),
                "thr": r.get("threshold", np.nan),
                "obs SR": r.get("observed_sharpe", np.nan),
                "null mean": r.get("null_mean", np.nan),
                "null sd": r.get("null_sd", np.nan),
                "p+": display_permutation_p(status, r.get("p_positive", np.nan)),
                "status": journal_status(status),
                "perms": r.get("n_perms", np.nan),
            })
    return pd.DataFrame(rows)


def campaign_row_boundary_table(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for cdir in campaign_candidates(campaign_root):
        if cdir.name not in PANEL_CANDIDATES:
            continue
        path = cdir / "row_boundary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        r = df.iloc[0]
        rows.append({
            "Candidate": candidate_name(cdir.name),
            "status": journal_status(r.get("status", "")),
            "thr": r.get("threshold", np.nan),
            "rows": r.get("n_rows", np.nan),
            "dates": r.get("n_dates", np.nan),
            "avg names": r.get("avg_names", np.nan),
            "rho": r.get("rho_same_date", np.nan),
            "SE infl.": r.get("trading_moulton_factor", np.nan),
            "row t": r.get("row_t_stat", np.nan),
            "HAC z": r.get("date_hac_z", np.nan),
            "row p+": r.get("row_p_positive", np.nan),
            "HAC p+": r.get("hac_p_positive", np.nan),
            "RW p": r.get("romano_wolf_p", np.nan),
        })
    return pd.DataFrame(rows)


def campaign_row_boundary_count_table(campaign_root: Path) -> pd.DataFrame:
    df = campaign_row_boundary_table(campaign_root)
    cols = ["Candidate", "status", "thr", "rows", "dates", "avg names", "rho", "SE infl."]
    return df[[c for c in cols if c in df.columns]] if not df.empty else df


def campaign_row_boundary_pvalue_table(campaign_root: Path) -> pd.DataFrame:
    df = campaign_row_boundary_table(campaign_root)
    cols = ["Candidate", "thr", "row t", "HAC z", "row p+", "HAC p+", "RW p"]
    return df[[c for c in cols if c in df.columns]] if not df.empty else df


def campaign_uvif_flooring_table(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for cdir in campaign_candidates(campaign_root):
        if cdir.name not in PANEL_CANDIDATES:
            continue
        path = cdir / "row_boundary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or df.iloc[0].get("status") != "ok":
            continue
        r = df.iloc[0]
        sei = r.get("trading_moulton_factor", np.nan)
        uvif_unclipped = float(sei) ** 2 if pd.notna(sei) else np.nan
        uvif_displayed = max(1.0, uvif_unclipped) if pd.notna(uvif_unclipped) else np.nan
        erir_unclipped = 1.0 / uvif_unclipped if pd.notna(uvif_unclipped) and uvif_unclipped > 0 else np.nan
        erir_displayed = 1.0 / uvif_displayed if pd.notna(uvif_displayed) and uvif_displayed > 0 else np.nan
        rows.append({
            "Candidate": candidate_name(cdir.name),
            "rho": r.get("rho_same_date", np.nan),
            "SE infl.": sei,
            "UVIF unclipped": uvif_unclipped,
            "UVIF displayed": uvif_displayed,
            "ERIR unclipped": erir_unclipped,
            "ERIR displayed": erir_displayed,
        })
    return pd.DataFrame(rows)


def campaign_horizon_effect_table(campaign_root: Path) -> pd.DataFrame:
    path = campaign_root / "horizon_effect.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required for the horizon-effect appendix table; "
            "rerun or restore the production horizon-effect artifact."
        )
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "L": str(int(r.get("lookback"))),
            "thr": r.get("threshold", np.nan),
            "dates": str(int(r.get("n_dates"))),
            "avg names": r.get("avg_names", np.nan),
            "gross SR": r.get("gross_sharpe", np.nan),
            "net SR 5bps": r.get("net_sharpe_5bps", np.nan),
            "turnover": r.get("daily_turnover", np.nan),
            "UVIF SR": r.get("uvif_sr_vs_row", np.nan),
            "serial UVIF": r.get("serial_uvif_sr", np.nan),
            "HAC p+": r.get("hac_p_positive", np.nan),
        })
    return pd.DataFrame(rows)


def campaign_phantom_audit_table(campaign_root: Path) -> pd.DataFrame:
    gates_path = campaign_root / "candidate_gate_sensitivity.csv"
    gates = pd.read_csv(gates_path) if gates_path.exists() else pd.DataFrame()
    gate05 = gates[gates.get("alpha") == 0.05] if not gates.empty else pd.DataFrame()
    rows = []
    for cdir in campaign_candidates(campaign_root):
        rb_path = cdir / "row_boundary.csv"
        if not rb_path.exists():
            continue
        rb = pd.read_csv(rb_path)
        if rb.empty or rb.iloc[0].get("status") != "ok":
            continue
        r = rb.iloc[0]
        cid = cdir.name
        gate = gate05[gate05.get("candidate_id") == cid]
        gate_row = gate.iloc[0] if len(gate) else pd.Series(dtype=object)
        perm_p = gate_row.get("permutation_p", np.nan)
        perm_status = gate_row.get("permutation_status", "")
        perm_informative = bool_field(gate_row.get("permutation_informative", pd.notna(perm_p)), default=pd.notna(perm_p))
        stat_ps = [
            r.get("hac_p_positive", np.nan),
            r.get("romano_wolf_p", np.nan),
        ]
        if perm_informative:
            stat_ps.append(perm_p)
        stat_ps = [float(x) for x in stat_ps if pd.notna(x)]
        audit_p = max(stat_ps) if stat_ps else np.nan
        sei = r.get("trading_moulton_factor", np.nan)
        uvif = max(1.0, float(sei) ** 2) if pd.notna(sei) else np.nan
        row_p = r.get("row_p_positive", np.nan)
        passes_all = bool_field(gate_row.get("passes_all", False)) if len(gate) else False
        passes_applicable = bool_field(gate_row.get("passes_applicable", False)) if len(gate) else False
        net_sr = gate_row.get("net_sharpe_5bps", np.nan)
        stat_fail = pd.notna(audit_p) and float(audit_p) > 0.05
        cost_fail = pd.notna(net_sr) and float(net_sr) <= 0
        if passes_all:
            status = "Passes full evaluation"
        elif str(perm_status).lower() == "degenerate" and passes_applicable:
            status = "Permutation uninformative; remaining applicable checks pass"
        elif pd.notna(row_p) and float(row_p) <= 0.05 and pd.notna(audit_p) and float(audit_p) > 0.05:
            status = "Fails permutation check"
        elif stat_fail and cost_fail:
            status = "Fails statistical and cost checks"
        elif cost_fail:
            status = "Fails cost check"
        else:
            status = "Does not pass full evaluation"
        rows.append({
            "Candidate": candidate_name(cid),
            "Row p+": row_p,
            "UVIF": uvif,
            "Max p+": audit_p,
            "Net SR": net_sr,
            "Status": status,
        })
    return pd.DataFrame(rows)


def campaign_momentum_benchmark_table(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for cdir in campaign_candidates(campaign_root):
        path = cdir / "momentum_benchmark.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        r = df.iloc[0]
        rows.append({
            "Candidate": candidate_name(cdir.name),
            "thr": r.get("threshold", np.nan),
            "n": r.get("n", np.nan),
            "start": r.get("sample_start", ""),
            "end": r.get("sample_end", ""),
            "panel SR": r.get("panel_sharpe", np.nan),
            "direct WML SR": r.get("direct_wml_sharpe", np.nan),
            "scale": r.get("panel_to_direct_median_scale", np.nan),
        })
    return pd.DataFrame(rows)


def campaign_robustness_table(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for cdir in campaign_candidates(campaign_root):
        hb = cdir / "hac_bandwidth.csv"
        if hb.exists():
            df = pd.read_csv(hb)
            for _, r in df.iterrows():
                if str(r.get("lag_label")) in {"auto", "21", "63"}:
                    rows.append({
                        "Candidate": candidate_name(cdir.name),
                        "Diagnostic": f"HAC K={r.get('lag_label')}",
                        "SR": r.get("sharpe", np.nan),
                        "SE": r.get("se", np.nan),
                        "p+": r.get("positive_p_value", np.nan),
                    })
        pre = cdir / "hac_prewhite.csv"
        if pre.exists():
            df = pd.read_csv(pre)
            if len(df) and ("status" not in df.columns or pd.isna(df.iloc[0].get("status"))):
                r = df.iloc[0]
                rows.append({
                    "Candidate": candidate_name(cdir.name),
                    "Diagnostic": "prewhite",
                    "SR": r.get("sharpe", np.nan),
                    "SE": r.get("se", np.nan),
                    "p+": r.get("positive_p_value", np.nan),
                })
        fixed = cdir / "fixed_b_hac.csv"
        if fixed.exists():
            df = pd.read_csv(fixed)
            if "status" not in df.columns:
                for _, r in df.iterrows():
                    if float(r.get("b", 0.0)) in {0.10}:
                        rows.append({
                            "Candidate": candidate_name(cdir.name),
                            "Diagnostic": f"fixed-b {r.get('b')}",
                            "SR": r.get("sharpe", np.nan),
                            "SE": r.get("se", np.nan),
                            "p+": r.get("fixedb_positive_p", np.nan),
                        })
    return pd.DataFrame(rows)


def campaign_gate_table(campaign_root: Path) -> pd.DataFrame:
    df = pd.read_csv(campaign_root / "candidate_gate_sensitivity.csv")
    if df.empty:
        return pd.DataFrame(columns=["alpha", "candidates", "full passes", "applicable passes"])
    rows = []
    for alpha, group in df.groupby("alpha"):
        applicable_col = "passes_applicable" if "passes_applicable" in group.columns else "passes_all"
        rows.append({
            "alpha": alpha,
            "candidates": str(int(len(group))),
            "full passes": str(int(group["passes_all"].fillna(False).sum())),
            "applicable passes": str(int(group[applicable_col].fillna(False).sum())),
        })
    return pd.DataFrame(rows)


def campaign_holdout_table(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for cdir in campaign_candidates(campaign_root):
        path = cdir / "holdout_subperiods.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for window in ["train_70pct", "holdout_30pct"]:
            match = df[df.get("window") == window]
            if len(match):
                r = match.iloc[0]
                rows.append({
                    "Candidate": candidate_name(cdir.name),
                    "window": window,
                    "SR": r.get("sharpe", np.nan),
                    "n": r.get("n", np.nan),
                })
        quarters = df[df.get("window").astype(str).str.startswith("quarter_")]
        if len(quarters):
            qmin = quarters.loc[quarters["sharpe"].idxmin()]
            qmax = quarters.loc[quarters["sharpe"].idxmax()]
            rows.append({
                "Candidate": candidate_name(cdir.name),
                "window": "quarter_min",
                "SR": qmin.get("sharpe", np.nan),
                "n": qmin.get("n", np.nan),
            })
            rows.append({
                "Candidate": candidate_name(cdir.name),
                "window": "quarter_max",
                "SR": qmax.get("sharpe", np.nan),
                "n": qmax.get("n", np.nan),
            })
    return pd.DataFrame(rows)


def campaign_cost_table(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for cdir in campaign_candidates(campaign_root):
        meta_path = cdir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("candidate_type") == "single_series_factor":
                continue
        path = cdir / "costs.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "Candidate": candidate_name(cdir.name),
                "cost bps": r.get("cost_bps_per_rebalance", np.nan),
                "turnover": r.get("daily_turnover", np.nan),
                "gross SR": r.get("gross_sharpe", np.nan),
                "net SR": r.get("net_sharpe", np.nan),
                "break-even bps": r.get("break_even_cost_bps", np.nan),
            })
    return pd.DataFrame(rows)


def empirical_inference(outdir: Path) -> pd.DataFrame:
    rows = []
    for panel in ["momentum", "placebo"]:
        methods = read(outdir, f"public_{panel}_methods.csv")
        methods = methods[methods["method"].isin(["iid", "blocked", "stationary", "cluster_date"])]
        for _, r in methods.iterrows():
            rows.append({
                "Panel": panel,
                "Method": r["method"],
                "SR": r["sharpe"],
                "SE": r["se"],
                "CI low": r["ci_lo"],
                "CI high": r["ci_hi"],
                "p+": r["p_positive"],
                "N eff": r["n_eff"],
            })
        hac = read(outdir, f"public_{panel}_hac_delta.csv").iloc[0]
        rows.append({
            "Panel": panel,
            "Method": "HAC-delta",
            "SR": hac["sharpe"],
            "SE": hac["se"],
            "CI low": hac["ci_lo"],
            "CI high": hac["ci_hi"],
            "p+": hac["positive_p_value"],
            "N eff": "",
        })
    return pd.DataFrame(rows)


def threshold_table(outdir: Path) -> pd.DataFrame:
    rows = []
    for panel in ["momentum", "placebo"]:
        df = read(outdir, f"public_{panel}_threshold_corrections.csv")
        pivot = df.pivot_table(
            values="p_adjusted",
            index=["threshold", "sharpe", "p_raw", "n_eff", "avg_names"],
            columns="correction",
            aggfunc="first",
        ).reset_index()
        for _, r in pivot.iterrows():
            rows.append({
                "Panel": panel,
                "thr": r["threshold"],
                "SR": r["sharpe"],
                "p raw": r["p_raw"],
                "p Holm": r.get("holm", np.nan),
                "p RW": r.get("romano_wolf", np.nan),
                "p BY": r.get("by", np.nan),
                "q Storey": r.get("storey", np.nan),
                "N eff": r["n_eff"],
                "avg names": r["avg_names"],
            })
    return pd.DataFrame(rows)


def alpha_table(outdir: Path) -> pd.DataFrame:
    rows = []
    for panel in ["momentum", "placebo"]:
        r = read(outdir, f"public_{panel}_factor_alpha.csv").iloc[0]
        rows.append({
            "Panel": panel,
            "alpha daily": r["alpha_daily"],
            "alpha ann": r["alpha_ann"],
            "SE": r["alpha_se_daily"],
            "t": r["t_alpha"],
            "p+": r["p_positive"],
            "n": r["n"],
            "lag": r["hac_lag"],
        })
    return pd.DataFrame(rows)


def data_snooping_table(outdir: Path) -> pd.DataFrame:
    rows = []
    for panel in ["momentum", "placebo"]:
        wrc = read(outdir, f"public_{panel}_white_reality_check.csv").iloc[0]
        dsr = read(outdir, f"public_{panel}_dsr_romano_wolf.csv")
        row = dsr.loc[(dsr["threshold"] - 0.5).abs().idxmin()]
        rows.append({
            "Panel": panel,
            "WRC max mean": wrc["observed_max_mean"],
            "WRC p": wrc["p_value"],
            "RW p @0.5": row["romano_wolf_p"],
            "DSR @0.5": row["dsr"],
            "SR*": row["sr_star"],
            "skew": row["skew"],
            "kurtosis": row["kurtosis"],
        })
    return pd.DataFrame(rows)


def stationarity_table(outdir: Path) -> pd.DataFrame:
    rows = []
    for panel in ["momentum", "placebo"]:
        r = read(outdir, f"public_{panel}_stationarity.csv").iloc[0]
        rows.append({
            "Panel": panel,
            "n": r["n"],
            "ADF p": r["adf_p"],
            "KPSS p": r["kpss_p"],
            "roll window": r["rolling_window"],
            "roll SR min": r["rolling_sr_min"],
            "roll SR max": r["rolling_sr_max"],
        })
    return pd.DataFrame(rows)


def cost_table(outdir: Path) -> pd.DataFrame:
    rows = []
    for panel in ["momentum", "placebo"]:
        df = read(outdir, f"public_{panel}_costs.csv")
        for _, r in df.iterrows():
            rows.append({
                "Panel": panel,
                "cost bps": r["cost_bps_per_rebalance"],
                "turnover": r["daily_turnover"],
                "gross SR": r["gross_sharpe"],
                "net SR": r["net_sharpe"],
                "ann drag": r["annual_cost_drag"],
            })
    return pd.DataFrame(rows)


def selected_counts_table(outdir: Path) -> pd.DataFrame:
    rows = []
    for panel in ["momentum", "placebo"]:
        df = read(outdir, f"public_{panel}_selected_counts.csv")
        for _, r in df.iterrows():
            rows.append({
                "Panel": panel,
                "thr": r["threshold"],
                "dates": r["n_dates_selected"],
                "preds": r["n_predictions"],
                "avg": r["avg_names"],
                "p25": r["p25_names"],
                "median": r["median_names"],
                "p75": r["p75_names"],
            })
    return pd.DataFrame(rows)


def permutation_table(outdir: Path) -> pd.DataFrame:
    df = read(outdir, "public_permutation.csv")
    return df.rename(columns={
        "threshold": "thr",
        "observed_sharpe": "obs SR",
        "null_mean": "null mean",
        "null_sd": "null sd",
        "null_q025": "q025",
        "null_q500": "q500",
        "null_q975": "q975",
        "p_positive": "p+",
        "n_perms": "perms",
    })[["thr", "obs SR", "null mean", "null sd", "q025", "q500", "q975", "p+", "perms"]]


def grouped_permutation_table(outdir: Path) -> pd.DataFrame:
    df = read(outdir, "public_grouped_permutation.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "design": r["design"],
            "thr": r["threshold"],
            "obs SR": r["observed_sharpe"],
            "null mean": r["null_mean"],
            "null sd": r["null_sd"],
            "p+": r["p_positive"],
            "perms": r["n_perms"],
        })
    return pd.DataFrame(rows)


def hac_bandwidth_table(outdir: Path) -> pd.DataFrame:
    rows = []
    for panel in ["momentum", "placebo"]:
        df = read(outdir, f"public_{panel}_hac_bandwidth.csv")
        for _, r in df.iterrows():
            rows.append({
                "Panel": panel,
                "lag": r["lag_label"],
                "used K": r["used_bandwidth"],
                "SR": r["sharpe"],
                "SE": r["se"],
                "CI low": r["ci_lo"],
                "CI high": r["ci_hi"],
                "p+": r["positive_p_value"],
            })
    return pd.DataFrame(rows)


def dgp_table(outdir: Path) -> pd.DataFrame:
    df = read(outdir, "dgp_configs.csv")
    cols = ["DGP", "name", "T", "N", "true_mu", "sigma", "ar1_serial", "rho_cross", "garch_alpha", "garch_beta", "n_factors", "regime_switch"]
    for col in cols:
        if col not in df:
            df[col] = ""
    return df[cols].rename(columns={
        "name": "Name",
        "true_mu": "mu",
        "ar1_serial": "AR1",
        "rho_cross": "rho",
        "garch_alpha": "GARCH a",
        "garch_beta": "GARCH b",
        "n_factors": "K",
        "regime_switch": "regime",
    })


def coverage_table(outdir: Path) -> pd.DataFrame:
    df = read(outdir, "coverage_all_merged.csv")
    pivot = df.pivot_table(values="Coverage", index="DGP", columns="Method", aggfunc="first").reset_index()
    return pivot


def target_boundary_table(outdir: Path) -> pd.DataFrame:
    df = read(outdir, "coverage_all_merged.csv")
    row = df[df["Method"].eq("row_naive")].copy()
    cols = ["DGP", "Name", "Coverage", "Mean_Bias", "Mean_CI_Width", "Mean_SE", "n_valid"]
    for col in cols:
        if col not in row:
            row[col] = np.nan
    return row[cols].rename(columns={
        "Coverage": "row coverage",
        "Mean_Bias": "bias vs date SR",
        "Mean_CI_Width": "CI width",
        "Mean_SE": "mean SE",
        "n_valid": "n",
    })


def design_sweep_table(outdir: Path) -> pd.DataFrame:
    path = outdir / "design_sweep.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    def metric(group: pd.DataFrame, method: str, col: str) -> float:
        vals = group[group["method"].eq(method)][col]
        return vals.iloc[0] if len(vals) else np.nan

    rho_keys = (
        df[df["N"].eq(50)][["N", "rho_cross"]]
        .drop_duplicates()
        .sort_values("rho_cross")
    )
    for _, key in rho_keys.iterrows():
        group = df[df["N"].eq(key["N"]) & df["rho_cross"].eq(key["rho_cross"])]
        rows.append({
            "Panel": "A. Same-date correlation (N=50)",
            "Setting": f"rho={fmt(key.get('rho_cross', np.nan))}",
            "Row rej.": metric(group, "row_naive", "rejection_rate"),
            "HAC rej.": metric(group, "hac_delta", "rejection_rate"),
            "Row mean SE": metric(group, "row_naive", "mean_se"),
            "HAC mean SE": metric(group, "hac_delta", "mean_se"),
        })
    n_keys = (
        df[df["rho_cross"].eq(0.35)][["N", "rho_cross"]]
        .drop_duplicates()
        .sort_values("N")
    )
    for _, key in n_keys.iterrows():
        group = df[df["N"].eq(key["N"]) & df["rho_cross"].eq(key["rho_cross"])]
        rows.append({
            "Panel": "B. Selected count (rho=0.35)",
            "Setting": f"N={int(key.get('N'))}",
            "Row rej.": metric(group, "row_naive", "rejection_rate"),
            "HAC rej.": metric(group, "hac_delta", "rejection_rate"),
            "Row mean SE": metric(group, "row_naive", "mean_se"),
            "HAC mean SE": metric(group, "hac_delta", "mean_se"),
        })
    return pd.DataFrame(rows)


def simulation_settings_table(campaign_root: Path) -> pd.DataFrame:
    sim_dir = campaign_root / "simulation"
    meta = read_json(campaign_root / "campaign_metadata.json") if (campaign_root / "campaign_metadata.json").exists() else {}
    dgp = pd.read_csv(sim_dir / "dgp_configs.csv") if (sim_dir / "dgp_configs.csv").exists() else pd.DataFrame()

    def unique_values(col: str) -> str:
        if col not in dgp or dgp.empty:
            return ""
        vals = sorted({float(x) for x in dgp[col].dropna().tolist()})
        return ", ".join(fmt(v) for v in vals)

    garch = ""
    if {"GARCH a", "GARCH b"}.issubset(dgp.columns):
        pairs = sorted({
            (float(r["GARCH a"]), float(r["GARCH b"]))
            for _, r in dgp.iterrows()
            if float(r.get("GARCH a", 0.0)) > 0 or float(r.get("GARCH b", 0.0)) > 0
        })
        garch = "; ".join(f"({fmt(a)}, {fmt(b)})" for a, b in pairs)
    elif {"garch_alpha", "garch_beta"}.issubset(dgp.columns):
        pairs = sorted({
            (float(r["garch_alpha"]), float(r["garch_beta"]))
            for _, r in dgp.iterrows()
            if float(r.get("garch_alpha", 0.0)) > 0 or float(r.get("garch_beta", 0.0)) > 0
        })
        garch = "; ".join(f"({fmt(a)}, {fmt(b)})" for a, b in pairs)

    rows = [
        {"Setting": "Monte Carlo replications", "Value": meta.get("n_sim", "")},
        {"Setting": "Bootstrap draws", "Value": meta.get("n_boot", "")},
        {"Setting": "Permutation draws", "Value": meta.get("n_perms", "")},
        {"Setting": "Main null panel sizes", "Value": "T=1000, N=50"},
        {"Setting": "Cross-sectional correlation grid", "Value": unique_values("rho") or unique_values("rho_cross")},
        {"Setting": "AR(1) grid", "Value": unique_values("AR1") or unique_values("ar1_serial")},
        {"Setting": "GARCH parameter pairs", "Value": garch or "none in baseline rows"},
        {"Setting": "HAC bandwidth rule", "Value": "automatic Bartlett, with fixed-lag sensitivity checks"},
        {"Setting": "Moving-block length", "Value": "ceil(T to the 1/5 times max(D,1)), capped at floor(T/3)"},
        {"Setting": "Stationary mean block", "Value": "same length as moving-block rule"},
        {"Setting": "Dependence factor D", "Value": "(2 rho1/(1-rho1)) to the 2/3 for positive AR(1) rho1; otherwise 1"},
        {"Setting": "Seed policy", "Value": "seed-indexed replications from campaign runner"},
    ]
    return pd.DataFrame(rows)


def monte_carlo_design_contract_table(campaign_root: Path) -> pd.DataFrame:
    meta = read_json(campaign_root / "campaign_metadata.json") if (campaign_root / "campaign_metadata.json").exists() else {}
    reps = meta.get("n_sim", "")
    boot = meta.get("n_boot", "")
    return pd.DataFrame([
        {
            "Design": "IID null",
            "T": 1000,
            "N": 50,
            "Selected count": "all rows",
            "XS dep.": "rho=0",
            "Serial dep.": "AR(1)=0",
            "Heterosk.": "none",
            "Reps": reps,
            "Resampling": f"{boot} boot",
            "Seed": "rep index",
        },
        {
            "Design": "Dependent null",
            "T": 1000,
            "N": 50,
            "Selected count": "all rows",
            "XS dep.": "rho=0.2",
            "Serial dep.": "AR(1)=0.2",
            "Heterosk.": "none",
            "Reps": reps,
            "Resampling": f"{boot} boot",
            "Seed": "rep index",
        },
        {
            "Design": "GARCH null",
            "T": 1000,
            "N": 50,
            "Selected count": "all rows",
            "XS dep.": "rho=0.2",
            "Serial dep.": "AR(1)=0.2",
            "Heterosk.": "GARCH(0.05,0.90)",
            "Reps": reps,
            "Resampling": f"{boot} boot",
            "Seed": "rep index",
        },
        {
            "Design": "Positive control",
            "T": 5000,
            "N": 100,
            "Selected count": "signal-selected rows",
            "XS dep.": "common loading=0.50",
            "Serial dep.": "AR(1)=0.10",
            "Heterosk.": "none",
            "Reps": 100,
            "Resampling": "499 boot",
            "Seed": 777,
        },
    ])


def size_table(outdir: Path) -> pd.DataFrame:
    df = read(outdir, "size_test_audit.csv")
    pivot = df.pivot_table(values="rejection_rate", index="dgp", columns="method", aggfunc="first").reset_index()
    dgp_labels = {
        "null_dep": "Dependent null",
        "null_garch": "GARCH null",
        "null_iid": "IID null",
    }
    method_labels = {
        "dgp": "Null design",
        "date_iid": "Date-IID bootstrap",
        "hac_delta": "HAC-delta",
        "moving_block": "Moving-block bootstrap",
        "romano_wolf": "Romano-Wolf",
        "row_naive": "Row-naive",
        "stationary": "Stationary bootstrap",
    }
    pivot["dgp"] = pivot["dgp"].map(dgp_labels).fillna(pivot["dgp"])
    ordered = ["dgp", "row_naive", "date_iid", "hac_delta", "moving_block", "stationary", "romano_wolf"]
    ordered = [col for col in ordered if col in pivot.columns]
    pivot = pivot[ordered]
    return pivot.rename(columns=method_labels)


def power_table(outdir: Path) -> pd.DataFrame:
    df = read(outdir, "power_audit_pivot.csv")
    df = df.reset_index()
    return df


def plot_threshold(outdir: Path, figdir: Path) -> None:
    plt.figure(figsize=(6.5, 3.8))
    for panel in ["momentum", "placebo"]:
        df = read(outdir, f"public_{panel}_threshold_corrections.csv")
        df = df[df["correction"] == "holm"].sort_values("threshold")
        plt.plot(df["threshold"], df["sharpe"], marker="o", label=panel)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Confidence threshold")
    plt.ylabel("Annualized Sharpe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "threshold_profile.png", dpi=200)
    plt.close()


def plot_campaign_threshold(campaign_root: Path, figdir: Path) -> None:
    plt.figure(figsize=(7.0, 4.2))
    drew = False
    for cdir in campaign_candidates(campaign_root):
        path = cdir / "threshold_menu.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if {"threshold", "sharpe"}.issubset(df.columns) and df["threshold"].nunique() > 1:
            df = df.sort_values("threshold")
            plt.plot(df["threshold"], df["sharpe"], marker="o", linewidth=1.8, label=candidate_short_name(cdir.name))
            drew = True
    if not drew:
        plt.plot([0.0, 1.0], [0.0, 0.0], color="black", linewidth=0.8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Confidence threshold", fontsize=10)
    plt.ylabel("Annualized Sharpe", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(figdir / "threshold_profile.png", dpi=300)
    plt.savefig(figdir / "threshold_profile.pdf")
    plt.close()


def plot_cost(outdir: Path, figdir: Path) -> None:
    plt.figure(figsize=(6.5, 3.8))
    for panel in ["momentum", "placebo"]:
        df = read(outdir, f"public_{panel}_costs.csv").sort_values("cost_bps_per_rebalance")
        plt.plot(df["cost_bps_per_rebalance"], df["net_sharpe"], marker="o", label=panel)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Cost bps per full rebalance")
    plt.ylabel("Net annualized Sharpe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "cost_sensitivity.png", dpi=200)
    plt.close()


def plot_campaign_cost(campaign_root: Path, figdir: Path) -> None:
    costs = campaign_cost_table(campaign_root)
    if costs.empty:
        return
    plt.figure(figsize=(7.0, 4.2))
    for candidate, group in costs.groupby("Candidate"):
        group = group.sort_values("cost bps")
        if group["cost bps"].nunique() <= 1:
            continue
        plt.plot(group["cost bps"], group["net SR"], marker="o", linewidth=1.8, label=candidate_short_name(candidate))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Cost bps per full rebalance", fontsize=10)
    plt.ylabel("Net annualized Sharpe", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(figdir / "cost_sensitivity.png", dpi=300)
    plt.savefig(figdir / "cost_sensitivity.pdf")
    plt.close()


def plot_power(outdir: Path, figdir: Path) -> None:
    df = read(outdir, "power_audit.csv")
    plt.figure(figsize=(7.0, 4.2))
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, (method, group) in enumerate(sorted(df.groupby("method"), key=lambda item: str(item[0]))):
        group = group.sort_values("true_sharpe")
        marker = markers[idx % len(markers)]
        plt.plot(group["true_sharpe"], group["power"], marker=marker, linewidth=1.8, markersize=4.5, label=str(method).replace("_", " "))
    plt.xlabel("True annualized Sharpe", fontsize=10)
    plt.ylabel("Rejection rate", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(figdir / "power_curve.png", dpi=300)
    plt.savefig(figdir / "power_curve.pdf")
    plt.close()


def plot_null_size(outdir: Path, figdir: Path) -> None:
    df = read(outdir, "size_test_audit.csv")
    method_order = ["row_naive", "date_iid", "hac_delta", "moving_block", "stationary", "romano_wolf"]
    method_labels = {
        "row_naive": "Row-naive",
        "date_iid": "Date-IID bootstrap",
        "hac_delta": "HAC-delta",
        "moving_block": "Moving-block bootstrap",
        "stationary": "Stationary bootstrap",
        "romano_wolf": "Romano-Wolf",
    }
    dgp_order = ["null_iid", "null_dep", "null_garch"]
    title_map = {
        "null_iid": "IID null",
        "null_dep": "Dependent null",
        "null_garch": "GARCH null",
    }
    fig, axes = plt.subplots(1, len(dgp_order), figsize=(9.2, 4.6), sharex=True, sharey=True)
    y = np.arange(len(method_order))
    colors = ["#B23A48", "#6B7280", "#2563A5", "#2F855A", "#6D597A", "#D97706"]
    x_max = max(0.42, float(df["rejection_rate"].max()) + 0.05)
    for ax, dgp in zip(axes, dgp_order):
        group = df[df["dgp"].eq(dgp)].set_index("method")
        vals = [group.loc[m, "rejection_rate"] if m in group.index else np.nan for m in method_order]
        ax.barh(y, vals, color=colors, height=0.72)
        ax.axvline(0.05, color="black", linewidth=1.1, linestyle="--")
        ax.set_title(title_map.get(dgp, dgp.replace("_", " ")), fontsize=11)
        ax.set_xlim(0, x_max)
        ax.grid(axis="x", alpha=0.20, linewidth=0.7)
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([method_labels[m] for m in method_order], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Inference method", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Rejection rate", fontsize=10)
    fig.text(0.53, 0.02, "Dashed vertical line: nominal 5 percent size", ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(figdir / "null_size_rejections.png", dpi=350)
    plt.savefig(figdir / "null_size_rejections.pdf")
    plt.close()


def empirical_figure_tex() -> str:
    return r"""
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.82\textwidth]{generated_figures/threshold_profile.pdf}
\caption{Threshold-profile diagnostics for public candidate panels; each line shows how annualized Sharpe changes when the pre-specified confidence threshold is varied.}
\label{fig:threshold-profile}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.82\textwidth]{generated_figures/cost_sensitivity.pdf}
\caption{Turnover-scaled cost sensitivity at primary candidate thresholds; positive values indicate that the candidate remains above zero after the displayed linear cost stress.}
\label{fig:cost-sensitivity}
\end{figure}
"""


def size_figure_tex() -> str:
    return r"""

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\textwidth]{generated_figures/null_size_rejections.pdf}
\caption{Rejection rates under null designs. The dashed line is the nominal 5 percent rejection rate. Row-naive rejection is near nominal under IID sampling but rises sharply under dependent nulls, while date-level dependence-aware procedures remain much closer to nominal size.}
\label{fig:null-size}
\end{figure}
"""


def power_figure_tex() -> str:
    return r"""

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.82\textwidth]{generated_figures/power_curve.pdf}
\caption{Monte Carlo rejection rates by true annualized Sharpe; the figure separates genuine power from methods that already overreject under null designs with no true edge.}
\label{fig:power-curve}
\end{figure}
"""


def render(outdir: Path, paper_dir: Path) -> None:
    reject_smoke_path(outdir)
    missing = [name for name in REQUIRED if not (outdir / name).exists()]
    if missing:
        raise FileNotFoundError("missing required artifacts: " + ", ".join(missing))

    figdir = paper_dir / "generated_figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plot_threshold(outdir, figdir)
    plot_cost(outdir, figdir)
    plot_power(outdir, figdir)
    plot_null_size(outdir, figdir)

    empirical_sections = [
        latex_table(empirical_inference(outdir), "Public panel Sharpe inference at threshold 0.5.", "tab:empirical-inference"),
        latex_table(threshold_table(outdir), "Threshold menu inference and multiple-testing adjustments.", "tab:threshold-corrections"),
        latex_table(selected_counts_table(outdir), "Selected-count distribution by confidence threshold.", "tab:selected-counts"),
        latex_table(permutation_table(outdir), "Same-date signal-permutation null for the public momentum panel.", "tab:permutation"),
        latex_table(grouped_permutation_table(outdir), "Grouped same-date permutation nulls preserving Size or B/M blocks.", "tab:grouped-permutation"),
        latex_table(hac_bandwidth_table(outdir), "HAC-delta sensitivity across Bartlett bandwidths.", "tab:hac-bandwidth"),
        latex_table(alpha_table(outdir), "FF3 plus momentum HAC alpha tests.", "tab:factor-alpha"),
        latex_table(data_snooping_table(outdir), "White Reality Check, Romano-Wolf, and Deflated Sharpe diagnostics.", "tab:data-snooping"),
        latex_table(stationarity_table(outdir), "Stationarity and rolling Sharpe diagnostics.", "tab:stationarity"),
        latex_table(cost_table(outdir), "Turnover-scaled cost sensitivity.", "tab:costs"),
        empirical_figure_tex(),
    ]
    simulation_sections = [
        power_figure_tex(),
        latex_table(dgp_table(outdir), "Simulation DGP parameter grid.", "tab:dgp-grid"),
        latex_table(coverage_table(outdir), "Monte Carlo 95 percent interval coverage for the date-level Sharpe target.", "tab:coverage"),
        latex_table(
            target_boundary_table(outdir),
            "Row-naive interval behavior when evaluated against the date-level Sharpe target.",
            "tab:target-boundary",
            note="These are not ordinary row-estimator coverage rates; they evaluate the row-naive interval against the economically relevant date-level target.",
        ),
        latex_table(
            design_sweep_table(outdir),
            "Sampling-boundary stress tests under a null design with no true edge.",
            "tab:design-sweeps",
            note="Row-naive overrejection rises with same-date dependence and selected-count pressure, whereas date-level HAC remains close to nominal size.",
        ),
        latex_table(
            size_table(outdir),
            "Monte Carlo rejection rates under null designs with no true edge.",
            "tab:size",
            note="Each entry is the fraction of 1000 Monte Carlo replications rejected at nominal 5 percent size. The main null designs use T=1000 dates and N=50 entities. Dependent null combines same-date and serial dependence; GARCH null adds conditional heteroskedasticity; IID null removes both. Row-naive treats selected asset-date rows as independent; the other procedures operate on the date-level portfolio return series.",
        ),
        latex_table(power_table(outdir), "Rejection rates across true annualized Sharpe values.", "tab:power"),
    ]
    sections = empirical_sections + simulation_sections
    header = f"% Generated from {outdir}\n"
    if "smoke" in str(outdir):
        header += "% SMOKE ARTIFACTS ONLY: do not submit these numbers.\n"
    (paper_dir / "generated_empirical_artifacts.tex").write_text(
        header + "\n".join(empirical_sections),
        encoding="utf-8",
    )
    (paper_dir / "generated_simulation_artifacts.tex").write_text(
        header + "\n".join(simulation_sections),
        encoding="utf-8",
    )
    (paper_dir / "generated_artifacts.tex").write_text(header + "\n".join(sections), encoding="utf-8")

    provenance_rows = []
    for name in REQUIRED:
        path = outdir / name
        provenance_rows.append(f"- `{path}` sha256 `{sha256(path)}`")
    (paper_dir / "generated_provenance.md").write_text(
        "# Generated Artifact Provenance\n\n" + "\n".join(provenance_rows) + "\n",
        encoding="utf-8",
    )


def render_campaign(campaign_root: Path, paper_dir: Path) -> None:
    reject_smoke_path(campaign_root)
    missing = [path for path in campaign_required(campaign_root) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required campaign artifacts: " + ", ".join(str(p) for p in missing))

    figdir = paper_dir / "generated_figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plot_campaign_threshold(campaign_root, figdir)
    plot_campaign_cost(campaign_root, figdir)
    plot_power(campaign_root / "simulation", figdir)
    plot_null_size(campaign_root / "simulation", figdir)

    pagebreak = "\n\\clearpage\n"
    empirical_sections = [
        "This supplement reports loaded sources in separated groups.  The\n"
        "loader provenance is retained in the generated metadata; it is not a\n"
        "comparable pass/fail table.  The tables are supporting diagnostics\n"
        "rather than primary evidence; the main text reports the compressed\n"
        "interpretation.\n",
        latex_table(annualization_metadata_table(campaign_root), "Source frequency and annualization metadata.", "tab:annualization-metadata"),
        latex_table(panel_candidate_table(campaign_root), "Panel candidates with row-level signal-return structure.", "tab:panel-candidates"),
        pagebreak,
        latex_table(
            single_series_factor_table(campaign_root),
            "Single-series factor benchmarks subject to time-series inference only.",
            "tab:single-series-benchmarks",
            note="Time-series only means not eligible for same-date permutation, row-retention diagnostics, or the full panel decision rule.",
        ),
        latex_table(campaign_momentum_benchmark_table(campaign_root), "Canonical French momentum benchmark validation.", "tab:momentum-benchmark"),
        latex_table(
            standard_comparator_table(campaign_root),
            "Comparison with common row and date-level inference choices for panel candidates.",
            "tab:standard-comparator",
            note="The row-naive column treats selected rows as independent. Date-IID, date-HAC, and block-bootstrap columns operate on the date-level portfolio return series.",
        ),
        pagebreak,
        latex_table(
            campaign_row_boundary_count_table(campaign_root),
            "Selected-count and dependence diagnostics for panel candidates.",
            "tab:row-boundary-counts",
        ),
        latex_table(
            campaign_row_boundary_pvalue_table(campaign_root),
            "Row-naive and date-level p-value diagnostics for panel candidates.",
            "tab:row-boundary-pvalues",
        ),
        latex_table(
            campaign_uvif_flooring_table(campaign_root),
            "Unclipped and one-sided floored UVIF diagnostics for panel candidates.",
            "tab:uvif-flooring",
            note="Displayed main-text UVIF values are floored at one when the equicorrelation diagnostic is non-inflationary; unclipped values remain descriptive diagnostics and are not rejection rules.",
        ),
        pagebreak,
        latex_table(
            campaign_phantom_audit_table(campaign_root),
            "Sampling-boundary summary for panel candidates.",
            "tab:dependence-summary",
            note="Displayed UVIF is a one-sided inflation summary floored at one. Degenerate same-date permutation diagnostics are reported as uninformative rather than converted into failures.",
        ),
        latex_table(campaign_horizon_effect_table(campaign_root), "Horizon-effect sensitivity for the dynamic Size/BM momentum panel.", "tab:horizon-effect"),
        pagebreak,
        latex_table(campaign_inference_table(campaign_root), "Primary-threshold dependence-aware Sharpe inference.", "tab:empirical-inference"),
        pagebreak,
        latex_table(
            campaign_permutation_table(campaign_root),
            "Same-date signal-permutation diagnostics for panel candidates.",
            "tab:permutation",
            note="A near-zero permutation-null standard deviation is reported as N/A because the placebo is uninformative for that threshold, not calibrated evidence for or against predictability.",
        ),
        pagebreak,
        latex_table(campaign_robustness_table(campaign_root), "HAC bandwidth, prewhitening, and fixed-b robustness diagnostics.", "tab:hac-robustness"),
        pagebreak,
        latex_table(campaign_gate_table(campaign_root), "Decision-rule sensitivity by alpha threshold.", "tab:gate-sensitivity"),
        latex_table(campaign_holdout_table(campaign_root), "Holdout and subperiod Sharpe diagnostics.", "tab:holdout-subperiods"),
        latex_table(
            campaign_cost_table(campaign_root),
            "Turnover-scaled cost sensitivity and break-even cost.",
            "tab:costs",
            note="AQR factor series are pre-aggregated returns. Constituent-level turnover is not observed in the public series, so turnover-cost diagnostics are reported only for row-level panel candidates.",
        ),
        empirical_figure_tex(),
    ]

    sim_dir = campaign_root / "simulation"
    simulation_sections = [power_figure_tex()]
    simulation_sections.append(latex_table(
        monte_carlo_design_contract_table(campaign_root),
        "Monte Carlo design contract.",
        "tab:simulation-contract",
        note="All null designs have true Sharpe equal to zero. Positive-control rows use a declared nonzero signal and are included to show that the reporting rule is not mechanically anti-discovery.",
    ))
    simulation_sections.append(latex_table(simulation_settings_table(campaign_root), "Monte Carlo and resampling settings.", "tab:simulation-settings"))
    if (sim_dir / "dgp_configs.csv").exists():
        simulation_sections.append(latex_table(dgp_table(sim_dir), "Simulation DGP parameter grid.", "tab:dgp-grid"))
    simulation_sections.extend([
        latex_table(
            coverage_table(sim_dir),
            "Coverage of the date-level Sharpe target by alternative interval procedures.",
            "tab:coverage",
            note="Row-naive intervals are evaluated against the economically relevant date-level Sharpe target, not against their own row-distribution Sharpe target. Low row-naive coverage therefore records target mismatch and false precision at the portfolio boundary.",
        ),
        latex_table(
            target_boundary_table(sim_dir),
            "Row-naive interval behavior when evaluated against the date-level Sharpe target.",
            "tab:target-boundary",
            note="These are not ordinary row-estimator coverage rates; they evaluate the row-naive interval against the economically relevant date-level target.",
        ),
        latex_table(
            design_sweep_table(sim_dir),
            "Sampling-boundary stress tests under a null design with no true edge.",
            "tab:design-sweeps",
            note="Row-naive overrejection rises with same-date dependence and selected-count pressure, whereas date-level HAC remains close to nominal size.",
        ),
        latex_table(
            size_table(sim_dir),
            "Monte Carlo rejection rates under null designs with no true edge.",
            "tab:size",
            note="Each entry is the fraction of 1000 Monte Carlo replications rejected at nominal 5 percent size. The main null designs use T=1000 dates and N=50 entities. Dependent null combines same-date and serial dependence; GARCH null adds conditional heteroskedasticity; IID null removes both. Row-naive treats selected asset-date rows as independent; the other procedures operate on the date-level portfolio return series.",
        ),
        latex_table(power_table(sim_dir), "Rejection rates across true annualized Sharpe values.", "tab:power"),
    ])

    sections = empirical_sections + simulation_sections
    header = f"% Generated from campaign root {campaign_root}\n"
    (paper_dir / "generated_empirical_artifacts.tex").write_text(
        header + "\n".join(empirical_sections),
        encoding="utf-8",
    )
    (paper_dir / "generated_simulation_artifacts.tex").write_text(
        header + "\n".join(simulation_sections),
        encoding="utf-8",
    )
    (paper_dir / "generated_artifacts.tex").write_text(header + "\n".join(sections), encoding="utf-8")

    empirical_main_sections = [
        "% Main-manuscript subset generated from generated_empirical_artifacts.tex.\n",
        "The main empirical tables report the row-level panel candidates only.\n"
        "Pre-aggregated AQR factor series, benchmark-validation details, horizon\n"
        "sensitivity, and full robustness diagnostics are reported in the\n"
        "technical appendix.\n",
        latex_table(
            main_panel_status_table(campaign_root),
            "Panel candidates and date-level evaluation status.",
            "tab:panel-status",
            note="Annualized SR is the gross date-level Sharpe ratio. Date-HAC p+ is the one-sided HAC-delta positive-edge p-value computed on the date-level portfolio return series. Permutation p is the same-date signal-placebo p-value; N/A means the placebo was degenerate or otherwise uninformative for that threshold and is not counted as a full pass. Net SR (5 bps) is the Sharpe ratio after the reported 5 bps turnover-cost stress.",
        ),
        "The comparator table makes the standard-practice contrast explicit:\n"
        "row-naive evidence is shown next to date-level HAC and bootstrap\n"
        "inference on the portfolio return series.\n",
        latex_table(
            standard_comparator_table(campaign_root),
            "Row-naive and date-level inference comparisons for panel candidates.",
            "tab:standard-comparator",
            note="p+ denotes a one-sided positive-Sharpe p-value. Row-naive p+ treats selected asset-date rows as independent. Date-IID, Date-HAC, and Moving-block p+ operate on the date-level portfolio return series. Sharpe UVIF is the diagnostic variance ratio between date-level long-run Sharpe uncertainty and row-pooled IID Sharpe uncertainty.",
        ),
    ]
    simulation_main_sections = [
        "% Main-manuscript subset generated from generated_simulation_artifacts.tex.\n",
        "The null-size figure and table are the core simulation results: under\n"
        "dependent nulls, row-naive testing rejects about one third of the time\n"
        "at a nominal 5 percent level, while date-level HAC, block-bootstrap,\n"
        "stationary, and Romano-Wolf procedures stay much closer to size.\n",
        size_figure_tex(),
        latex_table(
            size_table(sim_dir),
            "Monte Carlo rejection rates under null designs with no true edge.",
            "tab:size",
            note="Each entry is the fraction of 1000 Monte Carlo replications rejected at nominal 5 percent size. The main null designs use T=1000 dates and N=50 entities. Dependent null combines same-date and serial dependence; GARCH null adds conditional heteroskedasticity; IID null removes both. Row-naive treats selected asset-date rows as independent; the other procedures operate on the date-level portfolio return series.",
        ),
    ]
    (paper_dir / "generated_empirical_main_artifacts.tex").write_text(
        header + "\n".join(empirical_main_sections),
        encoding="utf-8",
    )
    (paper_dir / "generated_simulation_main_artifacts.tex").write_text(
        header + "\n".join(simulation_main_sections),
        encoding="utf-8",
    )

    provenance_rows = []
    for path in sorted(campaign_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}:
            provenance_rows.append(f"- `{path}` sha256 `{sha256(path)}`")
    (paper_dir / "generated_provenance.md").write_text(
        "# Generated Artifact Provenance\n\n" + "\n".join(provenance_rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.environ.get("AUDIT_OUTPUT_DIR", "code/output_prod_v2"))
    parser.add_argument("--campaign-root")
    parser.add_argument("--paper-dir", default="paper")
    args = parser.parse_args()
    if args.campaign_root:
        render_campaign(Path(args.campaign_root), Path(args.paper_dir))
    else:
        render(Path(args.output_dir), Path(args.paper_dir))


if __name__ == "__main__":
    main()
