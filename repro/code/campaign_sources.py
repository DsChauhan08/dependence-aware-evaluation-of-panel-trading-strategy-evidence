"""
Public data source registry and loaders for the full experiment campaign.

The loaders are intentionally conservative: a source that cannot be parsed is
returned as a recorded failure by the campaign runner rather than silently
dropping out of the search.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from public_data import (
    _download_zip_text,
    _parse_french_csv,
    build_momentum_panel,
    download_ff_factors,
    panel_from_signal,
)
from sddm_bootstrap import PanelData


REGISTRY_PATH = Path(__file__).with_name("campaign_registry.json")


class SourceLoadError(RuntimeError):
    """Raised when a registered public source cannot be loaded."""


@dataclass
class LoadedCandidate:
    spec: dict[str, Any]
    panel: PanelData
    returns: pd.DataFrame
    factors: pd.DataFrame | None
    groups: np.ndarray | None
    notes: list[str]


def load_registry(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read the frozen campaign registry."""
    registry_path = Path(path) if path is not None else REGISTRY_PATH
    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def registry_candidates(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Return candidate specs from the registry."""
    registry = load_registry(path)
    return list(registry.get("candidates", []))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download_bytes(url: str) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _download_zip_dataframe(url: str, columns_prefix: str) -> pd.DataFrame:
    df = _parse_french_csv(_download_zip_text(url))
    df.columns = [f"{columns_prefix}{i:02d}" for i in range(1, df.shape[1] + 1)]
    return df.dropna(how="all")


def _sample_tail(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    tail = int(spec.get("sample_tail", 0) or 0)
    if tail > 0:
        return df.iloc[-tail:].copy()
    return df.copy()


def _ranked_long_short_panel(returns: pd.DataFrame, groups: list[str] | None = None) -> PanelData:
    n = returns.shape[1]
    if n < 2:
        raise SourceLoadError("ranked long-short panel requires at least two portfolios")
    scores = 2.0 * np.arange(n, dtype=float) - float(n - 1)
    signal = np.tile(scores, (len(returns), 1))
    return panel_from_signal(returns, signal)


def _constant_long_panel(returns: pd.DataFrame) -> PanelData:
    values = returns.astype(float).copy()
    signal = np.ones(values.shape, dtype=float)
    realised = values.to_numpy(dtype=float).copy()
    confidence = np.ones(values.shape, dtype=float)
    active = np.isfinite(realised)
    realised[~active] = np.nan
    confidence[~active] = np.nan
    dates = np.array([np.datetime64(str(d.date())) for d in values.index])
    tickers = np.array(values.columns.astype(str))
    return PanelData(
        dates=dates,
        tickers=tickers,
        predictions=signal,
        realised=realised,
        confidence=confidence,
    )


def _load_french_ranked_portfolios(spec: dict[str, Any]) -> LoadedCandidate:
    returns = _download_zip_dataframe(spec["portfolio_url"], "MOM")
    returns = _sample_tail(returns, spec)
    panel = _ranked_long_short_panel(returns, spec.get("groups"))
    factors = download_ff_factors()
    groups = np.asarray(spec.get("groups", []), dtype=object) if spec.get("groups") else None
    if groups is not None and len(groups) != panel.N:
        raise SourceLoadError(f"group count {len(groups)} does not match panel width {panel.N}")
    return LoadedCandidate(spec=spec, panel=panel, returns=returns, factors=factors, groups=groups, notes=[])


def _load_french_dynamic_momentum(spec: dict[str, Any]) -> LoadedCandidate:
    returns = _download_zip_dataframe(spec["portfolio_url"], "P")
    returns = _sample_tail(returns, spec)
    panel = build_momentum_panel(
        returns,
        lookback=int(spec.get("lookback", 21)),
        skip=int(spec.get("skip", 2)),
        placebo=False,
    )
    factors = download_ff_factors()
    return LoadedCandidate(spec=spec, panel=panel, returns=returns, factors=factors, groups=None, notes=[])


def _load_stooq_etf_momentum(spec: dict[str, Any]) -> LoadedCandidate:
    frames = []
    notes: list[str] = []
    api_key = os.environ.get("STOOQ_APIKEY", "").strip()
    for symbol in spec.get("symbols", []):
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        if api_key:
            url += f"&apikey={api_key}"
        try:
            raw = _download_bytes(url)
            if raw.startswith(b"Get your apikey"):
                raise SourceLoadError("Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key")
            df = pd.read_csv(io.BytesIO(raw))
            if "Date" not in df or "Close" not in df:
                raise SourceLoadError(f"missing Date/Close columns for {symbol}")
            s = pd.Series(
                pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float),
                index=pd.to_datetime(df["Date"], errors="coerce"),
                name=symbol,
            ).dropna()
            frames.append(s)
        except Exception as exc:
            notes.append(f"{symbol}: {exc}")
    if len(frames) < 3:
        raise SourceLoadError("fewer than three Stooq series loaded: " + "; ".join(notes))
    prices = pd.concat(frames, axis=1).sort_index()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = _sample_tail(returns.dropna(how="all"), spec)
    panel = build_momentum_panel(
        returns,
        lookback=int(spec.get("lookback", 63)),
        skip=int(spec.get("skip", 2)),
        placebo=False,
    )
    return LoadedCandidate(spec=spec, panel=panel, returns=returns, factors=None, groups=None, notes=notes)


def _read_aqr_excel(url: str, cache_dir: Path | None = None) -> pd.DataFrame:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".xlsx")
        if cache_path.exists():
            data = cache_path.read_bytes()
        else:
            data = _download_bytes(url)
            cache_path.write_bytes(data)
    else:
        data = _download_bytes(url)

    try:
        xls = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    except ImportError as exc:
        raise SourceLoadError("AQR Excel loader requires openpyxl; install it before full AQR runs") from exc
    except zipfile.BadZipFile as exc:
        raise SourceLoadError("AQR URL did not return an xlsx workbook") from exc

    best = None
    for sheet in xls.sheet_names:
        preview = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl", nrows=80, header=None)
        header_idx = None
        for idx, row in preview.iterrows():
            vals = [str(x).strip().upper() for x in row.dropna().tolist()]
            if "DATE" in vals or "MONTH" in vals:
                header_idx = int(idx)
                break
        if header_idx is None:
            continue
        df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl", header=header_idx)
        df = df.dropna(axis=1, how="all")
        if len(df) > 20 and len(df.columns) >= 2:
            best = df
            break
    if best is None:
        raise SourceLoadError("no usable numeric sheet found in AQR workbook")
    return best


def _load_aqr_factor_excel(spec: dict[str, Any], cache_dir: Path | None = None) -> LoadedCandidate:
    raw = _read_aqr_excel(spec["source_url"], cache_dir=cache_dir)
    df = raw.copy()
    date_col = None
    for col in df.columns:
        label = str(col).strip().upper()
        if label not in {"DATE", "MONTH"} and col != df.columns[0]:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() > max(20, len(df) // 2):
            date_col = col
            df.index = parsed
            break
    if date_col is None:
        raise SourceLoadError("could not identify date column in AQR workbook")
    df = df.drop(columns=[date_col])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric = df.dropna(axis=1, how="all").dropna(how="all")
    if numeric.empty:
        raise SourceLoadError("AQR workbook contains no numeric return columns after parsing")

    preferred = spec.get("preferred_columns", [])
    selected = None
    for want in preferred:
        matches = [c for c in numeric.columns if want.lower() in str(c).lower()]
        if matches:
            selected = matches[:1]
            break
    if selected is None:
        selected = list(numeric.columns[:1])

    returns = numeric[selected].copy()
    returns.columns = [str(c).strip() for c in returns.columns]
    med = float(np.nanmedian(np.abs(returns.to_numpy(dtype=float))))
    if med > 1.0:
        returns = returns / 100.0
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if len(returns) < 30:
        raise SourceLoadError("AQR factor series has fewer than 30 usable rows")
    panel = _constant_long_panel(returns)
    notes = ["factor_series_source: cross-sectional signal permutation is not applicable"]
    return LoadedCandidate(spec=spec, panel=panel, returns=returns, factors=None, groups=None, notes=notes)


def load_candidate(spec: dict[str, Any], cache_dir: str | os.PathLike[str] | None = None) -> LoadedCandidate:
    """Load one registered public candidate."""
    loader = spec.get("loader")
    if loader == "french_ranked_portfolios":
        return _load_french_ranked_portfolios(spec)
    if loader == "french_dynamic_momentum":
        return _load_french_dynamic_momentum(spec)
    if loader == "stooq_etf_momentum":
        return _load_stooq_etf_momentum(spec)
    if loader == "aqr_factor_excel":
        return _load_aqr_factor_excel(spec, cache_dir=Path(cache_dir) if cache_dir else None)
    raise SourceLoadError(f"unknown source loader: {loader}")
