"""
Miscellaneous utility functions used across modules.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def to_utc(dt: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """Normalize any datetime-like value to UTC pandas Timestamp."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def floor_to_bar(dt: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Floor a timestamp to the start of its bar period."""
    return dt.floor(freq)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise if required OHLCV columns are missing."""
    required = {"open", "high", "low", "close", "volume"}
    cols = [str(c).lower() for c in df.columns]
    missing = required - set(cols)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase all columns and ensure standard OHLCV naming."""
    df = df.copy()
    # Handle MultiIndex columns (e.g. yfinance >= 0.2.36)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower().str.strip()
    rename_map = {
        "adj close": "close",
        "adjusted_close": "close",
        "vol":    "volume",
    }
    df.rename(columns=rename_map, inplace=True)
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame to a coarser timeframe.

    Args:
        df: DatetimeIndex DataFrame with open/high/low/close/volume
        rule: pandas resample rule, e.g. "4h", "1d"
    """
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    existing = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample(rule).agg(existing).dropna(subset=["close"])


def forward_fill_to_bar(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Reindex to a complete date range and forward-fill."""
    full_idx = pd.date_range(df.index[0], df.index[-1], freq=freq, tz=df.index.tz)
    return df.reindex(full_idx, method="ffill")


# ---------------------------------------------------------------------------
# Math / statistics helpers
# ---------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that returns fallback on zero/NaN denominator, NaN numerator, or inf result."""
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return fallback
    result = numerator / denominator
    if np.isinf(result):
        return fallback
    return result


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0.0
    return float(returns.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Return maximum drawdown as a positive fraction (0–1)."""
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(abs(drawdown.min()))


def calmar_ratio(cagr: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return cagr / max_dd


def annualised_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    n = len(equity)
    if n < 2 or equity.iloc[0] == 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = n / periods_per_year
    return float(total ** (1.0 / years) - 1.0)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def file_hash(path: Union[str, Path]) -> str:
    """SHA-256 of file contents — used for cache invalidation."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def flatten_dict(d: Dict, sep: str = ".", prefix: str = "") -> Dict:
    """Flatten nested dict to single-level with separator-joined keys."""
    items = {}
    for k, v in d.items():
        full_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, sep=sep, prefix=full_key))
        else:
            items[full_key] = v
    return items
