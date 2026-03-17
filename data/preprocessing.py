"""
OHLCV preprocessing pipeline.

Applied after loading, before feature engineering:
- Remove bad bars (zero/negative prices, extreme gaps)
- Fill short gaps with forward-fill (max 3 bars)
- Timezone enforcement
- Duplicate index removal
- Volume imputation for assets with unreliable volume data
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# Assets where volume data is unreliable / always zero
NO_VOLUME_ASSETS = {"XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"}
MAX_FFILL_GAPS = 3          # max consecutive bars to forward-fill
MAX_RETURN_SIGMA = 10.0     # returns > N sigma → flag as bad bar


def preprocess_ohlcv(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "",
) -> pd.DataFrame:
    """
    Clean and validate an OHLCV DataFrame.

    Args:
        df: Raw DataFrame with DatetimeIndex, columns: open high low close volume
        symbol: Asset symbol (used for asset-specific rules)
        timeframe: Timeframe string (informational)

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    n_raw = len(df)

    # 1. Ensure UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # 2. Sort
    df = df.sort_index()

    # 3. Remove duplicate timestamps
    dup_mask = df.index.duplicated(keep="last")
    if dup_mask.sum() > 0:
        log.debug("%s: removing %d duplicate timestamps", symbol, dup_mask.sum())
        df = df[~dup_mask]

    # 4. Remove bars with non-positive prices
    bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad_price.sum() > 0:
        log.debug("%s: removing %d zero/negative price bars", symbol, bad_price.sum())
        df = df[~bad_price]

    # 5. OHLC integrity: high >= low, high >= open, high >= close
    integrity = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"]  > df["open"]) |
        (df["low"]  > df["close"])
    )
    if integrity.sum() > 0:
        log.debug("%s: fixing %d OHLC integrity violations", symbol, integrity.sum())
        df.loc[integrity, "high"] = df.loc[integrity, ["open", "high", "close"]].max(axis=1)
        df.loc[integrity, "low"]  = df.loc[integrity, ["open", "low",  "close"]].min(axis=1)

    # 6. Flag extreme returns (possible data errors)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_mean = log_ret.mean()
    ret_std  = log_ret.std()
    if ret_std > 0:
        z_score = (log_ret - ret_mean) / ret_std
        extreme = z_score.abs() > MAX_RETURN_SIGMA
        if extreme.sum() > 0:
            log.warning(
                "%s %s: %d bars with |z-score| > %g (possible data errors, not removed)",
                symbol, timeframe, extreme.sum(), MAX_RETURN_SIGMA,
            )

    # 7. Volume: impute zero volume for no-volume assets
    if symbol in NO_VOLUME_ASSETS or df["volume"].sum() == 0:
        # Use ATR-proxy as volume stand-in (range as activity measure)
        df["volume"] = df["high"] - df["low"]

    # 8. Forward-fill short gaps (e.g. bank holidays, thin liquidity bars)
    df = df.ffill(limit=MAX_FFILL_GAPS)

    # 9. Drop any remaining NaNs
    n_before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < n_before:
        log.debug("%s: dropped %d NaN bars", symbol, n_before - len(df))

    log.debug(
        "%s %s: %d raw -> %d clean bars (removed %d)",
        symbol, timeframe, n_raw, len(df), n_raw - len(df),
    )
    return df


def compute_returns(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """Log returns of close price."""
    return np.log(df["close"] / df["close"].shift(periods))


def normalise_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume as multiple of rolling mean (volume shock indicator)."""
    vol_ma = df["volume"].rolling(window).mean()
    return df["volume"] / vol_ma.replace(0, np.nan)
