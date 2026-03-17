"""
Alpha features — strategy-specific entry/exit signals.

These are the features that drive individual strategy entry logic,
as opposed to regime_features which describe the macro environment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Donchian Channel: rolling high/low over `period` bars.
    Adds: donch_upper, donch_lower, donch_mid, donch_width_atr
    """
    out = df.copy()
    out[f"donch_upper_{period}"] = out["high"].rolling(period).max()
    out[f"donch_lower_{period}"] = out["low"].rolling(period).min()
    mid = (out[f"donch_upper_{period}"] + out[f"donch_lower_{period}"]) / 2
    out[f"donch_mid_{period}"]   = mid
    if "atr" in out.columns:
        out[f"donch_width_atr_{period}"] = (
            (out[f"donch_upper_{period}"] - out[f"donch_lower_{period}"]) /
            out["atr"].replace(0, np.nan)
        )
    return out


def add_ema_stack(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """
    EMA stack for trend direction signals.
    Adds columns: ema_{period} for each period in `periods`.
    """
    periods = periods or [9, 21, 50, 100, 200]
    out = df.copy()
    for p in periods:
        out[f"ema_{p}"] = out["close"].ewm(span=p, adjust=False).mean()
    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD with histogram."""
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    out["macd_line"]   = ema_fast - ema_slow
    out["macd_signal"] = out["macd_line"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"]   = out["macd_line"] - out["macd_signal"]
    return out


def add_session_range(df: pd.DataFrame, session_hours: int = 4) -> pd.DataFrame:
    """
    Opening range of each session window (high/low of first N hours).
    Adds: session_high, session_low, session_range_atr
    """
    out = df.copy()
    # Group by trading day, compute rolling session range
    daily_high = out["high"].resample("D").transform("max")
    daily_low  = out["low"].resample("D").transform("min")
    out["session_high"]  = daily_high
    out["session_low"]   = daily_low
    out["session_range"] = daily_high - daily_low
    if "atr" in out.columns:
        out["session_range_atr"] = out["session_range"] / out["atr"].replace(0, np.nan)
    return out


def add_mean_reversion_signals(
    df: pd.DataFrame,
    rsi_ob: float = 70.0,
    rsi_os: float = 30.0,
) -> pd.DataFrame:
    """
    Binary overbought/oversold and Bollinger Band touch signals.
    Requires rsi, bb_pct in df (from add_regime_features).
    """
    out = df.copy()
    if "rsi" in out.columns:
        out["rsi_overbought"]  = (out["rsi"] > rsi_ob).astype(float)
        out["rsi_oversold"]    = (out["rsi"] < rsi_os).astype(float)
    if "bb_pct" in out.columns:
        out["bb_upper_touch"]  = (out["bb_pct"] > 0.95).astype(float)
        out["bb_lower_touch"]  = (out["bb_pct"] < 0.05).astype(float)
    return out


def add_higher_tf_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-timeframe trend alignment from intraday data.

    Resamples to daily, computes trend direction via EMA50/EMA200 crossover
    and ADX strength, then maps back to the original timeframe.

    Adds:
        htf_trend:        +1 (bullish), -1 (bearish), 0 (neutral)
        htf_trend_strength: 0-1 score based on daily ADX and EMA separation
    """
    out = df.copy()

    # Resample to daily
    daily = out.resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])

    if len(daily) < 50:
        out["htf_trend"] = 0
        out["htf_trend_strength"] = 0.0
        return out

    # Daily EMA50 and EMA200
    ema50 = daily["close"].ewm(span=50, adjust=False).mean()
    ema200 = daily["close"].ewm(span=200, adjust=False).mean()

    # Daily ADX
    c, h, l = daily["close"], daily["high"], daily["low"]
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    plus_dm = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_d = tr.ewm(span=14, adjust=False).mean()
    di_p = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_d
    di_m = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_d
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    daily_adx = dx.ewm(span=14, adjust=False).mean()

    # Trend direction: EMA50 > EMA200 = bullish
    trend = pd.Series(0, index=daily.index)
    trend = trend.where(~(ema50 > ema200), 1)
    trend = trend.where(~(ema50 < ema200), -1)

    # Trend strength: ADX normalized (0-1) * EMA separation magnitude
    ema_sep = ((ema50 - ema200) / ema200).abs()
    strength = np.clip(daily_adx / 50, 0, 1) * np.clip(ema_sep * 20, 0, 1)

    # Map back to original timeframe (forward-fill daily values)
    trend_reindexed = trend.reindex(out.index, method="ffill")
    strength_reindexed = strength.reindex(out.index, method="ffill")

    out["htf_trend"] = trend_reindexed.fillna(0).astype(int)
    out["htf_trend_strength"] = strength_reindexed.fillna(0)
    return out


def add_momentum_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect momentum divergence (price vs RSI).

    Bullish divergence: price makes lower low but RSI makes higher low
    Bearish divergence: price makes higher high but RSI makes lower high

    Adds:
        mom_divergence: +1 (bullish), -1 (bearish), 0 (none)
    """
    out = df.copy()
    lookback = 20

    if "rsi" not in out.columns:
        out["mom_divergence"] = 0
        return out

    rsi = out["rsi"]
    close = out["close"]

    # Rolling min/max of price and RSI
    price_lo = close.rolling(lookback).min()
    price_hi = close.rolling(lookback).max()
    rsi_lo = rsi.rolling(lookback).min()
    rsi_hi = rsi.rolling(lookback).max()

    # Bullish: price near rolling low but RSI above its rolling low
    near_price_lo = (close - price_lo) / close.replace(0, np.nan) < 0.005
    rsi_higher = rsi > rsi_lo + 5
    bullish_div = near_price_lo & rsi_higher & (rsi < 40)

    # Bearish: price near rolling high but RSI below its rolling high
    near_price_hi = (price_hi - close) / close.replace(0, np.nan) < 0.005
    rsi_lower = rsi < rsi_hi - 5
    bearish_div = near_price_hi & rsi_lower & (rsi > 60)

    div = pd.Series(0, index=out.index)
    div[bullish_div] = 1
    div[bearish_div] = -1
    out["mom_divergence"] = div
    return out


def add_alpha_features(
    df: pd.DataFrame,
    donchian_periods: list = None,
    ema_periods: list = None,
    include_macd: bool = True,
) -> pd.DataFrame:
    """Apply all alpha features in sequence."""
    out = df.copy()
    out = add_ema_stack(out, periods=ema_periods)
    for p in (donchian_periods or [20, 50]):
        out = add_donchian(out, period=p)
    if include_macd:
        out = add_macd(out)
    out = add_session_range(out)
    out = add_mean_reversion_signals(out)
    out = add_higher_tf_trend(out)
    out = add_momentum_divergence(out)
    return out
