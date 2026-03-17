"""
Regime feature engineering.

These features are designed to describe the *current market environment*
and are the primary inputs to the regime detection models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_regime_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Compute all regime-detection features on an OHLCV DataFrame.

    Args:
        df:     OHLCV DataFrame (open, high, low, close, volume)
        config: Optional parameter overrides

    Returns:
        DataFrame with added feature columns (original columns preserved).
    """
    cfg = {
        "atr_period":        14,
        "vol_short":         10,
        "vol_long":          30,
        "adx_period":        14,
        "rsi_period":        14,
        "bb_period":         20,
        "bb_std":            2.0,
        "trend_period":      50,
        "slope_period":      20,
        "vol_ratio_period":  20,
        "range_period":      20,
    }
    if config:
        cfg.update(config)

    out = df.copy()
    c = out["close"]
    h = out["high"]
    l = out["low"]

    # --- Returns ---
    out["ret_1"]   = c.pct_change(1)
    out["ret_5"]   = c.pct_change(5)
    out["ret_20"]  = c.pct_change(20)
    out["log_ret"] = np.log(c / c.shift(1))

    # --- ATR (True Range) ---
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr"]          = tr.ewm(span=cfg["atr_period"], adjust=False).mean()
    out["atr_norm"]     = out["atr"] / c           # ATR as fraction of price
    out["atr_ratio"]    = (tr.rolling(cfg["vol_short"]).mean() /
                           tr.rolling(cfg["vol_long"]).mean())  # short/long ATR

    # --- Rolling volatility ---
    out["vol_short"] = out["log_ret"].rolling(cfg["vol_short"]).std() * np.sqrt(252)
    out["vol_long"]  = out["log_ret"].rolling(cfg["vol_long"]).std()  * np.sqrt(252)
    out["vol_ratio"] = out["vol_short"] / out["vol_long"].replace(0, np.nan)

    # --- ADX (Average Directional Index) ---
    plus_dm  = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_adx  = tr.ewm(span=cfg["adx_period"], adjust=False).mean()
    di_plus  = 100 * plus_dm.ewm(span=cfg["adx_period"],  adjust=False).mean() / atr_adx
    di_minus = 100 * minus_dm.ewm(span=cfg["adx_period"], adjust=False).mean() / atr_adx
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    out["adx"]      = dx.ewm(span=cfg["adx_period"], adjust=False).mean()
    out["di_plus"]  = di_plus
    out["di_minus"] = di_minus

    # --- RSI ---
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(span=cfg["rsi_period"], adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=cfg["rsi_period"], adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - 100 / (1 + rs)

    # --- Bollinger Bands ---
    bb_mid   = c.rolling(cfg["bb_period"]).mean()
    bb_std   = c.rolling(cfg["bb_period"]).std()
    bb_upper = bb_mid + cfg["bb_std"] * bb_std
    bb_lower = bb_mid - cfg["bb_std"] * bb_std
    out["bb_width"]   = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    out["bb_pct"]     = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_upper"]   = bb_upper
    out["bb_lower"]   = bb_lower

    # --- Trend slope (linear regression slope of log-price) ---
    def rolling_slope(series: pd.Series, w: int) -> pd.Series:
        x = np.arange(w)
        x_dm = x - x.mean()
        denom = (x_dm ** 2).sum()
        def _slope(y: np.ndarray) -> float:
            if np.isnan(y).any():
                return np.nan
            return float(np.dot(x_dm, y - y.mean()) / denom)
        return series.rolling(w).apply(_slope, raw=True)

    log_c = np.log(c)
    out["trend_slope"]  = rolling_slope(log_c, cfg["slope_period"])
    out["trend_slope_norm"] = out["trend_slope"] / out["atr_norm"].replace(0, np.nan)

    # --- Moving average spread (price distance from trend MA) ---
    ma_trend = c.rolling(cfg["trend_period"]).mean()
    out["ma_spread"]     = (c - ma_trend) / ma_trend.replace(0, np.nan)
    out["ema200_dist"]   = (c - c.ewm(span=200, adjust=False).mean()) / c

    # --- Range compression (current range vs historical range) ---
    bar_range = h - l
    out["range_norm"]   = bar_range / out["atr"]
    out["range_comp"]   = bar_range.rolling(cfg["range_period"]).min() / bar_range.rolling(cfg["range_period"]).max()

    # --- Volume shock ---
    vol_ma = out["volume"].rolling(cfg["vol_ratio_period"]).mean()
    out["vol_shock"] = out["volume"] / vol_ma.replace(0, np.nan)

    # --- Session volatility proxy (intrabar range relative to ATR) ---
    out["intrabar_range_ratio"] = bar_range / out["atr"].replace(0, np.nan)

    return out


def get_feature_columns() -> list:
    """Return the list of feature column names produced by add_regime_features."""
    return [
        "ret_1", "ret_5", "ret_20", "log_ret",
        "atr", "atr_norm", "atr_ratio",
        "vol_short", "vol_long", "vol_ratio",
        "adx", "di_plus", "di_minus",
        "rsi",
        "bb_width", "bb_pct",
        "trend_slope", "trend_slope_norm",
        "ma_spread", "ema200_dist",
        "range_norm", "range_comp",
        "vol_shock", "intrabar_range_ratio",
    ]
