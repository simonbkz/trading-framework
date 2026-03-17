"""
Risk features — position sizing inputs and risk environment indicators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_risk_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Compute risk-related features.

    Requires: atr, atr_norm (from regime_features) already in df.
    """
    cfg = {
        "vol_target_window": 21,
        "downside_vol_window": 21,
        "drawdown_window": 50,
        "correlation_window": 20,
    }
    if config:
        cfg.update(config)

    out = df.copy()
    c = out["close"]

    # --- Realised volatility ---
    log_ret = np.log(c / c.shift(1))
    out["realised_vol"]    = log_ret.rolling(cfg["vol_target_window"]).std() * np.sqrt(252)
    out["downside_vol"]    = (
        log_ret.where(log_ret < 0, 0)
        .rolling(cfg["downside_vol_window"])
        .std() * np.sqrt(252)
    )

    # --- Volatility regime ratio (current / 90-day) ---
    realised_90 = log_ret.rolling(90).std() * np.sqrt(252)
    out["vol_regime_ratio"] = out["realised_vol"] / realised_90.replace(0, np.nan)

    # --- Rolling drawdown (from rolling max within window) ---
    roll_max  = c.rolling(cfg["drawdown_window"]).max()
    out["roll_drawdown"] = (c - roll_max) / roll_max.replace(0, np.nan)

    # --- Gap risk (overnight gap as fraction of ATR) ---
    out["overnight_gap"] = (out["open"] - c.shift(1)) / out.get("atr", pd.Series(1, index=c.index))

    # --- Liquidity proxy: volume * price ($ volume) ---
    out["dollar_volume"] = out["volume"] * c

    # --- Tail risk: skewness and kurtosis of returns ---
    out["ret_skew"]  = log_ret.rolling(60).skew()
    out["ret_kurt"]  = log_ret.rolling(60).kurt()

    return out


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Full Kelly fraction.
    win_rate: fraction of winning trades [0,1]
    avg_win / avg_loss: average win/loss sizes (positive values)
    """
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    f = win_rate - (1 - win_rate) / b
    return max(0.0, f)


def half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    return kelly_fraction(win_rate, avg_win, avg_loss) * 0.5


def atr_position_size(
    equity: float,
    risk_pct: float,
    atr: float,
    atr_mult: float,
    pip_value: float,
    pip_size: float,
) -> float:
    """
    ATR-based position sizing.

    Risk amount / (ATR * multiplier * pip_value_per_pip)

    Returns: lot size (units)
    """
    risk_amount = equity * risk_pct / 100.0
    sl_distance = atr * atr_mult
    if sl_distance == 0 or pip_value == 0:
        return 0.0
    sl_pips = sl_distance / pip_size
    return risk_amount / (sl_pips * pip_value)
