"""
Trend Breakout Strategy (Donchian Channel).

Entry:
  - Donchian upper/lower breakout confirmed by volume
  - H4/daily context: close above/below EMA200 (trend filter)
  - ADX above threshold

Exit:
  - Fixed SL: breakout level - ATR * sl_mult
  - Fixed TP: entry + sl_dist * tp_rr
  - OR: close below trailing N-bar low (for longs)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Side, TradeProposal
from features.alpha_features import add_donchian


class TrendBreakoutStrategy(BaseStrategy):
    """Donchian channel breakout with volume + trend filters."""

    name = "trend_breakout"
    suitable_regimes = ["trend_up", "trend_down", "high_volatility"]

    def default_params(self) -> Dict:
        return {
            "don_period":    20,     # Donchian lookback
            "vol_mult":      1.5,    # volume > vol_mult * 20-bar avg
            "sl_atr_mult":   1.0,    # SL distance below breakout level
            "tp_rr":         2.0,    # TP as multiple of SL distance
            "adx_min":       20.0,   # minimum ADX for entry
            "ema_trend":     200,    # EMA for trend filter (0 = disabled)
            "pullback_bars": 6,      # bars to wait for retest (0 = no retest)
            "rsi_lo":        40,     # RSI lower bound for buy entry
            "rsi_hi":        70,     # RSI upper bound for buy entry
        }

    @property
    def param_schema(self) -> Dict:
        return {
            "don_period":    {"type": "int",   "low": 5,   "high": 50, "step": 5},
            "vol_mult":      {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5},
            "sl_atr_mult":   {"type": "float", "low": 0.5, "high": 3.0, "step": 0.5},
            "tp_rr":         {"type": "float", "low": 1.0, "high": 4.0, "step": 0.5},
            "adx_min":       {"type": "float", "low": 10,  "high": 35,  "step": 5},
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        side: Side = "long",
        asset: str = "",
        regime: str = "",
    ) -> pd.DataFrame:
        p = self.get_params(side)
        out = df.copy()
        out = add_donchian(out, period=p["don_period"])

        atr      = self._atr(out)
        don_up   = out[f"donch_upper_{p['don_period']}"].shift(1)   # confirmed bar
        don_lo   = out[f"donch_lower_{p['don_period']}"].shift(1)

        # Volume confirmation
        vol_ma  = out["volume"].rolling(20).mean()
        vol_ok  = out["volume"] > p["vol_mult"] * vol_ma

        # ADX filter
        adx_ok  = out.get("adx", pd.Series(100, index=out.index)) >= p["adx_min"]

        # Trend filter (EMA)
        ema_col = f"ema_{p['ema_trend']}"
        if p["ema_trend"] > 0 and ema_col not in out.columns:
            out[ema_col] = out["close"].ewm(span=p["ema_trend"], adjust=False).mean()
        if p["ema_trend"] > 0:
            trend_up   = out["close"] > out[ema_col]
            trend_down = out["close"] < out[ema_col]
        else:
            trend_up   = pd.Series(True, index=out.index)
            trend_down = pd.Series(True, index=out.index)

        # RSI filter
        rsi = out.get("rsi", pd.Series(50, index=out.index))
        rsi_lo, rsi_hi = p["rsi_lo"], p["rsi_hi"]

        # Higher-timeframe trend alignment
        htf_trend = out.get("htf_trend", pd.Series(0, index=out.index))

        # --- LONG signals ---
        if side == "long":
            htf_ok = htf_trend >= 0  # daily trend not bearish
            breakout = (out["close"] > don_up) & vol_ok & adx_ok & trend_up & htf_ok
            rsi_ok   = (rsi >= rsi_lo) & (rsi <= rsi_hi)
            signal   = breakout & rsi_ok
            sl_price = don_up - p["sl_atr_mult"] * atr
            sl_dist  = (out["close"] - sl_price).clip(lower=atr * 0.5)
            tp_price = out["close"] + sl_dist * p["tp_rr"]
            direction = 1

        # --- SHORT signals ---
        else:
            htf_ok = htf_trend <= 0  # daily trend not bullish
            breakout = (out["close"] < don_lo) & vol_ok & adx_ok & trend_down & htf_ok
            rsi_inv  = 100 - rsi
            rsi_ok   = (rsi_inv >= rsi_lo) & (rsi_inv <= rsi_hi)
            signal   = breakout & rsi_ok
            sl_price = don_lo + p["sl_atr_mult"] * atr
            sl_dist  = (sl_price - out["close"]).clip(lower=atr * 0.5)
            tp_price = out["close"] - sl_dist * p["tp_rr"]
            direction = -1

        out["signal"]   = 0
        out.loc[signal, "signal"] = direction
        out["sl"]       = sl_price
        out["tp"]       = tp_price
        base_strength = np.clip(
            (atr / out["close"].replace(0, np.nan)) * 100 *
            (out.get("adx", pd.Series(25, index=out.index)) / 25),
            0, 1,
        )

        # Macro multipliers (only apply when features are available)
        macro_mult = pd.Series(1.0, index=out.index)

        # VIX dampening: reduce strength in extreme fear
        vix_regime = out.get("vix_regime", pd.Series(1, index=out.index)).fillna(1)
        macro_mult *= pd.Series(
            np.where(vix_regime >= 3, 0.5, np.where(vix_regime >= 2, 0.75, 1.0)),
            index=out.index,
        )

        # DXY alignment: for forex, DXY strengthening → favor USD longs
        dxy_trend = out.get("dxy_trend", pd.Series(0, index=out.index)).fillna(0)
        if asset in ("EURUSD", "GBPUSD", "AUDUSD"):
            if side == "long":
                macro_mult *= np.where(dxy_trend == 1, 0.8, np.where(dxy_trend == -1, 1.15, 1.0))
            else:
                macro_mult *= np.where(dxy_trend == 1, 1.15, np.where(dxy_trend == -1, 0.8, 1.0))
        elif asset == "USDJPY":
            if side == "long":
                macro_mult *= np.where(dxy_trend == 1, 1.15, np.where(dxy_trend == -1, 0.8, 1.0))
            else:
                macro_mult *= np.where(dxy_trend == 1, 0.8, np.where(dxy_trend == -1, 1.15, 1.0))

        # Event risk dampening
        event_risk = out.get("event_risk_score", pd.Series(0, index=out.index)).fillna(0)
        macro_mult *= (1.0 - 0.3 * event_risk)

        # Fear/greed: trend strategies benefit from greed
        fg = out.get("fear_greed_proxy", pd.Series(0.5, index=out.index)).fillna(0.5)
        macro_mult *= (0.8 + 0.4 * fg)

        out["strength"] = np.where(signal, np.clip(base_strength * macro_mult, 0, 1), 0)
        return self._sanitize_signals(out)
