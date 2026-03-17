"""
Pullback Retest Strategy.

Enters on pullbacks within established trends — buying dips in uptrends,
selling rallies in downtrends. Higher win rate than breakout strategies
because entry is at better price levels.

Entry (Long):
  - EMA50 trending up (close > EMA50 > EMA100)
  - Price pulls back to EMA21 support zone (within 0.5 ATR)
  - RSI between 40-60 (not oversold, just pulled back)
  - Bullish candlestick pattern (close > open after touching zone)

Entry (Short): Mirror conditions

Exit:
  - SL: below recent swing low - ATR buffer
  - TP: previous swing high or R:R target
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Side


class PullbackRetestStrategy(BaseStrategy):

    name = "pullback_retest"
    suitable_regimes = ["trend_up", "trend_down", "range_bound"]

    def default_params(self) -> Dict:
        return {
            "ema_trend":     50,     # EMA for trend direction
            "ema_pullback":  21,     # EMA for pullback zone
            "pullback_atr":  0.3,    # tighter: how close to EMA pullback zone (ATR units)
            "sl_atr_mult":   1.0,    # SL below swing low (tighter for better R:R)
            "tp_rr":         2.5,    # reward:risk ratio (higher targets)
            "swing_lookback": 8,     # bars to find swing points
            "rsi_lo":        40,     # RSI lower bound (not too oversold)
            "rsi_hi":        60,     # RSI upper bound (not overbought)
            "adx_min":       20.0,   # minimum ADX (require clearer trend)
        }

    @property
    def param_schema(self) -> Dict:
        return {
            "ema_trend":     {"type": "int",   "low": 30,  "high": 80,  "step": 10},
            "ema_pullback":  {"type": "int",   "low": 10,  "high": 30,  "step": 5},
            "pullback_atr":  {"type": "float", "low": 0.2, "high": 1.5, "step": 0.3},
            "sl_atr_mult":   {"type": "float", "low": 0.5, "high": 2.5, "step": 0.5},
            "tp_rr":         {"type": "float", "low": 1.5, "high": 4.0, "step": 0.5},
            "adx_min":       {"type": "float", "low": 10,  "high": 25,  "step": 5},
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
        atr = self._atr(out)

        # EMAs for trend and pullback zone
        ema_trend = out["close"].ewm(span=p["ema_trend"], adjust=False).mean()
        ema_pb = out["close"].ewm(span=p["ema_pullback"], adjust=False).mean()
        ema_slow = out["close"].ewm(span=100, adjust=False).mean()

        # RSI
        if "rsi" in out.columns:
            rsi = out["rsi"]
        else:
            delta = out["close"].diff()
            gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
            rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        rsi_ok = (rsi >= p["rsi_lo"]) & (rsi <= p["rsi_hi"])

        # ADX filter
        adx = out.get("adx", pd.Series(20, index=out.index))
        adx_ok = adx >= p["adx_min"]

        # Swing points for SL
        swing_lo = out["low"].rolling(p["swing_lookback"]).min()
        swing_hi = out["high"].rolling(p["swing_lookback"]).max()

        # HTF trend alignment
        htf_trend = out.get("htf_trend", pd.Series(0, index=out.index))

        # Bullish/bearish candle
        bullish_candle = out["close"] > out["open"]
        bearish_candle = out["close"] < out["open"]

        if side == "long":
            # Uptrend: close > EMA_trend > EMA_slow
            trend_ok = (out["close"] > ema_trend) & (ema_trend > ema_slow)
            # Pullback: low touches EMA_pullback zone (within pullback_atr ATRs)
            pullback = (out["low"] - ema_pb).abs() <= p["pullback_atr"] * atr
            # Or: price pulled back below fast EMA but still above trend EMA
            pullback |= (out["low"] <= ema_pb) & (out["close"] > ema_trend * 0.99)
            # HTF not bearish
            htf_ok = htf_trend >= 0

            signal = trend_ok & pullback & rsi_ok & adx_ok & bullish_candle & htf_ok

            sl_price = swing_lo - p["sl_atr_mult"] * atr
            sl_dist = (out["close"] - sl_price).clip(lower=atr * 0.3)
            tp_price = out["close"] + sl_dist * p["tp_rr"]
            direction = 1
        else:
            # Downtrend: close < EMA_trend < EMA_slow
            trend_ok = (out["close"] < ema_trend) & (ema_trend < ema_slow)
            # Pullback toward EMA from below
            pullback = (ema_pb - out["high"]).abs() <= p["pullback_atr"] * atr
            pullback |= (out["high"] >= ema_pb) & (out["close"] < ema_trend * 1.01)
            htf_ok = htf_trend <= 0

            signal = trend_ok & pullback & rsi_ok & adx_ok & bearish_candle & htf_ok

            sl_price = swing_hi + p["sl_atr_mult"] * atr
            sl_dist = (sl_price - out["close"]).clip(lower=atr * 0.3)
            tp_price = out["close"] - sl_dist * p["tp_rr"]
            direction = -1

        out["signal"] = 0
        out.loc[signal, "signal"] = direction
        out["sl"] = sl_price
        out["tp"] = tp_price

        # Strength: trend quality + pullback precision + RSI position
        trend_strength = np.clip(
            (ema_trend - ema_slow).abs() / (atr * 5).replace(0, np.nan), 0, 1
        )
        # How close to the EMA zone (closer = stronger signal)
        pullback_quality = np.clip(
            1.0 - (out["close"] - ema_pb).abs() / (p["pullback_atr"] * atr * 2).replace(0, np.nan),
            0, 1
        )
        base_strength = (trend_strength * 0.5 + pullback_quality * 0.5).clip(0, 1)

        # Macro multipliers
        macro_mult = pd.Series(1.0, index=out.index)

        # VIX: pullbacks in moderate VIX are great
        vix_regime = out.get("vix_regime", pd.Series(1, index=out.index)).fillna(1)
        macro_mult *= pd.Series(
            np.where(vix_regime >= 3, 0.5, np.where(vix_regime == 0, 1.1, 1.0)),
            index=out.index,
        )

        # DXY for forex
        dxy_trend = out.get("dxy_trend", pd.Series(0, index=out.index)).fillna(0)
        if asset in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "EURGBP"):
            if side == "long":
                macro_mult *= np.where(dxy_trend == -1, 1.15, np.where(dxy_trend == 1, 0.8, 1.0))
            else:
                macro_mult *= np.where(dxy_trend == 1, 1.15, np.where(dxy_trend == -1, 0.8, 1.0))
        elif asset in ("USDJPY", "USDCAD"):
            if side == "long":
                macro_mult *= np.where(dxy_trend == 1, 1.15, np.where(dxy_trend == -1, 0.8, 1.0))
            else:
                macro_mult *= np.where(dxy_trend == 1, 0.8, np.where(dxy_trend == -1, 1.15, 1.0))

        # Event risk
        event_risk = out.get("event_risk_score", pd.Series(0, index=out.index)).fillna(0)
        macro_mult *= (1.0 - 0.2 * event_risk)

        out["strength"] = np.where(
            signal,
            np.clip(base_strength * macro_mult, 0, 1),
            0,
        )
        return self._sanitize_signals(out)
