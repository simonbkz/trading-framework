"""
Momentum Trend-Following Strategy.

Captures sustained directional moves using EMA alignment + MACD momentum.
Designed for high trade frequency across all asset classes.

Entry (Long):
  - EMA fast > EMA medium > EMA slow (trend alignment)
  - MACD histogram positive and rising (momentum accelerating)
  - Close above EMA fast (price in trend)
  - ATR above minimum threshold (enough volatility)

Entry (Short): Mirror conditions

Exit:
  - SL: swing low - ATR * sl_mult (long) / swing high + ATR * sl_mult (short)
  - TP: entry + sl_dist * tp_rr

Key: This strategy generates MORE signals than breakout strategies
because it enters on trend continuation, not just breakouts.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Side


class MomentumTrendStrategy(BaseStrategy):
    """EMA alignment + MACD momentum trend-following strategy."""

    name = "momentum_trend"
    suitable_regimes = ["trend_up", "trend_down", "high_volatility"]

    def default_params(self) -> Dict:
        return {
            "ema_fast":       9,
            "ema_medium":     21,
            "ema_slow":       50,
            "macd_fast":      12,
            "macd_slow":      26,
            "macd_signal":    9,
            "sl_atr_mult":    1.5,
            "tp_rr":          2.5,
            "adx_min":        18.0,   # minimum ADX for entry (lower than breakout)
            "swing_lookback": 5,      # bars to find swing low/high for SL
        }

    @property
    def param_schema(self) -> Dict:
        return {
            "ema_fast":    {"type": "int",   "low": 5,   "high": 15,  "step": 2},
            "ema_medium":  {"type": "int",   "low": 15,  "high": 30,  "step": 5},
            "ema_slow":    {"type": "int",   "low": 40,  "high": 80,  "step": 10},
            "sl_atr_mult": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.5},
            "tp_rr":       {"type": "float", "low": 1.5, "high": 4.0, "step": 0.5},
            "adx_min":     {"type": "float", "low": 10,  "high": 30,  "step": 5},
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

        # EMAs
        ema_f = out["close"].ewm(span=p["ema_fast"], adjust=False).mean()
        ema_m = out["close"].ewm(span=p["ema_medium"], adjust=False).mean()
        ema_s = out["close"].ewm(span=p["ema_slow"], adjust=False).mean()

        # MACD
        macd_line = (
            out["close"].ewm(span=p["macd_fast"], adjust=False).mean()
            - out["close"].ewm(span=p["macd_slow"], adjust=False).mean()
        )
        macd_signal = macd_line.ewm(span=p["macd_signal"], adjust=False).mean()
        macd_hist = macd_line - macd_signal
        macd_hist_rising = macd_hist > macd_hist.shift(1)

        # ADX filter
        adx = out.get("adx", pd.Series(25, index=out.index))
        adx_ok = adx >= p["adx_min"]

        # HTF trend alignment
        htf_trend = out.get("htf_trend", pd.Series(0, index=out.index))

        # Swing high/low for SL placement
        swing_lo = out["low"].rolling(p["swing_lookback"]).min()
        swing_hi = out["high"].rolling(p["swing_lookback"]).max()

        if side == "long":
            # EMA stack: fast > medium > slow
            ema_aligned = (ema_f > ema_m) & (ema_m > ema_s)
            # Price above fast EMA
            price_ok = out["close"] > ema_f
            # MACD positive and rising
            macd_ok = (macd_hist > 0) & macd_hist_rising
            # HTF not bearish
            htf_ok = htf_trend >= 0

            signal = ema_aligned & price_ok & macd_ok & adx_ok & htf_ok

            sl_price = swing_lo - p["sl_atr_mult"] * atr
            sl_dist = (out["close"] - sl_price).clip(lower=atr * 0.3)
            tp_price = out["close"] + sl_dist * p["tp_rr"]
            direction = 1
        else:
            # Short: EMA stack reversed
            ema_aligned = (ema_f < ema_m) & (ema_m < ema_s)
            price_ok = out["close"] < ema_f
            macd_ok = (macd_hist < 0) & (~macd_hist_rising)  # negative and falling
            htf_ok = htf_trend <= 0

            signal = ema_aligned & price_ok & macd_ok & adx_ok & htf_ok

            sl_price = swing_hi + p["sl_atr_mult"] * atr
            sl_dist = (sl_price - out["close"]).clip(lower=atr * 0.3)
            tp_price = out["close"] - sl_dist * p["tp_rr"]
            direction = -1

        out["signal"] = 0
        out.loc[signal, "signal"] = direction
        out["sl"] = sl_price
        out["tp"] = tp_price

        # Strength: ADX normalized + MACD magnitude + EMA spread
        adx_norm = np.clip((adx - p["adx_min"]) / 30, 0, 1)
        macd_norm = np.clip(
            macd_hist.abs() / (atr * out["close"] / 1000).replace(0, np.nan), 0, 1
        )
        ema_spread = np.clip(
            (ema_f - ema_s).abs() / (atr * 3).replace(0, np.nan), 0, 1
        )

        base_strength = (
            adx_norm * 0.4 + macd_norm * 0.3 + ema_spread * 0.3
        ).clip(0, 1)

        # Macro multipliers
        macro_mult = pd.Series(1.0, index=out.index)

        # VIX: trending markets work well with moderate fear
        vix_regime = out.get(
            "vix_regime", pd.Series(1, index=out.index)
        ).fillna(1)
        macro_mult *= pd.Series(
            np.where(vix_regime >= 3, 0.6, np.where(vix_regime == 0, 1.1, 1.0)),
            index=out.index,
        )

        # DXY alignment for forex
        dxy_trend = out.get(
            "dxy_trend", pd.Series(0, index=out.index)
        ).fillna(0)
        if asset in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "EURGBP"):
            if side == "long":
                macro_mult *= np.where(
                    dxy_trend == -1, 1.15, np.where(dxy_trend == 1, 0.8, 1.0)
                )
            else:
                macro_mult *= np.where(
                    dxy_trend == 1, 1.15, np.where(dxy_trend == -1, 0.8, 1.0)
                )
        elif asset in ("USDJPY", "USDCAD"):
            if side == "long":
                macro_mult *= np.where(
                    dxy_trend == 1, 1.15, np.where(dxy_trend == -1, 0.8, 1.0)
                )
            else:
                macro_mult *= np.where(
                    dxy_trend == 1, 0.8, np.where(dxy_trend == -1, 1.15, 1.0)
                )

        # Fear/greed: momentum works best with greed
        fg = out.get(
            "fear_greed_proxy", pd.Series(0.5, index=out.index)
        ).fillna(0.5)
        macro_mult *= (0.8 + 0.4 * fg)

        # Event risk dampening
        event_risk = out.get(
            "event_risk_score", pd.Series(0, index=out.index)
        ).fillna(0)
        macro_mult *= (1.0 - 0.3 * event_risk)

        out["strength"] = np.where(
            signal,
            np.clip(base_strength * macro_mult, 0, 1),
            0,
        )
        return self._sanitize_signals(out)
