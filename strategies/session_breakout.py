"""
Session Opening Breakout Strategy.

Enters when price breaks out of the prior session's range
during the first N hours of the new session.

Most effective on: London open, New York open.
Suitable regimes: trend_up, trend_down, high_volatility.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Side


class SessionBreakoutStrategy(BaseStrategy):

    name = "session_breakout"
    suitable_regimes = ["trend_up", "trend_down", "high_volatility", "range_bound"]

    def default_params(self) -> Dict:
        return {
            "reference_hours":  8,     # hours to define reference range (2 x 4H bars)
            "breakout_buffer":  0.0,   # price must exceed range by this fraction of ATR
            "sl_atr_mult":      1.5,
            "tp_rr":            7.0,
            "max_entry_hours":  8,     # only take entries within N hours of session open
            "session_open_hour": 7,    # session open in UTC (London = 7)
            "vol_mult":         1.2,
        }

    @property
    def param_schema(self) -> Dict:
        return {
            "reference_hours": {"type": "int",   "low": 4,   "high": 16, "step": 4},
            "sl_atr_mult":     {"type": "float", "low": 0.5, "high": 3.0, "step": 0.5},
            "tp_rr":           {"type": "float", "low": 1.5, "high": 4.0, "step": 0.5},
            "max_entry_hours": {"type": "int",   "low": 4,   "high": 12, "step": 4},
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

        # Reference range: rolling high/low of last `reference_hours` bars
        n = p["reference_hours"]
        ref_high = out["high"].rolling(n).max().shift(1)
        ref_low  = out["low"].rolling(n).min().shift(1)
        buffer   = p["breakout_buffer"] * atr

        # Session time filter
        hour = out.index.hour
        session_open = p["session_open_hour"]
        in_session_window = pd.Series(
            (hour >= session_open) & (hour < session_open + p["max_entry_hours"]),
            index=out.index,
        )

        # Volume confirmation — asset-class aware.
        # yfinance returns 0 volume for most forex pairs, making the filter
        # either trivially pass (0 >= 0) or trivially fail depending on bar.
        # For assets with unreliable volume (median == 0), skip the filter.
        vol_median = out["volume"].rolling(50, min_periods=20).median()
        has_reliable_volume = vol_median.iloc[-1] > 0 if len(vol_median) > 0 else False
        if has_reliable_volume:
            vol_ma = out["volume"].rolling(20).mean()
            vol_ok = out["volume"] >= p["vol_mult"] * vol_ma
        else:
            vol_ok = pd.Series(True, index=out.index)

        # Weekend gap filter: suppress breakout on the first bar after a gap
        # (e.g. Sunday open gap in forex). A gap-up past ref_high isn't a real
        # session breakout — it's a gap fill setup with different dynamics.
        time_delta = out.index.to_series().diff()
        is_gap_bar = time_delta > pd.Timedelta(hours=4)  # >4h gap = market was closed
        not_gap = ~is_gap_bar

        # Higher-timeframe trend alignment filter
        htf_trend = out.get("htf_trend", pd.Series(0, index=out.index))
        htf_strength = out.get("htf_trend_strength", pd.Series(0, index=out.index))

        if side == "long":
            # Only trade long when daily trend is not bearish (neutral or bullish)
            htf_ok = htf_trend >= 0
            breakout = (out["close"] > ref_high + buffer) & in_session_window & vol_ok & htf_ok & not_gap
            sl_price = ref_high - p["sl_atr_mult"] * atr
            sl_dist  = (out["close"] - sl_price).clip(lower=atr * 0.3)
            tp_price = out["close"] + sl_dist * p["tp_rr"]
            direction = 1
        else:
            # Only trade short when daily trend is not bullish
            htf_ok = htf_trend <= 0
            breakout = (out["close"] < ref_low - buffer) & in_session_window & vol_ok & htf_ok & not_gap
            sl_price = ref_low + p["sl_atr_mult"] * atr
            sl_dist  = (sl_price - out["close"]).clip(lower=atr * 0.3)
            tp_price = out["close"] - sl_dist * p["tp_rr"]
            direction = -1

        # One signal per breakout event: suppress consecutive-bar duplicates.
        # If the previous bar also triggered, this bar is a continuation of
        # the same breakout — skip it. Backtested: saves ~44R across portfolio
        # by avoiding worse-priced duplicate entries on the same move.
        breakout_shifted = breakout.shift(1, fill_value=False)
        breakout = breakout & ~breakout_shifted

        out["signal"]   = 0
        out.loc[breakout, "signal"] = direction
        out["sl"]       = sl_price
        out["tp"]       = tp_price
        out["ref_high"] = ref_high
        out["ref_low"]  = ref_low
        # Strength: breakout magnitude * HTF alignment bonus
        base_strength = np.clip((out["close"] - ref_high).abs() / atr.replace(0, np.nan), 0, 1)
        htf_bonus = 1.0 + 0.3 * htf_strength  # up to 30% boost for strong daily trend

        # Macro multipliers
        macro_mult = pd.Series(1.0, index=out.index)

        # VIX: reduce in extreme fear, slight boost in low vol (breakouts more reliable)
        vix_regime = out.get("vix_regime", pd.Series(1, index=out.index)).fillna(1)
        macro_mult *= pd.Series(
            np.where(vix_regime >= 3, 0.5, np.where(vix_regime == 0, 1.1, 1.0)),
            index=out.index,
        )

        # DXY alignment for forex
        dxy_trend = out.get("dxy_trend", pd.Series(0, index=out.index)).fillna(0)
        if asset in ("EURUSD", "GBPUSD", "AUDUSD"):
            if side == "long":
                macro_mult *= np.where(dxy_trend == -1, 1.15, np.where(dxy_trend == 1, 0.8, 1.0))
            else:
                macro_mult *= np.where(dxy_trend == 1, 1.15, np.where(dxy_trend == -1, 0.8, 1.0))
        elif asset == "USDJPY":
            if side == "long":
                macro_mult *= np.where(dxy_trend == 1, 1.15, np.where(dxy_trend == -1, 0.8, 1.0))
            else:
                macro_mult *= np.where(dxy_trend == 1, 0.8, np.where(dxy_trend == -1, 1.15, 1.0))

        # Event risk: avoid breakouts during high-risk periods (false breakouts)
        event_risk = out.get("event_risk_score", pd.Series(0, index=out.index)).fillna(0)
        macro_mult *= (1.0 - 0.4 * event_risk)

        # Volume sentiment: strong directional volume boosts breakout confidence
        vol_sent = out.get("volume_sentiment", pd.Series(0, index=out.index)).fillna(0)
        if side == "long":
            macro_mult *= np.where(vol_sent > 1.5, 1.15, 1.0)
        else:
            macro_mult *= np.where(vol_sent < -1.5, 1.15, 1.0)

        out["strength"] = np.where(
            breakout,
            np.clip(base_strength * htf_bonus * macro_mult, 0, 1),
            0,
        )
        return self._sanitize_signals(out)
