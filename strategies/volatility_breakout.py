"""
Volatility Breakout Strategy.

Enters when ATR expands significantly after a compression period.
Suitable for: high_volatility, low_volatility (anticipatory breakout).

Entry:
  - ATR ratio (current / historical avg) expands above threshold
  - Price breaks out of compression range (Donchian)
  - Volume spike confirms momentum

Exit:
  - Wide SL: compression range low/high - ATR buffer
  - TP: fixed R:R OR ATR expansion target
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Side
from features.alpha_features import add_donchian


class VolatilityBreakoutStrategy(BaseStrategy):

    name = "volatility_breakout"
    suitable_regimes = ["high_volatility", "low_volatility"]

    def default_params(self) -> Dict:
        return {
            "atr_ratio_min":   1.2,    # lowered from 1.3 for more signals
            "atr_ratio_max":   8.0,    # raised from 5.0 — allow larger moves (crypto/commodities)
            "compression_bars": 15,    # lowered from 20 for faster detection
            "don_period":      10,     # Donchian for breakout level
            "sl_atr_mult":     1.5,    # tightened from 2.0 for better R:R
            "tp_rr":           2.5,    # increased from 2.0
            "vol_mult":        1.5,    # lowered from 2.0 for more signals
            "min_range_pct":   0.001,  # lowered from 0.002 — allow smaller compressions
        }

    @property
    def param_schema(self) -> Dict:
        return {
            "atr_ratio_min":   {"type": "float", "low": 1.1, "high": 2.5, "step": 0.2},
            "compression_bars":{"type": "int",   "low": 10,  "high": 40,  "step": 5},
            "sl_atr_mult":     {"type": "float", "low": 1.0, "high": 3.5, "step": 0.5},
            "tp_rr":           {"type": "float", "low": 1.5, "high": 4.0, "step": 0.5},
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

        atr    = self._atr(out)
        atr_ma = atr.rolling(p["compression_bars"]).mean()

        # ATR expansion
        atr_ratio   = atr / atr_ma.replace(0, np.nan)
        expanding   = (atr_ratio >= p["atr_ratio_min"]) & (atr_ratio <= p["atr_ratio_max"])

        # Donchian breakout
        don_up = out[f"donch_upper_{p['don_period']}"].shift(1)
        don_lo = out[f"donch_lower_{p['don_period']}"].shift(1)

        # Volume spike
        vol_ma = out["volume"].rolling(20).mean()
        vol_ok = out["volume"] >= p["vol_mult"] * vol_ma

        # Minimum range size (filter micro-moves)
        range_ok = (don_up - don_lo) / out["close"] >= p["min_range_pct"]

        if side == "long":
            breakout = (out["close"] > don_up) & expanding & vol_ok & range_ok
            sl_price = don_up - p["sl_atr_mult"] * atr
            sl_dist  = (out["close"] - sl_price).clip(lower=atr * 0.5)
            tp_price = out["close"] + sl_dist * p["tp_rr"]
            direction = 1
        else:
            breakout = (out["close"] < don_lo) & expanding & vol_ok & range_ok
            sl_price = don_lo + p["sl_atr_mult"] * atr
            sl_dist  = (sl_price - out["close"]).clip(lower=atr * 0.5)
            tp_price = out["close"] - sl_dist * p["tp_rr"]
            direction = -1

        out["signal"]    = 0
        out.loc[breakout, "signal"] = direction
        out["sl"]        = sl_price
        out["tp"]        = tp_price
        out["atr_ratio"] = atr_ratio
        denom = max(p["atr_ratio_max"] - p["atr_ratio_min"], 1e-9)
        base_strength = np.clip((atr_ratio - p["atr_ratio_min"]) / denom, 0, 1)

        # Macro multipliers
        macro_mult = pd.Series(1.0, index=out.index)

        # VIX: vol breakouts work well when VIX is rising (momentum)
        vix_change = out.get("vix_change_5", pd.Series(0, index=out.index)).fillna(0)
        macro_mult *= np.where(vix_change > 0.1, 1.15, np.where(vix_change < -0.15, 0.85, 1.0))

        # Event risk: vol breakouts during events can be powerful OR traps
        event_risk = out.get("event_risk_score", pd.Series(0, index=out.index)).fillna(0)
        macro_mult *= (1.0 - 0.2 * event_risk)

        out["strength"] = np.where(breakout, np.clip(base_strength * macro_mult, 0, 1), 0)
        return self._sanitize_signals(out)
