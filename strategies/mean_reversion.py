"""
Mean Reversion Strategy (Bollinger Bands + RSI).

Entry:
  - Price touches or crosses lower BB (long) / upper BB (short)
  - RSI oversold/overbought confirmation
  - ADX below threshold (ranging market)
  - Volume spike (shock entry into BB band)

Exit:
  - SL: ATR-based beyond band
  - TP: Bollinger mid-band OR fixed R:R
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Side


class MeanReversionStrategy(BaseStrategy):

    name = "mean_reversion"
    suitable_regimes = ["range_bound", "mean_reverting", "low_volatility"]

    def default_params(self) -> Dict:
        return {
            "bb_period":   20,
            "bb_std":      2.0,
            "rsi_period":  14,
            "rsi_os":      32.0,     # RSI oversold for long entry
            "rsi_ob":      68.0,     # RSI overbought for short entry
            "adx_max":     28.0,     # max ADX — slight relaxation from 25
            "sl_atr_mult": 1.5,      # SL beyond BB band
            "tp_rr":       2.0,      # TP as R:R
            "use_midband_tp": True,  # True = TP at BB midband (higher WR)
            "vol_mult":    1.2,      # volume confirmation (require some conviction)
        }

    @property
    def param_schema(self) -> Dict:
        return {
            "bb_period":   {"type": "int",   "low": 10,  "high": 50, "step": 5},
            "bb_std":      {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5},
            "rsi_os":      {"type": "float", "low": 20,  "high": 40, "step": 5},
            "rsi_ob":      {"type": "float", "low": 60,  "high": 80, "step": 5},
            "adx_max":     {"type": "float", "low": 15,  "high": 35, "step": 5},
            "sl_atr_mult": {"type": "float", "low": 0.5, "high": 2.5, "step": 0.5},
            "tp_rr":       {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5},
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

        # Bollinger Bands
        ma  = out["close"].rolling(p["bb_period"]).mean()
        std = out["close"].rolling(p["bb_period"]).std()
        bb_upper = ma + p["bb_std"] * std
        bb_lower = ma - p["bb_std"] * std

        # RSI
        if "rsi" in out.columns:
            rsi = out["rsi"]
        else:
            delta = out["close"].diff()
            gain  = delta.clip(lower=0).ewm(span=p["rsi_period"], adjust=False).mean()
            loss  = (-delta.clip(upper=0)).ewm(span=p["rsi_period"], adjust=False).mean()
            rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        # ADX filter
        adx = out.get("adx", pd.Series(0, index=out.index))
        adx_ok = (adx <= p["adx_max"]) | (adx == 0)

        # Volume filter
        if p["vol_mult"] > 0:
            vol_ma = out["volume"].rolling(20).mean()
            vol_ok = out["volume"] >= p["vol_mult"] * vol_ma
        else:
            vol_ok = pd.Series(True, index=out.index)

        atr = self._atr(out)

        # Momentum divergence filter — prefer entries with confirming divergence
        mom_div = out.get("mom_divergence", pd.Series(0, index=out.index))

        # HTF trend — avoid mean reversion against strong daily trends
        htf_trend = out.get("htf_trend", pd.Series(0, index=out.index))
        htf_strength = out.get("htf_trend_strength", pd.Series(0, index=out.index))

        if side == "long":
            # Entry: price at or below lower BB + RSI oversold
            bb_touch = out["low"] <= bb_lower
            rsi_ok   = rsi <= p["rsi_os"]
            # Block long MR entries during strong bearish daily trend
            htf_ok   = ~((htf_trend == -1) & (htf_strength > 0.4))
            signal   = bb_touch & rsi_ok & adx_ok & vol_ok & htf_ok
            sl_price = bb_lower - p["sl_atr_mult"] * atr
            if p["use_midband_tp"]:
                tp_price = ma   # target: mean
            else:
                sl_dist  = (out["close"] - sl_price).clip(lower=atr * 0.3)
                tp_price = out["close"] + sl_dist * p["tp_rr"]
            direction = 1

        else:
            # Entry: price at or above upper BB + RSI overbought
            bb_touch = out["high"] >= bb_upper
            rsi_ok   = rsi >= p["rsi_ob"]
            # Block short MR entries during strong bullish daily trend
            htf_ok   = ~((htf_trend == 1) & (htf_strength > 0.4))
            signal   = bb_touch & rsi_ok & adx_ok & vol_ok & htf_ok
            sl_price = bb_upper + p["sl_atr_mult"] * atr
            if p["use_midband_tp"]:
                tp_price = ma
            else:
                sl_dist  = (sl_price - out["close"]).clip(lower=atr * 0.3)
                tp_price = out["close"] - sl_dist * p["tp_rr"]
            direction = -1

        out["signal"]   = 0
        out.loc[signal, "signal"] = direction
        out["sl"]       = sl_price
        out["tp"]       = tp_price
        out["bb_upper"] = bb_upper
        out["bb_lower"] = bb_lower
        out["bb_mid"]   = ma
        # Base strength: range quality (low ADX) + RSI extremity
        base_str = np.clip(
            (adx.rsub(p["adx_max"]) / max(p["adx_max"], 1e-9)) *
            abs(rsi - 50) / 50,
            0, 1,
        )
        # Boost strength by 30% when momentum divergence confirms the entry
        div_bonus = np.where(
            (side == "long") & (mom_div == 1) | (side == "short") & (mom_div == -1),
            1.3, 1.0,
        )

        # Macro multipliers for mean reversion
        macro_mult = pd.Series(1.0, index=out.index)

        # VIX: elevated VIX = MORE mean-reversion (overextended moves snap back)
        vix_regime = out.get("vix_regime", pd.Series(1, index=out.index)).fillna(1)
        macro_mult *= pd.Series(
            np.where(vix_regime >= 2, 1.2, np.where(vix_regime == 0, 0.8, 1.0)),
            index=out.index,
        )

        # Fear/greed: MR works best when sentiment is extreme (contrarian)
        fg = out.get("fear_greed_proxy", pd.Series(0.5, index=out.index)).fillna(0.5)
        if side == "long":
            # Buy when fearful (low fear/greed)
            macro_mult *= np.where(fg < 0.3, 1.2, np.where(fg > 0.7, 0.8, 1.0))
        else:
            # Sell when greedy (high fear/greed)
            macro_mult *= np.where(fg > 0.7, 1.2, np.where(fg < 0.3, 0.8, 1.0))

        # Event risk: MR during events is risky (moves can extend)
        event_risk = out.get("event_risk_score", pd.Series(0, index=out.index)).fillna(0)
        macro_mult *= (1.0 - 0.25 * event_risk)

        out["strength"] = np.where(
            signal,
            np.clip(base_str * div_bonus * macro_mult, 0, 1),
            0,
        )
        return self._sanitize_signals(out)
