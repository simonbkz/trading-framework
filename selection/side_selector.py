"""
Side selector — determines whether to trade LONG, SHORT, or both.

Takes regime direction bias and current market context into account.
Does NOT assume long == short symmetry.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from config.regimes import REGIME_DEFINITIONS
from utils.logger import get_logger

log = get_logger(__name__)


class SideSelector:
    """
    Decides which trade side(s) to evaluate for a given regime and asset.
    """

    def select(
        self,
        regime: str,
        df: pd.DataFrame,
        regime_proba: Optional[dict] = None,
    ) -> List[str]:
        """
        Return a list of side strings to try: ["long"], ["short"], or ["long","short"].

        Args:
            regime:       Current regime label
            df:           Recent OHLCV + features DataFrame
            regime_proba: {regime_label: probability} for nuanced routing

        Returns:
            List of sides to evaluate (never empty)
        """
        reg_def = REGIME_DEFINITIONS.get(regime)

        if reg_def is None:
            log.warning("Unknown regime '%s' — defaulting to both sides", regime)
            return ["long", "short"]

        bias = reg_def.direction_bias   # "long" | "short" | "both"

        # Hard regime direction
        if bias == "long":
            sides = ["long"]
        elif bias == "short":
            sides = ["short"]
        else:
            # "both" — use secondary signals to narrow down
            sides = self._narrow_by_context(df)

        log.debug("SideSelector: regime=%s bias=%s -> sides=%s", regime, bias, sides)
        return sides

    def _narrow_by_context(self, df: pd.DataFrame) -> List[str]:
        """
        When regime is "both", use short-term indicators to prefer one side.
        Falls back to both if signals are ambiguous.
        """
        if df.empty:
            return ["long", "short"]

        last = df.iloc[-1]
        signals = []

        # EMA trend
        if "ema_200_dist" in df.columns or "ema200_dist" in df.columns:
            col = "ema200_dist" if "ema200_dist" in df.columns else "ema_200_dist"
            dist = last.get(col, 0)
            signals.append("long" if dist > 0 else "short")

        # RSI
        if "rsi" in df.columns:
            rsi = last.get("rsi", 50)
            if rsi > 55:
                signals.append("long")
            elif rsi < 45:
                signals.append("short")

        # DI lines
        if "di_plus" in df.columns and "di_minus" in df.columns:
            signals.append("long" if last["di_plus"] > last["di_minus"] else "short")

        if not signals:
            return ["long", "short"]

        # Majority vote
        long_votes  = signals.count("long")
        short_votes = signals.count("short")

        if long_votes > short_votes:
            return ["long"]
        elif short_votes > long_votes:
            return ["short"]
        else:
            return ["long", "short"]
