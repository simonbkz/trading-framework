"""
Liquidity filter — detects and blocks thin / illiquid market conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


class LiquidityFilter:
    """
    Flags bars where market conditions are too thin to trade safely.

    Heuristics:
    - Very low volume relative to recent average
    - Very narrow spread / range (hibernation)
    - Weekend or holiday periods (zero volume / missing bars)
    """

    def __init__(
        self,
        min_volume_ratio: float = 0.20,    # reject if volume < 20% of 20-bar avg
        min_atr_ratio: float = 0.15,       # reject if ATR < 15% of 50-bar avg
        exclude_weekends: bool = True,
    ):
        self.min_volume_ratio  = min_volume_ratio
        self.min_atr_ratio     = min_atr_ratio
        self.exclude_weekends  = exclude_weekends

    def is_liquid(self, df: pd.DataFrame) -> pd.Series:
        """
        Return boolean Series — True where market is liquid enough to trade.

        Requires: volume, high, low, (optionally) atr columns in df.
        """
        liquid = pd.Series(True, index=df.index)

        # Weekend check
        if self.exclude_weekends:
            dow = df.index.dayofweek   # Mon=0, Sat=5, Sun=6
            liquid &= (dow < 5)

        # Volume check
        if "volume" in df.columns and df["volume"].sum() > 0:
            vol_ma = df["volume"].rolling(20, min_periods=5).mean()
            vol_ratio = df["volume"] / vol_ma.replace(0, np.nan)
            too_thin = vol_ratio < self.min_volume_ratio
            liquid &= ~too_thin.fillna(False)

        # ATR / range check
        if "atr" in df.columns:
            atr_ma = df["atr"].rolling(50, min_periods=10).mean()
            atr_ratio = df["atr"] / atr_ma.replace(0, np.nan)
            too_narrow = atr_ratio < self.min_atr_ratio
            liquid &= ~too_narrow.fillna(False)
        elif "high" in df.columns and "low" in df.columns:
            bar_range = df["high"] - df["low"]
            range_ma  = bar_range.rolling(50, min_periods=10).mean()
            range_rat = bar_range / range_ma.replace(0, np.nan)
            too_narrow = range_rat < self.min_atr_ratio
            liquid &= ~too_narrow.fillna(False)

        n_blocked = (~liquid).sum()
        if n_blocked > 0:
            log.debug("LiquidityFilter: blocked %d / %d bars", n_blocked, len(df))

        return liquid
