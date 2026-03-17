"""
Asset selector — ranks and selects the best asset to trade
given current regime and recent performance metrics.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.assets import ASSET_UNIVERSE, AssetConfig
from config.regimes import REGIME_DEFINITIONS
from utils.logger import get_logger

log = get_logger(__name__)


class AssetSelector:
    """
    Selects the best asset from the universe for the current regime.

    Scoring factors:
    - Regime suitability per asset class
    - Recent liquidity and volatility
    - Historical performance in this regime
    - Asset-class regime alignment
    """

    # Which asset classes perform best in each regime (rough heuristics)
    REGIME_ASSET_CLASS_AFFINITY: Dict[str, List[str]] = {
        "trend_up":       ["index", "crypto"],
        "trend_down":     ["forex", "commodity"],
        "range_bound":    ["forex"],
        "mean_reverting": ["forex", "commodity"],
        "high_volatility":["crypto", "commodity"],
        "low_volatility": ["forex"],
        "event_risk":     [],
    }

    def __init__(
        self,
        asset_universe: Optional[List[str]] = None,
        min_atr_pct: float = 0.001,    # skip if ATR/price < this
    ):
        self.universe    = asset_universe or list(ASSET_UNIVERSE.keys())
        self.min_atr_pct = min_atr_pct

    def rank(
        self,
        regime: str,
        market_data: Dict[str, pd.DataFrame],
        regime_confidence: float = 0.5,
        performance_history: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rank assets by suitability for the given regime.

        Args:
            regime:              Current market regime label
            market_data:         {symbol: df} with recent OHLCV + features
            regime_confidence:   Regime detection confidence [0, 1]
            performance_history: {symbol: recent_profit_factor} from backtesting

        Returns:
            List of (symbol, score) sorted by descending score
        """
        preferred_classes = self.REGIME_ASSET_CLASS_AFFINITY.get(regime, [])
        scores: Dict[str, float] = {}

        for sym in self.universe:
            if sym not in market_data:
                continue
            cfg = ASSET_UNIVERSE.get(sym)
            if cfg is None:
                continue

            df = market_data[sym]
            if df.empty:
                continue

            score = 0.0

            # 1. Asset class affinity
            if cfg.asset_class in preferred_classes:
                idx = preferred_classes.index(cfg.asset_class)
                score += (len(preferred_classes) - idx) * 2.0

            # 2. Recent liquidity (volatility proxy)
            if "atr_norm" in df.columns:
                atr_pct = float(df["atr_norm"].iloc[-20:].mean())
            elif "atr" in df.columns:
                atr_pct = float(df["atr"].iloc[-20:].mean() / df["close"].iloc[-1])
            else:
                atr_pct = 0.0

            if atr_pct < self.min_atr_pct:
                scores[sym] = -1.0   # too quiet
                continue
            score += min(atr_pct * 100, 5.0)   # cap contribution

            # 3. Historical performance in this regime
            if performance_history and sym in performance_history:
                pf = performance_history[sym]
                score += max(0, (pf - 1.0) * 3.0)

            # 4. Spread penalty
            score -= cfg.typical_spread_pips * 0.1

            scores[sym] = score * regime_confidence

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        log.debug("Asset ranking for regime '%s': %s", regime, ranked[:5])
        return ranked

    def select(
        self,
        regime: str,
        market_data: Dict[str, pd.DataFrame],
        **kwargs,
    ) -> Optional[str]:
        """Return the top-ranked asset, or None if none qualify."""
        ranked = self.rank(regime, market_data, **kwargs)
        valid  = [(sym, score) for sym, score in ranked if score > 0]
        if not valid:
            log.warning("No suitable assets found for regime '%s'", regime)
            return None
        best = valid[0][0]
        log.info("Selected asset '%s' (score=%.2f) for regime '%s'",
                 best, valid[0][1], regime)
        return best
