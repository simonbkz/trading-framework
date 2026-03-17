"""
Strategy selector — chooses the best strategy for the current regime and asset.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from config.regimes import preferred_strategies
from strategies.strategy_factory import STRATEGY_REGISTRY
from optimization.parameter_store import ParameterStore
from utils.logger import get_logger

log = get_logger(__name__)


class StrategySelector:
    """
    Selects the best strategy given regime, asset, and available parameters.

    Priority order:
    1. Strategy has optimised params in ParameterStore for this (asset, regime)
    2. Strategy is in the regime's preferred list
    3. Strategy has any stored params at all for this regime
    4. Fallback to regime default
    """

    def __init__(
        self,
        parameter_store: Optional[ParameterStore] = None,
    ):
        self.param_store = parameter_store

    def rank(
        self,
        regime: str,
        asset: str,
        side: str = "long",
        performance_history: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return list of (strategy_name, score) sorted by descending suitability.
        """
        preferred = preferred_strategies(regime)
        scores: Dict[str, float] = {}

        for strategy_name in STRATEGY_REGISTRY:
            score = 0.0

            # 1. Regime affinity
            if strategy_name in preferred:
                rank_bonus = (len(preferred) - preferred.index(strategy_name)) * 3.0
                score += rank_bonus

            # 2. Has optimised params in store
            if self.param_store:
                entry = self.param_store.get_entry(asset, regime, strategy_name, side)
                if entry:
                    score += entry.get("score", 0) * 0.5
                    score += 2.0   # bonus for having any params at all

            # 3. Historical performance
            if performance_history and strategy_name in performance_history:
                score += max(0, (performance_history[strategy_name] - 1.0) * 2.0)

            scores[strategy_name] = score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def select(
        self,
        regime: str,
        asset: str,
        side: str = "long",
        **kwargs,
    ) -> Optional[str]:
        ranked = self.rank(regime, asset, side, **kwargs)
        if not ranked:
            return None
        best = ranked[0][0]
        log.info(
            "Selected strategy '%s' (score=%.2f) for %s/%s/%s",
            best, ranked[0][1], asset, regime, side,
        )
        return best
