"""
Strategy factory — creates strategy instances by name.
Also manages the parameter registry (per asset/regime/strategy/side).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

from strategies.base_strategy import BaseStrategy, Side
from strategies.trend_breakout import TrendBreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy
from strategies.session_breakout import SessionBreakoutStrategy
from strategies.momentum_trend import MomentumTrendStrategy
from strategies.pullback_retest import PullbackRetestStrategy
from utils.logger import get_logger

log = get_logger(__name__)


# Registry of available strategies
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "trend_breakout":    TrendBreakoutStrategy,
    "mean_reversion":    MeanReversionStrategy,
    "volatility_breakout": VolatilityBreakoutStrategy,
    "session_breakout":  SessionBreakoutStrategy,
    "momentum_trend":    MomentumTrendStrategy,
    "pullback_retest":   PullbackRetestStrategy,
}


def get_strategy(
    name: str,
    params_long: Optional[Dict] = None,
    params_short: Optional[Dict] = None,
) -> BaseStrategy:
    """Instantiate a strategy by name with optional parameters."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    cls = STRATEGY_REGISTRY[name]
    return cls(params_long=params_long, params_short=params_short)


def list_strategies() -> List[str]:
    return list(STRATEGY_REGISTRY.keys())


def strategies_for_regime(regime: str) -> List[str]:
    """Return strategies suitable for a given regime."""
    from config.regimes import preferred_strategies
    return preferred_strategies(regime)


class ParameterRegistry:
    """
    Stores optimised parameter sets indexed by (asset, regime, strategy, side).

    Schema:
        registry[asset][regime][strategy][side] = {param_dict}
    """

    def __init__(self):
        self._store: Dict = {}

    def set(
        self,
        asset: str,
        regime: str,
        strategy: str,
        side: Side,
        params: Dict,
    ) -> None:
        self._store.setdefault(asset, {}) \
                   .setdefault(regime, {}) \
                   .setdefault(strategy, {})[side] = params

    def get(
        self,
        asset: str,
        regime: str,
        strategy: str,
        side: Side,
        fallback_to_defaults: bool = True,
    ) -> Optional[Dict]:
        try:
            return self._store[asset][regime][strategy][side]
        except KeyError:
            pass

        # Try asset-agnostic fallback
        try:
            return self._store["*"][regime][strategy][side]
        except KeyError:
            pass

        if fallback_to_defaults:
            log.debug(
                "No params found for %s/%s/%s/%s — using defaults",
                asset, regime, strategy, side,
            )
            strat = get_strategy(strategy)
            return strat.get_params(side)

        return None

    def get_strategy_with_params(
        self,
        asset: str,
        regime: str,
        strategy: str,
        side: Side,
    ) -> BaseStrategy:
        """Return a strategy instance pre-loaded with the best known params."""
        params = self.get(asset, regime, strategy, side)
        return get_strategy(strategy, params_long=params, params_short=params)

    def save(self, path: str) -> None:
        from utils.helpers import save_json
        save_json(self._store, path)
        log.info("ParameterRegistry saved to %s", path)

    def load(self, path: str) -> "ParameterRegistry":
        from utils.helpers import load_json
        self._store = load_json(path)
        log.info("ParameterRegistry loaded from %s", path)
        return self

    def keys(self) -> List[Tuple]:
        """Return all (asset, regime, strategy, side) tuples in registry."""
        keys = []
        for asset, regimes in self._store.items():
            for regime, strategies in regimes.items():
                for strategy, sides in strategies.items():
                    for side in sides:
                        keys.append((asset, regime, strategy, side))
        return keys

    def summary(self) -> str:
        rows = [f"  {'/'.join(k)}" for k in self.keys()]
        return "ParameterRegistry ({} entries):\n{}".format(len(rows), "\n".join(rows))
