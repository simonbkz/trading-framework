"""
Regime definitions, labels, and regime→strategy routing table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Canonical regime identifiers
# ---------------------------------------------------------------------------
TREND_UP         = "trend_up"
TREND_DOWN       = "trend_down"
RANGE_BOUND      = "range_bound"
MEAN_REVERTING   = "mean_reverting"
HIGH_VOLATILITY  = "high_volatility"
LOW_VOLATILITY   = "low_volatility"
EVENT_RISK       = "event_risk"
CHOPPY           = "choppy"              # low ADX, no clear direction

ALL_REGIMES: List[str] = [
    TREND_UP,
    TREND_DOWN,
    RANGE_BOUND,
    MEAN_REVERTING,
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    EVENT_RISK,
    CHOPPY,
]


@dataclass
class RegimeDefinition:
    label: str
    description: str
    preferred_strategies: List[str]    # ordered by priority
    tradable: bool = True              # False → skip trading entirely
    direction_bias: str = "both"       # long | short | both


REGIME_DEFINITIONS: Dict[str, RegimeDefinition] = {
    # All regimes use proven strategies — let strategy-level filters decide
    TREND_UP: RegimeDefinition(
        label=TREND_UP,
        description="Sustained upward price trend with momentum",
        preferred_strategies=["session_breakout"],
        direction_bias="both",
    ),
    TREND_DOWN: RegimeDefinition(
        label=TREND_DOWN,
        description="Sustained downward price trend with strong momentum",
        preferred_strategies=["session_breakout"],
        direction_bias="both",
    ),
    RANGE_BOUND: RegimeDefinition(
        label=RANGE_BOUND,
        description="Price oscillating between support and resistance",
        preferred_strategies=["session_breakout"],
        direction_bias="both",
    ),
    MEAN_REVERTING: RegimeDefinition(
        label=MEAN_REVERTING,
        description="Strong mean-reversion dynamics, overshoots snap back",
        preferred_strategies=["session_breakout"],
        direction_bias="both",
    ),
    HIGH_VOLATILITY: RegimeDefinition(
        label=HIGH_VOLATILITY,
        description="Elevated volatility — larger moves, wider spreads",
        preferred_strategies=["session_breakout"],
        direction_bias="both",
    ),
    LOW_VOLATILITY: RegimeDefinition(
        label=LOW_VOLATILITY,
        description="Compressed volatility — potential breakout building",
        preferred_strategies=["session_breakout"],
        direction_bias="both",
    ),
    EVENT_RISK: RegimeDefinition(
        label=EVENT_RISK,
        description="High-impact news imminent",
        preferred_strategies=["session_breakout"],
        tradable=True,
        direction_bias="both",
    ),
    CHOPPY: RegimeDefinition(
        label=CHOPPY,
        description="Low ADX, no directional conviction",
        preferred_strategies=["session_breakout"],
        tradable=True,
        direction_bias="both",
    ),
}


def get_regime_def(label: str) -> RegimeDefinition:
    if label not in REGIME_DEFINITIONS:
        raise ValueError(f"Unknown regime '{label}'")
    return REGIME_DEFINITIONS[label]


def preferred_strategies(regime: str) -> List[str]:
    return REGIME_DEFINITIONS.get(regime, RegimeDefinition("", "", [])).preferred_strategies


def is_tradable(regime: str) -> bool:
    return REGIME_DEFINITIONS.get(regime, RegimeDefinition("", "", [])).tradable
