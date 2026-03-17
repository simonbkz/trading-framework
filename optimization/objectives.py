"""
Optimization objective functions.

Each function takes a trades DataFrame and returns a scalar score to maximise.
A higher score is always better.

Penalties are applied for:
- Too few trades (< min_trades)
- Extreme overtrading
- Unstable parameter sets (checked externally via perturbation)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.helpers import safe_div, max_drawdown, annualised_return, sharpe_ratio
from validation.metrics import compute_trade_metrics, compute_equity_metrics


def objective_calmar(
    trades: pd.DataFrame,
    equity: pd.Series,
    min_trades: int = 20,
    penalty_weight: float = 1.0,
) -> float:
    """
    Calmar ratio = CAGR / MaxDrawdown.
    Penalised for insufficient trades.
    """
    if len(trades) < min_trades:
        return -1.0 * penalty_weight

    em = compute_equity_metrics(equity)
    mdd = em["max_drawdown"]
    if mdd == 0:
        return 0.0
    score = em["cagr"] / mdd
    return float(score)


def objective_sharpe(
    trades: pd.DataFrame,
    equity: pd.Series,
    min_trades: int = 20,
    penalty_weight: float = 1.0,
) -> float:
    if len(trades) < min_trades:
        return -5.0 * penalty_weight
    returns = equity.pct_change().dropna()
    return float(sharpe_ratio(returns))


def objective_sortino(
    trades: pd.DataFrame,
    equity: pd.Series,
    min_trades: int = 20,
) -> float:
    from utils.helpers import sortino_ratio
    if len(trades) < min_trades:
        return -5.0
    returns = equity.pct_change().dropna()
    return float(sortino_ratio(returns))


def objective_profit_factor(
    trades: pd.DataFrame,
    equity: pd.Series,
    min_trades: int = 20,
    max_pf: float = 5.0,
) -> float:
    if len(trades) < min_trades:
        return 0.0
    tm = compute_trade_metrics(trades)
    return float(min(tm["profit_factor"], max_pf))


def objective_composite(
    trades: pd.DataFrame,
    equity: pd.Series,
    min_trades: int = 20,
    sharpe_weight: float = 0.4,
    calmar_weight: float = 0.4,
    pf_weight:     float = 0.2,
) -> float:
    """
    Weighted composite of Sharpe, Calmar, and Profit Factor.
    All components normalised to comparable scales.
    """
    if len(trades) < min_trades:
        return -10.0

    em  = compute_equity_metrics(equity)
    tm  = compute_trade_metrics(trades)

    sharpe  = em["sharpe_ratio"]
    calmar  = em["calmar_ratio"]
    pf      = min(tm["profit_factor"], 5.0) - 1.0  # center around 0

    score = (sharpe_weight * sharpe +
             calmar_weight * calmar +
             pf_weight     * pf)
    return float(score)


OBJECTIVE_MAP = {
    "calmar":         objective_calmar,
    "sharpe":         objective_sharpe,
    "sortino":        objective_sortino,
    "profit_factor":  objective_profit_factor,
    "composite":      objective_composite,
}


def get_objective(name: str):
    if name not in OBJECTIVE_MAP:
        raise ValueError(f"Unknown objective '{name}'. Available: {list(OBJECTIVE_MAP)}")
    return OBJECTIVE_MAP[name]
