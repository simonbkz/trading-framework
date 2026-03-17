"""
Performance metrics for backtests and live monitoring.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import math

from utils.helpers import (
    safe_div, sharpe_ratio, sortino_ratio, max_drawdown,
    calmar_ratio, annualised_return,
)


def _safe_round(val: float, ndigits: int = 4, fallback: float = 0.0) -> float:
    """Round a value, replacing NaN/inf with fallback."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return fallback
    return round(float(val), ndigits)


def compute_trade_metrics(trades: pd.DataFrame) -> Dict:
    """
    Compute full trade-level metrics.

    Args:
        trades: DataFrame with columns:
            entry_time, exit_time, side, entry, exit, pnl_pct,
            (optional) regime, strategy, asset

    Returns:
        Dict of metric names → values
    """
    if trades.empty:
        return _empty_metrics()

    pnl = trades["pnl_pct"].dropna()
    n   = len(pnl)
    wins    = pnl[pnl > 0]
    losses  = pnl[pnl <= 0]

    gross_profit = wins.sum()
    gross_loss   = abs(losses.sum())
    profit_factor = safe_div(gross_profit, gross_loss)

    win_rate   = safe_div(len(wins), n)
    avg_win    = wins.mean()    if len(wins) > 0  else 0.0
    avg_loss   = abs(losses.mean()) if len(losses) > 0 else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Duration
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        durations = (
            pd.to_datetime(trades["exit_time"]) -
            pd.to_datetime(trades["entry_time"])
        ).dt.total_seconds() / 3600
        avg_duration_hours = durations.mean()
    else:
        avg_duration_hours = np.nan

    # Equity curve from cumulative P&L
    equity = (1 + pnl / 100).cumprod() * 100
    max_dd = max_drawdown(equity)

    return {
        "n_trades":            n,
        "win_rate":            _safe_round(win_rate, 4),
        "profit_factor":       _safe_round(profit_factor, 3),
        "expectancy_pct":      _safe_round(expectancy, 4),
        "avg_win_pct":         _safe_round(avg_win, 4),
        "avg_loss_pct":        _safe_round(avg_loss, 4),
        "gross_profit_pct":    _safe_round(gross_profit, 3),
        "gross_loss_pct":      _safe_round(gross_loss, 3),
        "max_drawdown_pct":    _safe_round(max_dd * 100, 2),
        "avg_duration_hours":  _safe_round(avg_duration_hours, 1) if not np.isnan(avg_duration_hours) else 0.0,
        "total_return_pct":    _safe_round(float(pnl.sum()), 3),
    }


def compute_equity_metrics(
    equity: pd.Series,
    periods_per_year: int = 252,
) -> Dict:
    """
    Compute equity-curve level metrics.

    Args:
        equity: pd.Series of equity values (e.g. starting at 100)
        periods_per_year: bars per year for annualisation
    """
    returns = equity.pct_change().dropna()
    cagr    = annualised_return(equity, periods_per_year)
    mdd     = max_drawdown(equity)
    sharpe  = sharpe_ratio(returns, periods_per_year)
    sortino = sortino_ratio(returns, periods_per_year)
    calmar  = calmar_ratio(cagr, mdd)

    # Recovery factor
    total_return = safe_div(equity.iloc[-1] - equity.iloc[0], equity.iloc[0])
    recovery_factor = safe_div(total_return, mdd)

    # Longest drawdown
    roll_max = equity.cummax()
    dd_series = (equity - roll_max) / roll_max
    in_drawdown = dd_series < 0
    changes = in_drawdown.astype(int).diff().fillna(0)
    starts  = equity.index[changes == 1]
    ends    = equity.index[changes == -1]
    if in_drawdown.iloc[-1]:
        ends = ends.append(pd.Index([equity.index[-1]]))
    max_dd_duration = max(
        [(e - s).days for s, e in zip(starts, ends)], default=0
    )

    return {
        "cagr":                 _safe_round(cagr, 4),
        "max_drawdown":         _safe_round(mdd, 4),
        "sharpe_ratio":         _safe_round(sharpe, 3),
        "sortino_ratio":        _safe_round(sortino, 3),
        "calmar_ratio":         _safe_round(calmar, 3),
        "recovery_factor":      _safe_round(recovery_factor, 3),
        "total_return":         _safe_round(total_return, 4),
        "max_drawdown_days":    max_dd_duration,
    }


def compute_metrics_by_group(
    trades: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    Compute trade metrics broken down by a grouping column.

    Args:
        trades:    Trade DataFrame with pnl_pct column
        group_col: Column to group by (e.g. 'regime', 'strategy', 'asset')

    Returns:
        DataFrame indexed by group values
    """
    results = []
    for group_val, grp in trades.groupby(group_col):
        m = compute_trade_metrics(grp)
        m[group_col] = group_val
        results.append(m)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index(group_col)
    return df


def _empty_metrics() -> Dict:
    return {
        "n_trades":            0,
        "win_rate":            0.0,
        "profit_factor":       0.0,
        "expectancy_pct":      0.0,
        "avg_win_pct":         0.0,
        "avg_loss_pct":        0.0,
        "gross_profit_pct":    0.0,
        "gross_loss_pct":      0.0,
        "max_drawdown_pct":    0.0,
        "avg_duration_hours":  0.0,
        "total_return_pct":    0.0,
    }
