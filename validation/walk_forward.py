"""
Walk-forward validation.

Splits data into expanding/rolling train + fixed test windows,
optimizes on train, evaluates on test — exactly as we'd do in production.

Prevents lookahead bias by strict time ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Side
from validation.metrics import compute_trade_metrics, compute_equity_metrics
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict
    train_score: float
    test_trades: pd.DataFrame
    test_equity: pd.Series
    test_metrics: Dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    folds: List[WalkForwardFold]
    combined_trades: pd.DataFrame
    combined_equity: pd.Series
    overall_metrics: Dict
    avg_test_score: float
    strategy_name: str
    side: str
    objective: str

    def summary(self) -> str:
        lines = [
            f"Walk-Forward: {self.strategy_name}/{self.side} | {len(self.folds)} folds",
            f"Objective: {self.objective} | Avg test score: {self.avg_test_score:.4f}",
            "",
        ]
        for fold in self.folds:
            tm = fold.test_metrics
            lines.append(
                f"  Fold {fold.fold_id}: "
                f"{fold.train_start.date()} -> {fold.test_end.date()} | "
                f"params={fold.best_params} | "
                f"T={tm.get('n_trades',0)} PF={tm.get('profit_factor',0):.2f} "
                f"WR={tm.get('win_rate',0)*100:.1f}%"
            )
        lines.extend(["", "OVERALL:", ""])
        for k, v in self.overall_metrics.items():
            lines.append(f"  {k:25s}: {v}")
        return "\n".join(lines)


class WalkForwardValidator:
    """
    Rolling or expanding window walk-forward validator.

    Args:
        n_folds:         number of test periods
        test_size_bars:  bars per test period
        train_min_bars:  minimum training bars
        expanding:       True = expanding window; False = rolling (fixed train size)
        purge_gap_bars:  embargo bars between train and test (prevent leakage)
    """

    def __init__(
        self,
        n_folds: int = 5,
        test_size_bars: Optional[int] = None,
        train_min_bars: int = 500,
        expanding: bool = True,
        purge_gap_bars: int = 10,
    ):
        self.n_folds        = n_folds
        self.test_size_bars = test_size_bars
        self.train_min_bars = train_min_bars
        self.expanding      = expanding
        self.purge_gap_bars = purge_gap_bars

    def run(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        side: Side = "long",
        objective: str = "calmar",
        opt_method: str = "optuna",
        opt_trials: int = 100,
        min_trades: int = 15,
    ) -> WalkForwardResult:
        """Run full walk-forward validation."""
        folds = self._create_folds(df)
        fold_results = []
        all_trades = []

        for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
            train_df = df.iloc[train_idx]
            test_df  = df.iloc[test_idx]

            log.info(
                "WF Fold %d/%d: train=%s->%s (%d bars), test=%s->%s (%d bars)",
                fold_id, len(folds),
                train_df.index[0].date(), train_df.index[-1].date(), len(train_df),
                test_df.index[0].date(),  test_df.index[-1].date(),  len(test_df),
            )

            # Optimize on train (lazy import to avoid circular dependency)
            from optimization.optimizer import Optimizer
            optimizer = Optimizer(
                strategy=strategy,
                df=train_df,
                method=opt_method,
                n_trials=opt_trials,
                min_trades=min_trades,
            )
            opt_result = optimizer.run(side=side, objective=objective)
            best_params = opt_result.best_params

            # Evaluate on test
            strategy.set_params(best_params, side=side)
            test_signals = strategy.generate_signals(test_df, side=side)
            test_trades, test_equity = optimizer._simulate(test_signals, side)
            test_metrics = compute_trade_metrics(test_trades)

            if not test_trades.empty:
                all_trades.append(test_trades)

            fold_results.append(WalkForwardFold(
                fold_id    = fold_id,
                train_start= train_df.index[0],
                train_end  = train_df.index[-1],
                test_start = test_df.index[0],
                test_end   = test_df.index[-1],
                best_params= best_params,
                train_score= opt_result.best_score,
                test_trades= test_trades,
                test_equity= test_equity,
                test_metrics=test_metrics,
            ))

        # Combine all test trades
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
        else:
            combined_trades = pd.DataFrame()

        # Build combined equity
        if not combined_trades.empty:
            pnl = combined_trades["pnl_pct"]
            combined_equity = (1 + pnl / 100).cumprod() * 100
        else:
            combined_equity = pd.Series([100.0])

        overall_metrics = compute_trade_metrics(combined_trades)

        from optimization.objectives import get_objective
        avg_test_score = np.mean([
            get_objective(objective)(
                fold.test_trades,
                fold.test_equity,
                min_trades=min_trades,
            )
            for fold in fold_results
            if not fold.test_trades.empty
        ]) if fold_results else 0.0

        result = WalkForwardResult(
            folds           = fold_results,
            combined_trades = combined_trades,
            combined_equity = combined_equity,
            overall_metrics = overall_metrics,
            avg_test_score  = float(avg_test_score),
            strategy_name   = strategy.name,
            side            = side,
            objective       = objective,
        )

        log.info("\n%s", result.summary())
        return result

    def _create_folds(
        self,
        df: pd.DataFrame,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create (train_indices, test_indices) for each fold.
        """
        n = len(df)
        test_size = self.test_size_bars or max(50, n // (self.n_folds + 1))
        folds = []

        for i in range(self.n_folds):
            test_end   = n - i * test_size
            test_start = test_end - test_size
            if test_start < self.train_min_bars + self.purge_gap_bars:
                break

            train_end   = test_start - self.purge_gap_bars
            train_start = 0 if self.expanding else max(0, train_end - test_size * 4)

            if train_end - train_start < self.train_min_bars:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx  = np.arange(test_start, test_end)
            folds.append((train_idx, test_idx))

        return list(reversed(folds))   # chronological order
