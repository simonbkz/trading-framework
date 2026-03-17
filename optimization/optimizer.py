"""
Parameter optimizer — runs grid, random, or Bayesian (Optuna) search.

Uses a vectorized backtest simulation to evaluate each parameter set.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimization.objectives import get_objective
from optimization.search_space import SearchSpace
from strategies.base_strategy import BaseStrategy, Side
from utils.logger import get_logger

log = get_logger(__name__)


class OptimizationResult:
    def __init__(
        self,
        best_params: Dict,
        best_score: float,
        all_results: List[Dict],
        objective: str,
        n_trials: int,
        elapsed_secs: float,
    ):
        self.best_params   = best_params
        self.best_score    = best_score
        self.all_results   = all_results
        self.objective     = objective
        self.n_trials      = n_trials
        self.elapsed_secs  = elapsed_secs

    def top_n(self, n: int = 10) -> List[Dict]:
        return sorted(self.all_results, key=lambda x: x["score"], reverse=True)[:n]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.all_results).sort_values("score", ascending=False)

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(objective={self.objective}, "
            f"best_score={self.best_score:.4f}, "
            f"best_params={self.best_params}, "
            f"n_trials={self.n_trials}, "
            f"elapsed={self.elapsed_secs:.1f}s)"
        )


class Optimizer:
    """
    Strategy parameter optimizer.

    Usage:
        opt = Optimizer(strategy, df, method="optuna", n_trials=200)
        result = opt.run(side="long", objective="calmar")
        best_params = result.best_params
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        method: str = "optuna",   # grid | random | optuna
        n_trials: int = 200,
        n_jobs: int = 1,
        min_trades: int = 20,
        commission_pct: float = 0.0001,
        slippage_pct: float = 0.0002,
    ):
        self.strategy       = strategy
        self.df             = df
        self.method         = method
        self.n_trials       = n_trials
        self.n_jobs         = n_jobs
        self.min_trades     = min_trades
        self.commission_pct = commission_pct
        self.slippage_pct   = slippage_pct

    def run(
        self,
        side: Side = "long",
        objective: str = "calmar",
        search_space: Optional[SearchSpace] = None,
    ) -> OptimizationResult:
        """Run optimization and return best parameters."""
        obj_fn = get_objective(objective)
        space  = search_space or SearchSpace(self.strategy.param_schema)

        log.info(
            "Optimizing %s/%s via %s (%d trials, objective=%s)",
            self.strategy.name, side, self.method, self.n_trials, objective,
        )

        t0 = time.time()
        results = []

        if self.method == "grid":
            results = self._run_grid(space, side, obj_fn)
        elif self.method == "random":
            results = self._run_random(space, side, obj_fn)
        elif self.method == "optuna":
            results = self._run_optuna(space, side, obj_fn)
        else:
            raise ValueError(f"Unknown method '{self.method}'")

        elapsed = time.time() - t0
        best = max(results, key=lambda x: x["score"]) if results else {"params": {}, "score": -np.inf}

        log.info(
            "Optimization complete: %d trials in %.1fs | best_score=%.4f | best_params=%s",
            len(results), elapsed, best["score"], best["params"],
        )

        return OptimizationResult(
            best_params  = best["params"],
            best_score   = best["score"],
            all_results  = results,
            objective    = objective,
            n_trials     = len(results),
            elapsed_secs = elapsed,
        )

    def _evaluate(
        self,
        params: Dict,
        side: Side,
        obj_fn: Callable,
    ) -> float:
        try:
            self.strategy.set_params(params, side=side)
            signals = self.strategy.generate_signals(self.df, side=side)
            trades, equity = self._simulate(signals, side)
            if len(trades) < self.min_trades:
                return -(self.min_trades - len(trades)) * 0.1
            return obj_fn(trades, equity, min_trades=self.min_trades)
        except Exception as exc:
            log.debug("Evaluation failed for params %s: %s", params, exc)
            return -np.inf

    def _run_grid(self, space: SearchSpace, side: Side, obj_fn: Callable) -> List[Dict]:
        results = []
        total = space.n_grid_points
        log.info("Grid search: %d combinations", total)
        for i, params in enumerate(space.grid_points()):
            score = self._evaluate(params, side, obj_fn)
            results.append({"params": params, "score": score})
            if (i + 1) % max(1, total // 20) == 0:
                log.debug("Grid progress: %d/%d", i + 1, total)
        return results

    def _run_random(self, space: SearchSpace, side: Side, obj_fn: Callable) -> List[Dict]:
        rng = np.random.default_rng(42)
        results = []
        for _ in range(self.n_trials):
            params = space.random_sample(rng)
            score  = self._evaluate(params, side, obj_fn)
            results.append({"params": params, "score": score})
        return results

    def _run_optuna(self, space: SearchSpace, side: Side, obj_fn: Callable) -> List[Dict]:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            log.warning("Optuna not installed — falling back to random search")
            return self._run_random(space, side, obj_fn)

        results = []

        def objective(trial):
            params = space.suggest_optuna(trial)
            score  = self._evaluate(params, side, obj_fn)
            results.append({"params": params, "score": score})
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=max(1, self.n_jobs),
            show_progress_bar=False,
        )
        return results

    def _simulate(
        self,
        signals: pd.DataFrame,
        side: Side,
        initial_equity: float = 10000.0,
        risk_pct: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Vectorized trade simulation on signal DataFrame.

        Returns:
            trades: DataFrame with columns entry_time, exit_time, side, pnl_pct
            equity: pd.Series of equity curve
        """
        direction = 1 if side == "long" else -1
        sig_mask  = signals["signal"] == direction

        trade_rows = []
        equity = initial_equity
        equity_curve = []

        in_trade    = False
        entry_price = 0.0
        sl_price    = 0.0
        tp_price    = 0.0
        entry_time  = None

        for ts, row in signals.iterrows():
            if not in_trade:
                if row.get("signal", 0) == direction:
                    in_trade    = True
                    entry_price = float(row["close"]) * (
                        1 + self.slippage_pct if side == "long" else 1 - self.slippage_pct
                    )
                    sl_price    = float(row.get("sl", 0))
                    tp_price    = float(row.get("tp", 0))
                    entry_time  = ts
            else:
                # Check exit conditions
                exit_price = None
                if side == "long":
                    if row["low"] <= sl_price:
                        exit_price = sl_price * (1 - self.slippage_pct)
                    elif row["high"] >= tp_price:
                        exit_price = tp_price * (1 - self.slippage_pct)
                else:
                    if row["high"] >= sl_price:
                        exit_price = sl_price * (1 + self.slippage_pct)
                    elif row["low"] <= tp_price:
                        exit_price = tp_price * (1 + self.slippage_pct)

                if exit_price is not None:
                    if side == "long":
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price * 100
                    # Subtract round-trip commission
                    pnl_pct -= 2 * self.commission_pct * 100

                    equity *= (1 + pnl_pct / 100)
                    trade_rows.append({
                        "entry_time": entry_time,
                        "exit_time":  ts,
                        "side":       side,
                        "entry":      entry_price,
                        "exit":       exit_price,
                        "pnl_pct":    pnl_pct,
                    })
                    in_trade = False

            equity_curve.append((ts, equity))

        trades = pd.DataFrame(trade_rows)
        eq_series = pd.Series(
            [e for _, e in equity_curve],
            index=[t for t, _ in equity_curve],
        )
        return trades, eq_series
