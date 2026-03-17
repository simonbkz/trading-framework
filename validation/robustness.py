"""
Robustness checks — parameter perturbation tests.

A good parameter set should not be fragile: small changes to any
parameter should not cause large drops in performance.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Side
from validation.metrics import compute_trade_metrics
from utils.logger import get_logger

log = get_logger(__name__)


class RobustnessChecker:
    """
    Tests parameter sensitivity via perturbation analysis.

    For each parameter, nudges it ±10% and ±20% while holding
    others constant. Reports performance degradation.

    A set is "robust" if performance drops < 30% under ±10% perturbation.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        perturbation_pcts: List[float] = None,
        robust_threshold: float = 0.30,   # max allowed PF degradation
    ):
        self.strategy        = strategy
        self.df              = df
        self.perturbations   = perturbation_pcts or [-0.20, -0.10, +0.10, +0.20]
        self.threshold       = robust_threshold

    def check(
        self,
        base_params: Dict,
        side: Side = "long",
    ) -> dict:
        """
        Run robustness check on a parameter set.

        Returns:
            dict with:
                is_robust: bool
                base_score: float (profit factor)
                perturbation_results: list of per-param results
                sensitivity_scores: {param_name: sensitivity_pct}
        """
        from optimization.optimizer import Optimizer

        opt = Optimizer(self.strategy, self.df)

        # Baseline
        self.strategy.set_params(base_params, side=side)
        base_signals = self.strategy.generate_signals(self.df, side=side)
        base_trades, base_equity = opt._simulate(base_signals, side)
        base_tm   = compute_trade_metrics(base_trades)
        base_pf   = base_tm["profit_factor"]

        results = []
        sensitivity = {}

        for param_name, base_val in base_params.items():
            if not isinstance(base_val, (int, float)):
                continue

            param_sensitivities = []
            for pct in self.perturbations:
                if isinstance(base_val, int):
                    perturbed = max(1, round(base_val * (1 + pct)))
                else:
                    perturbed = base_val * (1 + pct)

                test_params = {**base_params, param_name: perturbed}
                self.strategy.set_params(test_params, side=side)
                try:
                    sigs = self.strategy.generate_signals(self.df, side=side)
                    trades, equity = opt._simulate(sigs, side)
                    tm  = compute_trade_metrics(trades)
                    pf  = tm["profit_factor"]
                except Exception:
                    pf = 0.0

                degradation = (base_pf - pf) / max(base_pf, 1e-6)
                param_sensitivities.append(degradation)

                results.append({
                    "param":       param_name,
                    "perturbation": pct,
                    "base_val":    base_val,
                    "perturbed_val": perturbed,
                    "base_pf":     base_pf,
                    "test_pf":     pf,
                    "degradation": round(degradation, 4),
                })

            sensitivity[param_name] = float(np.mean(np.abs(param_sensitivities)))

        # Restore original params
        self.strategy.set_params(base_params, side=side)

        is_robust = all(v < self.threshold for v in sensitivity.values())

        return {
            "is_robust":            is_robust,
            "base_profit_factor":   round(base_pf, 3),
            "sensitivity_scores":   {k: round(v, 4) for k, v in sensitivity.items()},
            "perturbation_results": results,
            "robust_threshold":     self.threshold,
        }
