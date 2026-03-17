"""
Search space definitions for parameter optimization.

Supports:
- Grid: explicit list of values per parameter
- Random: uniform/log-uniform sampling
- Optuna: suggest_int / suggest_float calls

SearchSpace is generated from strategy.param_schema.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, Generator, List, Optional, Tuple


class SearchSpace:
    """
    Defines the parameter search space for a strategy.

    Schema format (matches strategy.param_schema):
        {
            "param_name": {
                "type":  "int" | "float" | "categorical",
                "low":   ...,
                "high":  ...,
                "step":  ...,           # for grid
                "log":   True/False,    # log scale for random/optuna
                "values": [...],        # for categorical
            }
        }
    """

    def __init__(self, schema: Dict[str, Dict]):
        self.schema = schema

    def grid_points(self) -> List[Dict]:
        """Generate all grid combinations."""
        param_grids = {}
        for name, spec in self.schema.items():
            if spec["type"] == "categorical":
                param_grids[name] = spec["values"]
            elif spec["type"] == "int":
                lo   = int(spec["low"])
                hi   = int(spec["high"])
                step = int(spec.get("step", 1))
                param_grids[name] = list(range(lo, hi + 1, step))
            elif spec["type"] == "float":
                lo   = float(spec["low"])
                hi   = float(spec["high"])
                step = float(spec.get("step", (hi - lo) / 10))
                vals = []
                v = lo
                while v <= hi + 1e-9:
                    vals.append(round(v, 6))
                    v += step
                param_grids[name] = vals

        keys = list(param_grids)
        for combo in itertools.product(*param_grids.values()):
            yield dict(zip(keys, combo))

    def random_sample(self, rng=None) -> Dict:
        """Sample one random parameter combination."""
        import numpy as np
        rng = rng or np.random.default_rng()
        params = {}
        for name, spec in self.schema.items():
            if spec["type"] == "categorical":
                params[name] = rng.choice(spec["values"])
            elif spec["type"] == "int":
                params[name] = int(rng.integers(spec["low"], spec["high"] + 1))
            elif spec["type"] == "float":
                lo, hi = spec["low"], spec["high"]
                if spec.get("log", False):
                    import math
                    log_lo = math.log(lo)
                    log_hi = math.log(hi)
                    params[name] = float(np.exp(rng.uniform(log_lo, log_hi)))
                else:
                    params[name] = float(rng.uniform(lo, hi))
        return params

    def suggest_optuna(self, trial) -> Dict:
        """Generate params using an optuna trial object."""
        params = {}
        for name, spec in self.schema.items():
            if spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["values"])
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                                 step=spec.get("step", 1))
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"],
                    step=spec.get("step", None),
                    log=spec.get("log", False),
                )
        return params

    @property
    def n_grid_points(self) -> int:
        """Total number of grid combinations."""
        total = 1
        for name, spec in self.schema.items():
            if spec["type"] == "categorical":
                total *= len(spec["values"])
            elif spec["type"] == "int":
                lo, hi = int(spec["low"]), int(spec["high"])
                step = int(spec.get("step", 1))
                total *= len(range(lo, hi + 1, step))
            elif spec["type"] == "float":
                lo, hi = float(spec["low"]), float(spec["high"])
                step = float(spec.get("step", (hi - lo) / 10))
                n = max(1, round((hi - lo) / step) + 1)
                total *= n
        return total

    def __repr__(self) -> str:
        return f"SearchSpace({list(self.schema.keys())}, n_grid={self.n_grid_points})"
