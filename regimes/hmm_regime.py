"""
Hidden Markov Model regime detector using hmmlearn.

The HMM operates on returns + volatility features and discovers
latent states. The states are then mapped to human-readable regime
labels via the RegimeMapper.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from regimes.base_regime_model import BaseRegimeModel
from config.regimes import ALL_REGIMES
from utils.logger import get_logger

log = get_logger(__name__)


class HMMRegimeModel(BaseRegimeModel):
    """
    Gaussian HMM regime detector.

    States are unlabelled after fitting — use RegimeMapper to assign
    human-readable labels based on state statistics.
    """

    name = "hmm"

    HMM_FEATURES = [
        "log_ret",
        "vol_short",
        "atr_norm",
        "adx",
        "trend_slope_norm",
        "bb_width",
    ]

    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
        regimes: Optional[List[str]] = None,
    ):
        super().__init__(regimes)
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self._hmm = None
        self._state_to_regime: dict = {}

    def fit(
        self,
        features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
    ) -> "HMMRegimeModel":
        try:
            from hmmlearn import hmm as hmmlib
        except ImportError:
            raise ImportError("Install hmmlearn: pip install hmmlearn")

        X = self._prepare_X(features)

        self._hmm = hmmlib.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self._hmm.fit(X)

        # Map states to regimes based on state statistics
        self._state_to_regime = self._auto_map_states(X)
        self._fitted = True
        log.info(
            "HMM fitted: %d states, convergence=%s, state_map=%s",
            self.n_states,
            self._hmm.monitor_.converged,
            self._state_to_regime,
        )
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        X = self._prepare_X(features)
        # Get posterior probabilities for each state
        _, posteriors = self._hmm.score_samples(X)   # (n_samples, n_states)

        # Map state columns to regime labels (one-hot expand if n_states < n_regimes)
        n_regimes = len(self.regimes)
        proba = np.zeros((len(X), n_regimes))
        for state_idx, regime_label in self._state_to_regime.items():
            if regime_label in self.regimes:
                col_idx = self.regimes.index(regime_label)
                proba[:, col_idx] += posteriors[:, state_idx]

        # Renormalise rows that have zero mass (unmapped states)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        proba = proba / row_sums

        return self._to_prob_frame(proba, features.index, self.regimes)

    def _prepare_X(self, features: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.HMM_FEATURES if c in features.columns]
        if not cols:
            raise ValueError(
                f"None of HMM features {self.HMM_FEATURES} found in DataFrame. "
                f"Run add_regime_features() first."
            )
        X = features[cols].dropna().values.astype(np.float64)
        return X

    def _auto_map_states(self, X: np.ndarray) -> dict:
        """
        Heuristic mapping of HMM states to regime labels using
        the mean of each feature per state.

        Assignments:
          - highest ADX mean → trend (up/down by slope sign)
          - highest vol mean → high_volatility
          - lowest vol mean  → low_volatility
          - remainder        → range_bound / mean_reverting
        """
        state_seqs = self._hmm.predict(X)
        feat_names = [c for c in self.HMM_FEATURES if c in
                      ["log_ret", "vol_short", "atr_norm", "adx", "trend_slope_norm", "bb_width"]]

        # Build state stats
        stats = {}
        for s in range(self.n_states):
            mask = state_seqs == s
            if mask.sum() < 5:
                stats[s] = {"vol": 0, "adx": 0, "ret": 0, "slope": 0}
                continue
            Xs = X[mask]
            idx = {n: i for i, n in enumerate(feat_names)}
            stats[s] = {
                "vol":   Xs[:, idx.get("vol_short", 0)].mean() if "vol_short" in idx else 0,
                "adx":   Xs[:, idx.get("adx", 0)].mean()       if "adx" in idx else 0,
                "ret":   Xs[:, idx.get("log_ret", 0)].mean()   if "log_ret" in idx else 0,
                "slope": Xs[:, idx.get("trend_slope_norm", 0)].mean() if "trend_slope_norm" in idx else 0,
            }

        sorted_by_adx = sorted(stats, key=lambda s: stats[s]["adx"], reverse=True)
        sorted_by_vol = sorted(stats, key=lambda s: stats[s]["vol"], reverse=True)

        mapping = {}
        assigned = set()

        # Highest ADX → trending
        trend_state = sorted_by_adx[0]
        if stats[trend_state]["slope"] >= 0:
            mapping[trend_state] = "trend_up"
        else:
            mapping[trend_state] = "trend_down"
        assigned.add(trend_state)

        # Second highest ADX (if n_states >= 4) → other trend direction or high_vol
        if self.n_states >= 4 and len(sorted_by_adx) > 1:
            s2 = next(s for s in sorted_by_adx[1:] if s not in assigned)
            if stats[s2]["adx"] > 20:
                mapping[s2] = "trend_down" if mapping[trend_state] == "trend_up" else "trend_up"
            else:
                mapping[s2] = "high_volatility"
            assigned.add(s2)

        # Highest vol (not yet assigned) → high_volatility
        for s in sorted_by_vol:
            if s not in assigned:
                mapping[s] = "high_volatility"
                assigned.add(s)
                break

        # Lowest vol → low_volatility
        for s in reversed(sorted_by_vol):
            if s not in assigned:
                mapping[s] = "low_volatility"
                assigned.add(s)
                break

        # Remainder → range_bound
        for s in range(self.n_states):
            if s not in assigned:
                mapping[s] = "range_bound"

        return mapping
