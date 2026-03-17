"""
Abstract base class for all regime detection models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.regimes import ALL_REGIMES
from utils.logger import get_logger

log = get_logger(__name__)


class BaseRegimeModel(ABC):
    """
    All regime models must implement fit() and predict_proba().

    Output convention:
        predict_proba() → pd.DataFrame with columns = regime labels,
                          values in [0, 1], each row sums to ≈ 1.
        predict()       → Series of most-likely regime labels.
    """

    name: str = "base"
    feature_columns: List[str] = []

    def __init__(self, regimes: List[str] = None):
        self.regimes = regimes or ALL_REGIMES
        self._fitted = False

    @abstractmethod
    def fit(self, features: pd.DataFrame, labels: Optional[pd.Series] = None) -> "BaseRegimeModel":
        """Fit the model on feature data."""
        ...

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Return per-regime probabilities for each row in features.

        Returns:
            DataFrame with same index as features,
            columns = self.regimes, values in [0, 1].
        """
        ...

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return most-likely regime label for each row."""
        proba = self.predict_proba(features)
        return proba.idxmax(axis=1).rename("regime")

    def predict_with_confidence(
        self, features: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Return (regime_series, confidence_series).
        Confidence = probability of the predicted regime.
        """
        proba = self.predict_proba(features)
        regime = proba.idxmax(axis=1).rename("regime")
        confidence = proba.max(axis=1).rename("confidence")
        return regime, confidence

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. Call fit() first."
            )

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Drop NaN rows and warn if significant fraction is missing."""
        n_total = len(features)
        features = features.dropna()
        n_dropped = n_total - len(features)
        if n_dropped > 0:
            frac = n_dropped / n_total
            level = log.warning if frac > 0.1 else log.debug
            level("Dropped %d/%d rows (%.1f%%) with NaN before regime fitting",
                  n_dropped, n_total, frac * 100)
        return features

    @staticmethod
    def _to_prob_frame(
        proba_array: np.ndarray,
        index: pd.Index,
        regimes: List[str],
    ) -> pd.DataFrame:
        """Convert a numpy probability array to a labelled DataFrame."""
        return pd.DataFrame(proba_array, index=index, columns=regimes)

    def save(self, path: str) -> None:
        """Persist model to disk via joblib."""
        import joblib
        joblib.dump(self, path)
        log.info("Saved %s to %s", self.name, path)

    @classmethod
    def load(cls, path: str) -> "BaseRegimeModel":
        import joblib
        model = joblib.load(path)
        log.info("Loaded %s from %s", model.name, path)
        return model
