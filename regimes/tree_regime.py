"""
Tree-based supervised regime classifier (LightGBM / scikit-learn).

Requires labelled training data produced by RegimeLabeller or manually.
Falls back to scikit-learn RandomForest if LightGBM is not installed.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

from regimes.base_regime_model import BaseRegimeModel
from features.regime_features import get_feature_columns
from config.regimes import ALL_REGIMES
from utils.logger import get_logger

log = get_logger(__name__)


class TreeRegimeModel(BaseRegimeModel):
    """
    Supervised multi-class regime classifier.

    Uses LightGBM when available, falls back to sklearn GradientBoostingClassifier.
    Requires labelled training data.
    """

    name = "tree"

    def __init__(
        self,
        regimes: Optional[List[str]] = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        super().__init__(regimes)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._clf = None
        self._feature_cols: List[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "TreeRegimeModel":
        """
        Fit classifier.

        Args:
            features: Feature DataFrame (output of add_regime_features)
            labels:   Series of regime label strings aligned to features
        """
        X, y = self._prepare_Xy(features, labels)
        self._clf = self._make_classifier()
        self._clf.fit(X, y)
        self._fitted = True
        log.info(
            "Tree regime model fitted on %d samples, %d features, %d classes",
            len(X), X.shape[1], len(np.unique(y)),
        )
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        X = features[self._feature_cols].fillna(0).values
        raw_proba = self._clf.predict_proba(X)  # (n, n_classes)
        classes   = list(self._clf.classes_)

        # Build full probability array across all possible regimes
        n = len(X)
        proba = np.zeros((n, len(self.regimes)))
        for i, regime in enumerate(self.regimes):
            if regime in classes:
                col = classes.index(regime)
                proba[:, i] = raw_proba[:, col]

        # Renormalise
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        proba = proba / row_sums
        return self._to_prob_frame(proba, features.index, self.regimes)

    def feature_importances(self) -> pd.Series:
        self._check_fitted()
        imp = self._clf.feature_importances_
        return pd.Series(imp, index=self._feature_cols).sort_values(ascending=False)

    def _prepare_Xy(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ):
        # Use available feature columns
        self._feature_cols = [c for c in get_feature_columns() if c in features.columns]
        if not self._feature_cols:
            # Fallback: use numeric columns
            self._feature_cols = features.select_dtypes(include="number").columns.tolist()

        df = features[self._feature_cols].join(labels.rename("_label"))
        df = df.dropna()
        X = df[self._feature_cols].values
        y = df["_label"].values
        return X, y

    def _make_classifier(self):
        try:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            log.debug("Using LightGBM classifier")
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(
                n_estimators=min(self.n_estimators, 100),
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            log.debug("LightGBM not installed — using sklearn GradientBoostingClassifier")
        return clf


class RegimeLabeller:
    """
    Rule-based labeller for generating training data for TreeRegimeModel.

    Uses regime_features to assign labels based on indicator thresholds.
    These labels are approximate but sufficient to train a reasonable classifier.
    """

    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign regime labels to each bar using rule-based thresholds.

        Requires columns from add_regime_features().
        """
        labels = pd.Series(index=df.index, dtype=str)
        labels[:] = "range_bound"   # default

        has = lambda col: col in df.columns

        # High volatility: vol_ratio > 1.5 AND ATR expanding
        if has("vol_ratio") and has("atr_ratio"):
            hi_vol = (df["vol_ratio"] > 1.5) & (df["atr_ratio"] > 1.2)
            labels[hi_vol] = "high_volatility"

        # Low volatility: vol_ratio < 0.7 AND range compressed
        if has("vol_ratio") and has("range_comp"):
            lo_vol = (df["vol_ratio"] < 0.7) & (df["range_comp"] > 0.5)
            labels[lo_vol] = "low_volatility"

        # Trend up: ADX > 25 AND DI+ > DI- AND positive slope
        if has("adx") and has("di_plus") and has("di_minus") and has("trend_slope"):
            trend_up = (
                (df["adx"] > 25) &
                (df["di_plus"] > df["di_minus"]) &
                (df["trend_slope"] > 0)
            )
            labels[trend_up] = "trend_up"

        # Trend down: ADX > 25 AND DI- > DI+ AND negative slope
        if has("adx") and has("di_plus") and has("di_minus") and has("trend_slope"):
            trend_dn = (
                (df["adx"] > 25) &
                (df["di_minus"] > df["di_plus"]) &
                (df["trend_slope"] < 0)
            )
            labels[trend_dn] = "trend_down"

        # Mean reverting: RSI extreme + BB touch + low ADX
        if has("rsi") and has("bb_pct") and has("adx"):
            mr = (
                (df["adx"] < 20) &
                ((df["rsi"] > 65) | (df["rsi"] < 35)) &
                ((df["bb_pct"] > 0.9) | (df["bb_pct"] < 0.1))
            )
            labels[mr] = "mean_reverting"

        return labels
