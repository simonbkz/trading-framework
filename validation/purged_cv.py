"""
Purged K-Fold Cross-Validation for time-series data.

Based on de Prado's "Advances in Financial Machine Learning":
- Embargo period between train and test folds
- No leakage through overlapping labels
- Suitable for training the regime classifier
"""
from __future__ import annotations

from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils.logger import get_logger

log = get_logger(__name__)


class PurgedKFold:
    """
    Purged K-Fold splitter that respects time ordering and adds
    an embargo gap between training and test sets.

    Yields (train_indices, test_indices) tuples suitable for sklearn.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 10,   # bars removed between train and test
    ):
        self.n_splits  = n_splits
        self.purge_gap = purge_gap

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups=None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_idx, test_idx) for each fold."""
        n = len(X)
        fold_size = n // self.n_splits

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end   = test_start + fold_size if fold < self.n_splits - 1 else n

            # Test indices
            test_idx = np.arange(test_start, test_end)

            # Train indices: everything outside the test window + purge gap
            train_mask = np.ones(n, dtype=bool)
            purge_start = max(0, test_start - self.purge_gap)
            purge_end   = min(n, test_end   + self.purge_gap)
            train_mask[purge_start:purge_end] = False
            train_idx = np.where(train_mask)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def purged_cross_val_score(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    scoring,
    n_splits: int = 5,
    purge_gap: int = 10,
) -> np.ndarray:
    """
    Cross-validate using purged K-Fold.

    Args:
        estimator: sklearn-compatible estimator with fit/score
        X:         feature DataFrame
        y:         target Series
        scoring:   sklearn scorer or callable
        n_splits:  number of folds
        purge_gap: embargo bars

    Returns:
        Array of test scores per fold
    """
    from sklearn.base import clone

    cv = PurgedKFold(n_splits=n_splits, purge_gap=purge_gap)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = clone(estimator)
        clf.fit(X_train, y_train)

        if callable(scoring):
            score = scoring(clf, X_test, y_test)
        else:
            from sklearn.metrics import check_scoring
            scorer = check_scoring(clf, scoring=scoring)
            score = scorer(clf, X_test, y_test)

        scores.append(score)
        log.debug("Purged CV fold %d: score=%.4f", fold + 1, score)

    return np.array(scores)
