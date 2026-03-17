"""
Ensemble regime model — combines HMM + Tree model probabilities.

Supports soft voting (weighted average of probabilities) and
hard voting (majority rule across predicted labels).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from regimes.base_regime_model import BaseRegimeModel
from utils.logger import get_logger

log = get_logger(__name__)


class EnsembleRegimeModel(BaseRegimeModel):
    """
    Combines multiple BaseRegimeModel instances via weighted soft voting.

    Usage:
        hmm  = HMMRegimeModel(n_states=4).fit(features)
        tree = TreeRegimeModel().fit(features, labels)
        ens  = EnsembleRegimeModel([hmm, tree], weights=[0.4, 0.6])
        proba = ens.predict_proba(features)
    """

    name = "ensemble"

    def __init__(
        self,
        models: Optional[List[BaseRegimeModel]] = None,
        weights: Optional[List[float]] = None,
        regimes: Optional[List[str]] = None,
    ):
        super().__init__(regimes)
        self.models: List[BaseRegimeModel] = models or []
        if weights is not None:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = None   # equal weighting
        self._fitted = len(self.models) > 0 and all(m._fitted for m in self.models)

    def add_model(
        self,
        model: BaseRegimeModel,
        weight: float = 1.0,
    ) -> "EnsembleRegimeModel":
        self.models.append(model)
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)
        # Renormalise
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        self._fitted = all(m._fitted for m in self.models)
        return self

    def fit(
        self,
        features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
    ) -> "EnsembleRegimeModel":
        """Fit all unfitted sub-models."""
        for model in self.models:
            if not model._fitted:
                log.info("Fitting sub-model: %s", model.name)
                model.fit(features, labels)
        self._fitted = True
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("No models in ensemble. Add models with add_model().")

        weights = self.weights or [1.0 / len(self.models)] * len(self.models)
        weighted_proba = None

        for model, weight in zip(self.models, weights):
            try:
                proba = model.predict_proba(features)
            except Exception as exc:
                log.warning("Sub-model %s failed: %s — skipping", model.name, exc)
                continue

            # Align columns to self.regimes
            for r in self.regimes:
                if r not in proba.columns:
                    proba[r] = 0.0
            proba = proba[self.regimes]

            if weighted_proba is None:
                weighted_proba = proba.values * weight
            else:
                weighted_proba += proba.values * weight

        if weighted_proba is None:
            # All models failed — return uniform distribution
            log.error("All ensemble sub-models failed. Returning uniform probabilities.")
            n = len(features)
            k = len(self.regimes)
            weighted_proba = np.full((n, k), 1.0 / k)

        # Renormalise rows
        row_sums = weighted_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        proba_norm = weighted_proba / row_sums

        return self._to_prob_frame(proba_norm, features.index, self.regimes)

    def calibrate_weights(
        self,
        features: pd.DataFrame,
        true_labels: pd.Series,
    ) -> None:
        """
        Optimise model weights using log-loss on validation data.
        Updates self.weights in-place.
        """
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss

        probas = []
        for model in self.models:
            try:
                p = model.predict_proba(features)
                for r in self.regimes:
                    if r not in p.columns:
                        p[r] = 0.0
                probas.append(p[self.regimes].values)
            except Exception:
                probas.append(None)

        valid = [(i, p) for i, p in enumerate(probas) if p is not None]
        if len(valid) < 2:
            log.warning("Cannot calibrate weights with < 2 valid models")
            return

        y = pd.Categorical(true_labels, categories=self.regimes).codes

        def loss(w):
            w = np.abs(w) / np.sum(np.abs(w))
            combined = sum(wi * p for wi, (_, p) in zip(w, valid))
            combined = np.clip(combined, 1e-7, 1.0)
            return log_loss(y, combined, labels=list(range(len(self.regimes))))

        w0 = np.ones(len(valid)) / len(valid)
        res = minimize(loss, w0, method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-4})
        opt_w = np.abs(res.x) / np.sum(np.abs(res.x))

        full_weights = [0.0] * len(self.models)
        for rank, (orig_idx, _) in enumerate(valid):
            full_weights[orig_idx] = opt_w[rank]
        total = sum(full_weights)
        self.weights = [w / total for w in full_weights]
        log.info("Calibrated ensemble weights: %s",
                 {m.name: f"{w:.3f}" for m, w in zip(self.models, self.weights)})
