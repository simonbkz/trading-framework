"""
RegimeService — the main entry point for regime detection.

Orchestrates: feature engineering → model inference → mapping → output.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from config.settings import SETTINGS
from config.regimes import ALL_REGIMES
from features.regime_features import add_regime_features
from regimes.hmm_regime import HMMRegimeModel
from regimes.tree_regime import TreeRegimeModel, RegimeLabeller
from regimes.ensemble_regime import EnsembleRegimeModel
from regimes.transformer_regime import TransformerRegimeModel
from regimes.regime_mapper import RegimeMapper, RegimeResult
from utils.logger import get_logger

log = get_logger(__name__)


class RegimeService:
    """
    Detects market regimes from OHLCV data.

    Usage:
        svc = RegimeService()
        svc.fit(df)                     # train on historical data
        result = svc.detect(df)         # detect on new data
        latest = svc.latest(df)         # get single RegimeResult
    """

    def __init__(
        self,
        use_ensemble: bool = True,
        hmm_states: int = 4,
        smooth_window: int = 5,
        min_confidence: float = 0.40,
        model_dir: Optional[Path] = None,
        use_transformer: bool = False,
    ):
        self.use_ensemble = use_ensemble
        self.hmm_states = hmm_states
        self.use_transformer = use_transformer
        self.mapper = RegimeMapper(
            smooth_window=smooth_window,
            min_confidence=min_confidence,
        )
        self.model_dir = model_dir or Path("models")
        self._model: Optional[EnsembleRegimeModel | HMMRegimeModel] = None

    def fit(
        self,
        df: pd.DataFrame,
        labels: Optional[pd.Series] = None,
    ) -> "RegimeService":
        """
        Fit regime model on OHLCV DataFrame.

        Args:
            df:     OHLCV data
            labels: Optional manual regime labels (for supervised tree model).
                    If None, labels are generated via RegimeLabeller.
        """
        log.info("Engineering regime features (%d bars)", len(df))
        features = add_regime_features(df)
        features = features.dropna()

        if labels is None:
            log.info("Auto-labelling regime training data with rule-based labeller")
            labeller = RegimeLabeller()
            labels = labeller.label(features).reindex(features.index)

        if self.use_ensemble:
            self._model = self._build_ensemble(features, labels)
        else:
            self._model = HMMRegimeModel(n_states=self.hmm_states).fit(features)

        log.info("Regime model fitted successfully (type=%s)", type(self._model).__name__)
        return self

    def detect(
        self,
        df: pd.DataFrame,
        asset: str = "UNKNOWN",
        news_blocked: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Run regime detection on OHLCV data.

        Returns:
            DataFrame with columns: regime, confidence, smoothed_regime, is_tradable,
            plus p_{regime_name} probability columns.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        features = add_regime_features(df)
        # Keep only rows with full features; align back to input index
        valid = features.dropna()
        proba_df = self._model.predict_proba(valid)
        proba_df = proba_df.reindex(df.index)   # realign (NaN for leading bars)
        proba_df = proba_df.ffill()             # forward-fill the leading NaNs

        result = self.mapper.map_series(proba_df, asset=asset, news_blocked=news_blocked)
        return result

    def latest(
        self,
        df: pd.DataFrame,
        asset: str = "UNKNOWN",
        news_blocked: bool = False,
    ) -> RegimeResult:
        """Return RegimeResult for the most recent bar."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        features = add_regime_features(df)
        valid = features.dropna()
        if valid.empty:
            raise ValueError("No valid bars after feature engineering")

        proba_df = self._model.predict_proba(valid)
        return self.mapper.map_latest(proba_df, asset=asset, news_blocked=news_blocked)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.model_dir / "regime_model.pkl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._model:
            self._model.save(str(path))

    def load(self, path: Optional[Path] = None) -> "RegimeService":
        path = path or self.model_dir / "regime_model.pkl"
        self._model = EnsembleRegimeModel.load(str(path))
        return self

    def _build_ensemble(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> EnsembleRegimeModel:
        hmm  = HMMRegimeModel(n_states=self.hmm_states).fit(features)
        tree = TreeRegimeModel().fit(features, labels)

        if self.use_transformer:
            try:
                transformer = TransformerRegimeModel(
                    seq_len=60, d_model=64, n_heads=4,
                    n_layers=2, epochs=50, batch_size=64,
                ).fit(features, labels)
                ens = EnsembleRegimeModel(
                    models=[hmm, tree, transformer],
                    weights=[0.25, 0.50, 0.25],
                )
                log.info("Ensemble: HMM(0.25) + LightGBM(0.50) + Transformer(0.25)")
            except Exception as exc:
                log.warning("Transformer training failed (%s), using HMM+Tree only", exc)
                ens = EnsembleRegimeModel(
                    models=[hmm, tree],
                    weights=[0.35, 0.65],
                )
        else:
            ens = EnsembleRegimeModel(
                models=[hmm, tree],
                weights=[0.35, 0.65],
            )

        ens._fitted = True
        return ens
