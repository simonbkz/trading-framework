"""
RegimeMapper — maps raw model output to structured RegimeResult objects.

Adds smoothing (mode filter), persistence check, and event_risk injection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from config.regimes import ALL_REGIMES, is_tradable
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RegimeResult:
    timestamp: datetime
    asset: str
    regime_probabilities: Dict[str, float]
    predicted_regime: str
    confidence: float
    is_tradable: bool = True
    smoothed_regime: Optional[str] = None  # regime after mode-smoothing

    def to_dict(self) -> dict:
        return {
            "timestamp":            self.timestamp.isoformat(),
            "asset":                self.asset,
            "regime_probabilities": {k: round(v, 4) for k, v in self.regime_probabilities.items()},
            "predicted_regime":     self.predicted_regime,
            "confidence":           round(self.confidence, 4),
            "smoothed_regime":      self.smoothed_regime or self.predicted_regime,
            "is_tradable":          self.is_tradable,
        }


class RegimeMapper:
    """
    Post-processes raw regime probabilities into RegimeResult objects.

    Features:
    - Smoothing: majority vote over last `smooth_window` bars
    - Persistence: minimum bars in state before switching
    - event_risk injection: if news_blocked is True, forces event_risk regime
    """

    def __init__(
        self,
        smooth_window: int = 5,
        min_persistence_bars: int = 3,
        min_confidence: float = 0.40,
    ):
        self.smooth_window = smooth_window
        self.min_persistence_bars = min_persistence_bars
        self.min_confidence = min_confidence

    def map_series(
        self,
        proba_df: pd.DataFrame,
        asset: str,
        news_blocked: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Convert probability DataFrame to labelled regime series.

        Args:
            proba_df:     DataFrame(index=DatetimeIndex, columns=regimes, values in [0,1])
            asset:        Symbol string for logging
            news_blocked: Boolean series aligned to proba_df.index — True → event_risk

        Returns:
            DataFrame with columns:
                regime, confidence, smoothed_regime, is_tradable
        """
        raw_regime    = proba_df.idxmax(axis=1)
        confidence    = proba_df.max(axis=1)

        # Smooth: rolling mode
        smoothed = self._rolling_mode(raw_regime, self.smooth_window)

        # Persistence filter
        smoothed = self._apply_persistence(smoothed, self.min_persistence_bars)

        # Low-confidence → keep previous label (expressed as NaN → ffill)
        low_conf_mask = confidence < self.min_confidence
        smoothed[low_conf_mask] = pd.NA
        smoothed = smoothed.ffill()

        # News override
        if news_blocked is not None:
            news_aligned = news_blocked.reindex(proba_df.index, fill_value=False)
            smoothed[news_aligned] = "event_risk"

        tradable = smoothed.map(is_tradable).fillna(True)

        result = pd.DataFrame({
            "regime":         raw_regime,
            "confidence":     confidence,
            "smoothed_regime":smoothed,
            "is_tradable":    tradable,
        })

        # Attach probability columns
        for col in proba_df.columns:
            result[f"p_{col}"] = proba_df[col]

        return result

    def map_latest(
        self,
        proba_df: pd.DataFrame,
        asset: str,
        news_blocked: bool = False,
    ) -> RegimeResult:
        """Map the most recent bar to a RegimeResult."""
        mapped = self.map_series(
            proba_df,
            asset=asset,
            news_blocked=pd.Series([news_blocked], index=[proba_df.index[-1]]),
        )
        last = mapped.iloc[-1]
        last_proba = proba_df.iloc[-1].to_dict()
        return RegimeResult(
            timestamp          = proba_df.index[-1].to_pydatetime(),
            asset              = asset,
            regime_probabilities = last_proba,
            predicted_regime   = last["regime"],
            confidence         = float(last["confidence"]),
            is_tradable        = bool(last["is_tradable"]),
            smoothed_regime    = last["smoothed_regime"],
        )

    @staticmethod
    def _rolling_mode(series: pd.Series, window: int) -> pd.Series:
        """Rolling mode for string/categorical series."""
        result = series.copy()
        values = series.values
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_vals = values[start:i + 1]
            # Count occurrences and pick most frequent
            counts: dict = {}
            for v in window_vals:
                counts[v] = counts.get(v, 0) + 1
            result.iloc[i] = max(counts, key=counts.get)
        return result

    @staticmethod
    def _apply_persistence(series: pd.Series, min_bars: int) -> pd.Series:
        """
        Suppress regime switches that last fewer than min_bars bars.
        A switch is only accepted if the new regime persists for >= min_bars bars.
        """
        result = series.copy()
        current = series.iloc[0]
        count = 1
        pending = None
        pending_count = 0

        for i in range(1, len(series)):
            val = series.iloc[i]
            if val == current:
                count += 1
                pending = None
                pending_count = 0
            else:
                if pending == val:
                    pending_count += 1
                    result.iloc[i] = current   # still showing old regime
                    if pending_count >= min_bars:
                        current = val
                        count = pending_count
                        pending = None
                        pending_count = 0
                        # Backfill the pending period
                        result.iloc[i - pending_count + 1: i + 1] = current
                else:
                    pending = val
                    pending_count = 1
                    result.iloc[i] = current
        return result
