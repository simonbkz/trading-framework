"""Tests for regime detection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from features.regime_features import add_regime_features
from regimes.tree_regime import RegimeLabeller
from regimes.regime_mapper import RegimeMapper
from config.regimes import ALL_REGIMES


def make_ohlcv(n=500) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    np.random.seed(7)
    close = 1.1 + np.random.randn(n).cumsum() * 0.001
    return pd.DataFrame({
        "open":   close - 0.0003,
        "high":   close + 0.001,
        "low":    close - 0.001,
        "close":  close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    }, index=idx)


def test_regime_labeller_produces_valid_labels():
    df  = make_ohlcv()
    df  = add_regime_features(df)
    labeller = RegimeLabeller()
    labels = labeller.label(df)
    assert len(labels) == len(df)
    for label in labels.dropna():
        assert label in ALL_REGIMES, f"Unexpected label: {label}"


def test_regime_mapper_returns_dataframe():
    df    = make_ohlcv(200)
    df    = add_regime_features(df)
    labeller = RegimeLabeller()
    labels = labeller.label(df)

    # Build fake probability frame
    n = len(df)
    proba = pd.DataFrame(
        np.random.dirichlet(np.ones(len(ALL_REGIMES)), n),
        index=df.index,
        columns=ALL_REGIMES,
    )
    mapper = RegimeMapper()
    result = mapper.map_series(proba, asset="EURUSD")
    assert "regime" in result.columns
    assert "confidence" in result.columns
    assert "smoothed_regime" in result.columns


def test_regime_mapper_event_risk_override():
    df    = make_ohlcv(50)
    df    = add_regime_features(df)
    n     = len(df)
    proba = pd.DataFrame(
        np.ones((n, len(ALL_REGIMES))) / len(ALL_REGIMES),
        index=df.index, columns=ALL_REGIMES,
    )
    mapper = RegimeMapper()
    # Block all rows
    blocked = pd.Series(True, index=df.index)
    result  = mapper.map_series(proba, asset="TEST", news_blocked=blocked)
    assert (result["smoothed_regime"] == "event_risk").all()


def test_hmm_regime_model_fits():
    pytest.importorskip("hmmlearn")
    df = make_ohlcv(500)
    df = add_regime_features(df)
    from regimes.hmm_regime import HMMRegimeModel
    model = HMMRegimeModel(n_states=4)
    model.fit(df.dropna())
    assert model._fitted
    proba = model.predict_proba(df.dropna())
    assert proba.shape[1] == len(ALL_REGIMES)
    assert (proba.sum(axis=1) - 1.0).abs().max() < 0.01


def test_tree_regime_model_fits():
    df     = make_ohlcv(500)
    df     = add_regime_features(df)
    labeller = RegimeLabeller()
    labels = labeller.label(df.dropna())
    from regimes.tree_regime import TreeRegimeModel
    model  = TreeRegimeModel()
    model.fit(df.dropna(), labels)
    assert model._fitted
    proba  = model.predict_proba(df.dropna())
    assert proba.shape[1] == len(ALL_REGIMES)
