"""Tests for feature engineering."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from features.regime_features import add_regime_features, get_feature_columns
from features.alpha_features import add_donchian, add_ema_stack, add_macd


def make_ohlcv(n=200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    np.random.seed(0)
    close = 100 + np.random.randn(n).cumsum()
    return pd.DataFrame({
        "open":   close - 0.1,
        "high":   close + 0.5,
        "low":    close - 0.5,
        "close":  close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    }, index=idx)


def test_regime_features_added():
    df = make_ohlcv()
    result = add_regime_features(df)
    for col in ["atr", "adx", "rsi", "bb_width", "trend_slope"]:
        assert col in result.columns, f"Missing column: {col}"


def test_regime_features_shape():
    df = make_ohlcv()
    result = add_regime_features(df)
    assert len(result) == len(df)


def test_donchian_columns():
    df = make_ohlcv()
    result = add_donchian(df, period=20)
    assert "donch_upper_20" in result.columns
    assert "donch_lower_20" in result.columns
    valid = result.dropna(subset=["donch_upper_20", "donch_lower_20"])
    assert (valid["donch_upper_20"] >= valid["donch_lower_20"]).all()


def test_ema_stack():
    df = make_ohlcv()
    result = add_ema_stack(df, periods=[9, 21, 50])
    for p in [9, 21, 50]:
        assert f"ema_{p}" in result.columns


def test_macd():
    df = make_ohlcv()
    result = add_macd(df)
    assert "macd_line" in result.columns
    assert "macd_hist" in result.columns


def test_atr_positive():
    df = make_ohlcv()
    result = add_regime_features(df)
    assert (result["atr"].dropna() > 0).all()


def test_rsi_range():
    df = make_ohlcv()
    result = add_regime_features(df)
    rsi = result["rsi"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()
