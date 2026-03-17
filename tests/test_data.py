"""Tests for data loading and preprocessing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.preprocessing import preprocess_ohlcv
from utils.helpers import normalize_ohlcv_columns, validate_ohlcv


def make_ohlcv(n=100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    np.random.seed(42)
    close = 1.1 + np.random.randn(n).cumsum() * 0.001
    return pd.DataFrame({
        "open":   close - np.abs(np.random.randn(n) * 0.0005),
        "high":   close + np.abs(np.random.randn(n) * 0.001),
        "low":    close - np.abs(np.random.randn(n) * 0.001),
        "close":  close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    }, index=idx)


def test_validate_ohlcv_passes():
    df = make_ohlcv()
    validate_ohlcv(df)   # should not raise


def test_validate_ohlcv_fails_on_missing():
    df = make_ohlcv().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing"):
        validate_ohlcv(df)


def test_normalize_columns():
    df = make_ohlcv()
    df.columns = ["Open", "High", "Low", "Adj Close", "Volume"]
    df = normalize_ohlcv_columns(df)
    assert "close" in df.columns
    assert "open" in df.columns


def test_preprocess_removes_negatives():
    df = make_ohlcv()
    df.iloc[5, df.columns.get_loc("close")] = -1.0
    clean = preprocess_ohlcv(df)
    assert (clean["close"] > 0).all()


def test_preprocess_removes_duplicates():
    df = make_ohlcv()
    df_dup = pd.concat([df, df.iloc[:5]])
    clean = preprocess_ohlcv(df_dup)
    assert clean.index.is_unique


def test_preprocess_utc_index():
    df = make_ohlcv()
    df.index = df.index.tz_localize(None)
    clean = preprocess_ohlcv(df)
    assert clean.index.tz is not None
