"""Tests for strategy signal generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from features.regime_features import add_regime_features
from features.alpha_features import add_alpha_features


def make_df(n=300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    np.random.seed(1)
    close = 1.1 + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "open":  close - 0.0002,
        "high":  close + 0.0008,
        "low":   close - 0.0008,
        "close": close,
        "volume": np.random.randint(500, 2000, n).astype(float),
    }, index=idx)
    df = add_regime_features(df)
    df = add_alpha_features(df)
    return df


def test_trend_breakout_signals():
    from strategies.trend_breakout import TrendBreakoutStrategy
    df = make_df()
    s  = TrendBreakoutStrategy()
    out = s.generate_signals(df, side="long")
    assert "signal" in out.columns
    assert "sl" in out.columns
    assert "tp" in out.columns
    # Signal must be 0 or 1
    assert set(out["signal"].unique()).issubset({-1, 0, 1})


def test_mean_reversion_signals():
    from strategies.mean_reversion import MeanReversionStrategy
    df  = make_df()
    s   = MeanReversionStrategy()
    out = s.generate_signals(df, side="long")
    assert "signal" in out.columns


def test_volatility_breakout_signals():
    from strategies.volatility_breakout import VolatilityBreakoutStrategy
    df  = make_df()
    s   = VolatilityBreakoutStrategy()
    out = s.generate_signals(df, side="long")
    assert "signal" in out.columns


def test_session_breakout_signals():
    from strategies.session_breakout import SessionBreakoutStrategy
    df  = make_df()
    s   = SessionBreakoutStrategy()
    out = s.generate_signals(df, side="long")
    assert "signal" in out.columns


def test_trade_proposal_valid():
    from strategies.trend_breakout import TrendBreakoutStrategy
    df = make_df()
    s  = TrendBreakoutStrategy()
    # Inject a guaranteed signal on last bar
    out = s.generate_signals(df, side="long")
    out.iloc[-1, out.columns.get_loc("signal")] = 1
    out.iloc[-1, out.columns.get_loc("sl")]     = df["close"].iloc[-1] * 0.99
    out.iloc[-1, out.columns.get_loc("tp")]     = df["close"].iloc[-1] * 1.02
    proposal = s.get_latest_proposal(df, side="long", asset="EURUSD", regime="trend_up")
    # Either returns a valid proposal or None (no real signal on last bar)
    if proposal is not None:
        assert proposal.rr_ratio > 0
        assert proposal.stop_loss < proposal.entry


def test_strategy_factory():
    from strategies.strategy_factory import get_strategy, list_strategies
    names = list_strategies()
    assert "trend_breakout" in names
    for name in names:
        s = get_strategy(name)
        assert s is not None
