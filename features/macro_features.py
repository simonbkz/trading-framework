"""
Macro & sentiment features — external data dimensions.

Adds VIX (fear), DXY (dollar strength), yield curve, cross-asset
correlation, and sentiment scores to enrich strategy signal quality.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# Cache macro data to avoid repeated downloads within one session
_macro_cache: Dict[str, pd.DataFrame] = {}


def _fetch_macro_series(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    interval: str = "1h",
) -> pd.Series:
    """Download a single macro series from yfinance, cached."""
    cache_key = f"{ticker}_{start}_{end}_{interval}"
    if cache_key in _macro_cache:
        return _macro_cache[cache_key]

    try:
        import yfinance as yf
        data = yf.download(
            ticker, start=start, end=end,
            interval=interval, progress=False, auto_adjust=True,
        )
        if hasattr(data.columns, "levels"):
            data.columns = data.columns.get_level_values(0)
        series = data["Close"].squeeze()
        _macro_cache[cache_key] = series
        log.info("Fetched %s: %d bars", ticker, len(series))
        return series
    except Exception as exc:
        log.warning("Failed to fetch %s: %s", ticker, exc)
        return pd.Series(dtype=float)


def add_vix_features(df: pd.DataFrame, start: str = "2024-01-01") -> pd.DataFrame:
    """
    Add VIX-derived features: level, regime, z-score, term structure proxy.

    Features added:
        vix_level:     raw VIX value mapped to asset timeframe
        vix_zscore:    20-period z-score of VIX (spikes = regime change signal)
        vix_regime:    0=low(<15), 1=normal(15-25), 2=elevated(25-35), 3=extreme(>35)
        vix_change_5:  5-bar VIX change (acceleration of fear)
        vix_ma_ratio:  VIX / VIX 20-bar MA (>1 = fear rising)
    """
    out = df.copy()
    vix = _fetch_macro_series("^VIX", start=start)

    if vix.empty:
        for col in ["vix_level", "vix_zscore", "vix_regime", "vix_change_5", "vix_ma_ratio"]:
            out[col] = np.nan
        return out

    # Resample VIX to match df frequency and forward-fill
    vix_aligned = vix.reindex(out.index, method="ffill")

    out["vix_level"] = vix_aligned
    vix_ma = vix_aligned.rolling(20, min_periods=5).mean()
    vix_std = vix_aligned.rolling(20, min_periods=5).std()
    out["vix_zscore"] = (vix_aligned - vix_ma) / vix_std.replace(0, np.nan)
    out["vix_change_5"] = vix_aligned.pct_change(5)
    out["vix_ma_ratio"] = vix_aligned / vix_ma.replace(0, np.nan)

    # Regime buckets
    out["vix_regime"] = pd.cut(
        vix_aligned,
        bins=[-np.inf, 15, 25, 35, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(float)

    return out


def add_dxy_features(df: pd.DataFrame, start: str = "2024-01-01") -> pd.DataFrame:
    """
    Add US Dollar Index features — critical for forex pair direction.

    Features added:
        dxy_level:       raw DXY value
        dxy_trend:       +1 (strengthening), -1 (weakening), 0 (flat)
        dxy_zscore:      20-period z-score
        dxy_momentum:    10-bar rate of change
        dxy_vs_ema50:    distance from EMA50 (% of price)
    """
    out = df.copy()
    dxy = _fetch_macro_series("DX-Y.NYB", start=start)

    if dxy.empty:
        for col in ["dxy_level", "dxy_trend", "dxy_zscore", "dxy_momentum", "dxy_vs_ema50"]:
            out[col] = np.nan
        return out

    dxy_aligned = dxy.reindex(out.index, method="ffill")

    out["dxy_level"] = dxy_aligned
    dxy_ma20 = dxy_aligned.rolling(20, min_periods=5).mean()
    dxy_std = dxy_aligned.rolling(20, min_periods=5).std()
    out["dxy_zscore"] = (dxy_aligned - dxy_ma20) / dxy_std.replace(0, np.nan)
    out["dxy_momentum"] = dxy_aligned.pct_change(10)

    dxy_ema50 = dxy_aligned.ewm(span=50, adjust=False).mean()
    out["dxy_vs_ema50"] = (dxy_aligned - dxy_ema50) / dxy_ema50.replace(0, np.nan)

    # Trend: EMA10 vs EMA30
    ema10 = dxy_aligned.ewm(span=10, adjust=False).mean()
    ema30 = dxy_aligned.ewm(span=30, adjust=False).mean()
    trend = pd.Series(0, index=out.index)
    trend[ema10 > ema30 * 1.001] = 1
    trend[ema10 < ema30 * 0.999] = -1
    out["dxy_trend"] = trend

    return out


def add_yield_features(df: pd.DataFrame, start: str = "2024-01-01") -> pd.DataFrame:
    """
    Add US Treasury yield features — rate environment proxy.

    Features added:
        yield_10y:        raw 10Y yield
        yield_change_5:   5-bar change in yield
        yield_zscore:     20-period z-score
    """
    out = df.copy()
    tnx = _fetch_macro_series("^TNX", start=start)

    if tnx.empty:
        for col in ["yield_10y", "yield_change_5", "yield_zscore"]:
            out[col] = np.nan
        return out

    tnx_aligned = tnx.reindex(out.index, method="ffill")

    out["yield_10y"] = tnx_aligned
    out["yield_change_5"] = tnx_aligned.diff(5)
    tnx_ma = tnx_aligned.rolling(20, min_periods=5).mean()
    tnx_std = tnx_aligned.rolling(20, min_periods=5).std()
    out["yield_zscore"] = (tnx_aligned - tnx_ma) / tnx_std.replace(0, np.nan)

    return out


def add_cross_asset_features(
    df: pd.DataFrame,
    asset: str,
    all_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Add cross-asset correlation and relative strength features.

    Features added:
        corr_to_gold:    20-bar rolling correlation to XAUUSD returns
        corr_to_dxy:     20-bar rolling correlation to DXY returns
        relative_vol:    asset vol / average universe vol (>1 = more volatile)
        asset_momentum_rank: rank of asset's 20-bar return vs universe
    """
    out = df.copy()
    ret = out["close"].pct_change()

    # Correlation to DXY
    if "dxy_level" in out.columns:
        dxy_ret = out["dxy_level"].pct_change()
        out["corr_to_dxy"] = ret.rolling(20, min_periods=10).corr(dxy_ret)
    else:
        out["corr_to_dxy"] = np.nan

    # Correlation to Gold (from XAUUSD in all_data if available)
    if all_data and "XAUUSD" in all_data and asset != "XAUUSD":
        gold_ret = all_data["XAUUSD"]["close"].pct_change().reindex(out.index)
        out["corr_to_gold"] = ret.rolling(20, min_periods=10).corr(gold_ret)
    else:
        out["corr_to_gold"] = 0.0 if asset == "XAUUSD" else np.nan

    # Relative volatility vs universe
    if all_data and len(all_data) > 1:
        asset_vol = ret.rolling(20, min_periods=10).std()
        all_vols = []
        for a, adf in all_data.items():
            if a != asset:
                v = adf["close"].pct_change().rolling(20, min_periods=10).std()
                v = v.reindex(out.index)
                all_vols.append(v)
        if all_vols:
            avg_vol = pd.concat(all_vols, axis=1).mean(axis=1)
            out["relative_vol"] = asset_vol / avg_vol.replace(0, np.nan)
        else:
            out["relative_vol"] = 1.0
    else:
        out["relative_vol"] = 1.0

    return out


def add_sentiment_features(df: pd.DataFrame, asset: str = "EURUSD") -> pd.DataFrame:
    """
    Add rule-based sentiment proxy features derived from price action.

    Since we can't reliably scrape live news in backtesting, we compute
    sentiment proxies from market behavior that correlate with news-driven moves:

    Features added:
        gap_sentiment:      overnight gap direction/magnitude (news proxy)
        volume_sentiment:   volume spike + direction = institutional flow proxy
        momentum_breadth:   short vs medium momentum agreement (0-1)
        fear_greed_proxy:   composite sentiment 0 (fear) to 1 (greed)
    """
    out = df.copy()

    # Gap sentiment: large gaps often follow overnight news
    gap = (out["open"] - out["close"].shift(1)) / out["close"].shift(1)
    gap_ma = gap.rolling(20, min_periods=5).mean()
    gap_std = gap.rolling(20, min_periods=5).std().replace(0, np.nan)
    out["gap_sentiment"] = (gap - gap_ma) / gap_std  # z-score of gap

    # Volume sentiment: high volume + direction = conviction
    if "volume" in out.columns:
        vol_ma = out["volume"].rolling(20, min_periods=5).mean()
        vol_ratio = out["volume"] / vol_ma.replace(0, np.nan)
        price_dir = np.sign(out["close"] - out["open"])
        out["volume_sentiment"] = vol_ratio * price_dir
    else:
        out["volume_sentiment"] = 0.0

    # Momentum breadth: agreement between short and medium-term momentum
    ret5 = out["close"].pct_change(5)
    ret20 = out["close"].pct_change(20)
    agreement = (np.sign(ret5) == np.sign(ret20)).astype(float)
    # Strength = both positive and magnitude aligned
    out["momentum_breadth"] = agreement * np.clip(
        np.abs(ret5 / ret20.replace(0, np.nan)), 0, 2
    ) / 2

    # Fear/greed proxy: composite of vol, momentum, mean-reversion signals
    fear_greed = pd.Series(0.5, index=out.index)  # neutral baseline

    # Low VIX / low vol → greed
    if "vix_regime" in out.columns:
        fear_greed -= out["vix_regime"].fillna(1) * 0.1  # high VIX → fear

    # Strong uptrend → greed
    if "htf_trend" in out.columns:
        fear_greed += out["htf_trend"].fillna(0) * 0.15

    # RSI extremes
    if "rsi" in out.columns:
        rsi_norm = (out["rsi"].fillna(50) - 50) / 50  # -1 to +1
        fear_greed += rsi_norm * 0.1

    # Volume conviction
    fear_greed += out["volume_sentiment"].fillna(0).clip(-1, 1) * 0.05

    out["fear_greed_proxy"] = fear_greed.clip(0, 1)

    return out


def add_event_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based event risk features.

    Markets tend to be riskier around:
    - US trading session open/close
    - Month-end / quarter-end rebalancing
    - First Friday of month (NFP)
    - FOMC meeting days (typically 8x per year)

    Features added:
        is_month_end:     1 if within last 2 days of month
        is_quarter_end:   1 if within last 3 days of quarter
        is_nfp_week:      1 if first full week of month (NFP typically first Friday)
        day_of_week:      0=Mon to 4=Fri (weekend effect)
        hour_bucket:      session bucket (0=Asia, 1=London, 2=NY, 3=late)
        event_risk_score: composite 0-1 risk score
    """
    out = df.copy()
    idx = out.index

    # Calendar features
    out["day_of_week"] = idx.dayofweek
    out["is_month_end"] = ((idx.day >= (idx + pd.offsets.MonthEnd(0)).day - 1) |
                           (idx.is_month_start)).astype(float)

    # Simpler month-end: last 2 days
    days_to_month_end = pd.Series(
        [(pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthEnd(0)).day - d.day
         for d in idx],
        index=idx,
    )
    out["is_month_end"] = (days_to_month_end <= 2).astype(float)
    out["is_quarter_end"] = ((days_to_month_end <= 3) & (idx.month % 3 == 0)).astype(float)

    # NFP week: first 7 days of month
    out["is_nfp_week"] = (idx.day <= 7).astype(float)

    # Session bucket from hour
    hour = idx.hour
    out["hour_bucket"] = pd.cut(
        hour,
        bins=[-1, 7, 12, 20, 24],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Composite event risk score
    risk = pd.Series(0.0, index=idx)
    risk += out["is_month_end"] * 0.2
    risk += out["is_quarter_end"] * 0.15
    risk += out["is_nfp_week"] * 0.25
    # Monday/Friday slightly riskier
    risk += ((out["day_of_week"] == 0) | (out["day_of_week"] == 4)).astype(float) * 0.1

    out["event_risk_score"] = risk.clip(0, 1)

    return out


def add_all_macro_features(
    df: pd.DataFrame,
    asset: str = "UNKNOWN",
    all_data: Optional[Dict[str, pd.DataFrame]] = None,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Apply all macro/sentiment features in sequence."""
    out = df.copy()
    out = add_vix_features(out, start=start)
    out = add_dxy_features(out, start=start)
    out = add_yield_features(out, start=start)
    out = add_cross_asset_features(out, asset=asset, all_data=all_data)
    out = add_sentiment_features(out, asset=asset)
    out = add_event_risk_features(out)
    return out
