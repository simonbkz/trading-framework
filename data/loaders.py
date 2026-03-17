"""
Provider-agnostic data loader.

Returns unified OHLCV DataFrames with:
  - DatetimeIndex in UTC
  - Columns: open, high, low, close, volume  (all lowercase)
  - No gaps for non-trading periods (caller decides on reindexing)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from config.assets import get_asset, AssetConfig
from config.providers import get_provider
from data.cache import DataCache
from utils.helpers import normalize_ohlcv_columns
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _load_yfinance(
    ticker: str,
    start: str,
    end: str,
    interval: str,
) -> pd.DataFrame:
    """Load via yfinance. interval: 1m 5m 15m 30m 1h 4h 1d 1wk 1mo"""
    import datetime as dt
    import yfinance as yf

    # yfinance doesn't natively support 4h — download 1h and resample
    resample_to_4h = False
    yf_interval = interval
    if interval in ("4h", "4H"):
        yf_interval = "1h"
        resample_to_4h = True

    # yfinance limits intraday data to the last 730 days
    intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
    if yf_interval in intraday_intervals:
        max_lookback = dt.date.today() - dt.timedelta(days=729)
        req_start = dt.date.fromisoformat(str(start)[:10])
        if req_start < max_lookback:
            log.info(
                "yfinance %s: clamping start from %s to %s (730-day intraday limit)",
                ticker, req_start, max_lookback,
            )
            start = max_lookback.isoformat()

    df = yf.download(ticker, start=start, end=end, interval=yf_interval,
                     auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    # yfinance >= 0.2.36 returns MultiIndex columns (Price, Ticker); flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = normalize_ohlcv_columns(df)
    # Ensure UTC index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()

    # Resample 1h -> 4h if needed
    if resample_to_4h:
        df = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open"])
        log.info("Resampled 1h -> 4h: %d bars for %s", len(df), ticker)

    return df


def _load_alpha_vantage(
    symbol: str,
    start: str,
    end: str,
    interval: str,
) -> pd.DataFrame:
    """Load via Alpha Vantage REST API."""
    import requests
    from utils.env_loader import require_env
    api_key = require_env("ALPHA_VANTAGE_API_KEY", "Alpha Vantage API key")

    # Map interval to AV function
    av_intervals = {"1m": "1min", "5m": "5min", "15m": "15min",
                    "30m": "30min", "1h": "60min"}
    if interval in av_intervals:
        func = "TIME_SERIES_INTRADAY"
        params = {
            "function": func,
            "symbol": symbol,
            "interval": av_intervals[interval],
            "outputsize": "full",
            "apikey": api_key,
            "datatype": "json",
        }
    elif interval in ("1d", "D"):
        params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol,
                  "outputsize": "full", "apikey": api_key, "datatype": "json"}
    else:
        raise ValueError(f"Alpha Vantage: unsupported interval '{interval}'")

    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Find time series key
    ts_key = [k for k in data if "Time Series" in k]
    if not ts_key:
        raise ValueError(f"Alpha Vantage error: {data.get('Note', data)}")
    ts = data[ts_key[0]]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.split(". ")[1].lower() for c in df.columns]
    df = df.astype(float).sort_index()
    if "adjusted_close" in df.columns:
        df["close"] = df["adjusted_close"]
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def _load_twelve_data(
    symbol: str,
    start: str,
    end: str,
    interval: str,
) -> pd.DataFrame:
    """Load via Twelve Data REST API."""
    import requests
    from utils.env_loader import require_env
    api_key = require_env("TWELVE_DATA_API_KEY", "Twelve Data API key")

    td_intervals = {"1m": "1min", "5m": "5min", "15m": "15min",
                    "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1day"}
    td_interval = td_intervals.get(interval, interval)

    params = {
        "symbol": symbol,
        "interval": td_interval,
        "start_date": start,
        "end_date": end,
        "outputsize": 5000,
        "apikey": api_key,
        "format": "json",
    }
    resp = requests.get("https://api.twelvedata.com/time_series",
                        params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "values" not in data:
        raise ValueError(f"Twelve Data error: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df.index = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop(columns=["datetime"])
    df.columns = [c.lower() for c in df.columns]
    df = df.astype(float).sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _load_polygon(
    symbol: str,
    start: str,
    end: str,
    interval: str,
) -> pd.DataFrame:
    """Load via Polygon.io REST API."""
    import requests
    from utils.env_loader import require_env
    api_key = require_env("POLYGON_API_KEY", "Polygon.io API key")

    mul_map = {"1m": (1,"minute"), "5m": (5,"minute"), "15m": (15,"minute"),
               "30m": (30,"minute"), "1h": (1,"hour"), "4h": (4,"hour"), "1d": (1,"day")}
    if interval not in mul_map:
        raise ValueError(f"Polygon: unsupported interval '{interval}'")
    mult, span = mul_map[interval]

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("resultsCount", 0) == 0:
        raise ValueError(f"Polygon returned no results for {symbol}")
    df = pd.DataFrame(data["results"])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def _load_csv(
    symbol: str,
    csv_dir: Path,
    start: str,
    end: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Load from CSV file.
    Expected filename pattern: {symbol}_{timeframe}.csv or {symbol}.csv
    Expected columns: datetime/date/time + open/high/low/close/volume
    """
    candidates = [
        csv_dir / f"{symbol}_{timeframe}.csv",
        csv_dir / f"{symbol}.csv",
        csv_dir / f"{symbol.lower()}_{timeframe}.csv",
        csv_dir / f"{symbol.lower()}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No CSV found for {symbol} in {csv_dir}. "
            f"Expected one of: {[str(c) for c in candidates]}"
        )

    df = pd.read_csv(path)
    # Detect datetime column
    dt_col = next(
        (c for c in df.columns if c.lower() in ("datetime", "date", "time", "timestamp")),
        df.columns[0],
    )
    df.index = pd.to_datetime(df[dt_col])
    df = df.drop(columns=[dt_col])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = normalize_ohlcv_columns(df)
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df.loc[start:end]


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    start: str = "2022-01-01",
    end: Optional[str] = None,
    provider: str = "yfinance",
    cache: Optional[DataCache] = None,
    cache_max_age_hours: float = 4.0,
    csv_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol.

    Args:
        symbol:    Asset symbol as defined in config/assets.py (e.g. "EURUSD").
        timeframe: Data frequency: 1m 5m 15m 30m 1h 4h 1d
        start:     ISO date string "YYYY-MM-DD"
        end:       ISO date string; defaults to today
        provider:  yfinance | alpha_vantage | twelve_data | polygon | csv
        cache:     DataCache instance; None to disable caching
        cache_max_age_hours: Max cache age before re-fetch
        csv_dir:   Override CSV search directory (for csv provider)

    Returns:
        pd.DataFrame with DatetimeIndex(UTC), columns: open high low close volume
    """
    import datetime as dt
    if end is None:
        # Use tomorrow so yfinance includes all of today's intraday bars
        # (yfinance treats 'end' as exclusive — end='2026-03-16' excludes March 16)
        end = (dt.date.today() + dt.timedelta(days=1)).isoformat()

    asset: AssetConfig = get_asset(symbol)

    # Check cache first
    if cache is not None:
        cached = cache.get(symbol, timeframe, start, end, max_age_hours=cache_max_age_hours)
        if cached is not None:
            return cached

    log.info("Fetching %s %s from %s to %s via %s", symbol, timeframe, start, end, provider)

    if provider == "yfinance":
        ticker = asset.yfinance_ticker
        df = _load_yfinance(ticker, start, end, timeframe)

    elif provider == "alpha_vantage":
        df = _load_alpha_vantage(asset.alpha_vantage_symbol, start, end, timeframe)

    elif provider == "twelve_data":
        df = _load_twelve_data(asset.twelve_data_symbol, start, end, timeframe)

    elif provider == "polygon":
        df = _load_polygon(asset.polygon_ticker, start, end, timeframe)

    elif provider == "csv":
        _dir = csv_dir or Path(os.getenv("MARKET_DATA_CSV_DIR", "./data/market"))
        df = _load_csv(symbol, _dir, start, end, timeframe)

    else:
        raise ValueError(f"Unknown provider '{provider}'")

    log.info("Loaded %d bars for %s %s", len(df), symbol, timeframe)

    # Store in cache
    if cache is not None and not df.empty:
        cache.put(df, symbol, timeframe, start, end)

    return df
