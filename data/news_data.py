"""
Economic calendar / news event loader.

Supported sources:
  - CSV file (fallback, always available)
  - Trading Economics API (requires API key)

CSV format (events.csv):
  datetime,currency,impact,event
  2024-01-26 13:30:00,USD,high,US GDP Q4 Advance
  2024-01-26 13:30:00,USD,high,US PCE Price Index
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config.settings import SETTINGS
from utils.logger import get_logger

log = get_logger(__name__)

# Expected CSV columns
REQUIRED_COLS = {"datetime", "currency", "impact", "event"}
IMPACT_LEVELS = {"high": 3, "medium": 2, "low": 1}


def load_news_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load economic calendar from CSV.

    Returns DataFrame with columns:
        datetime (UTC, DatetimeIndex), currency, impact, event, impact_score
    """
    path = path or SETTINGS.data.news_csv_path
    path = Path(path)

    if not path.exists():
        log.warning("News CSV not found at %s — returning empty calendar", path)
        return _empty_calendar()

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"News CSV missing columns: {missing}. Found: {list(df.columns)}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert("UTC")

    df["impact"] = df["impact"].str.lower().str.strip()
    df["impact_score"] = df["impact"].map(IMPACT_LEVELS).fillna(0).astype(int)

    df = df.sort_values("datetime").reset_index(drop=True)
    log.info("Loaded %d news events from %s", len(df), path)
    return df


def load_news_trading_economics(
    start: str,
    end: str,
    currencies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load economic calendar from Trading Economics API.
    Requires TRADING_ECONOMICS_API_KEY and TRADING_ECONOMICS_API_SECRET env vars.
    """
    try:
        import tradingeconomics as te
        api_key    = os.environ["TRADING_ECONOMICS_API_KEY"]
        api_secret = os.environ.get("TRADING_ECONOMICS_API_SECRET", "")
        te.login(api_key + ":" + api_secret)
    except ImportError:
        raise ImportError("Install tradingeconomics: pip install tradingeconomics")
    except KeyError as e:
        raise EnvironmentError(f"Missing environment variable: {e}")

    cal = te.getCalendarData(d1=start, d2=end, output_type="df")
    if cal is None or cal.empty:
        log.warning("Trading Economics returned no calendar data for %s to %s", start, end)
        return _empty_calendar()

    cal.columns = cal.columns.str.lower()
    cal = cal.rename(columns={
        "date": "datetime",
        "country": "currency",   # TE uses country; we store under 'currency'
        "importance": "impact",
        "category": "event",
    })

    if "datetime" in cal.columns:
        cal["datetime"] = pd.to_datetime(cal["datetime"], utc=True)
    cal["impact"] = (
        cal["impact"]
        .astype(str)
        .str.replace("3", "high")
        .str.replace("2", "medium")
        .str.replace("1", "low")
        .str.lower()
    )
    cal["impact_score"] = cal["impact"].map(IMPACT_LEVELS).fillna(0).astype(int)

    if currencies:
        cal = cal[cal["currency"].isin(currencies)]

    return cal[["datetime", "currency", "impact", "impact_score", "event"]].sort_values("datetime")


def _empty_calendar() -> pd.DataFrame:
    return pd.DataFrame(columns=["datetime", "currency", "impact", "impact_score", "event"])


def get_news_events(
    start: str,
    end: str,
    currencies: Optional[List[str]] = None,
    provider: str = "csv",
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Unified news data accessor.

    Args:
        provider: "csv" or "trading_economics"
    """
    if provider == "trading_economics":
        try:
            return load_news_trading_economics(start, end, currencies)
        except Exception as exc:
            log.warning("Trading Economics failed (%s) — falling back to CSV", exc)

    return load_news_csv(csv_path)
