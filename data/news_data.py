"""
Economic calendar / news event loader.

Supported sources:
  - CSV file (fallback, always available)
  - Trading Economics API (requires API key)
  - JBlanked API (free, no key required)
  - Alpha Vantage news sentiment (free, 25 req/day)
  - Alternative.me Fear & Greed Index (free, no key)

CSV format (events.csv):
  datetime,currency,impact,event
  2024-01-26 13:30:00,USD,high,US GDP Q4 Advance
  2024-01-26 13:30:00,USD,high,US PCE Price Index
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from config.settings import SETTINGS
from utils.logger import get_logger

log = get_logger(__name__)

# Expected CSV columns
REQUIRED_COLS = {"datetime", "currency", "impact", "event"}
IMPACT_LEVELS = {"high": 3, "medium": 2, "low": 1}

# Currency mapping: which currencies affect which instruments
INSTRUMENT_CURRENCIES = {
    "XAUUSD": ["USD"],
    "XAGUSD": ["USD"],
    "BTCUSD": ["USD"],
    "ETHUSD": ["USD"],
    "EURJPY": ["EUR", "JPY"],
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
}


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


def load_news_jblanked(
    currencies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load today's economic calendar from JBlanked API (free, no key).

    Sources: MQL5, ForexFactory, FXStreet aggregated.
    Rate limit: 1 request/second.
    """
    # JBlanked API requires auth as of March 2026. Try pip-installed package first,
    # then fall back to direct API with user-agent header.
    urls = [
        "https://www.jblanked.com/news/api/mql5/calendar/today/",
    ]
    all_events = []

    # Try jb-news Python package first (pip install jb-news)
    try:
        from jb_news import JBlanked
        jb = JBlanked()
        data = jb.mql5_calendar_today()
        if data:
            for ev in data:
                currency = str(ev.get("currency", "")).upper().strip()
                impact_raw = str(ev.get("importance", "low")).lower()
                if impact_raw in ("3", "high", "red"):
                    impact = "high"
                elif impact_raw in ("2", "medium", "orange"):
                    impact = "medium"
                else:
                    impact = "low"
                all_events.append({
                    "datetime": ev.get("date", ev.get("datetime", "")),
                    "currency": currency[:3],
                    "impact": impact,
                    "event": ev.get("name", ev.get("title", "Unknown")),
                })
    except ImportError:
        log.debug("jb-news package not installed — trying HTTP endpoints")
    except Exception as exc:
        log.warning("jb-news package failed: %s — trying HTTP endpoints", exc)

    # Fallback to HTTP API
    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "TradingFramework/1.0"})
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list):
                events = data
            elif isinstance(data, dict):
                events = data.get("events", data.get("data", []))
            else:
                continue

            for ev in events:
                # Normalize fields across MQL5 and FF formats
                dt_str = ev.get("date") or ev.get("datetime") or ev.get("time", "")
                currency = ev.get("currency", ev.get("country", "")).upper().strip()
                impact_raw = str(ev.get("impact", ev.get("importance", "low"))).lower()
                event_name = ev.get("title", ev.get("event", ev.get("name", "Unknown")))

                # Normalize impact
                if impact_raw in ("3", "high", "red"):
                    impact = "high"
                elif impact_raw in ("2", "medium", "orange", "moderate"):
                    impact = "medium"
                else:
                    impact = "low"

                all_events.append({
                    "datetime": dt_str,
                    "currency": currency[:3] if len(currency) >= 3 else currency,
                    "impact": impact,
                    "event": event_name,
                })
        except Exception as exc:
            log.warning("JBlanked API %s failed: %s", url, exc)

    if not all_events:
        log.warning("JBlanked returned no events")
        return _empty_calendar()

    df = pd.DataFrame(all_events)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["impact_score"] = df["impact"].map(IMPACT_LEVELS).fillna(0).astype(int)

    if currencies:
        df = df[df["currency"].isin(currencies)]

    df = df.drop_duplicates(subset=["datetime", "currency", "event"])
    df = df.sort_values("datetime").reset_index(drop=True)
    log.info("JBlanked: loaded %d events (%d high impact)",
             len(df), (df["impact"] == "high").sum())
    return df


def load_fear_greed_index() -> Dict:
    """
    Fetch the crypto Fear & Greed Index from Alternative.me (free, no key).

    Returns dict with:
        value: int 0-100 (0=extreme fear, 100=extreme greed)
        label: str ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")
        timestamp: datetime
        signal_mult: float (multiplier for crypto signal strength)
    """
    try:
        # Try multiple endpoints (Alternative.me has changed URLs over time)
        fgi_urls = [
            "https://api.alternative.me/fgi/?limit=1",
            "https://api.alternative.me/fgi/?limit=1&format=json",
        ]
        data = None
        for fgi_url in fgi_urls:
            try:
                resp = requests.get(fgi_url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()["data"][0]
                    break
            except Exception:
                continue

        if data is None:
            raise ConnectionError("All Fear & Greed endpoints failed")

        data = data  # already set above
        value = int(data["value"])
        label = data["value_classification"]

        # Compute signal multiplier:
        # Extreme fear (<20): contrarian bullish boost, bearish reduction
        # Extreme greed (>80): contrarian bearish boost, bullish reduction
        # Neutral (30-70): no adjustment
        if value < 20:
            long_mult, short_mult = 1.3, 0.7
        elif value < 30:
            long_mult, short_mult = 1.15, 0.85
        elif value > 80:
            long_mult, short_mult = 0.6, 1.3
        elif value > 70:
            long_mult, short_mult = 0.85, 1.15
        else:
            long_mult, short_mult = 1.0, 1.0

        result = {
            "value": value,
            "label": label,
            "timestamp": datetime.fromtimestamp(int(data["timestamp"]), tz=timezone.utc),
            "long_mult": long_mult,
            "short_mult": short_mult,
        }
        log.info("Fear & Greed Index: %d (%s) → long=%.2fx, short=%.2fx",
                 value, label, long_mult, short_mult)
        return result
    except Exception as exc:
        log.warning("Fear & Greed API failed: %s — returning neutral", exc)
        return {"value": 50, "label": "Neutral", "timestamp": datetime.now(tz=timezone.utc),
                "long_mult": 1.0, "short_mult": 1.0}


def load_alpha_vantage_sentiment(
    tickers: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Fetch news sentiment from Alpha Vantage (free, 25 req/day).

    Returns dict of ticker/topic -> sentiment score (-1.0 to +1.0).
    Positive = bullish, negative = bearish.
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        log.debug("Alpha Vantage API key not configured — skipping sentiment")
        return {}

    # Map asset tickers to AV topics for better coverage.
    # AV topics: blockchain, economy_fiscal, economy_monetary, finance,
    #            energy_transportation, financial_markets, etc.
    default_topics = ["blockchain", "economy_monetary", "financial_markets"]
    topics = topics or default_topics
    topics_param = ",".join(topics)

    try:
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": topics_param,
            "sort": "LATEST",
            "limit": "50",
            "apikey": api_key,
        }
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if "feed" not in data:
            log.warning("Alpha Vantage returned no feed: %s", data.get("Note", data.get("Information", "unknown")))
            return {}

        # Aggregate sentiment per ticker, grouping related tickers.
        # Map AV ticker formats to our instrument names.
        ticker_map = {
            "BTC": "BTCUSD", "CRYPTO:BTC": "BTCUSD",
            "ETH": "ETHUSD", "CRYPTO:ETH": "ETHUSD",
            "FOREX:XAUUSD": "XAUUSD", "GOLD": "XAUUSD",
            "FOREX:XAGUSD": "XAGUSD", "SILVER": "XAGUSD",
            "FOREX:EURJPY": "EURJPY",
        }
        instrument_scores: Dict[str, List[float]] = {}
        for article in data["feed"]:
            for ts in article.get("ticker_sentiment", []):
                ticker = ts["ticker"]
                score = float(ts["ticker_sentiment_score"])
                # Try direct mapping
                mapped = ticker_map.get(ticker)
                if not mapped:
                    # Try partial match (e.g. "BTC" in "BTCUSD")
                    for key, inst in ticker_map.items():
                        if key in ticker or ticker in key:
                            mapped = inst
                            break
                if mapped:
                    instrument_scores.setdefault(mapped, []).append(score)

        result = {}
        for instrument, scores in instrument_scores.items():
            avg = sum(scores) / len(scores)
            result[instrument] = round(avg, 4)

        log.info("Alpha Vantage sentiment: %s", result)
        return result
    except Exception as exc:
        log.warning("Alpha Vantage sentiment failed: %s", exc)
        return {}


def get_sentiment_overlay(asset: str, side: str) -> float:
    """
    Get a combined sentiment multiplier for an asset and trade direction.

    Returns a float multiplier (0.5 to 1.5) to apply to signal strength.
    Uses Fear & Greed for crypto, Alpha Vantage for forex/metals.
    """
    mult = 1.0

    # Crypto: use Fear & Greed Index
    if asset in ("BTCUSD", "ETHUSD"):
        fgi = load_fear_greed_index()
        mult *= fgi["long_mult"] if side == "long" else fgi["short_mult"]

    # All assets: use Alpha Vantage sentiment if available
    if asset in ("XAUUSD", "BTCUSD", "ETHUSD", "XAGUSD", "EURJPY"):
        sentiments = load_alpha_vantage_sentiment()
        score = sentiments.get(asset, 0.0)
        # Sentiment alignment: bullish sentiment + long = boost, bearish + long = reduce
        if side == "long":
            mult *= 1.0 + 0.2 * score  # +-20% based on sentiment
        else:
            mult *= 1.0 - 0.2 * score

    return max(0.5, min(1.5, mult))


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
        provider: "csv", "trading_economics", or "jblanked"
    """
    if provider == "jblanked":
        try:
            return load_news_jblanked(currencies)
        except Exception as exc:
            log.warning("JBlanked failed (%s) — falling back to CSV", exc)

    if provider == "trading_economics":
        try:
            return load_news_trading_economics(start, end, currencies)
        except Exception as exc:
            log.warning("Trading Economics failed (%s) — falling back to CSV", exc)

    return load_news_csv(csv_path)
