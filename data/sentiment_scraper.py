"""
News & sentiment data scraper — pulls live sentiment from free APIs.

Sources:
    1. Crypto Fear & Greed Index (alternative.me) — free, no key
    2. Yahoo Finance news headlines — free via RSS
    3. Simple keyword sentiment scoring for news headlines

For backtesting: uses price-action proxies (see macro_features.py).
For live trading: scrapes real-time data and produces a sentiment score
that the strategy can use to boost or reduce signal confidence.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

# Keyword-based sentiment scoring for financial headlines
BULLISH_KEYWORDS = [
    "rally", "surge", "soar", "jump", "gain", "rise", "bull", "breakout",
    "record high", "all-time high", "beat", "exceed", "strong", "boom",
    "upgrade", "outperform", "recovery", "rebound", "momentum", "growth",
    "dovish", "cut rate", "stimulus", "easing", "optimism", "confidence",
]

BEARISH_KEYWORDS = [
    "crash", "plunge", "drop", "fall", "decline", "bear", "breakdown",
    "record low", "miss", "weak", "recession", "downturn", "sell-off",
    "downgrade", "underperform", "crisis", "risk", "fear", "panic",
    "hawkish", "rate hike", "tightening", "inflation", "tariff", "war",
    "sanctions", "default", "collapse", "volatile", "uncertainty",
]

# Asset-specific keywords
ASSET_KEYWORDS = {
    "XAUUSD": ["gold", "precious metal", "safe haven", "xau"],
    "XAGUSD": ["silver", "precious metal", "xag"],
    "BTCUSD": ["bitcoin", "btc", "crypto", "cryptocurrency", "digital asset"],
    "ETHUSD": ["ethereum", "eth", "crypto", "defi", "smart contract"],
    "USOIL": ["oil", "crude", "wti", "opec", "petroleum", "energy"],
    "EURUSD": ["euro", "eur", "ecb", "eurozone"],
    "USDJPY": ["yen", "jpy", "boj", "japan"],
    "GBPJPY": ["pound", "gbp", "boe", "sterling"],
}


class SentimentScraper:
    """Scrapes and scores sentiment from multiple free sources."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/sentiment_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, dict] = {}

    def get_crypto_fear_greed(self) -> dict:
        """
        Fetch Crypto Fear & Greed Index from alternative.me.
        Returns: {value: 0-100, classification: str, timestamp: str}
        Free, no API key needed, 1 request/min limit.
        """
        try:
            import requests
            url = "https://api.alternative.me/fng/?limit=30&format=json"
            resp = requests.get(url, timeout=10)
            data = resp.json()

            if "data" in data:
                entries = data["data"]
                result = {
                    "current": int(entries[0]["value"]),
                    "classification": entries[0]["value_classification"],
                    "timestamp": entries[0]["timestamp"],
                    "history": [
                        {"value": int(e["value"]), "date": e["timestamp"]}
                        for e in entries[:30]
                    ],
                }
                # Compute trend (is fear/greed rising or falling?)
                if len(entries) >= 7:
                    recent_avg = np.mean([int(e["value"]) for e in entries[:3]])
                    older_avg = np.mean([int(e["value"]) for e in entries[3:7]])
                    result["trend"] = "rising" if recent_avg > older_avg else "falling"
                else:
                    result["trend"] = "neutral"

                log.info("Crypto F&G: %d (%s), trend=%s",
                         result["current"], result["classification"], result["trend"])
                return result
        except Exception as exc:
            log.warning("Crypto F&G fetch failed: %s", exc)

        return {"current": 50, "classification": "Neutral", "trend": "neutral", "history": []}

    def get_yahoo_news(self, asset: str) -> List[dict]:
        """
        Fetch recent news headlines from Yahoo Finance RSS for an asset.
        Free, no API key needed.
        """
        ticker_map = {
            "XAUUSD": "GC=F",
            "XAGUSD": "SI=F",
            "BTCUSD": "BTC-USD",
            "ETHUSD": "ETH-USD",
            "USOIL": "CL=F",
            "EURUSD": "EURUSD=X",
            "USDJPY": "USDJPY=X",
            "GBPJPY": "GBPJPY=X",
        }
        ticker = ticker_map.get(asset, asset)

        try:
            import requests
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            resp = requests.get(url, timeout=10)

            # Parse simple XML for titles
            headlines = []
            for match in re.finditer(r"<title><!\[CDATA\[(.*?)\]\]></title>", resp.text):
                title = match.group(1)
                if title and "Yahoo" not in title:
                    headlines.append({"title": title, "source": "yahoo"})

            # Fallback: try without CDATA
            if not headlines:
                for match in re.finditer(r"<title>(.*?)</title>", resp.text):
                    title = match.group(1)
                    if title and "Yahoo" not in title and "RSS" not in title:
                        headlines.append({"title": title, "source": "yahoo"})

            log.info("Yahoo news for %s: %d headlines", asset, len(headlines))
            return headlines[:20]  # limit to 20 most recent

        except Exception as exc:
            log.warning("Yahoo news fetch failed for %s: %s", asset, exc)
            return []

    def score_headline(self, headline: str, asset: str = "") -> float:
        """
        Score a headline from -1 (very bearish) to +1 (very bullish).
        Uses keyword matching with asset-specific weighting.
        """
        text = headline.lower()
        bull_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
        bear_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text)

        # Asset relevance bonus
        relevance = 0.5  # base relevance
        if asset in ASSET_KEYWORDS:
            for kw in ASSET_KEYWORDS[asset]:
                if kw in text:
                    relevance = 1.0
                    break

        total = bull_count + bear_count
        if total == 0:
            return 0.0

        raw_score = (bull_count - bear_count) / total  # -1 to +1
        return raw_score * relevance

    def get_sentiment_score(self, asset: str) -> dict:
        """
        Compute aggregate sentiment score for an asset.

        Returns:
            {
                score: float (-1 to +1),
                confidence: float (0 to 1),
                n_headlines: int,
                crypto_fg: int (0-100, only for crypto),
                regime_hint: str (risk_on, risk_off, neutral),
            }
        """
        headlines = self.get_yahoo_news(asset)
        scores = [self.score_headline(h["title"], asset) for h in headlines]

        # Filter out zero scores (irrelevant headlines)
        nonzero = [s for s in scores if abs(s) > 0.05]

        if nonzero:
            avg_score = np.mean(nonzero)
            confidence = min(1.0, len(nonzero) / 5.0)  # 5+ relevant headlines = full confidence
        else:
            avg_score = 0.0
            confidence = 0.0

        # Add crypto fear/greed for crypto assets
        crypto_fg = 50
        if asset in ("BTCUSD", "ETHUSD"):
            fg_data = self.get_crypto_fear_greed()
            crypto_fg = fg_data["current"]
            # Convert 0-100 to -1 to +1 scale
            fg_score = (crypto_fg - 50) / 50
            # Blend with headline sentiment
            if confidence > 0:
                avg_score = avg_score * 0.6 + fg_score * 0.4
            else:
                avg_score = fg_score
                confidence = 0.7  # F&G is reasonably reliable

        # Derive regime hint
        if avg_score > 0.3:
            regime_hint = "risk_on"
        elif avg_score < -0.3:
            regime_hint = "risk_off"
        else:
            regime_hint = "neutral"

        result = {
            "score": float(np.clip(avg_score, -1, 1)),
            "confidence": float(confidence),
            "n_headlines": len(headlines),
            "crypto_fg": crypto_fg,
            "regime_hint": regime_hint,
            "headlines": [h["title"] for h in headlines[:5]],
        }

        # Cache the result
        self._cache[asset] = {**result, "timestamp": datetime.utcnow().isoformat()}
        self._save_cache()

        return result

    def get_portfolio_sentiment(self, assets: List[str]) -> dict:
        """
        Get aggregate sentiment for the whole portfolio.

        Returns:
            {
                overall: float (-1 to +1),
                by_asset: {asset: score_dict},
                regime_suggestion: str,
                risk_multiplier: float (0.5 to 1.5),
            }
        """
        by_asset = {}
        scores = []
        for asset in assets:
            try:
                sent = self.get_sentiment_score(asset)
                by_asset[asset] = sent
                if sent["confidence"] > 0.2:
                    scores.append(sent["score"])
            except Exception as exc:
                log.warning("Sentiment failed for %s: %s", asset, exc)

        overall = float(np.mean(scores)) if scores else 0.0

        # Risk multiplier: scale position size by sentiment
        # Strong bullish sentiment → larger positions (1.3x)
        # Strong bearish/fear → smaller positions (0.7x)
        # Neutral → 1.0x
        risk_multiplier = 1.0 + overall * 0.3  # range: 0.7 to 1.3

        if overall > 0.3:
            regime_suggestion = "risk_on"
        elif overall < -0.3:
            regime_suggestion = "risk_off"
        else:
            regime_suggestion = "neutral"

        return {
            "overall": overall,
            "by_asset": by_asset,
            "regime_suggestion": regime_suggestion,
            "risk_multiplier": float(np.clip(risk_multiplier, 0.5, 1.5)),
        }

    def _save_cache(self):
        """Save sentiment cache to disk for inspection."""
        cache_path = self.cache_dir / "latest_sentiment.json"
        try:
            # Convert non-serializable types
            cache_data = {}
            for k, v in self._cache.items():
                cache_data[k] = {
                    key: val for key, val in v.items()
                    if isinstance(val, (str, int, float, list, bool))
                }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass

    def load_cache(self) -> Dict:
        """Load cached sentiment data."""
        cache_path = self.cache_dir / "latest_sentiment.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return {}
