"""
MarketDataService — the main entry point for market data in the framework.

Handles:
- Multi-asset data loading
- Multiple timeframe resolution (e.g. H1 signals + H4 context)
- Unified bar alignment
- Automatic provider fallback chain
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.assets import DEFAULT_ASSETS
from config.settings import SETTINGS
from data.cache import DataCache
from data.loaders import load_ohlcv
from data.preprocessing import preprocess_ohlcv
from utils.logger import get_logger

log = get_logger(__name__)


class MarketDataService:
    """
    Facade for loading and pre-processing market data.

    Usage:
        svc = MarketDataService()
        df = svc.get("EURUSD", "1h", start="2023-01-01")
        multi = svc.get_multi(["EURUSD", "USDJPY"], "1h")
    """

    PROVIDER_FALLBACK_CHAIN = ["yfinance", "csv"]

    def __init__(
        self,
        provider: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_max_age_hours: float = 4.0,
        csv_dir: Optional[Path] = None,
    ):
        self.provider = provider or SETTINGS.data.provider
        self.cache = DataCache(cache_dir or SETTINGS.data.cache_dir)
        self.cache_max_age_hours = cache_max_age_hours
        self.csv_dir = csv_dir or SETTINGS.data.market_csv_dir

    def get(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: str = "2022-01-01",
        end: Optional[str] = None,
        preprocess: bool = True,
    ) -> pd.DataFrame:
        """
        Load OHLCV for a single symbol, with preprocessing.

        Falls back to yfinance → csv if primary provider fails.
        """
        providers = self._build_fallback_chain()

        for prov in providers:
            try:
                df = load_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    provider=prov,
                    cache=self.cache,
                    cache_max_age_hours=self.cache_max_age_hours,
                    csv_dir=self.csv_dir,
                )
                if df is not None and not df.empty:
                    if preprocess:
                        df = preprocess_ohlcv(df, symbol=symbol, timeframe=timeframe)
                    return df
            except (ImportError, EnvironmentError):
                log.warning("Provider '%s' unavailable for %s, trying fallback", prov, symbol)
            except Exception as exc:
                log.warning("Provider '%s' failed for %s: %s — trying fallback", prov, symbol, exc)

        raise RuntimeError(
            f"All providers failed for {symbol} {timeframe}. "
            "Check your data provider configuration or supply a CSV file."
        )

    def get_multi(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        start: str = "2022-01-01",
        end: Optional[str] = None,
        preprocess: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple symbols, returning {symbol: df} dict."""
        symbols = symbols or DEFAULT_ASSETS
        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                result[sym] = self.get(sym, timeframe, start, end, preprocess)
            except Exception as exc:
                log.error("Failed to load %s: %s", sym, exc)
        return result

    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str],
        start: str = "2022-01-01",
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load one symbol at multiple timeframes."""
        return {
            tf: self.get(symbol, tf, start, end)
            for tf in timeframes
        }

    def get_aligned_pair(
        self,
        symbol: str,
        signal_tf: str = "1h",
        context_tf: str = "4h",
        start: str = "2022-01-01",
        end: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (signal_df, context_df) with context bars merged onto signal bars.
        Context columns are prefixed with '{context_tf}_'.
        """
        sig = self.get(symbol, signal_tf, start, end)
        ctx = self.get(symbol, context_tf, start, end)

        # Rename context columns
        ctx_renamed = ctx.add_prefix(f"{context_tf}_")

        # Merge-as-of: for each signal bar, take the last available context bar
        sig = sig.sort_index()
        ctx_renamed = ctx_renamed.sort_index()
        merged = pd.merge_asof(
            sig.reset_index(),
            ctx_renamed.reset_index().rename(columns={"index": "ctx_time"}),
            left_on="index",
            right_on="ctx_time",
            direction="backward",
        ).set_index("index")
        merged.index.name = None
        return sig, merged

    def _build_fallback_chain(self) -> List[str]:
        chain = [self.provider]
        for p in self.PROVIDER_FALLBACK_CHAIN:
            if p != self.provider:
                chain.append(p)
        return chain
