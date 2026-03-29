"""
Disk-based OHLCV cache using Parquet files.

Cache key: {symbol}_{timeframe}_{start}_{end}.parquet
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


class DataCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, symbol: str, timeframe: str, start: str, end: str) -> str:
        raw = f"{symbol}_{timeframe}_{start}_{end}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _path(self, symbol: str, timeframe: str, start: str, end: str) -> Path:
        key = self._key(symbol, timeframe, start, end)
        return self.cache_dir / f"{symbol}_{timeframe}_{key}.parquet"

    def get(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        max_age_hours: float = 4.0,
    ) -> Optional[pd.DataFrame]:
        path = self._path(symbol, timeframe, start, end)
        if not path.exists():
            return None

        age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        if age_hours > max_age_hours:
            log.debug("Cache expired for %s %s (%.1fh old)", symbol, timeframe, age_hours)
            return None

        try:
            df = pd.read_parquet(path)
            log.debug("Cache hit: %s %s (%d bars)", symbol, timeframe, len(df))
            return df
        except Exception as exc:
            log.warning("Cache read failed for %s: %s", path, exc)
            return None

    def put(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> None:
        path = self._path(symbol, timeframe, start, end)
        try:
            df.to_parquet(path)
            log.debug("Cached %s %s -> %s", symbol, timeframe, path.name)
        except Exception as exc:
            log.warning("Cache write failed: %s", exc)

    def age_hours(self, symbol: str, timeframe: str, start: str, end: str) -> float:
        """Return age of cache file in hours, or inf if not found."""
        path = self._path(symbol, timeframe, start, end)
        if not path.exists():
            return float("inf")
        return (datetime.now().timestamp() - path.stat().st_mtime) / 3600

    def invalidate(self, symbol: str, timeframe: Optional[str] = None) -> int:
        """Remove cached files for a symbol (and optionally timeframe)."""
        pattern = f"{symbol}_*" if timeframe is None else f"{symbol}_{timeframe}_*"
        removed = 0
        for f in self.cache_dir.glob(pattern + ".parquet"):
            f.unlink()
            removed += 1
        return removed

    def clear_all(self) -> int:
        removed = sum(1 for f in self.cache_dir.glob("*.parquet") if f.unlink() is None)
        return removed
