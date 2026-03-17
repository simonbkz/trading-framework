"""
News event filter — blocks trading around high-impact economic events.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from config.settings import SETTINGS
from data.news_data import get_news_events
from utils.logger import get_logger

log = get_logger(__name__)


class NewsFilter:
    """
    Marks periods around high-impact news events as non-tradable.

    Usage:
        nf = NewsFilter()
        nf.load("2023-01-01", "2025-12-31", currencies=["USD", "EUR"])
        blocked = nf.is_blocked(df.index)   # boolean Series
    """

    def __init__(
        self,
        pre_event_minutes: int = None,
        post_event_minutes: int = None,
        block_high: bool = None,
        block_medium: bool = None,
        currencies: Optional[List[str]] = None,
    ):
        s = SETTINGS.news_filter
        self.pre_minutes    = pre_event_minutes  or s.pre_event_minutes
        self.post_minutes   = post_event_minutes or s.post_event_minutes
        self.block_high     = block_high   if block_high   is not None else s.block_high_impact
        self.block_medium   = block_medium if block_medium is not None else s.block_medium_impact
        self.currencies     = currencies or s.currencies_of_interest
        self._events: Optional[pd.DataFrame] = None
        self._blocked_ranges: list = []   # list of (start, end) tuples

    def load(
        self,
        start: str,
        end: str,
        provider: str = "csv",
        csv_path: Optional[Path] = None,
    ) -> "NewsFilter":
        """Load economic calendar and build blocked time ranges."""
        events = get_news_events(
            start=start,
            end=end,
            currencies=self.currencies,
            provider=provider,
            csv_path=csv_path,
        )
        self._events = events

        # Select events to block
        if self.block_high and self.block_medium:
            mask = events["impact"].isin(["high", "medium"])
        elif self.block_high:
            mask = events["impact"] == "high"
        elif self.block_medium:
            mask = events["impact"] == "medium"
        else:
            mask = pd.Series(False, index=events.index)

        relevant = events[mask]
        self._blocked_ranges = []
        for _, row in relevant.iterrows():
            ev_time = row["datetime"]
            block_start = ev_time - pd.Timedelta(minutes=self.pre_minutes)
            block_end   = ev_time + pd.Timedelta(minutes=self.post_minutes)
            self._blocked_ranges.append((block_start, block_end))

        log.info(
            "NewsFilter: loaded %d events, blocking %d windows (%d/%d minutes pre/post)",
            len(events), len(self._blocked_ranges),
            self.pre_minutes, self.post_minutes,
        )
        return self

    def is_blocked(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        """
        Return a boolean Series — True where trading is blocked.

        Args:
            timestamps: DatetimeIndex to check

        Returns:
            pd.Series(bool) aligned to timestamps
        """
        if not self._blocked_ranges:
            return pd.Series(False, index=timestamps)

        ts = pd.Series(timestamps, index=timestamps)
        blocked = pd.Series(False, index=timestamps)

        for start, end in self._blocked_ranges:
            blocked |= (ts >= start) & (ts <= end)

        n_blocked = blocked.sum()
        if n_blocked > 0:
            log.debug("NewsFilter: %d bars blocked out of %d", n_blocked, len(timestamps))

        return blocked

    def next_event(self, from_time: pd.Timestamp) -> Optional[dict]:
        """Return the next scheduled event after `from_time`."""
        if self._events is None or self._events.empty:
            return None
        future = self._events[self._events["datetime"] > from_time]
        if future.empty:
            return None
        row = future.iloc[0]
        return {
            "datetime": row["datetime"],
            "currency": row["currency"],
            "impact":   row["impact"],
            "event":    row["event"],
            "minutes_until": int((row["datetime"] - from_time).total_seconds() / 60),
        }

    def upcoming_blocks(self, from_time: pd.Timestamp, hours_ahead: float = 4.0) -> list:
        """Return all blocked windows within `hours_ahead` hours of now."""
        horizon = from_time + pd.Timedelta(hours=hours_ahead)
        return [
            (s, e) for s, e in self._blocked_ranges
            if s >= from_time and s <= horizon
        ]
