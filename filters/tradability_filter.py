"""
TradabilityFilter — master filter that combines all sub-filters.

Returns a single boolean Series indicating where all conditions
are met for trading.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from filters.news_filter import NewsFilter
from filters.session_filter import SessionFilter
from filters.liquidity_filter import LiquidityFilter
from utils.logger import get_logger

log = get_logger(__name__)


class TradabilityFilter:
    """
    Combines news, session, and liquidity filters.

    Usage:
        tf = TradabilityFilter()
        tf.load_news("2023-01-01", "2025-12-31")
        can_trade = tf.is_tradable(df)   # boolean Series
    """

    def __init__(
        self,
        sessions: Optional[list] = None,
        news_pre_minutes: int = 60,
        news_post_minutes: int = 30,
        block_high_impact: bool = True,
        block_medium_impact: bool = False,
        min_volume_ratio: float = 0.20,
        exclude_weekends: bool = True,
    ):
        self.news_filter      = NewsFilter(
            pre_event_minutes=news_pre_minutes,
            post_event_minutes=news_post_minutes,
            block_high=block_high_impact,
            block_medium=block_medium_impact,
        )
        self.session_filter   = SessionFilter(sessions=sessions)
        self.liquidity_filter = LiquidityFilter(
            min_volume_ratio=min_volume_ratio,
            exclude_weekends=exclude_weekends,
        )
        self._news_loaded = False

    def load_news(
        self,
        start: str,
        end: str,
        provider: str = "csv",
    ) -> "TradabilityFilter":
        self.news_filter.load(start=start, end=end, provider=provider)
        self._news_loaded = True
        return self

    def is_tradable(
        self,
        df: pd.DataFrame,
        apply_session: bool = True,
        apply_news: bool = True,
        apply_liquidity: bool = True,
    ) -> pd.Series:
        """
        Return boolean Series — True where all enabled filters pass.

        Args:
            df:              OHLCV DataFrame
            apply_session:   Whether to apply session filter
            apply_news:      Whether to apply news filter (requires load_news())
            apply_liquidity: Whether to apply liquidity filter

        Returns:
            pd.Series(bool) aligned to df.index
        """
        tradable = pd.Series(True, index=df.index)

        if apply_session:
            tradable &= self.session_filter.is_active(df.index)

        if apply_news and self._news_loaded:
            tradable &= ~self.news_filter.is_blocked(df.index)

        if apply_liquidity:
            tradable &= self.liquidity_filter.is_liquid(df)

        pct = tradable.mean() * 100
        log.debug(
            "TradabilityFilter: %.1f%% of %d bars are tradable",
            pct, len(df),
        )
        return tradable

    def filter_df(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Return only tradable rows of df."""
        mask = self.is_tradable(df, **kwargs)
        return df[mask]

    def summary(self, df: pd.DataFrame) -> dict:
        """Return filter breakdown statistics."""
        sess  = self.session_filter.is_active(df.index)
        news  = ~self.news_filter.is_blocked(df.index) if self._news_loaded else pd.Series(True, index=df.index)
        liq   = self.liquidity_filter.is_liquid(df)
        all_f = sess & news & liq
        return {
            "total_bars":       len(df),
            "session_ok":       int(sess.sum()),
            "news_ok":          int(news.sum()),
            "liquidity_ok":     int(liq.sum()),
            "all_filters_pass": int(all_f.sum()),
            "tradable_pct":     round(all_f.mean() * 100, 1),
        }
