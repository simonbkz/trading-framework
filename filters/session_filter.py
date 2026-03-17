"""
Trading session filter — restricts entry to preferred market hours.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

from config.settings import SETTINGS
from utils.logger import get_logger

log = get_logger(__name__)

# Standard session windows (UTC hours)
SESSION_UTC: Dict[str, Tuple[int, int]] = {
    "sydney":   (21, 6),    # 21:00–06:00 UTC
    "tokyo":    (0,  9),    # 00:00–09:00 UTC
    "london":   (7,  16),   # 07:00–16:00 UTC
    "new_york": (12, 21),   # 12:00–21:00 UTC
}


class SessionFilter:
    """
    Filters a DatetimeIndex to include only bars within active sessions.
    """

    def __init__(
        self,
        sessions: Optional[List[str]] = None,
        timezone: str = "UTC",
    ):
        self.sessions  = sessions or SETTINGS.session.preferred_sessions
        self.timezone  = timezone

    def is_active(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        """
        Return boolean Series — True where at least one preferred session is active.
        """
        if timestamps.tz is None:
            ts_utc = timestamps.tz_localize("UTC")
        else:
            ts_utc = timestamps.tz_convert("UTC")

        active = pd.Series(False, index=timestamps)

        for session in self.sessions:
            if session not in SESSION_UTC:
                log.warning("Unknown session '%s' — skipping", session)
                continue
            start_h, end_h = SESSION_UTC[session]
            hour = ts_utc.hour

            if start_h < end_h:
                # Normal: e.g. 07–16
                mask = (hour >= start_h) & (hour < end_h)
            else:
                # Wraps midnight: e.g. 21–06
                mask = (hour >= start_h) | (hour < end_h)

            active |= pd.Series(mask, index=timestamps)

        return active

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows of df that fall inside active sessions."""
        mask = self.is_active(df.index)
        return df[mask]

    def session_of(self, ts: pd.Timestamp) -> List[str]:
        """Return which sessions are active at a given timestamp."""
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        active = []
        for name, (start_h, end_h) in SESSION_UTC.items():
            h = ts.hour
            if start_h < end_h:
                in_session = start_h <= h < end_h
            else:
                in_session = h >= start_h or h < end_h
            if in_session:
                active.append(name)
        return active
