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

# Session definitions using local exchange timezones for DST awareness.
# Hours are in LOCAL exchange time — converted to UTC dynamically.
SESSION_LOCAL: Dict[str, Tuple[str, int, int]] = {
    #             (timezone,              local_start, local_end)
    "sydney":    ("Australia/Sydney",     10, 16),   # 10:00–16:00 AEST/AEDT
    "tokyo":     ("Asia/Tokyo",            9, 15),   # 09:00–15:00 JST (no DST)
    "london":    ("Europe/London",         8, 16),   # 08:00–16:00 GMT/BST
    "new_york":  ("America/New_York",      8, 17),   # 08:00–17:00 EST/EDT
}

# Fallback fixed UTC windows (used when pytz is unavailable)
SESSION_UTC: Dict[str, Tuple[int, int]] = {
    "sydney":   (21, 6),
    "tokyo":    (0,  9),
    "london":   (7,  16),
    "new_york": (12, 21),
}


class SessionFilter:
    """
    Filters a DatetimeIndex to include only bars within active sessions.
    Uses DST-aware local exchange times when possible.
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
        Uses local exchange timezone to handle DST shifts automatically.
        """
        if timestamps.tz is None:
            ts_utc = timestamps.tz_localize("UTC")
        else:
            ts_utc = timestamps.tz_convert("UTC")

        active = pd.Series(False, index=timestamps)

        for session in self.sessions:
            if session in SESSION_LOCAL:
                tz_name, start_h, end_h = SESSION_LOCAL[session]
                try:
                    local_tz = pytz.timezone(tz_name)
                    ts_local = ts_utc.tz_convert(local_tz)
                    hour = ts_local.hour
                except Exception:
                    # Fallback to fixed UTC if conversion fails
                    start_h, end_h = SESSION_UTC.get(session, (0, 24))
                    hour = ts_utc.hour
            elif session in SESSION_UTC:
                start_h, end_h = SESSION_UTC[session]
                hour = ts_utc.hour
            else:
                log.warning("Unknown session '%s' — skipping", session)
                continue

            if start_h < end_h:
                mask = (hour >= start_h) & (hour < end_h)
            else:
                mask = (hour >= start_h) | (hour < end_h)

            active |= pd.Series(mask, index=timestamps)

        return active

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows of df that fall inside active sessions."""
        mask = self.is_active(df.index)
        return df[mask]

    def session_of(self, ts: pd.Timestamp) -> List[str]:
        """Return which sessions are active at a given timestamp (DST-aware)."""
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        active = []
        for name, (tz_name, start_h, end_h) in SESSION_LOCAL.items():
            try:
                local_ts = ts.tz_convert(pytz.timezone(tz_name))
                h = local_ts.hour
            except Exception:
                h = ts.hour
                start_h, end_h = SESSION_UTC.get(name, (0, 24))
            if start_h < end_h:
                in_session = start_h <= h < end_h
            else:
                in_session = h >= start_h or h < end_h
            if in_session:
                active.append(name)
        return active
