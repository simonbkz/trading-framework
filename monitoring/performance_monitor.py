"""
Live performance monitoring — tracks equity, open positions, and rolling metrics.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from validation.metrics import compute_trade_metrics, compute_equity_metrics
from utils.logger import get_logger

log = get_logger(__name__)


class PerformanceMonitor:
    """
    Tracks the framework's performance over time.

    Stores:
    - Equity snapshots
    - Closed trades
    - Rolling performance metrics (last 20 trades)
    """

    def __init__(
        self,
        initial_equity: float = 10000.0,
        rolling_window: int = 20,
    ):
        self.initial_equity = initial_equity
        self.rolling_window = rolling_window
        self._equity_history: List[dict] = [
            {"timestamp": datetime.now(tz=timezone.utc).isoformat(), "equity": initial_equity}
        ]
        self._closed_trades: List[dict] = []

    def update_equity(self, equity: float) -> None:
        self._equity_history.append({
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "equity": equity,
        })

    def record_trade(self, trade: dict) -> None:
        self._closed_trades.append({**trade,
            "recorded_at": datetime.now(tz=timezone.utc).isoformat()
        })

    @property
    def equity_series(self) -> pd.Series:
        df = pd.DataFrame(self._equity_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")["equity"]

    @property
    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._closed_trades)

    def rolling_metrics(self) -> dict:
        """Metrics on the last `rolling_window` trades."""
        if not self._closed_trades:
            return {}
        recent = pd.DataFrame(self._closed_trades[-self.rolling_window:])
        return compute_trade_metrics(recent)

    def full_metrics(self) -> dict:
        """Metrics on all closed trades + equity curve."""
        tm = compute_trade_metrics(self.trades_df) if self._closed_trades else {}
        eq = self.equity_series
        em = compute_equity_metrics(eq) if len(eq) > 1 else {}
        return {**tm, **em}

    def is_performance_degraded(
        self,
        min_profit_factor: float = 0.80,
        check_last_n: int = 20,
    ) -> bool:
        """True if rolling PF has dropped below threshold (re-optimize signal)."""
        m = self.rolling_metrics()
        pf = m.get("profit_factor", 1.0)
        if pf < min_profit_factor and m.get("n_trades", 0) >= check_last_n:
            log.warning(
                "Performance degraded: rolling PF=%.2f < %.2f — consider re-optimizing",
                pf, min_profit_factor,
            )
            return True
        return False

    def save(self, path: Path) -> None:
        from utils.helpers import save_json
        data = {
            "equity_history": self._equity_history,
            "closed_trades":  self._closed_trades,
            "metrics":        self.full_metrics(),
            "saved_at":       datetime.now(tz=timezone.utc).isoformat(),
        }
        save_json(data, path)
        log.info("PerformanceMonitor saved to %s", path)

    def load(self, path: Path) -> "PerformanceMonitor":
        from utils.helpers import load_json
        data = load_json(path)
        self._equity_history = data.get("equity_history", [])
        self._closed_trades  = data.get("closed_trades", [])
        return self
