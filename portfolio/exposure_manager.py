"""
Exposure manager — tracks and limits portfolio-wide exposure.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


class ExposureManager:
    """
    Tracks open positions and computes portfolio exposure metrics.
    """

    def __init__(self):
        self._positions: Dict[str, dict] = {}   # ticket → position data

    def add(self, ticket: str, position: dict) -> None:
        """Register an open position."""
        self._positions[ticket] = position

    def remove(self, ticket: str) -> None:
        """Close/remove a position."""
        self._positions.pop(ticket, None)

    @property
    def open_count(self) -> int:
        return len(self._positions)

    @property
    def open_positions(self) -> List[dict]:
        return list(self._positions.values())

    def exposure_by_asset(self) -> Dict[str, float]:
        """Return total lot exposure per asset."""
        exp: Dict[str, float] = {}
        for pos in self._positions.values():
            asset = pos.get("asset", "")
            exp[asset] = exp.get(asset, 0) + pos.get("lots", 0)
        return exp

    def exposure_by_direction(self) -> Dict[str, float]:
        """Return total lot exposure per direction (long/short)."""
        exp = {"long": 0.0, "short": 0.0}
        for pos in self._positions.values():
            side = pos.get("side", "long")
            exp[side] = exp.get(side, 0) + pos.get("lots", 0)
        return exp

    def net_exposure_pct(self, equity: float) -> float:
        """Net exposure as % of equity (long - short in nominal USD terms)."""
        if equity == 0:
            return 0.0
        net = 0.0
        for pos in self._positions.values():
            val = pos.get("lots", 0) * pos.get("entry", 0)
            if pos.get("side") == "long":
                net += val
            else:
                net -= val
        return (net / equity) * 100

    def summary(self) -> dict:
        return {
            "open_positions": self.open_count,
            "by_asset":       self.exposure_by_asset(),
            "by_direction":   self.exposure_by_direction(),
        }
