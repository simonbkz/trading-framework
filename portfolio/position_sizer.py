"""
Position sizer — computes lot sizes given risk parameters and trade proposal.
"""
from __future__ import annotations

from typing import Optional

from config.assets import get_asset
from config.settings import SETTINGS
from strategies.base_strategy import TradeProposal
from utils.logger import get_logger

log = get_logger(__name__)


class PositionSizer:
    """
    ATR / fixed-risk position sizing.

    Inputs:  equity, risk%, SL distance
    Output:  lot size (units)
    """

    def __init__(
        self,
        risk_pct: float = None,
        commission_pct: float = 0.0001,
        slippage_pct: float = 0.0002,
    ):
        self.risk_pct       = risk_pct or SETTINGS.risk.default_risk_pct
        self.commission_pct = commission_pct
        self.slippage_pct   = slippage_pct

    def compute(
        self,
        proposal: TradeProposal,
        equity: float,
        risk_pct_override: Optional[float] = None,
    ) -> dict:
        """
        Compute position size for a trade proposal.

        Returns:
            {
                "lots":          float  (position size in standard lots or units),
                "risk_amount":   float  (USD at risk),
                "sl_distance":   float  (price distance to SL),
                "tp_distance":   float  (price distance to TP),
                "risk_pct":      float  (actual risk % of equity),
            }
        """
        risk_pct = (risk_pct_override or self.risk_pct) / 100.0
        risk_amount = equity * risk_pct

        sl_dist = proposal.risk_pips
        if sl_dist <= 0:
            log.warning("Invalid SL distance for %s — returning zero size", proposal.asset)
            return {"lots": 0.0, "risk_amount": 0.0, "sl_distance": 0, "tp_distance": 0, "risk_pct": 0}

        # Guard: reject near-zero equity
        if equity < 10:
            log.warning("Equity too low ($%.2f) for %s — returning zero size", equity, proposal.asset)
            return {"lots": 0.0, "risk_amount": 0.0, "sl_distance": 0, "tp_distance": 0, "risk_pct": 0}

        try:
            cfg = get_asset(proposal.asset)
            sl_pips     = sl_dist / cfg.pip_size
            pip_value   = cfg.pip_value_usd
            lots = risk_amount / (sl_pips * pip_value)
        except Exception:
            # Fallback: fractional position
            lots = risk_amount / (sl_dist * 100000)

        # Clamp lots to prevent oversized positions from tiny SL distances
        MAX_LOTS = 50.0
        lots = min(MAX_LOTS, max(0.01, round(lots, 2)))

        return {
            "lots":          lots,
            "risk_amount":   round(equity * risk_pct, 2),
            "sl_distance":   round(sl_dist, 6),
            "tp_distance":   round(proposal.reward_pips, 6),
            "risk_pct":      round(risk_pct * 100, 3),
        }
