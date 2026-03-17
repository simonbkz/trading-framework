"""
Risk engine — enforces portfolio-level risk controls before order approval.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from config.settings import SETTINGS
from strategies.base_strategy import TradeProposal
from portfolio.position_sizer import PositionSizer
from utils.logger import get_logger

log = get_logger(__name__)


class RiskEngine:
    """
    Gate that approves or rejects trade proposals based on portfolio risk rules.

    Rules enforced:
    - Max open trades
    - Daily loss guard
    - Max drawdown halt
    - Max asset exposure
    - Min R:R requirement
    - Volatility targeting (if configured)
    """

    def __init__(
        self,
        risk_pct: Optional[float] = None,
        max_open_trades: Optional[int] = None,
        max_daily_loss_pct: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        max_asset_exposure_pct: Optional[float] = None,
        min_rr_ratio: Optional[float] = None,
    ):
        r = SETTINGS.risk
        self.risk_pct              = risk_pct or r.default_risk_pct
        self.max_open_trades       = max_open_trades or r.max_open_trades
        self.max_daily_loss_pct    = max_daily_loss_pct or r.max_daily_loss_pct
        self.max_drawdown_pct      = max_drawdown_pct or r.max_drawdown_pct
        self.max_asset_exposure    = max_asset_exposure_pct or r.max_asset_exposure_pct
        self.min_rr_ratio          = min_rr_ratio if min_rr_ratio is not None else getattr(r, 'min_rr_ratio', 2.0)
        self.sizer                 = PositionSizer(risk_pct=self.risk_pct)

        # State (updated externally by portfolio monitor)
        self.current_equity: float  = 10000.0
        self.peak_equity: float     = 10000.0
        self.day_start_equity: float = 10000.0
        self.open_trades: List[Dict] = []
        self.halted: bool = False

    def approve(
        self,
        proposal: TradeProposal,
    ) -> Dict:
        """
        Evaluate a trade proposal.

        Returns:
            {
                "approved":  bool,
                "reason":    str (explanation if rejected),
                "lots":      float (if approved),
                "risk_pct":  float,
            }
        """
        if self.halted:
            return self._reject("Trading halted — daily/drawdown limit exceeded")

        # 1. R:R check
        if proposal.rr_ratio < self.min_rr_ratio:
            return self._reject(
                f"R:R {proposal.rr_ratio:.2f} < min {self.min_rr_ratio:.2f}"
            )

        # 2. Max open trades
        if len(self.open_trades) >= self.max_open_trades:
            return self._reject(
                f"Max open trades ({self.max_open_trades}) reached"
            )

        # 3. Asset exposure limit
        asset_lots = sum(
            t.get("lots", 0) for t in self.open_trades
            if t.get("asset") == proposal.asset
        )
        asset_exposure_pct = (asset_lots * proposal.entry / self.current_equity) * 100
        if asset_exposure_pct > self.max_asset_exposure:
            return self._reject(
                f"Asset exposure {asset_exposure_pct:.1f}% > max {self.max_asset_exposure:.1f}%"
            )

        # 4. Daily loss guard
        daily_pnl_pct = (
            (self.current_equity - self.day_start_equity) / self.day_start_equity * 100
        )
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            self.halted = True
            return self._reject(
                f"Daily loss limit hit: {daily_pnl_pct:.2f}% <= -{self.max_daily_loss_pct}%"
            )

        # 5. Drawdown guard
        drawdown_pct = (
            (self.current_equity - self.peak_equity) / self.peak_equity * 100
        )
        if drawdown_pct <= -self.max_drawdown_pct:
            self.halted = True
            return self._reject(
                f"Max drawdown hit: {drawdown_pct:.2f}% <= -{self.max_drawdown_pct}%"
            )

        # 6. Size the position
        sizing = self.sizer.compute(proposal, self.current_equity)

        # Reject if position sizer returned zero lots (invalid SL distance)
        if sizing["lots"] <= 0:
            return self._reject("Position sizer returned zero lots (invalid SL distance)")

        log.info(
            "APPROVED: %s %s | lots=%.2f risk=%.2f%% RR=%.2f",
            proposal.asset, proposal.side,
            sizing["lots"], sizing["risk_pct"], proposal.rr_ratio,
        )

        return {
            "approved": True,
            "reason":   "OK",
            "lots":     sizing["lots"],
            "risk_pct": sizing["risk_pct"],
            "risk_amount": sizing["risk_amount"],
        }

    def update_state(
        self,
        equity: float,
        open_trades: Optional[List[Dict]] = None,
    ) -> None:
        """Update internal state from portfolio monitor."""
        self.current_equity = equity
        self.peak_equity    = max(self.peak_equity, equity)
        if open_trades is not None:
            self.open_trades = open_trades
        # Reset halt at start of new day (caller manages this)

    def reset_daily(self, equity: float) -> None:
        self.day_start_equity = equity
        self.halted           = False

    @staticmethod
    def _reject(reason: str) -> Dict:
        log.info("REJECTED: %s", reason)
        return {"approved": False, "reason": reason, "lots": 0.0, "risk_pct": 0.0}
