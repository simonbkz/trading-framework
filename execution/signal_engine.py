"""
Signal engine — assembles the final signal from a trade proposal
and writes it to disk for MT5 pickup.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import SETTINGS
from execution.order_schema import Order, MT5Signal, save_signal_batch
from portfolio.risk_engine import RiskEngine
from strategies.base_strategy import TradeProposal
from utils.logger import get_logger

log = get_logger(__name__)


class SignalEngine:
    """
    Converts TradeProposals → Orders → MT5Signals → JSON files.

    Usage:
        engine = SignalEngine(risk_engine)
        signals = engine.process(proposals)
        engine.write_latest(signals[0])
    """

    def __init__(
        self,
        risk_engine: Optional[RiskEngine] = None,
        output_path: Optional[Path] = None,
        paper_mode: bool = True,
    ):
        self.risk_engine = risk_engine or RiskEngine()
        self.output_path = output_path or SETTINGS.execution.signal_output_path
        self.paper_mode  = paper_mode
        self._history: List[Dict] = []

    def process(
        self,
        proposals: List[TradeProposal],
        equity: float = 10000.0,
        regime_confidence: float = 0.5,
    ) -> List[MT5Signal]:
        """
        Process proposals through risk engine and produce MT5Signal list.

        Args:
            proposals:         List of TradeProposals from router
            equity:            Current account equity
            regime_confidence: For the signal payload

        Returns:
            List of approved MT5Signal objects
        """
        self.risk_engine.update_state(equity=equity)
        signals = []

        for proposal in proposals:
            decision = self.risk_engine.approve(proposal)

            if not decision["approved"]:
                log.info("Proposal rejected: %s — %s", proposal.asset, decision["reason"])
                continue

            order = Order.from_proposal(
                proposal,
                lots=decision["lots"],
                risk_pct=decision["risk_pct"],
            )
            signal = MT5Signal.from_order(order, regime_confidence=regime_confidence)

            # Final guardrail: reject signals with invalid numeric fields
            if not signal.validate():
                log.warning(
                    "Signal rejected (invalid values): %s lots=%.4f entry=%.6f sl=%.6f tp=%.6f",
                    signal.asset, signal.lots, signal.entry, signal.stop_loss, signal.take_profit,
                )
                continue

            signals.append(signal)

            self._history.append({
                "timestamp": signal.timestamp,
                "asset":     signal.asset,
                "side":      signal.side,
                "strategy":  signal.strategy,
                "regime":    signal.regime,
                "lots":      signal.lots,
                "entry":     signal.entry,
                "order_id":  signal.order_id,
            })

        log.info(
            "SignalEngine: %d proposals -> %d approved signals",
            len(proposals), len(signals),
        )
        return signals

    def write_latest(self, signal: MT5Signal) -> None:
        """Write a single signal to the latest_signal.json (legacy wrapper)."""
        self.write_all([signal])

    def write_all(self, signals: List[MT5Signal]) -> None:
        """
        Write all approved signals to latest_signal.json as a batch.

        Applies deduplication: only the strongest signal per asset is kept.
        Also mirrors to MQL5/Files/ if the path is configured.
        """
        deduped = self._deduplicate(signals)
        path = Path(self.output_path)
        save_signal_batch(deduped, path)
        self._mirror_to_mt5(deduped)
        for sig in deduped:
            log.info(
                "Signal written: %s %s %s | entry=%.5f lots=%.4f [%s]",
                sig.asset, sig.side.upper(), sig.strategy,
                sig.entry, sig.lots, sig.order_id,
            )
        log.info("Wrote %d signal(s) to %s", len(deduped), path)

    def write_no_signal(self) -> None:
        """Write an explicit 'no signal' batch payload."""
        path = Path(self.output_path)
        save_signal_batch([], path)
        self._mirror_to_mt5([])

    def _mirror_to_mt5(self, signals: list) -> None:
        """Write a copy to MQL5/Files/ so the EA can read it from its sandbox."""
        mt5_path = SETTINGS.execution.mt5_files_path
        if mt5_path is None:
            return
        try:
            mt5_file = Path(mt5_path) / "latest_signal.json"
            save_signal_batch(signals, mt5_file)
            log.debug("Mirrored signal to MT5: %s", mt5_file)
        except Exception as exc:
            log.warning("Failed to mirror signal to MT5 Files: %s", exc)

    @staticmethod
    def _deduplicate(signals: List[MT5Signal]) -> List[MT5Signal]:
        """
        Keep only the best signal per asset (highest signal_strength).

        Prevents conflicting long+short on the same instrument.
        """
        best: dict = {}
        for sig in signals:
            key = sig.asset
            if key not in best or sig.signal_strength > best[key].signal_strength:
                best[key] = sig
        return list(best.values())

    def history_df(self):
        """Return signal history as a DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._history)


class PaperExecutor:
    """
    Simulates order execution in paper mode — tracks fills without real orders.
    """

    def __init__(self):
        self._positions: Dict[str, dict] = {}
        self._closed: List[dict] = []

    def execute(self, signal: MT5Signal) -> dict:
        """
        Simulate order fill at signal entry price.
        Returns fill result dict.
        """
        import uuid
        ticket = str(uuid.uuid4())[:8]
        fill = {
            "order_id":   signal.order_id,
            "ticket":     ticket,
            "status":     "filled",
            "fill_price": signal.entry,
            "fill_time":  datetime.now(tz=timezone.utc).isoformat(),
            "lots":       signal.lots,
            "error_code": 0,
            "error_msg":  "",
        }
        self._positions[ticket] = {
            "ticket":     ticket,
            "asset":      signal.asset,
            "side":       signal.side,
            "entry":      signal.entry,
            "sl":         signal.stop_loss,
            "tp":         signal.take_profit,
            "lots":       signal.lots,
            "strategy":   signal.strategy,
            "regime":     signal.regime,
            "opened_at":  fill["fill_time"],
        }
        log.info(
            "[PAPER] Filled: %s %s %.2f lots @ %.5f SL=%.5f TP=%.5f",
            signal.asset, signal.side.upper(), signal.lots,
            signal.entry, signal.stop_loss, signal.take_profit,
        )
        return fill

    def check_exits(self, current_prices: Dict[str, float]) -> List[dict]:
        """
        Check if any open positions hit their SL or TP.

        Args:
            current_prices: {symbol: current_price}

        Returns:
            List of closed trade dicts
        """
        newly_closed = []
        for ticket, pos in list(self._positions.items()):
            price = current_prices.get(pos["asset"])
            if price is None:
                continue
            closed = False
            exit_price = price
            exit_reason = "market"

            if pos["side"] == "long":
                if price <= pos["sl"]:
                    exit_reason = "sl"
                    exit_price  = pos["sl"]
                    closed = True
                elif price >= pos["tp"]:
                    exit_reason = "tp"
                    exit_price  = pos["tp"]
                    closed = True
            else:
                if price >= pos["sl"]:
                    exit_reason = "sl"
                    exit_price  = pos["sl"]
                    closed = True
                elif price <= pos["tp"]:
                    exit_reason = "tp"
                    exit_price  = pos["tp"]
                    closed = True

            if closed:
                if pos["side"] == "long":
                    pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
                else:
                    pnl_pct = (pos["entry"] - exit_price) / pos["entry"] * 100

                closed_trade = {**pos, "exit_price": exit_price,
                                "exit_reason": exit_reason, "pnl_pct": pnl_pct,
                                "closed_at": datetime.now(tz=timezone.utc).isoformat()}
                self._closed.append(closed_trade)
                del self._positions[ticket]
                newly_closed.append(closed_trade)
                log.info(
                    "[PAPER] Closed: %s %s @ %.5f (%s) PnL=%.2f%%",
                    pos["asset"], pos["side"].upper(), exit_price,
                    exit_reason, pnl_pct,
                )
        return newly_closed

    @property
    def open_positions(self) -> List[dict]:
        return list(self._positions.values())

    @property
    def closed_trades(self) -> List[dict]:
        return self._closed
