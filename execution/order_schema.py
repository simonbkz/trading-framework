"""
Order schema definitions.

The OrderSchema is the internal canonical order representation.
The MT5BridgeSchema is the JSON format written to disk for MT5 pickup.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal, Optional

OrderType = Literal["market", "limit", "stop"]
OrderSide = Literal["long", "short"]
OrderStatus = Literal["pending", "filled", "cancelled", "rejected", "expired"]


@dataclass
class Order:
    """Internal order representation."""
    order_id: str
    timestamp: str                      # ISO UTC
    asset: str
    side: OrderSide
    order_type: OrderType
    lots: float
    entry: float
    stop_loss: float
    take_profit: float
    strategy: str
    regime: str
    confidence: float
    signal_strength: float
    risk_pct: float
    status: OrderStatus = "pending"
    fill_price: Optional[float] = None
    fill_time: Optional[str] = None
    notes: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_proposal(
        cls,
        proposal,
        lots: float,
        risk_pct: float,
        order_id: Optional[str] = None,
    ) -> "Order":
        import uuid
        return cls(
            order_id       = order_id or str(uuid.uuid4())[:8],
            timestamp      = datetime.now(tz=timezone.utc).isoformat(),
            asset          = proposal.asset,
            side           = proposal.side,
            order_type     = "market",
            lots           = lots,
            entry          = proposal.entry,
            stop_loss      = proposal.stop_loss,
            take_profit    = proposal.take_profit,
            strategy       = proposal.strategy,
            regime         = proposal.regime,
            confidence     = proposal.confidence,
            signal_strength= proposal.signal_strength,
            risk_pct       = risk_pct,
            notes          = proposal.notes,
        )


@dataclass
class MT5Signal:
    """
    JSON payload written to disk and picked up by the MT5 EA bridge.

    MT5 EA reads this file on every tick via FileOpen/FileReadString.
    """
    timestamp: str
    asset: str
    regime: str
    regime_confidence: float
    strategy: str
    side: OrderSide
    order_type: OrderType
    entry: float
    stop_loss: float
    take_profit: float
    lots: float
    risk_pct: float
    signal_strength: float
    order_id: str
    action: str = "open"     # open | close | update | none
    magic_number: int = 99999
    comment: str = ""
    expiry_minutes: int = 60  # signal valid for N minutes
    # Broker-agnostic distances: EA can recalculate SL/TP from actual fill price
    # This handles quote differences between data provider (yfinance) and broker (XM)
    sl_distance: float = 0.0   # absolute price distance from entry to SL
    tp_distance: float = 0.0   # absolute price distance from entry to TP
    sl_pips: float = 0.0       # SL distance in pips (for forex)
    tp_pips: float = 0.0       # TP distance in pips (for forex)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Replace any NaN/inf/None with safe defaults for JSON
        for key, val in d.items():
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                d[key] = 0.0
            elif val is None:
                d[key] = "" if isinstance(self.__dataclass_fields__[key].default, str) else 0.0
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def validate(self) -> bool:
        """Return True if all critical numeric fields are valid (non-zero, finite)."""
        for field_name in ("entry", "stop_loss", "take_profit", "lots"):
            val = getattr(self, field_name)
            if val <= 0 or math.isnan(val) or math.isinf(val):
                return False
        return True

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def from_order(cls, order: Order, regime_confidence: float = 0.5) -> "MT5Signal":
        # Compute price distances for broker-agnostic execution
        # The EA should use these distances from actual fill price rather than
        # absolute levels, since data provider (yfinance) and broker (XM) may differ
        entry = order.entry
        sl_dist = abs(entry - order.stop_loss) if entry > 0 and order.stop_loss > 0 else 0
        tp_dist = abs(order.take_profit - entry) if entry > 0 and order.take_profit > 0 else 0

        # Pip calculation: for JPY pairs pip=0.01, others pip=0.0001
        is_jpy = "JPY" in order.asset.upper()
        pip_size = 0.01 if is_jpy else 0.0001
        # For commodities/crypto, pips don't apply — distance in price units
        is_forex = any(ccy in order.asset.upper() for ccy in ("USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"))
        sl_pips = sl_dist / pip_size if is_forex and pip_size > 0 else 0
        tp_pips = tp_dist / pip_size if is_forex and pip_size > 0 else 0

        return cls(
            timestamp         = order.timestamp,
            asset             = order.asset,
            regime            = order.regime,
            regime_confidence = regime_confidence,
            strategy          = order.strategy,
            side              = order.side,
            order_type        = order.order_type,
            entry             = order.entry,
            stop_loss         = order.stop_loss,
            take_profit       = order.take_profit,
            lots              = order.lots,
            risk_pct          = order.risk_pct,
            signal_strength   = order.signal_strength,
            order_id          = order.order_id,
            comment           = f"Regime:{order.regime}|Strat:{order.strategy}",
            sl_distance       = round(sl_dist, 6),
            tp_distance       = round(tp_dist, 6),
            sl_pips           = round(sl_pips, 1),
            tp_pips           = round(tp_pips, 1),
        )


def save_signal_batch(signals: list, path: Path) -> None:
    """
    Write a multi-signal payload to disk for MT5 pickup.

    Format:
        {
            "timestamp": "...",
            "count": N,
            "action": "open" | "none",
            "signals": [ {signal1}, {signal2}, ... ]
        }

    When no signals are present, writes an explicit "none" payload
    so the EA always sees valid JSON.
    """
    now = datetime.now(tz=timezone.utc).isoformat()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if signals:
        payload = {
            "timestamp": now,
            "count":     len(signals),
            "action":    "open",
            "signals":   [s.to_dict() for s in signals],
        }
    else:
        payload = {
            "timestamp": now,
            "count":     0,
            "action":    "none",
            "signals":   [],
        }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_mt5_signal(path: Path) -> Optional[MT5Signal]:
    """Load and deserialise the latest signal from disk (legacy single-signal)."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Support both old single-signal and new batch format
    if "signals" in data and isinstance(data["signals"], list):
        if not data["signals"]:
            return None
        return MT5Signal(**data["signals"][0])
    return MT5Signal(**data)


def load_signal_batch(path: Path) -> list:
    """Load all signals from a batch payload."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "signals" in data and isinstance(data["signals"], list):
        return [MT5Signal(**s) for s in data["signals"]]
    # Legacy single-signal format
    if "action" in data and data["action"] != "none":
        return [MT5Signal(**data)]
    return []
