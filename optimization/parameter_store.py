"""
Parameter store — persists and retrieves optimized parameter sets.

Backed by a JSON file. Supports versioning and timestamp tracking.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_STORE_PATH = Path("parameters") / "parameter_store.json"


class ParameterStore:
    """
    Persistent storage for optimized strategy parameters.

    Key structure:
        {asset} / {regime} / {strategy} / {side}

    Each entry contains:
        params:     the parameter dict
        score:      optimization objective score
        objective:  metric used for optimization
        n_trades:   number of trades in the optimization period
        updated_at: ISO timestamp of last update
        notes:      free-text notes
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or DEFAULT_STORE_PATH)
        self._store: Dict = {}
        if self.path.exists():
            self.load()

    def set(
        self,
        asset: str,
        regime: str,
        strategy: str,
        side: str,
        params: Dict,
        score: float = 0.0,
        objective: str = "",
        n_trades: int = 0,
        notes: str = "",
    ) -> None:
        self._store \
            .setdefault(asset, {}) \
            .setdefault(regime, {}) \
            .setdefault(strategy, {})[side] = {
                "params":     params,
                "score":      round(score, 6),
                "objective":  objective,
                "n_trades":   n_trades,
                "updated_at": datetime.utcnow().isoformat(),
                "notes":      notes,
            }
        self.save()
        log.info("ParameterStore: saved %s/%s/%s/%s (score=%.4f)", asset, regime, strategy, side, score)

    def get(
        self,
        asset: str,
        regime: str,
        strategy: str,
        side: str,
    ) -> Optional[Dict]:
        """Return stored params dict or None if not found."""
        try:
            entry = self._store[asset][regime][strategy][side]
            return entry["params"]
        except KeyError:
            return None

    def get_entry(
        self,
        asset: str,
        regime: str,
        strategy: str,
        side: str,
    ) -> Optional[Dict]:
        """Return the full entry (params + metadata)."""
        try:
            return self._store[asset][regime][strategy][side]
        except KeyError:
            return None

    def keys(self) -> List[Tuple[str, str, str, str]]:
        """Return all (asset, regime, strategy, side) keys."""
        out = []
        for asset, regimes in self._store.items():
            for regime, strategies in regimes.items():
                for strategy, sides in strategies.items():
                    for side in sides:
                        out.append((asset, regime, strategy, side))
        return out

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._store, fh, indent=2, default=str)

    def load(self) -> "ParameterStore":
        with open(self.path, "r", encoding="utf-8") as fh:
            self._store = json.load(fh)
        log.info("ParameterStore loaded from %s (%d entries)", self.path, len(self.keys()))
        return self

    def export_csv(self, out_path: Optional[Path] = None) -> None:
        """Export all entries to a flat CSV for inspection."""
        import pandas as pd
        rows = []
        for asset, regime, strategy, side in self.keys():
            entry = self._store[asset][regime][strategy][side]
            row = {
                "asset": asset, "regime": regime,
                "strategy": strategy, "side": side,
                "score": entry.get("score"),
                "objective": entry.get("objective"),
                "n_trades": entry.get("n_trades"),
                "updated_at": entry.get("updated_at"),
            }
            row.update({f"p_{k}": v for k, v in entry.get("params", {}).items()})
            rows.append(row)

        df = pd.DataFrame(rows)
        out = out_path or self.path.with_suffix(".csv")
        df.to_csv(out, index=False)
        log.info("ParameterStore exported to %s", out)

    def __repr__(self) -> str:
        return f"ParameterStore({self.path}, {len(self.keys())} entries)"
