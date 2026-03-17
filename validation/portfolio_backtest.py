"""
Multi-asset portfolio backtester.

Two-phase approach for speed:
    Phase 1: Pre-compute regime labels and strategy signals for all assets
             (vectorized, fast)
    Phase 2: Simulate portfolio allocation bar-by-bar using pre-computed signals
             (lightweight — just position management, no ML calls)

Usage:
    bt = PortfolioBacktester(market_data, regime_svc, param_store)
    result = bt.run(equity=10000)
    print(result.summary())
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import SETTINGS
from config.regimes import REGIME_DEFINITIONS, preferred_strategies
from strategies.strategy_factory import get_strategy, list_strategies
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ClosedTrade:
    """Record of a completed trade."""
    asset: str
    side: str
    strategy: str
    regime: str
    entry_price: float
    exit_price: float
    lots: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl_pct: float
    exit_reason: str  # "sl", "tp", "end_of_data"


@dataclass
class OpenPosition:
    """Tracks a single open position during simulation."""
    asset: str
    side: str
    strategy: str
    regime: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lots: float
    entry_time: pd.Timestamp
    risk_pct_effective: float = 2.0   # actual % of equity risked on this trade
    initial_risk_dist: float = 0.0    # original SL distance (before trailing)
    highest_price: float = 0.0   # for trailing stop (long)
    lowest_price: float = 1e12   # for trailing stop (short)


@dataclass
class PortfolioBacktestResult:
    """Results from a portfolio backtest run."""
    closed_trades: List[ClosedTrade]
    equity_curve: pd.Series
    initial_equity: float
    final_equity: float
    elapsed_secs: float
    asset_breakdown: Dict[str, Dict]
    strategy_breakdown: Dict[str, Dict]
    regime_breakdown: Dict[str, Dict]

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self.closed_trades:
            return pd.DataFrame()
        rows = []
        for t in self.closed_trades:
            rows.append({
                "asset": t.asset, "side": t.side, "strategy": t.strategy,
                "regime": t.regime, "entry": t.entry_price, "exit": t.exit_price,
                "lots": t.lots, "entry_time": t.entry_time,
                "exit_time": t.exit_time, "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
            })
        return pd.DataFrame(rows)

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.initial_equity - 1) * 100

    @property
    def n_trades(self) -> int:
        return len(self.closed_trades)

    def summary(self) -> str:
        from validation.metrics import compute_trade_metrics, compute_equity_metrics

        lines = ["=" * 60, "PORTFOLIO BACKTEST RESULTS", "=" * 60]
        lines.append(f"Duration:       {self.elapsed_secs:.1f}s")
        lines.append(f"Total trades:   {self.n_trades}")
        lines.append(f"Initial equity: ${self.initial_equity:,.0f}")
        lines.append(f"Final equity:   ${self.final_equity:,.0f}")
        lines.append(f"Total return:   {self.total_return_pct:+.2f}%")
        lines.append("")

        if self.n_trades > 0:
            df = self.trades_df
            tm = compute_trade_metrics(df)
            em = compute_equity_metrics(self.equity_curve)
            lines.append(f"Win rate:       {tm.get('win_rate', 0) * 100:.1f}%")
            lines.append(f"Profit factor:  {tm.get('profit_factor', 0):.2f}")
            lines.append(f"Sharpe ratio:   {em.get('sharpe_ratio', 0):.2f}")
            # Compute MaxDD directly from equity curve
            eq = self.equity_curve
            eq_peak = eq.expanding().max()
            eq_dd = ((eq - eq_peak) / eq_peak * 100).min()
            lines.append(f"Max drawdown:   {eq_dd:.1f}%")
            lines.append(f"Avg trade:      {tm.get('avg_pnl_pct', tm.get('expectancy_pct', 0)):.2f}%")
            lines.append(f"Avg winner:     {tm.get('avg_win_pct', 0):.2f}%")
            lines.append(f"Avg loser:      {tm.get('avg_loss_pct', 0):.2f}%")
            lines.append("")

            lines.append("--- BY ASSET ---")
            for asset, stats in sorted(self.asset_breakdown.items()):
                lines.append(
                    f"  {asset:10s}: {stats['n_trades']:3d} trades | "
                    f"WR={stats['win_rate']:.0f}% | "
                    f"PnL={stats['total_pnl']:+.2f}%"
                )

            lines.append("")
            lines.append("--- BY STRATEGY ---")
            for strat, stats in sorted(self.strategy_breakdown.items()):
                lines.append(
                    f"  {strat:20s}: {stats['n_trades']:3d} trades | "
                    f"WR={stats['win_rate']:.0f}% | "
                    f"PnL={stats['total_pnl']:+.2f}%"
                )

            lines.append("")
            lines.append("--- BY REGIME ---")
            for regime, stats in sorted(self.regime_breakdown.items()):
                lines.append(
                    f"  {regime:20s}: {stats['n_trades']:3d} trades | "
                    f"WR={stats['win_rate']:.0f}% | "
                    f"PnL={stats['total_pnl']:+.2f}%"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


class PortfolioBacktester:
    """
    Two-phase multi-asset portfolio backtester.

    Phase 1 (vectorized): Pre-compute regimes and strategy signals per asset.
    Phase 2 (sequential): Walk forward through bars, manage positions, track equity.
    """

    def __init__(
        self,
        market_data: Dict[str, pd.DataFrame],
        regime_service,
        parameter_store=None,
        tradability_filter=None,
        max_open_trades: int = 12,
        risk_pct: float = 2.0,              # risk per trade as % of equity
        commission_pct: float = 0.0001,
        slippage_pct: float = 0.0002,
        min_rr: float = 1.5,
        warmup_bars: int = 200,
        trailing_stop_atr: float = 0.0,     # disabled — trailing destroys edge
        trailing_activate_rr: float = 0.0,
        cooldown_bars: int = 0,             # no cooldown — capture all signals
        max_positions_per_asset: int = 2,   # pyramiding: up to 2 per asset
    ):
        self.market_data   = market_data
        self.regime_svc    = regime_service
        self.param_store   = parameter_store
        self.tf_filter     = tradability_filter
        self.max_open      = max_open_trades
        self.risk_pct      = risk_pct
        self.commission_pct = commission_pct
        self.slippage_pct  = slippage_pct
        self.min_rr        = min_rr
        self.warmup_bars   = warmup_bars
        self.trailing_stop_atr = trailing_stop_atr
        self.trailing_activate_rr = trailing_activate_rr
        self.cooldown_bars = cooldown_bars
        self.max_per_asset = max_positions_per_asset

    def run(self, equity: float = 10000.0) -> PortfolioBacktestResult:
        t0 = time.time()
        assets = list(self.market_data.keys())

        # --- Phase 1: Pre-compute regimes and signals ---
        log.info("Phase 1: Pre-computing regimes and signals for %d assets...", len(assets))
        asset_signals = {}  # {asset: DataFrame with regime, signal, sl, tp, strength columns}

        for asset in assets:
            df = self.market_data[asset]
            signals_df = self._precompute_signals(asset, df)
            if signals_df is not None and len(signals_df) > 0:
                asset_signals[asset] = signals_df
                n_entries = (signals_df["signal"] != 0).sum()
                log.info("  %s: %d bars, %d entry signals", asset, len(signals_df), n_entries)

        if not asset_signals:
            log.warning("No signals generated for any asset")
            return self._empty_result(equity, time.time() - t0)

        # --- Phase 2: Simulate portfolio bar-by-bar ---
        log.info("Phase 2: Simulating portfolio...")
        result = self._simulate_portfolio(asset_signals, equity)
        result.elapsed_secs = time.time() - t0

        log.info("\n%s", result.summary())
        return result

    def _precompute_signals(self, asset: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Run regime detection and all suitable strategies on full history.

        Returns a DataFrame with columns:
            regime, signal (+1/-1/0), sl, tp, strength, strategy, side
        """
        try:
            regime_result = self.regime_svc.detect(df, asset=asset)
        except Exception as exc:
            log.warning("Regime detection failed for %s: %s", asset, exc)
            return None

        regime_col = regime_result["smoothed_regime"]

        # For each bar, pick the best strategy based on regime
        best_signals = pd.DataFrame(index=df.index)
        best_signals["regime"] = regime_col
        best_signals["signal"] = 0
        best_signals["sl"] = np.nan
        best_signals["tp"] = np.nan
        best_signals["strength"] = 0.0
        best_signals["strategy"] = ""
        best_signals["side"] = ""
        best_signals["close"] = df["close"]
        best_signals["high"] = df["high"]
        best_signals["low"] = df["low"]
        best_signals["atr"] = df.get("atr", pd.Series(0, index=df.index))

        # Generate signals for each strategy and pick the best per regime
        for strat_name in list_strategies():
            for side in ["long", "short"]:
                # Use optimized parameters from parameter store if available
                params_dict = None
                if self.param_store is not None:
                    params_dict = self.param_store.get(asset, "all", strat_name, side)
                    # Skip combos with very negative optimization scores
                    entry = self.param_store.get_entry(asset, "all", strat_name, side)
                    if entry and entry.get("score", 0) < -2.0:
                        continue

                if side == "long":
                    strategy = get_strategy(strat_name, params_long=params_dict)
                else:
                    strategy = get_strategy(strat_name, params_short=params_dict)

                try:
                    sig_df = strategy.generate_signals(df, side=side, asset=asset)
                except Exception:
                    continue

                direction = 1 if side == "long" else -1
                has_signal = sig_df["signal"] == direction

                if not has_signal.any():
                    continue

                # Only use this strategy where the regime prefers it
                for regime_label, regime_def in REGIME_DEFINITIONS.items():
                    if not regime_def.tradable:
                        continue
                    if strat_name not in regime_def.preferred_strategies:
                        continue

                    # Check direction bias
                    if regime_def.direction_bias == "long" and side != "long":
                        continue
                    if regime_def.direction_bias == "short" and side != "short":
                        continue

                    # Mask: bar has signal AND regime matches
                    mask = has_signal & (regime_col == regime_label)
                    if not mask.any():
                        continue

                    # Update where this signal is stronger than existing
                    sig_strength = sig_df.loc[mask, "strength"].fillna(0.5)
                    existing_strength = best_signals.loc[mask, "strength"]
                    better = sig_strength > existing_strength

                    update_idx = better[better].index
                    if len(update_idx) == 0:
                        continue

                    best_signals.loc[update_idx, "signal"] = direction
                    best_signals.loc[update_idx, "sl"] = sig_df.loc[update_idx, "sl"]
                    best_signals.loc[update_idx, "tp"] = sig_df.loc[update_idx, "tp"]
                    best_signals.loc[update_idx, "strength"] = sig_strength.loc[update_idx]
                    best_signals.loc[update_idx, "strategy"] = strat_name
                    best_signals.loc[update_idx, "side"] = side

        return best_signals

    def _simulate_portfolio(
        self,
        asset_signals: Dict[str, pd.DataFrame],
        initial_equity: float,
    ) -> PortfolioBacktestResult:
        """Walk forward through all bars, managing positions across assets."""

        # Build unified time index
        all_idx = pd.DatetimeIndex([])
        for sig_df in asset_signals.values():
            all_idx = all_idx.union(sig_df.index)
        all_idx = all_idx.sort_values()

        if len(all_idx) <= self.warmup_bars:
            return self._empty_result(initial_equity, 0)

        eval_idx = all_idx[self.warmup_bars:]

        open_positions: List[OpenPosition] = []
        closed_trades: List[ClosedTrade] = []
        equity = initial_equity
        peak_equity = initial_equity
        equity_points = []
        # Cooldown tracking: {asset: bar_index when cooldown expires}
        cooldown_until: Dict[str, int] = {}

        for bar_i, ts in enumerate(eval_idx):
            # --- Check exits ---
            newly_closed = []
            for pos in open_positions:
                sig_df = asset_signals.get(pos.asset)
                if sig_df is None or ts not in sig_df.index:
                    continue

                row = sig_df.loc[ts]
                exit_price, exit_reason = self._check_exit(pos, row)

                if exit_price is not None:
                    pnl_pct = self._compute_pnl(pos, exit_price)
                    equity *= (1 + pnl_pct / 100)
                    closed_trades.append(ClosedTrade(
                        asset=pos.asset, side=pos.side, strategy=pos.strategy,
                        regime=pos.regime, entry_price=pos.entry_price,
                        exit_price=exit_price, lots=pos.lots,
                        entry_time=pos.entry_time, exit_time=ts,
                        pnl_pct=pnl_pct, exit_reason=exit_reason,
                    ))
                    newly_closed.append(pos)
                    # Set cooldown after a loss
                    if pnl_pct < 0 and self.cooldown_bars > 0:
                        cooldown_until[pos.asset] = bar_i + self.cooldown_bars

            for pos in newly_closed:
                open_positions.remove(pos)

            # --- Check entries ---
            if len(open_positions) < self.max_open:
                for asset, sig_df in asset_signals.items():
                    if len(open_positions) >= self.max_open:
                        break
                    if ts not in sig_df.index:
                        continue
                    row = sig_df.loc[ts]
                    signal = row["signal"]

                    # Pyramiding: allow up to max_per_asset positions per asset
                    asset_positions = [p for p in open_positions if p.asset == asset]
                    if len(asset_positions) >= self.max_per_asset:
                        continue
                    # If already have a position, only add if same direction (pyramid)
                    if asset_positions:
                        existing_side = asset_positions[0].side
                        current_side = "long" if signal == 1 else "short"
                        if existing_side != current_side:
                            continue
                    # Skip if asset is in cooldown after a loss
                    if asset in cooldown_until and bar_i < cooldown_until[asset]:
                        continue
                    if signal == 0:
                        continue

                    sl_val = row["sl"]
                    tp_val = row["tp"]
                    close_val = row["close"]

                    if pd.isna(sl_val) or pd.isna(tp_val) or sl_val <= 0 or tp_val <= 0:
                        continue

                    side = "long" if signal == 1 else "short"

                    # R:R check
                    if side == "long":
                        risk_dist = close_val - sl_val
                        reward_dist = tp_val - close_val
                    else:
                        risk_dist = sl_val - close_val
                        reward_dist = close_val - tp_val

                    if risk_dist <= 0 or reward_dist <= 0:
                        continue

                    rr = reward_dist / risk_dist
                    if rr < self.min_rr:
                        continue

                    # Fixed risk per trade — no scaling
                    effective_risk_pct = self.risk_pct

                    # Slippage on entry
                    if side == "long":
                        entry = close_val * (1 + self.slippage_pct)
                    else:
                        entry = close_val * (1 - self.slippage_pct)

                    open_positions.append(OpenPosition(
                        asset=asset, side=side, strategy=str(row["strategy"]),
                        regime=str(row["regime"]), entry_price=entry,
                        stop_loss=sl_val, take_profit=tp_val, lots=0,
                        entry_time=ts,
                        risk_pct_effective=effective_risk_pct,
                        initial_risk_dist=risk_dist,
                    ))

            peak_equity = max(peak_equity, equity)
            equity_points.append((ts, equity))

            if (bar_i + 1) % 1000 == 0:
                dd = (equity - peak_equity) / peak_equity * 100
                log.info(
                    "  Bar %d/%d | equity=$%.0f | trades=%d | open=%d | DD=%.1f%%",
                    bar_i + 1, len(eval_idx), equity,
                    len(closed_trades), len(open_positions), dd,
                )

        # Force-close remaining positions
        for pos in open_positions:
            sig_df = asset_signals.get(pos.asset)
            if sig_df is not None and len(sig_df) > 0:
                last_price = float(sig_df["close"].iloc[-1])
                pnl_pct = self._compute_pnl(pos, last_price)
                equity *= (1 + pnl_pct / 100)
                closed_trades.append(ClosedTrade(
                    asset=pos.asset, side=pos.side, strategy=pos.strategy,
                    regime=pos.regime, entry_price=pos.entry_price,
                    exit_price=last_price, lots=pos.lots,
                    entry_time=pos.entry_time, exit_time=sig_df.index[-1],
                    pnl_pct=pnl_pct, exit_reason="end_of_data",
                ))
                equity_points.append((sig_df.index[-1], equity))

        eq_series = pd.Series(
            [e for _, e in equity_points],
            index=pd.DatetimeIndex([t for t, _ in equity_points]),
        )

        return PortfolioBacktestResult(
            closed_trades=closed_trades,
            equity_curve=eq_series,
            initial_equity=initial_equity,
            final_equity=equity,
            elapsed_secs=0,
            asset_breakdown=self._breakdown_by(closed_trades, "asset"),
            strategy_breakdown=self._breakdown_by(closed_trades, "strategy"),
            regime_breakdown=self._breakdown_by(closed_trades, "regime"),
        )

    def _check_exit(self, pos: OpenPosition, row) -> Tuple[Optional[float], str]:
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row.get("atr", 0)) if "atr" in row.index else 0

        if pos.side == "long":
            # Update highest price for trailing stop
            pos.highest_price = max(pos.highest_price, high)

            # Check fixed SL first
            if low <= pos.stop_loss:
                return pos.stop_loss * (1 - self.slippage_pct), "sl"
            # Check TP
            if high >= pos.take_profit:
                return pos.take_profit * (1 - self.slippage_pct), "tp"
            # Trailing stop: activate once price moves trailing_activate_rr * risk in profit
            if self.trailing_stop_atr > 0 and atr > 0:
                risk_dist = pos.entry_price - pos.stop_loss
                profit = pos.highest_price - pos.entry_price
                if profit >= self.trailing_activate_rr * risk_dist:
                    trail_sl = pos.highest_price - self.trailing_stop_atr * atr
                    if trail_sl > pos.stop_loss and low <= trail_sl:
                        return trail_sl * (1 - self.slippage_pct), "trail"
                    # Tighten the SL to trailing level
                    if trail_sl > pos.stop_loss:
                        pos.stop_loss = trail_sl
        else:
            # Update lowest price for trailing stop
            pos.lowest_price = min(pos.lowest_price, low)

            if high >= pos.stop_loss:
                return pos.stop_loss * (1 + self.slippage_pct), "sl"
            if low <= pos.take_profit:
                return pos.take_profit * (1 + self.slippage_pct), "tp"
            # Trailing stop for shorts
            if self.trailing_stop_atr > 0 and atr > 0:
                risk_dist = pos.stop_loss - pos.entry_price
                profit = pos.entry_price - pos.lowest_price
                if profit >= self.trailing_activate_rr * risk_dist:
                    trail_sl = pos.lowest_price + self.trailing_stop_atr * atr
                    if trail_sl < pos.stop_loss and high >= trail_sl:
                        return trail_sl * (1 + self.slippage_pct), "trail"
                    if trail_sl < pos.stop_loss:
                        pos.stop_loss = trail_sl
        return None, ""

    def _compute_pnl(self, pos: OpenPosition, exit_price: float) -> float:
        """Risk-based PnL: if you risk X% and price moves 1R, you gain/lose X%."""
        if pos.initial_risk_dist <= 0:
            return 0.0
        if pos.side == "long":
            price_move = exit_price - pos.entry_price
        else:
            price_move = pos.entry_price - exit_price
        # R-multiple: how many "risk units" the trade moved
        r_multiple = price_move / pos.initial_risk_dist
        # PnL = effective risk * R-multiple (minus commission)
        pnl = pos.risk_pct_effective * r_multiple - 2 * self.commission_pct * 100
        return pnl

    @staticmethod
    def _breakdown_by(trades: List[ClosedTrade], field: str) -> Dict[str, Dict]:
        groups: Dict[str, List[ClosedTrade]] = {}
        for t in trades:
            key = getattr(t, field)
            groups.setdefault(key, []).append(t)

        result = {}
        for key, group in groups.items():
            wins = [t for t in group if t.pnl_pct > 0]
            total_pnl = sum(t.pnl_pct for t in group)
            result[key] = {
                "n_trades": len(group),
                "win_rate": len(wins) / len(group) * 100 if group else 0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(group) if group else 0,
            }
        return result

    def _empty_result(self, equity: float, elapsed: float) -> PortfolioBacktestResult:
        return PortfolioBacktestResult(
            closed_trades=[], equity_curve=pd.Series([equity]),
            initial_equity=equity, final_equity=equity,
            elapsed_secs=elapsed, asset_breakdown={},
            strategy_breakdown={}, regime_breakdown={},
        )
