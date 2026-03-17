"""SOTA techniques sweep: partial TP, time exit, vol targeting, equity curve filter."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from data.market_data import MarketDataService
from features.regime_features import add_regime_features
from features.alpha_features import add_alpha_features
from regimes.regime_service import RegimeService
from optimization.parameter_store import ParameterStore
from validation.portfolio_backtest import PortfolioBacktester
import validation.portfolio_backtest as vpb
import strategies.session_breakout as sb
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnhancedPosition:
    """Extended position with partial TP tracking."""
    asset: str
    side: str
    strategy: str
    regime: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lots: float
    entry_time: pd.Timestamp
    risk_pct_effective: float
    initial_risk_dist: float
    highest_price: float = 0.0
    lowest_price: float = 1e12
    # Partial TP tracking
    partial_tp_hit: bool = False
    partial_tp_price: float = 0.0
    # Time-based exit
    entry_bar: int = 0
    # Original risk for scaling
    original_risk_pct: float = 0.0


def enhanced_simulate(self, asset_signals, initial_equity,
                       partial_tp_r=0.0, partial_tp_frac=0.5,
                       max_bars=0, vol_target=0.0,
                       eq_ma_period=0, eq_ma_risk_mult=0.5):
    """Enhanced simulation with SOTA techniques.

    Args:
        partial_tp_r: R-multiple at which to take partial profit (0=disabled)
        partial_tp_frac: Fraction of risk to take at partial TP (0.5 = half)
        max_bars: Max bars to hold a trade (0=disabled)
        vol_target: Target annualized vol for risk scaling (0=disabled)
        eq_ma_period: Equity curve MA period for filter (0=disabled)
        eq_ma_risk_mult: Risk multiplier when below equity MA
    """
    all_idx = pd.DatetimeIndex([])
    for sig_df in asset_signals.values():
        all_idx = all_idx.union(sig_df.index)
    all_idx = all_idx.sort_values()
    if len(all_idx) <= self.warmup_bars:
        return self._empty_result(initial_equity, 0)
    eval_idx = all_idx[self.warmup_bars:]
    open_positions = []
    closed_trades = []
    equity, peak_equity = initial_equity, initial_equity
    equity_points = []
    cooldown_until = {}

    # For volatility targeting
    recent_returns = []

    # For equity curve filter
    equity_history = []

    for bar_i, ts in enumerate(eval_idx):
        # === CHECK EXITS ===
        newly_closed = []
        for pos in open_positions:
            sig_df = asset_signals.get(pos.asset)
            if sig_df is None or ts not in sig_df.index:
                continue
            row = sig_df.loc[ts]
            high = float(row["high"])
            low = float(row["low"])

            exit_price = None
            exit_reason = ""

            if pos.side == "long":
                pos.highest_price = max(pos.highest_price, high)
                # Check SL
                if low <= pos.stop_loss:
                    exit_price = pos.stop_loss * (1 - self.slippage_pct)
                    exit_reason = "sl"
                # Check TP
                elif high >= pos.take_profit:
                    exit_price = pos.take_profit * (1 - self.slippage_pct)
                    exit_reason = "tp"
                # Check partial TP (move SL to breakeven)
                elif partial_tp_r > 0 and not pos.partial_tp_hit:
                    partial_price = pos.entry_price + partial_tp_r * pos.initial_risk_dist
                    if high >= partial_price:
                        pos.partial_tp_hit = True
                        pos.partial_tp_price = partial_price
                        # Move SL to breakeven + small buffer
                        pos.stop_loss = pos.entry_price + 0.1 * pos.initial_risk_dist
                        # Record partial profit
                        partial_pnl = pos.risk_pct_effective * partial_tp_frac * partial_tp_r
                        equity *= (1 + partial_pnl / 100)
                        # Reduce remaining risk
                        pos.risk_pct_effective *= (1 - partial_tp_frac)
            else:
                pos.lowest_price = min(pos.lowest_price, low)
                if high >= pos.stop_loss:
                    exit_price = pos.stop_loss * (1 + self.slippage_pct)
                    exit_reason = "sl"
                elif low <= pos.take_profit:
                    exit_price = pos.take_profit * (1 + self.slippage_pct)
                    exit_reason = "tp"
                elif partial_tp_r > 0 and not pos.partial_tp_hit:
                    partial_price = pos.entry_price - partial_tp_r * pos.initial_risk_dist
                    if low <= partial_price:
                        pos.partial_tp_hit = True
                        pos.partial_tp_price = partial_price
                        pos.stop_loss = pos.entry_price - 0.1 * pos.initial_risk_dist
                        partial_pnl = pos.risk_pct_effective * partial_tp_frac * partial_tp_r
                        equity *= (1 + partial_pnl / 100)
                        pos.risk_pct_effective *= (1 - partial_tp_frac)

            # Time-based exit
            if exit_price is None and max_bars > 0:
                bars_held = bar_i - pos.entry_bar
                if bars_held >= max_bars:
                    exit_price = float(row["close"])
                    exit_reason = "time"

            if exit_price is not None:
                pnl_pct = self._compute_pnl(pos, exit_price)
                equity *= (1 + pnl_pct / 100)
                closed_trades.append(vpb.ClosedTrade(
                    asset=pos.asset, side=pos.side, strategy=pos.strategy,
                    regime=pos.regime, entry_price=pos.entry_price,
                    exit_price=exit_price, lots=pos.lots,
                    entry_time=pos.entry_time, exit_time=ts,
                    pnl_pct=pnl_pct, exit_reason=exit_reason))
                newly_closed.append(pos)
                if pnl_pct < 0 and self.cooldown_bars > 0:
                    cooldown_until[pos.asset] = bar_i + self.cooldown_bars
                # Track returns for vol targeting
                recent_returns.append(pnl_pct)

        for pos in newly_closed:
            open_positions.remove(pos)

        # === DETERMINE CURRENT RISK ===
        current_risk = self.risk_pct

        # Volatility targeting: scale risk inversely to realized vol
        if vol_target > 0 and len(recent_returns) >= 20:
            recent_vol = np.std(recent_returns[-50:]) * np.sqrt(252)  # annualized
            if recent_vol > 0:
                vol_ratio = vol_target / recent_vol
                current_risk = self.risk_pct * np.clip(vol_ratio, 0.3, 2.0)

        # Equity curve filter: reduce risk when below equity MA
        if eq_ma_period > 0 and len(equity_history) >= eq_ma_period:
            eq_ma = np.mean(equity_history[-eq_ma_period:])
            if equity < eq_ma:
                current_risk *= eq_ma_risk_mult

        # === CHECK ENTRIES ===
        if len(open_positions) < self.max_open:
            for asset, sig_df in asset_signals.items():
                if len(open_positions) >= self.max_open:
                    break
                if ts not in sig_df.index:
                    continue
                asset_positions = [p for p in open_positions if p.asset == asset]
                if len(asset_positions) >= self.max_per_asset:
                    continue
                if asset in cooldown_until and bar_i < cooldown_until[asset]:
                    continue
                row = sig_df.loc[ts]
                signal = row["signal"]
                if signal == 0:
                    continue
                sl_val, tp_val, close_val = row["sl"], row["tp"], row["close"]
                if pd.isna(sl_val) or pd.isna(tp_val) or sl_val <= 0 or tp_val <= 0:
                    continue
                side = "long" if signal == 1 else "short"
                if side == "long":
                    risk_dist = close_val - sl_val
                    reward_dist = tp_val - close_val
                else:
                    risk_dist = sl_val - close_val
                    reward_dist = close_val - tp_val
                if risk_dist <= 0 or reward_dist <= 0:
                    continue
                if reward_dist / risk_dist < self.min_rr:
                    continue
                entry = close_val * (1 + self.slippage_pct) if side == "long" else close_val * (1 - self.slippage_pct)
                open_positions.append(EnhancedPosition(
                    asset=asset, side=side, strategy=str(row["strategy"]),
                    regime=str(row["regime"]), entry_price=entry,
                    stop_loss=sl_val, take_profit=tp_val, lots=0,
                    entry_time=ts,
                    risk_pct_effective=current_risk,
                    initial_risk_dist=risk_dist,
                    entry_bar=bar_i,
                    original_risk_pct=current_risk))

        peak_equity = max(peak_equity, equity)
        equity_points.append((ts, equity))
        equity_history.append(equity)

    # Force-close remaining
    for pos in open_positions:
        sig_df = asset_signals.get(pos.asset)
        if sig_df is not None and len(sig_df) > 0:
            last_price = float(sig_df["close"].iloc[-1])
            pnl_pct = self._compute_pnl(pos, last_price)
            equity *= (1 + pnl_pct / 100)
            closed_trades.append(vpb.ClosedTrade(
                asset=pos.asset, side=pos.side, strategy=pos.strategy,
                regime=pos.regime, entry_price=pos.entry_price,
                exit_price=last_price, lots=pos.lots,
                entry_time=pos.entry_time, exit_time=sig_df.index[-1],
                pnl_pct=pnl_pct, exit_reason="end_of_data"))
            equity_points.append((sig_df.index[-1], equity))

    eq_series = pd.Series([e for _, e in equity_points],
                          index=pd.DatetimeIndex([t for t, _ in equity_points]))
    return vpb.PortfolioBacktestResult(
        closed_trades=closed_trades, equity_curve=eq_series,
        initial_equity=initial_equity, final_equity=equity, elapsed_secs=0,
        asset_breakdown=self._breakdown_by(closed_trades, "asset"),
        strategy_breakdown=self._breakdown_by(closed_trades, "strategy"),
        regime_breakdown=self._breakdown_by(closed_trades, "regime"))


if __name__ == "__main__":
    svc = MarketDataService(provider='yfinance')
    regime_svc = RegimeService()
    regime_svc.load(Path('models') / 'regime_model.pkl')
    param_store = ParameterStore()

    def make_params(self):
        return {"reference_hours": 8, "breakout_buffer": 0.0,
                "sl_atr_mult": 1.5, "tp_rr": 7.0,
                "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2}
    sb.SessionBreakoutStrategy.default_params = make_params

    assets = ['XAGUSD', 'XAUUSD', 'ETHUSD', 'USOIL', 'BTCUSD']
    market_data = {}
    for asset in assets:
        df = svc.get(asset, timeframe='1h', start='2024-06-01')
        df = add_regime_features(df)
        df = add_alpha_features(df)
        market_data[asset] = df
        print(f"  Loaded {asset}: {len(df)} bars")

    results = []

    def run_enhanced(label, risk=1.5, partial_tp_r=0.0, partial_tp_frac=0.5,
                     max_bars=0, vol_target=0.0, eq_ma_period=0, eq_ma_risk_mult=0.5,
                     mpa=2, max_open=12):
        # Monkey-patch with closure
        def sim(self, asset_signals, initial_equity):
            return enhanced_simulate(self, asset_signals, initial_equity,
                                     partial_tp_r=partial_tp_r,
                                     partial_tp_frac=partial_tp_frac,
                                     max_bars=max_bars,
                                     vol_target=vol_target,
                                     eq_ma_period=eq_ma_period,
                                     eq_ma_risk_mult=eq_ma_risk_mult)
        vpb.PortfolioBacktester._simulate_portfolio = sim

        bt = PortfolioBacktester(
            market_data=market_data, regime_service=regime_svc,
            parameter_store=param_store, max_open_trades=max_open,
            risk_pct=risk, trailing_stop_atr=0.0, trailing_activate_rr=0,
            cooldown_bars=0, min_rr=1.5, max_positions_per_asset=mpa)
        r = bt.run(equity=10000)
        trades = r.trades_df
        if len(trades) == 0:
            return
        wr = (trades.pnl_pct > 0).mean() * 100
        eq = r.equity_curve
        dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
        results.append((label, r.total_return_pct, r.n_trades, wr, dd))
        print(f"  {label}: {r.total_return_pct:+.1f}% | {r.n_trades} trades | WR: {wr:.1f}% | DD: {dd:.1f}%")

    # ============================================================
    # BASELINE
    # ============================================================
    print("\n=== BASELINE ===")
    run_enhanced("BASELINE 1.5%")

    # ============================================================
    # TEST 1: Partial Take-Profit
    # ============================================================
    print("\n=== PARTIAL TAKE-PROFIT ===")
    for partial_r in [2.0, 3.0, 3.5, 4.0, 5.0]:
        for frac in [0.3, 0.5]:
            run_enhanced(f"partTP@{partial_r:.0f}R/{frac:.0f}%",
                        partial_tp_r=partial_r, partial_tp_frac=frac)

    # ============================================================
    # TEST 2: Time-based Exit
    # ============================================================
    print("\n=== TIME-BASED EXIT ===")
    for max_b in [48, 72, 96, 120, 168]:
        run_enhanced(f"timeExit@{max_b}bars", max_bars=max_b)

    # ============================================================
    # TEST 3: Volatility Targeting
    # ============================================================
    print("\n=== VOLATILITY TARGETING ===")
    for vt in [15, 20, 25, 30, 40]:
        run_enhanced(f"volTarget@{vt}%", vol_target=vt)

    # ============================================================
    # TEST 4: Equity Curve Filter
    # ============================================================
    print("\n=== EQUITY CURVE FILTER ===")
    for ma_per in [50, 100, 200]:
        for mult in [0.3, 0.5, 0.7]:
            run_enhanced(f"eqMA{ma_per}/{mult:.1f}x",
                        eq_ma_period=ma_per, eq_ma_risk_mult=mult)

    # ============================================================
    # TEST 5: Combinations of best techniques
    # ============================================================
    print("\n=== COMBINATIONS ===")
    # Partial TP + equity curve filter
    run_enhanced("partTP3R+eqMA100",
                partial_tp_r=3.0, partial_tp_frac=0.5,
                eq_ma_period=100, eq_ma_risk_mult=0.5)

    # Partial TP + higher risk (since partial TP reduces DD)
    for risk in [1.75, 2.0, 2.25]:
        run_enhanced(f"partTP3R/{risk:.2f}%",
                    risk=risk, partial_tp_r=3.0, partial_tp_frac=0.5)

    # Partial TP + vol targeting
    run_enhanced("partTP3R+volT25",
                partial_tp_r=3.0, partial_tp_frac=0.5, vol_target=25)

    # Time exit + partial TP
    run_enhanced("partTP3R+time96",
                partial_tp_r=3.0, partial_tp_frac=0.5, max_bars=96)

    # Kitchen sink: partial TP + eq MA + higher risk
    for risk in [1.75, 2.0]:
        run_enhanced(f"ALL/{risk:.2f}%",
                    risk=risk, partial_tp_r=3.0, partial_tp_frac=0.5,
                    eq_ma_period=100, eq_ma_risk_mult=0.5)

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 72)
    print("ALL RESULTS (sorted by return, DD < 50%)")
    print("=" * 72)
    # Sort by return, prioritize DD < 50%
    results.sort(key=lambda x: x[1] if x[4] > -50 else -99999, reverse=True)
    print("{:<28s} {:>10s} {:>6s} {:>5s} {:>7s}".format('Config', 'Return', 'Trades', 'WR', 'MaxDD'))
    print('-' * 72)
    for label, ret, nt, wr, dd in results:
        flag = " ***" if ret > 3322 and dd > -50 else (" **" if dd > -50 and ret > 2000 else "")
        print("{:<28s} {:>+9.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%{}".format(label, ret, nt, wr, dd, flag))

    print("\n=== BEST: MAX RETURN WITH DD < 50% ===")
    under50 = [(l, r, n, w, d) for l, r, n, w, d in results if d > -50]
    under50.sort(key=lambda x: x[1], reverse=True)
    for label, ret, nt, wr, dd in under50[:5]:
        print("{:<28s} {:>+9.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%".format(label, ret, nt, wr, dd))
