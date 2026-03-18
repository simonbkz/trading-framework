"""
Backtest comparison: multi-strategy regime routing vs session_breakout only.
Also tests layered optimizations (volatility scaling, macro features, etc).
"""
import sys, warnings, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from data.market_data import MarketDataService
from features.regime_features import add_regime_features
from features.alpha_features import add_alpha_features
from regimes.regime_service import RegimeService
from optimization.parameter_store import ParameterStore
from validation.portfolio_backtest import PortfolioBacktester, PortfolioBacktestResult
import validation.portfolio_backtest as vpb
from strategies.strategy_factory import list_strategies, get_strategy
from config.regimes import REGIME_DEFINITIONS


def fixed_risk_simulate(self, asset_signals, initial_equity):
    """Standard fixed-risk simulation matching final_verification.py."""
    all_idx = pd.DatetimeIndex([])
    for sig_df in asset_signals.values():
        all_idx = all_idx.union(sig_df.index)
    all_idx = all_idx.sort_values()
    if len(all_idx) <= self.warmup_bars:
        return self._empty_result(initial_equity, 0)
    eval_idx = all_idx[self.warmup_bars:]
    open_positions, closed_trades = [], []
    equity, peak_equity = initial_equity, initial_equity
    equity_points = []
    cooldown_until = {}

    for bar_i, ts in enumerate(eval_idx):
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
                closed_trades.append(vpb.ClosedTrade(
                    asset=pos.asset, side=pos.side, strategy=pos.strategy,
                    regime=pos.regime, entry_price=pos.entry_price,
                    exit_price=exit_price, lots=pos.lots,
                    entry_time=pos.entry_time, exit_time=ts,
                    pnl_pct=pnl_pct, exit_reason=exit_reason))
                newly_closed.append(pos)
                if pnl_pct < 0 and self.cooldown_bars > 0:
                    cooldown_until[pos.asset] = bar_i + self.cooldown_bars
        for pos in newly_closed:
            open_positions.remove(pos)

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
                open_positions.append(vpb.OpenPosition(
                    asset=asset, side=side, strategy=str(row.get("strategy", "session_breakout")),
                    regime=str(row.get("regime", "unknown")), entry_price=entry,
                    stop_loss=sl_val, take_profit=tp_val, lots=0,
                    entry_time=ts,
                    risk_pct_effective=self.risk_pct,
                    initial_risk_dist=risk_dist))
        peak_equity = max(peak_equity, equity)
        equity_points.append((ts, equity))

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


def print_result(label, result):
    """Pretty-print a backtest result."""
    trades = result.trades_df
    if len(trades) == 0:
        print(f"\n{label}: NO TRADES")
        return
    wr = (trades.pnl_pct > 0).mean() * 100
    eq = result.equity_curve
    dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
    avg_win = trades[trades.pnl_pct > 0].pnl_pct.mean() if (trades.pnl_pct > 0).any() else 0
    avg_loss = trades[trades.pnl_pct <= 0].pnl_pct.mean() if (trades.pnl_pct <= 0).any() else 0
    pf = trades[trades.pnl_pct > 0].pnl_pct.sum() / abs(trades[trades.pnl_pct <= 0].pnl_pct.sum()) if (trades.pnl_pct <= 0).any() else 999

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Return:        {result.total_return_pct:+.1f}%")
    print(f"  Max Drawdown:  {dd:.1f}%")
    print(f"  Trades:        {result.n_trades}")
    print(f"  Win Rate:      {wr:.1f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Avg Winner:    {avg_win:+.2f}%")
    print(f"  Avg Loser:     {avg_loss:+.2f}%")
    print(f"  $10K -> ${result.final_equity:,.0f}")

    # Strategy breakdown
    if hasattr(result, 'strategy_breakdown') and result.strategy_breakdown:
        print(f"\n  Strategy Breakdown:")
        for strat, stats in sorted(result.strategy_breakdown.items()):
            s_trades = [t for t in result.closed_trades if t.strategy == strat]
            s_pnl = sum(t.pnl_pct for t in s_trades)
            s_wr = sum(1 for t in s_trades if t.pnl_pct > 0) / max(len(s_trades), 1) * 100
            print(f"    {strat:25s}: {len(s_trades):3d} trades, PnL={s_pnl:+.1f}%, WR={s_wr:.0f}%")

    # Asset breakdown
    print(f"\n  Asset Breakdown:")
    for asset in sorted(set(t.asset for t in result.closed_trades)):
        a_trades = [t for t in result.closed_trades if t.asset == asset]
        a_pnl = sum(t.pnl_pct for t in a_trades)
        a_wr = sum(1 for t in a_trades if t.pnl_pct > 0) / max(len(a_trades), 1) * 100
        print(f"    {asset:10s}: {len(a_trades):3d} trades, PnL={a_pnl:+.1f}%, WR={a_wr:.0f}%")

    # Regime breakdown
    print(f"\n  Regime Breakdown:")
    for regime in sorted(set(t.regime for t in result.closed_trades)):
        r_trades = [t for t in result.closed_trades if t.regime == regime]
        r_pnl = sum(t.pnl_pct for t in r_trades)
        r_wr = sum(1 for t in r_trades if t.pnl_pct > 0) / max(len(r_trades), 1) * 100
        print(f"    {regime:20s}: {len(r_trades):3d} trades, PnL={r_pnl:+.1f}%, WR={r_wr:.0f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    vpb.PortfolioBacktester._simulate_portfolio = fixed_risk_simulate

    # Load data
    svc = MarketDataService(provider='yfinance')
    regime_svc = RegimeService()
    regime_svc.load(Path('models') / 'regime_model.pkl')
    param_store = ParameterStore()

    assets = ['XAUUSD', 'BTCUSD', 'XAGUSD', 'ETHUSD', 'EURJPY']
    market_data = {}
    print("Loading market data...")
    for asset in assets:
        df = svc.get(asset, timeframe='1h', start='2024-06-01')
        df = add_regime_features(df)
        df = add_alpha_features(df)
        market_data[asset] = df
        print(f"  {asset}: {len(df)} bars")

    # Also add macro features for one of the tests
    market_data_macro = {}
    try:
        from features.macro_features import add_all_macro_features
        print("\nAdding macro features...")
        for asset, df in market_data.items():
            df_m = add_all_macro_features(df.copy(), asset=asset, start='2024-06-01')
            market_data_macro[asset] = df_m
            print(f"  {asset}: +macro features")
    except Exception as e:
        print(f"  Macro features failed: {e}")
        market_data_macro = market_data

    # ======================================================================
    # TEST 1: Baseline — session_breakout only (old config)
    # ======================================================================
    print("\n\nRunning TEST 1: Baseline (session_breakout only)...")

    # Temporarily revert regime routing to session_breakout only
    saved_defs = {}
    for regime_label, regime_def in REGIME_DEFINITIONS.items():
        saved_defs[regime_label] = regime_def.preferred_strategies
        regime_def.preferred_strategies = ["session_breakout"]

    bt1 = PortfolioBacktester(
        market_data=market_data, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=2.0, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
    result1 = bt1.run(equity=10000)
    print_result("TEST 1: Baseline (session_breakout only)", result1)

    # Restore multi-strategy routing
    for regime_label, strats in saved_defs.items():
        REGIME_DEFINITIONS[regime_label].preferred_strategies = strats

    # ======================================================================
    # TEST 2: Multi-strategy regime routing (new config)
    # ======================================================================
    print("\n\nRunning TEST 2: Multi-strategy regime routing...")
    bt2 = PortfolioBacktester(
        market_data=market_data, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=2.0, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
    result2 = bt2.run(equity=10000)
    print_result("TEST 2: Multi-strategy regime routing", result2)

    # ======================================================================
    # TEST 3: Multi-strategy + macro features
    # ======================================================================
    print("\n\nRunning TEST 3: Multi-strategy + macro features...")
    bt3 = PortfolioBacktester(
        market_data=market_data_macro, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=2.0, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
    result3 = bt3.run(equity=10000)
    print_result("TEST 3: Multi-strategy + macro features", result3)

    # ======================================================================
    # TEST 4: Multi-strategy + max_per_asset=1 (safer)
    # ======================================================================
    print("\n\nRunning TEST 4: Multi-strategy, max_per_asset=1 (lower DD)...")
    bt4 = PortfolioBacktester(
        market_data=market_data, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=2.0, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=1)
    result4 = bt4.run(equity=10000)
    print_result("TEST 4: Multi-strategy, max_per_asset=1", result4)

    # ======================================================================
    # TEST 5: Multi-strategy + 1.5% risk (conservative)
    # ======================================================================
    print("\n\nRunning TEST 5: Multi-strategy, 1.5% risk...")
    bt5 = PortfolioBacktester(
        market_data=market_data, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=1.5, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
    result5 = bt5.run(equity=10000)
    print_result("TEST 5: Multi-strategy, 1.5% risk", result5)

    # ======================================================================
    # TEST 6: Multi-strategy + macro + max_per_asset=1 + 1.5% (best DD)
    # ======================================================================
    print("\n\nRunning TEST 6: Conservative optimized (macro+mpa1+1.5%)...")
    bt6 = PortfolioBacktester(
        market_data=market_data_macro, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=1.5, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=1)
    result6 = bt6.run(equity=10000)
    print_result("TEST 6: Conservative (macro + mpa=1 + 1.5% risk)", result6)

    # ======================================================================
    # SUMMARY TABLE
    # ======================================================================
    print("\n\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Test':<45} {'Return':>10} {'MaxDD':>8} {'Trades':>7} {'WR':>6} {'PF':>6} {'Final$':>12}")
    print("-" * 80)

    for label, res in [
        ("1. Baseline (session only)", result1),
        ("2. Multi-strategy routing", result2),
        ("3. Multi-strat + macro", result3),
        ("4. Multi-strat, mpa=1", result4),
        ("5. Multi-strat, 1.5% risk", result5),
        ("6. Conservative optimized", result6),
    ]:
        t = res.trades_df
        if len(t) == 0:
            print(f"{label:<45} {'N/A':>10}")
            continue
        wr = (t.pnl_pct > 0).mean() * 100
        eq = res.equity_curve
        dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
        pf = t[t.pnl_pct > 0].pnl_pct.sum() / abs(t[t.pnl_pct <= 0].pnl_pct.sum()) if (t.pnl_pct <= 0).any() else 999
        print(f"{label:<45} {res.total_return_pct:>+9.0f}% {dd:>+7.1f}% {res.n_trades:>7} {wr:>5.1f}% {pf:>5.2f} ${res.final_equity:>11,.0f}")

    print("=" * 80)
