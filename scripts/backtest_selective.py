"""
Selective strategy tests: find the best combination.
Based on initial findings:
- session_breakout is the dominant alpha source (tp_rr=7.0)
- momentum_trend is promising (+195%, 39% WR)
- pullback_retest, trend_breakout, mean_reversion are value destroyers

Tests:
A. session_breakout only (baseline)
B. session_breakout + momentum_trend (selective)
C. session_breakout + momentum_trend with tp_rr=7.0
D. session_breakout with direction-biased regimes
"""
import sys, warnings
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
from validation.portfolio_backtest import PortfolioBacktester
import validation.portfolio_backtest as vpb
import strategies.session_breakout as sb
import strategies.momentum_trend as mt
from config.regimes import REGIME_DEFINITIONS


def fixed_risk_simulate(self, asset_signals, initial_equity):
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
    print(f"  Avg Winner:    {avg_win:+.2f}%  Avg Loser: {avg_loss:+.2f}%")
    print(f"  $10K -> ${result.final_equity:,.0f}")

    # Strategy breakdown
    for strat in sorted(set(t.strategy for t in result.closed_trades)):
        s_trades = [t for t in result.closed_trades if t.strategy == strat]
        s_pnl = sum(t.pnl_pct for t in s_trades)
        s_wr = sum(1 for t in s_trades if t.pnl_pct > 0) / max(len(s_trades), 1) * 100
        print(f"    {strat:25s}: {len(s_trades):3d} trades, PnL={s_pnl:+.1f}%, WR={s_wr:.0f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    vpb.PortfolioBacktester._simulate_portfolio = fixed_risk_simulate

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

    # Save original momentum_trend params
    orig_mt_params = mt.MomentumTrendStrategy.default_params

    # ======================================================================
    # TEST A: Baseline — session_breakout only
    # ======================================================================
    print("\n\nTEST A: session_breakout only (baseline)...")
    for rd in REGIME_DEFINITIONS.values():
        rd.preferred_strategies = ["session_breakout"]

    bt = PortfolioBacktester(
        market_data=market_data, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=2.0, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
    rA = bt.run(equity=10000)
    print_result("TEST A: session_breakout only", rA)

    # ======================================================================
    # TEST B: session_breakout + momentum_trend (default tp_rr=2.5)
    # ======================================================================
    print("\n\nTEST B: session_breakout + momentum_trend (tp_rr=2.5)...")
    for rd in REGIME_DEFINITIONS.values():
        rd.preferred_strategies = ["session_breakout"]
    # Add momentum_trend to trend regimes only
    REGIME_DEFINITIONS["trend_up"].preferred_strategies = ["session_breakout", "momentum_trend"]
    REGIME_DEFINITIONS["trend_down"].preferred_strategies = ["session_breakout", "momentum_trend"]
    REGIME_DEFINITIONS["high_volatility"].preferred_strategies = ["session_breakout", "momentum_trend"]

    bt = PortfolioBacktester(
        market_data=market_data, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=2.0, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
    rB = bt.run(equity=10000)
    print_result("TEST B: session + momentum (tp_rr=2.5)", rB)

    # ======================================================================
    # TEST C: session_breakout + momentum_trend with tp_rr=7.0
    # ======================================================================
    print("\n\nTEST C: session + momentum with tp_rr=7.0...")

    def mt_params_7r(self):
        return {"ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
                "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                "sl_atr_mult": 1.5, "tp_rr": 7.0,
                "adx_min": 18.0, "swing_lookback": 5}
    mt.MomentumTrendStrategy.default_params = mt_params_7r

    bt = PortfolioBacktester(
        market_data=market_data, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=2.0, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
    rC = bt.run(equity=10000)
    print_result("TEST C: session + momentum (tp_rr=7.0)", rC)

    # ======================================================================
    # TEST D: session + momentum(7R), max_per_asset=1
    # ======================================================================
    print("\n\nTEST D: session + momentum(7R), mpa=1...")
    bt = PortfolioBacktester(
        market_data=market_data, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=2.0, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=1)
    rD = bt.run(equity=10000)
    print_result("TEST D: session + momentum(7R), mpa=1", rD)

    # ======================================================================
    # TEST E: session + momentum(7R) with direction bias
    # ======================================================================
    print("\n\nTEST E: session + momentum(7R) + direction bias...")
    REGIME_DEFINITIONS["trend_up"].direction_bias = "long"
    REGIME_DEFINITIONS["trend_down"].direction_bias = "short"

    bt = PortfolioBacktester(
        market_data=market_data, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=2.0, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
    rE = bt.run(equity=10000)
    print_result("TEST E: session + momentum(7R) + direction bias", rE)

    # Reset direction bias
    REGIME_DEFINITIONS["trend_up"].direction_bias = "both"
    REGIME_DEFINITIONS["trend_down"].direction_bias = "both"

    # ======================================================================
    # TEST F: session_breakout only with tp_rr sweep (5, 7, 9, 10)
    # ======================================================================
    for rd in REGIME_DEFINITIONS.values():
        rd.preferred_strategies = ["session_breakout"]
    mt.MomentumTrendStrategy.default_params = orig_mt_params

    for tp_rr in [5.0, 7.0, 9.0, 10.0]:
        print(f"\n\nTEST F-{tp_rr}: session_breakout tp_rr={tp_rr}...")
        def make_params(self, _rr=tp_rr):
            return {"reference_hours": 8, "breakout_buffer": 0.0,
                    "sl_atr_mult": 1.5, "tp_rr": _rr,
                    "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2}
        sb.SessionBreakoutStrategy.default_params = make_params

        bt = PortfolioBacktester(
            market_data=market_data, regime_service=regime_svc,
            parameter_store=param_store, max_open_trades=12,
            risk_pct=2.0, trailing_stop_atr=0.0, trailing_activate_rr=0,
            cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
        result = bt.run(equity=10000)
        print_result(f"TEST F: session tp_rr={tp_rr}", result)

    # Restore default
    def orig_sb_params(self):
        return {"reference_hours": 8, "breakout_buffer": 0.0,
                "sl_atr_mult": 1.5, "tp_rr": 7.0,
                "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2}
    sb.SessionBreakoutStrategy.default_params = orig_sb_params

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Test':<50} {'Return':>10} {'MaxDD':>8} {'Trades':>7} {'WR':>6} {'PF':>6} {'Final$':>12}")
    print("-" * 90)

    for label, res in [
        ("A. Baseline (session only, tp=7R)", rA),
        ("B. session + momentum (tp=2.5R)", rB),
        ("C. session + momentum (tp=7R)", rC),
        ("D. session + momentum(7R), mpa=1", rD),
        ("E. session + momentum(7R) + dir bias", rE),
    ]:
        t = res.trades_df
        if len(t) == 0:
            print(f"{label:<50} {'N/A':>10}")
            continue
        wr = (t.pnl_pct > 0).mean() * 100
        eq = res.equity_curve
        dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
        pf = t[t.pnl_pct > 0].pnl_pct.sum() / abs(t[t.pnl_pct <= 0].pnl_pct.sum()) if (t.pnl_pct <= 0).any() else 999
        print(f"{label:<50} {res.total_return_pct:>+9.0f}% {dd:>+7.1f}% {res.n_trades:>7} {wr:>5.1f}% {pf:>5.2f} ${res.final_equity:>11,.0f}")
    print("=" * 90)
