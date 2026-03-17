"""Next-level sweep: 6th asset, session timing, asymmetric risk."""
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
                    asset=asset, side=side, strategy=str(row["strategy"]),
                    regime=str(row["regime"]), entry_price=entry,
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


if __name__ == "__main__":
    vpb.PortfolioBacktester._simulate_portfolio = fixed_risk_simulate
    svc = MarketDataService(provider='yfinance')
    regime_svc = RegimeService()
    regime_svc.load(Path('models') / 'regime_model.pkl')
    param_store = ParameterStore()

    # Load all candidate assets
    all_assets = ['XAGUSD', 'XAUUSD', 'ETHUSD', 'USOIL', 'BTCUSD', 'EURJPY', 'GBPJPY']
    all_data = {}
    for asset in all_assets:
        try:
            df = svc.get(asset, timeframe='1h', start='2024-06-01')
            df = add_regime_features(df)
            df = add_alpha_features(df)
            all_data[asset] = df
            print(f"  Loaded {asset}: {len(df)} bars")
        except Exception as e:
            print(f"  SKIP {asset}: {e}")

    results = []
    base5 = ['XAGUSD', 'XAUUSD', 'ETHUSD', 'USOIL', 'BTCUSD']

    # ============================================================
    # TEST 1: Baseline confirmation (5 assets, tp7, risk 1.5%)
    # ============================================================
    print("\n=== TEST 1: Baseline ===")
    def make_params(tp=7.0, sl=1.5, session_hour=7, ref_hours=8, max_entry=8):
        def d(self):
            return {"reference_hours": ref_hours, "breakout_buffer": 0.0,
                    "sl_atr_mult": sl, "tp_rr": tp,
                    "max_entry_hours": max_entry, "session_open_hour": session_hour, "vol_mult": 1.2}
        return d

    sb.SessionBreakoutStrategy.default_params = make_params(tp=7.0)
    md = {a: all_data[a] for a in base5 if a in all_data}
    bt = PortfolioBacktester(market_data=md, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=12,
        risk_pct=1.5, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
    r = bt.run(equity=10000)
    trades = r.trades_df
    if len(trades) > 0:
        wr = (trades.pnl_pct > 0).mean() * 100
        eq = r.equity_curve
        dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
        results.append(("BASELINE 5a/tp7/1.5%", r.total_return_pct, r.n_trades, wr, dd))
        print(f"  Return: {r.total_return_pct:+.1f}% | {r.n_trades} trades | WR: {wr:.1f}% | DD: {dd:.1f}%")

    # ============================================================
    # TEST 2: Add 6th asset (EURJPY, GBPJPY)
    # ============================================================
    print("\n=== TEST 2: 6th Asset ===")
    for extra in ['EURJPY', 'GBPJPY']:
        if extra not in all_data:
            continue
        sb.SessionBreakoutStrategy.default_params = make_params(tp=7.0)
        assets6 = base5 + [extra]
        md = {a: all_data[a] for a in assets6 if a in all_data}
        for risk in [1.0, 1.25, 1.5]:
            bt = PortfolioBacktester(market_data=md, regime_service=regime_svc,
                parameter_store=param_store, max_open_trades=14,
                risk_pct=risk, trailing_stop_atr=0.0, trailing_activate_rr=0,
                cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
            r = bt.run(equity=10000)
            trades = r.trades_df
            if len(trades) > 0:
                wr = (trades.pnl_pct > 0).mean() * 100
                eq = r.equity_curve
                dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
                label = f"6a+{extra}/{risk:.2f}%"
                results.append((label, r.total_return_pct, r.n_trades, wr, dd))

    # ============================================================
    # TEST 3: Session timing variations
    # ============================================================
    print("\n=== TEST 3: Session Timing ===")
    for session_hour in [6, 7, 8, 9]:
        for ref_hours in [6, 8, 10]:
            sb.SessionBreakoutStrategy.default_params = make_params(
                tp=7.0, session_hour=session_hour, ref_hours=ref_hours)
            md = {a: all_data[a] for a in base5 if a in all_data}
            bt = PortfolioBacktester(market_data=md, regime_service=regime_svc,
                parameter_store=param_store, max_open_trades=12,
                risk_pct=1.5, trailing_stop_atr=0.0, trailing_activate_rr=0,
                cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
            r = bt.run(equity=10000)
            trades = r.trades_df
            if len(trades) > 0:
                wr = (trades.pnl_pct > 0).mean() * 100
                eq = r.equity_curve
                dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
                label = f"ses{session_hour}/ref{ref_hours}"
                results.append((label, r.total_return_pct, r.n_trades, wr, dd))

    # ============================================================
    # TEST 4: SL multiplier fine-tuning
    # ============================================================
    print("\n=== TEST 4: SL Fine-tuning ===")
    for sl_mult in [1.0, 1.25, 1.5, 1.75, 2.0]:
        sb.SessionBreakoutStrategy.default_params = make_params(tp=7.0, sl=sl_mult)
        md = {a: all_data[a] for a in base5 if a in all_data}
        bt = PortfolioBacktester(market_data=md, regime_service=regime_svc,
            parameter_store=param_store, max_open_trades=12,
            risk_pct=1.5, trailing_stop_atr=0.0, trailing_activate_rr=0,
            cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
        r = bt.run(equity=10000)
        trades = r.trades_df
        if len(trades) > 0:
            wr = (trades.pnl_pct > 0).mean() * 100
            eq = r.equity_curve
            dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
            label = f"sl{sl_mult:.2f}/tp7"
            results.append((label, r.total_return_pct, r.n_trades, wr, dd))

    # ============================================================
    # TEST 5: TP ratio sweep around 7 (finer grid)
    # ============================================================
    print("\n=== TEST 5: TP Fine-tuning ===")
    for tp in [6.0, 6.5, 7.0, 7.5, 8.0, 9.0]:
        sb.SessionBreakoutStrategy.default_params = make_params(tp=tp)
        md = {a: all_data[a] for a in base5 if a in all_data}
        bt = PortfolioBacktester(market_data=md, regime_service=regime_svc,
            parameter_store=param_store, max_open_trades=12,
            risk_pct=1.5, trailing_stop_atr=0.0, trailing_activate_rr=0,
            cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
        r = bt.run(equity=10000)
        trades = r.trades_df
        if len(trades) > 0:
            wr = (trades.pnl_pct > 0).mean() * 100
            eq = r.equity_curve
            dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
            label = f"tp{tp:.1f}/1.5%"
            results.append((label, r.total_return_pct, r.n_trades, wr, dd))

    # ============================================================
    # TEST 6: Higher risk with 50% DD tolerance
    # ============================================================
    print("\n=== TEST 6: Higher Risk (50% DD ok) ===")
    sb.SessionBreakoutStrategy.default_params = make_params(tp=7.0)
    md = {a: all_data[a] for a in base5 if a in all_data}
    for risk in [1.5, 1.75, 2.0, 2.25, 2.5]:
        bt = PortfolioBacktester(market_data=md, regime_service=regime_svc,
            parameter_store=param_store, max_open_trades=12,
            risk_pct=risk, trailing_stop_atr=0.0, trailing_activate_rr=0,
            cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
        r = bt.run(equity=10000)
        trades = r.trades_df
        if len(trades) > 0:
            wr = (trades.pnl_pct > 0).mean() * 100
            eq = r.equity_curve
            dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
            label = f"risk{risk:.2f}%/5a"
            results.append((label, r.total_return_pct, r.n_trades, wr, dd))

    # ============================================================
    # TEST 7: Pyramiding 3 per asset
    # ============================================================
    print("\n=== TEST 7: Pyramiding 3 ===")
    for risk in [1.0, 1.25, 1.5]:
        bt = PortfolioBacktester(market_data=md, regime_service=regime_svc,
            parameter_store=param_store, max_open_trades=15,
            risk_pct=risk, trailing_stop_atr=0.0, trailing_activate_rr=0,
            cooldown_bars=0, min_rr=1.5, max_positions_per_asset=3)
        r = bt.run(equity=10000)
        trades = r.trades_df
        if len(trades) > 0:
            wr = (trades.pnl_pct > 0).mean() * 100
            eq = r.equity_curve
            dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
            label = f"pyr3/{risk:.2f}%"
            results.append((label, r.total_return_pct, r.n_trades, wr, dd))

    # ============================================================
    # TEST 8: max_entry_hours variation
    # ============================================================
    print("\n=== TEST 8: Entry Window ===")
    for max_entry in [4, 6, 8, 10, 12]:
        sb.SessionBreakoutStrategy.default_params = make_params(tp=7.0, max_entry=max_entry)
        md = {a: all_data[a] for a in base5 if a in all_data}
        bt = PortfolioBacktester(market_data=md, regime_service=regime_svc,
            parameter_store=param_store, max_open_trades=12,
            risk_pct=1.5, trailing_stop_atr=0.0, trailing_activate_rr=0,
            cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
        r = bt.run(equity=10000)
        trades = r.trades_df
        if len(trades) > 0:
            wr = (trades.pnl_pct > 0).mean() * 100
            eq = r.equity_curve
            dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
            label = f"entry{max_entry}h"
            results.append((label, r.total_return_pct, r.n_trades, wr, dd))

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 72)
    print("{:<28s} {:>10s} {:>6s} {:>5s} {:>7s}".format('Config', 'Return', 'Trades', 'WR', 'MaxDD'))
    print('-' * 72)
    results.sort(key=lambda x: x[1], reverse=True)
    for label, ret, nt, wr, dd in results:
        flag = " ***" if ret > 3322 and dd > -50 else (" **" if dd > -50 else "")
        print("{:<28s} {:>+9.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%{}".format(label, ret, nt, wr, dd, flag))

    # Best configs under 50% DD
    print("\n=== BEST CONFIGS (DD > -50%) ===")
    filtered = [(l, r, n, w, d) for l, r, n, w, d in results if d > -50]
    filtered.sort(key=lambda x: x[1], reverse=True)
    for label, ret, nt, wr, dd in filtered[:10]:
        print("{:<28s} {:>+9.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%".format(label, ret, nt, wr, dd))
