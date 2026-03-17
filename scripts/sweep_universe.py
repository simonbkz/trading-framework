"""Test every available asset pair/index to find the best portfolio mix."""
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
from itertools import combinations


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


def run_backtest(md, regime_svc, param_store, risk=1.5, max_open=12, mpa=2):
    bt = PortfolioBacktester(
        market_data=md, regime_service=regime_svc,
        parameter_store=param_store, max_open_trades=max_open,
        risk_pct=risk, trailing_stop_atr=0.0, trailing_activate_rr=0,
        cooldown_bars=0, min_rr=1.5, max_positions_per_asset=mpa)
    r = bt.run(equity=10000)
    trades = r.trades_df
    if len(trades) == 0:
        return None
    wr = (trades.pnl_pct > 0).mean() * 100
    eq = r.equity_curve
    dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
    return (r.total_return_pct, r.n_trades, wr, dd)


if __name__ == "__main__":
    vpb.PortfolioBacktester._simulate_portfolio = fixed_risk_simulate
    svc = MarketDataService(provider='yfinance')
    regime_svc = RegimeService()
    regime_svc.load(Path('models') / 'regime_model.pkl')
    param_store = ParameterStore()

    def make_params(self):
        return {"reference_hours": 8, "breakout_buffer": 0.0,
                "sl_atr_mult": 1.5, "tp_rr": 7.0,
                "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2}
    sb.SessionBreakoutStrategy.default_params = make_params

    # Full universe of assets to test
    universe = [
        'XAGUSD', 'XAUUSD', 'ETHUSD', 'USOIL', 'BTCUSD',  # current best
        'EURJPY', 'GBPJPY', 'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD',  # forex
        'NAS100', 'SPX500', 'US30',  # indices (yfinance tickers)
        'NGAS',  # natural gas
    ]

    # yfinance ticker mapping
    yf_tickers = {
        'NAS100': '^IXIC', 'SPX500': '^GSPC', 'US30': '^DJI',
        'NGAS': 'NG=F',
    }

    print("=== LOADING ALL ASSETS ===")
    all_data = {}
    for asset in universe:
        try:
            df = svc.get(asset, timeframe='1h', start='2024-06-01')
            df = add_regime_features(df)
            df = add_alpha_features(df)
            all_data[asset] = df
            print(f"  OK {asset}: {len(df)} bars")
        except Exception as e:
            print(f"  SKIP {asset}: {e}")

    available = list(all_data.keys())
    print(f"\nAvailable: {available}")

    # ============================================================
    # STEP 1: Test each individual asset at 1.5% risk
    # ============================================================
    print("\n=== INDIVIDUAL ASSET PERFORMANCE ===")
    individual = []
    for asset in available:
        md = {asset: all_data[asset]}
        res = run_backtest(md, regime_svc, param_store, risk=1.5, max_open=4, mpa=2)
        if res:
            ret, nt, wr, dd = res
            individual.append((asset, ret, nt, wr, dd))
            print(f"  {asset:10s}: {ret:>+8.1f}% | {nt:>4d} trades | WR: {wr:>4.1f}% | DD: {dd:>+6.1f}%")

    # Sort by return
    individual.sort(key=lambda x: x[1], reverse=True)
    print("\nRanked by return:")
    for asset, ret, nt, wr, dd in individual:
        print(f"  {asset:10s}: {ret:>+8.1f}%  DD: {dd:>+6.1f}%")

    # ============================================================
    # STEP 2: Test best combos of 5, 6, 7 assets
    # ============================================================
    # Only use assets with positive returns individually
    positive_assets = [a for a, r, n, w, d in individual if r > 0]
    print(f"\nPositive-return assets: {positive_assets}")

    # If we have enough, test combinations
    results = []

    if len(positive_assets) >= 5:
        print("\n=== TESTING 5-ASSET COMBOS (from positive assets) ===")
        combos5 = list(combinations(positive_assets, 5))
        print(f"Testing {len(combos5)} combos...")
        for combo in combos5:
            md = {a: all_data[a] for a in combo}
            res = run_backtest(md, regime_svc, param_store, risk=1.5, max_open=12, mpa=2)
            if res:
                ret, nt, wr, dd = res
                label = "+".join(combo)
                results.append((label, 5, ret, nt, wr, dd))

    if len(positive_assets) >= 6:
        print("\n=== TESTING 6-ASSET COMBOS (from top assets) ===")
        # Only test top 8 to keep combos manageable
        top8 = positive_assets[:8]
        combos6 = list(combinations(top8, 6))
        print(f"Testing {len(combos6)} combos...")
        for combo in combos6:
            md = {a: all_data[a] for a in combo}
            res = run_backtest(md, regime_svc, param_store, risk=1.5, max_open=14, mpa=2)
            if res:
                ret, nt, wr, dd = res
                label = "+".join(combo)
                results.append((label, 6, ret, nt, wr, dd))

    if len(positive_assets) >= 7:
        print("\n=== TESTING 7-ASSET COMBOS (from top assets) ===")
        top8 = positive_assets[:8]
        combos7 = list(combinations(top8, 7))
        print(f"Testing {len(combos7)} combos...")
        for combo in combos7:
            md = {a: all_data[a] for a in combo}
            res = run_backtest(md, regime_svc, param_store, risk=1.5, max_open=16, mpa=2)
            if res:
                ret, nt, wr, dd = res
                label = "+".join(combo)
                results.append((label, 7, ret, nt, wr, dd))

    # ============================================================
    # RESULTS: Best configs under 50% DD, sorted by return
    # ============================================================
    print("\n" + "=" * 90)
    print("ALL RESULTS (sorted by return, DD < 50%)")
    print("=" * 90)
    under50 = [(l, n, r, t, w, d) for l, n, r, t, w, d in results if d > -50]
    under50.sort(key=lambda x: x[2], reverse=True)
    print(f"\n{len(under50)} configs with DD > -50% out of {len(results)} total\n")
    print("{:<55s} {:>3s} {:>9s} {:>5s} {:>5s} {:>7s}".format('Assets', '#A', 'Return', 'Trds', 'WR', 'MaxDD'))
    print('-' * 90)
    for label, na, ret, nt, wr, dd in under50[:30]:
        print("{:<55s} {:>2d}  {:>+8.1f}% {:>5d} {:>4.1f}% {:>+6.1f}%".format(label, na, ret, nt, wr, dd))

    # Also show best regardless of DD
    print("\n\nTOP 10 BY RETURN (any DD)")
    results.sort(key=lambda x: x[2], reverse=True)
    for label, na, ret, nt, wr, dd in results[:10]:
        flag = " <<<" if dd > -50 else ""
        print("{:<55s} {:>2d}  {:>+8.1f}% {:>5d} {:>4.1f}% {:>+6.1f}%{}".format(label, na, ret, nt, wr, dd, flag))
