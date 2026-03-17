"""Try different SL mult and additional assets to push returns."""
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
    op, ct = [], []
    equity, peak = initial_equity, initial_equity
    ep, cu = [], {}
    for bi, ts in enumerate(eval_idx):
        nc = []
        for pos in op:
            sd = asset_signals.get(pos.asset)
            if sd is None or ts not in sd.index: continue
            row = sd.loc[ts]
            ex, er = self._check_exit(pos, row)
            if ex is not None:
                pnl = self._compute_pnl(pos, ex)
                equity *= (1 + pnl / 100)
                ct.append(vpb.ClosedTrade(asset=pos.asset, side=pos.side, strategy=pos.strategy,
                    regime=pos.regime, entry_price=pos.entry_price, exit_price=ex, lots=pos.lots,
                    entry_time=pos.entry_time, exit_time=ts, pnl_pct=pnl, exit_reason=er))
                nc.append(pos)
                if pnl < 0 and self.cooldown_bars > 0: cu[pos.asset] = bi + self.cooldown_bars
        for p in nc: op.remove(p)
        if len(op) < self.max_open:
            for asset, sd in asset_signals.items():
                if len(op) >= self.max_open: break
                if ts not in sd.index: continue
                ap = [p for p in op if p.asset == asset]
                if len(ap) >= self.max_per_asset: continue
                if asset in cu and bi < cu[asset]: continue
                row = sd.loc[ts]
                if row["signal"] == 0: continue
                sl, tp, cl = row["sl"], row["tp"], row["close"]
                if pd.isna(sl) or pd.isna(tp) or sl <= 0 or tp <= 0: continue
                side = "long" if row["signal"] == 1 else "short"
                rd = (cl - sl) if side == "long" else (sl - cl)
                rwd = (tp - cl) if side == "long" else (cl - tp)
                if rd <= 0 or rwd <= 0 or rwd/rd < self.min_rr: continue
                entry = cl * (1 + self.slippage_pct) if side == "long" else cl * (1 - self.slippage_pct)
                op.append(vpb.OpenPosition(asset=asset, side=side, strategy=str(row["strategy"]),
                    regime=str(row["regime"]), entry_price=entry, stop_loss=sl, take_profit=tp,
                    lots=0, entry_time=ts, risk_pct_effective=self.risk_pct, initial_risk_dist=rd))
        peak = max(peak, equity)
        ep.append((ts, equity))
    for pos in op:
        sd = asset_signals.get(pos.asset)
        if sd is not None and len(sd) > 0:
            lp = float(sd["close"].iloc[-1])
            pnl = self._compute_pnl(pos, lp)
            equity *= (1 + pnl / 100)
            ct.append(vpb.ClosedTrade(asset=pos.asset, side=pos.side, strategy=pos.strategy,
                regime=pos.regime, entry_price=pos.entry_price, exit_price=lp, lots=pos.lots,
                entry_time=pos.entry_time, exit_time=sd.index[-1], pnl_pct=pnl, exit_reason="end_of_data"))
            ep.append((sd.index[-1], equity))
    eq = pd.Series([e for _, e in ep], index=pd.DatetimeIndex([t for t, _ in ep]))
    return vpb.PortfolioBacktestResult(closed_trades=ct, equity_curve=eq,
        initial_equity=initial_equity, final_equity=equity, elapsed_secs=0,
        asset_breakdown=self._breakdown_by(ct, "asset"),
        strategy_breakdown=self._breakdown_by(ct, "strategy"),
        regime_breakdown=self._breakdown_by(ct, "regime"))

if __name__ == "__main__":
    vpb.PortfolioBacktester._simulate_portfolio = fixed_risk_simulate
    svc = MarketDataService(provider='yfinance')
    regime_svc = RegimeService()
    regime_svc.load(Path('models') / 'regime_model.pkl')
    param_store = ParameterStore()

    results = []

    # Test 1: Different SL multipliers with tp7, 5 assets
    assets5 = ['XAGUSD', 'XAUUSD', 'ETHUSD', 'USOIL', 'BTCUSD']
    for sl_mult in [1.0, 1.25, 1.5, 2.0]:
        def make_d(slm):
            def d(self):
                return {"reference_hours": 8, "breakout_buffer": 0.0,
                        "sl_atr_mult": slm, "tp_rr": 7.0,
                        "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2}
            return d
        sb.SessionBreakoutStrategy.default_params = make_d(sl_mult)
        market_data = {}
        for asset in assets5:
            df = svc.get(asset, timeframe='1h', start='2024-06-01')
            df = add_regime_features(df)
            df = add_alpha_features(df)
            market_data[asset] = df

        for risk in [0.75, 0.90, 1.0]:
            bt = PortfolioBacktester(
                market_data=market_data, regime_service=regime_svc,
                parameter_store=param_store, max_open_trades=12,
                risk_pct=risk, trailing_stop_atr=0.0, trailing_activate_rr=0,
                cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
            result = bt.run(equity=10000)
            trades = result.trades_df
            if len(trades) == 0 or 'pnl_pct' not in trades.columns: continue
            wr = (trades.pnl_pct > 0).mean() * 100
            eq = result.equity_curve
            dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
            label = "sl%.1f/tp7/%.2f%%/5a" % (sl_mult, risk)
            results.append((label, result.total_return_pct, result.n_trades, wr, dd))

    # Test 2: Add 6th/7th asset with best SL config
    def make_tp7(self):
        return {"reference_hours": 8, "breakout_buffer": 0.0,
                "sl_atr_mult": 1.5, "tp_rr": 7.0,
                "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2}
    sb.SessionBreakoutStrategy.default_params = make_tp7

    extra_assets = ['GBPJPY', 'EURJPY', 'EURUSD']
    for extra in extra_assets:
        assets6 = assets5 + [extra]
        market_data = {}
        for asset in assets6:
            try:
                df = svc.get(asset, timeframe='1h', start='2024-06-01')
                df = add_regime_features(df)
                df = add_alpha_features(df)
                market_data[asset] = df
            except: pass

        for risk in [0.80, 1.0]:
            bt = PortfolioBacktester(
                market_data=market_data, regime_service=regime_svc,
                parameter_store=param_store, max_open_trades=12,
                risk_pct=risk, trailing_stop_atr=0.0, trailing_activate_rr=0,
                cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
            result = bt.run(equity=10000)
            trades = result.trades_df
            if len(trades) == 0 or 'pnl_pct' not in trades.columns: continue
            wr = (trades.pnl_pct > 0).mean() * 100
            eq = result.equity_curve
            dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
            label = "tp7/%.2f%%/+%s" % (risk, extra)
            results.append((label, result.total_return_pct, result.n_trades, wr, dd))

    print("{:<28s} {:>10s} {:>6s} {:>5s} {:>7s}".format('Config', 'Return', 'Trades', 'WR', 'MaxDD'))
    print('-' * 60)
    results.sort(key=lambda x: -x[1] if x[4] > -35 else -99999)
    for label, ret, nt, wr, dd in results:
        flag = " **" if dd > -30 else ""
        print("{:<28s} {:>+9.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%{}".format(label, ret, nt, wr, dd, flag))
