"""Test soft DD protection: gentle risk scaling to allow recovery."""
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

def soft_dd_simulate(self, asset_signals, initial_equity):
    """Soft DD protection: reduce to 70% at worst, boost 25% at highs."""
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

                # Soft DD protection
                dd_pct = (peak - equity) / peak if peak > 0 else 0
                if dd_pct > 0.20:
                    dd_mult = 0.70
                elif dd_pct > 0.12:
                    dd_mult = 0.85
                elif dd_pct < 0.03:
                    dd_mult = 1.25  # boost at equity highs
                else:
                    dd_mult = 1.0
                effective_risk = self.risk_pct * dd_mult

                entry = cl * (1 + self.slippage_pct) if side == "long" else cl * (1 - self.slippage_pct)
                op.append(vpb.OpenPosition(asset=asset, side=side, strategy=str(row["strategy"]),
                    regime=str(row["regime"]), entry_price=entry, stop_loss=sl, take_profit=tp,
                    lots=0, entry_time=ts, risk_pct_effective=effective_risk, initial_risk_dist=rd))
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
    vpb.PortfolioBacktester._simulate_portfolio = soft_dd_simulate

    def make_tp7(self):
        return {"reference_hours": 8, "breakout_buffer": 0.0,
                "sl_atr_mult": 1.5, "tp_rr": 7.0,
                "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2}
    sb.SessionBreakoutStrategy.default_params = make_tp7

    assets = ['XAGUSD', 'XAUUSD', 'ETHUSD', 'USOIL', 'BTCUSD']
    svc = MarketDataService(provider='yfinance')
    market_data = {}
    for asset in assets:
        df = svc.get(asset, timeframe='1h', start='2024-06-01')
        df = add_regime_features(df)
        df = add_alpha_features(df)
        market_data[asset] = df

    regime_svc = RegimeService()
    regime_svc.load(Path('models') / 'regime_model.pkl')
    param_store = ParameterStore()

    print("{:<28s} {:>10s} {:>6s} {:>5s} {:>7s}".format('Config', 'Return', 'Trades', 'WR', 'MaxDD'))
    print('-' * 60)

    for risk in [0.80, 0.90, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]:
        for mpa, mt in [(2, 12), (3, 15)]:
            bt = PortfolioBacktester(
                market_data=market_data, regime_service=regime_svc,
                parameter_store=param_store, max_open_trades=mt,
                risk_pct=risk, trailing_stop_atr=0.0, trailing_activate_rr=0,
                cooldown_bars=0, min_rr=1.5, max_positions_per_asset=mpa)
            result = bt.run(equity=10000)
            trades = result.trades_df
            if len(trades) == 0 or 'pnl_pct' not in trades.columns: continue
            wr = (trades.pnl_pct > 0).mean() * 100
            eq = result.equity_curve
            dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
            label = "soft/%.2f%%/pyr%d/%dt" % (risk, mpa, mt)
            flag = " <<<" if result.total_return_pct > 3200 and dd > -30 else (" **" if dd > -30 else "")
            print("{:<28s} {:>+9.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%{}".format(
                label, result.total_return_pct, result.n_trades, wr, dd, flag))

    # Also test with EURJPY added
    print("\n--- With EURJPY added ---")
    try:
        df = svc.get('EURJPY', timeframe='1h', start='2024-06-01')
        df = add_regime_features(df)
        df = add_alpha_features(df)
        market_data['EURJPY'] = df
    except: pass

    for risk in [0.80, 1.0, 1.2, 1.5]:
        for mpa, mt in [(2, 12), (3, 15)]:
            bt = PortfolioBacktester(
                market_data=market_data, regime_service=regime_svc,
                parameter_store=param_store, max_open_trades=mt,
                risk_pct=risk, trailing_stop_atr=0.0, trailing_activate_rr=0,
                cooldown_bars=0, min_rr=1.5, max_positions_per_asset=mpa)
            result = bt.run(equity=10000)
            trades = result.trades_df
            if len(trades) == 0 or 'pnl_pct' not in trades.columns: continue
            wr = (trades.pnl_pct > 0).mean() * 100
            eq = result.equity_curve
            dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
            label = "+JPY/%.2f%%/pyr%d/%dt" % (risk, mpa, mt)
            flag = " <<<" if result.total_return_pct > 3200 and dd > -30 else (" **" if dd > -30 else "")
            print("{:<28s} {:>+9.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%{}".format(
                label, result.total_return_pct, result.n_trades, wr, dd, flag))
