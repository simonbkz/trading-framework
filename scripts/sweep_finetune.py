"""Fine-tune around tp6-7, risk 0.8-1.2%, pyr2-3."""
import sys, warnings, time
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
    equity_points, cooldown_until = [], {}
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
                ap = [p for p in open_positions if p.asset == asset]
                if len(ap) >= self.max_per_asset:
                    continue
                if asset in cooldown_until and bar_i < cooldown_until[asset]:
                    continue
                row = sig_df.loc[ts]
                if row["signal"] == 0:
                    continue
                sl, tp, cl = row["sl"], row["tp"], row["close"]
                if pd.isna(sl) or pd.isna(tp) or sl <= 0 or tp <= 0:
                    continue
                side = "long" if row["signal"] == 1 else "short"
                rd = (cl - sl) if side == "long" else (sl - cl)
                rwd = (tp - cl) if side == "long" else (cl - tp)
                if rd <= 0 or rwd <= 0 or rwd/rd < self.min_rr:
                    continue
                entry = cl * (1 + self.slippage_pct) if side == "long" else cl * (1 - self.slippage_pct)
                open_positions.append(vpb.OpenPosition(
                    asset=asset, side=side, strategy=str(row["strategy"]),
                    regime=str(row["regime"]), entry_price=entry,
                    stop_loss=sl, take_profit=tp, lots=0, entry_time=ts,
                    risk_pct_effective=self.risk_pct, initial_risk_dist=rd))
        peak_equity = max(peak_equity, equity)
        equity_points.append((ts, equity))
    for pos in open_positions:
        sig_df = asset_signals.get(pos.asset)
        if sig_df is not None and len(sig_df) > 0:
            lp = float(sig_df["close"].iloc[-1])
            pnl_pct = self._compute_pnl(pos, lp)
            equity *= (1 + pnl_pct / 100)
            closed_trades.append(vpb.ClosedTrade(
                asset=pos.asset, side=pos.side, strategy=pos.strategy,
                regime=pos.regime, entry_price=pos.entry_price,
                exit_price=lp, lots=pos.lots, entry_time=pos.entry_time,
                exit_time=sig_df.index[-1], pnl_pct=pnl_pct, exit_reason="end_of_data"))
            equity_points.append((sig_df.index[-1], equity))
    eq_s = pd.Series([e for _, e in equity_points], index=pd.DatetimeIndex([t for t, _ in equity_points]))
    return vpb.PortfolioBacktestResult(
        closed_trades=closed_trades, equity_curve=eq_s,
        initial_equity=initial_equity, final_equity=equity, elapsed_secs=0,
        asset_breakdown=self._breakdown_by(closed_trades, "asset"),
        strategy_breakdown=self._breakdown_by(closed_trades, "strategy"),
        regime_breakdown=self._breakdown_by(closed_trades, "regime"))

if __name__ == "__main__":
    vpb.PortfolioBacktester._simulate_portfolio = fixed_risk_simulate
    assets = ['XAGUSD', 'XAUUSD', 'ETHUSD', 'USOIL', 'BTCUSD']
    svc = MarketDataService(provider='yfinance')
    regime_svc = RegimeService()
    regime_svc.load(Path('models') / 'regime_model.pkl')
    param_store = ParameterStore()
    results = []

    for tp_rr in [6.0, 7.0]:
        def make_d(tprr):
            def d(self):
                return {"reference_hours": 8, "breakout_buffer": 0.0,
                        "sl_atr_mult": 1.5, "tp_rr": tprr,
                        "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2}
            return d
        sb.SessionBreakoutStrategy.default_params = make_d(tp_rr)
        market_data = {}
        for asset in assets:
            df = svc.get(asset, timeframe='1h', start='2024-06-01')
            df = add_regime_features(df)
            df = add_alpha_features(df)
            market_data[asset] = df

        for risk in [0.80, 0.90, 1.00, 1.10, 1.20, 1.30]:
            for mpa, mt in [(2, 10), (2, 12), (3, 12), (3, 15)]:
                bt = PortfolioBacktester(
                    market_data=market_data, regime_service=regime_svc,
                    parameter_store=param_store, max_open_trades=mt,
                    risk_pct=risk, trailing_stop_atr=0.0,
                    trailing_activate_rr=0, cooldown_bars=0,
                    min_rr=1.5, max_positions_per_asset=mpa)
                result = bt.run(equity=10000)
                trades = result.trades_df
                if len(trades) == 0 or 'pnl_pct' not in trades.columns:
                    continue
                wr = (trades.pnl_pct > 0).mean() * 100
                eq = result.equity_curve
                dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
                ret = result.total_return_pct
                label = "tp%.0f/%.2f%%/pyr%d/%dt" % (tp_rr, risk, mpa, mt)
                results.append((label, ret, result.n_trades, wr, dd))

    print("{:<25s} {:>10s} {:>6s} {:>5s} {:>7s}".format('Config', 'Return', 'Trades', 'WR', 'MaxDD'))
    print('-' * 58)
    results.sort(key=lambda x: -x[1] if x[4] > -35 else -99999)
    for label, ret, nt, wr, dd in results:
        flag = " <<<" if ret > 3200 and dd > -30 else (" **" if dd > -30 else "")
        print("{:<25s} {:>+9.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%{}".format(label, ret, nt, wr, dd, flag))
