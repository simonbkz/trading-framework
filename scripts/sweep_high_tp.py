"""Explore high TP ratios (6-12R) with low risk for DD control."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    """Fixed risk per trade, no DD scaling."""
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
                    pnl_pct=pnl_pct, exit_reason=exit_reason,
                ))
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
                    initial_risk_dist=risk_dist,
                ))
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
                pnl_pct=pnl_pct, exit_reason="end_of_data",
            ))
            equity_points.append((sig_df.index[-1], equity))

    eq_series = pd.Series([e for _, e in equity_points],
                          index=pd.DatetimeIndex([t for t, _ in equity_points]))
    return vpb.PortfolioBacktestResult(
        closed_trades=closed_trades, equity_curve=eq_series,
        initial_equity=initial_equity, final_equity=equity, elapsed_secs=0,
        asset_breakdown=self._breakdown_by(closed_trades, "asset"),
        strategy_breakdown=self._breakdown_by(closed_trades, "strategy"),
        regime_breakdown=self._breakdown_by(closed_trades, "regime"),
    )


if __name__ == "__main__":
    vpb.PortfolioBacktester._simulate_portfolio = fixed_risk_simulate

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

    print("=== HIGH TP RATIO SWEEP (5 best assets) ===")
    header = "{:<30s} {:>8s} {:>6s} {:>5s} {:>7s}".format('Config', 'Return', 'Trades', 'WR', 'MaxDD')
    print(header)
    print('-' * 62)

    for tp_rr in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]:
        def make_defaults(tprr):
            def d(self):
                return {
                    "reference_hours": 8, "breakout_buffer": 0.0,
                    "sl_atr_mult": 1.5, "tp_rr": tprr,
                    "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2,
                }
            return d
        sb.SessionBreakoutStrategy.default_params = make_defaults(tp_rr)

        for risk in [0.5, 0.75, 1.0, 1.5, 2.0]:
            for mpa, mt in [(1, 8), (2, 12), (3, 15)]:
                bt = PortfolioBacktester(
                    market_data=market_data, regime_service=regime_svc,
                    parameter_store=param_store, max_open_trades=mt,
                    risk_pct=risk, trailing_stop_atr=0.0,
                    trailing_activate_rr=0, cooldown_bars=0,
                    min_rr=1.5, max_positions_per_asset=mpa,
                )
                result = bt.run(equity=10000)
                trades = result.trades_df
                if len(trades) == 0 or 'pnl_pct' not in trades.columns:
                    continue
                wr = (trades.pnl_pct > 0).mean() * 100
                eq = result.equity_curve
                dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()

                # Only print promising configs
                ret = result.total_return_pct
                if ret > 500 or (ret > 100 and dd > -35):
                    label = "tp%.0f/%.2f%%/pyr%d/%dt" % (tp_rr, risk, mpa, mt)
                    print("{:<30s} {:>+8.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%".format(
                        label, ret, result.n_trades, wr, dd))

    # Focused: find configs near 3200% with best DD
    print("\n=== BEST CANDIDATES (near 3200% target) ===")
    for tp_rr in [6.0, 7.0, 8.0]:
        def make_defaults2(tprr):
            def d(self):
                return {
                    "reference_hours": 8, "breakout_buffer": 0.0,
                    "sl_atr_mult": 1.5, "tp_rr": tprr,
                    "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2,
                }
            return d
        sb.SessionBreakoutStrategy.default_params = make_defaults2(tp_rr)

        for risk in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            for mpa, mt in [(2, 12), (3, 15)]:
                bt = PortfolioBacktester(
                    market_data=market_data, regime_service=regime_svc,
                    parameter_store=param_store, max_open_trades=mt,
                    risk_pct=risk, trailing_stop_atr=0.0,
                    trailing_activate_rr=0, cooldown_bars=0,
                    min_rr=1.5, max_positions_per_asset=mpa,
                )
                result = bt.run(equity=10000)
                trades = result.trades_df
                if len(trades) == 0 or 'pnl_pct' not in trades.columns:
                    continue
                wr = (trades.pnl_pct > 0).mean() * 100
                eq = result.equity_curve
                dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
                ret = result.total_return_pct

                if ret > 2000 or (ret > 1000 and dd > -40):
                    label = "tp%.0f/%.1f%%/pyr%d/%dt" % (tp_rr, risk, mpa, mt)
                    print("{:<30s} {:>+8.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%".format(
                        label, ret, result.n_trades, wr, dd))
