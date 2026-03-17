"""Sweep the new best asset combo (EURJPY instead of USOIL) with risk levels."""
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
from dataclasses import dataclass


@dataclass
class EnhancedPosition:
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
    partial_tp_hit: bool = False
    entry_bar: int = 0


def enhanced_simulate(self, asset_signals, initial_equity,
                       partial_tp_r=0.0, partial_tp_frac=0.5):
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
            high, low = float(row["high"]), float(row["low"])
            exit_price, exit_reason = None, ""

            if pos.side == "long":
                pos.highest_price = max(pos.highest_price, high)
                if low <= pos.stop_loss:
                    exit_price = pos.stop_loss * (1 - self.slippage_pct)
                    exit_reason = "sl"
                elif high >= pos.take_profit:
                    exit_price = pos.take_profit * (1 - self.slippage_pct)
                    exit_reason = "tp"
                elif partial_tp_r > 0 and not pos.partial_tp_hit:
                    pp = pos.entry_price + partial_tp_r * pos.initial_risk_dist
                    if high >= pp:
                        pos.partial_tp_hit = True
                        pos.stop_loss = pos.entry_price + 0.1 * pos.initial_risk_dist
                        partial_pnl = pos.risk_pct_effective * partial_tp_frac * partial_tp_r
                        equity *= (1 + partial_pnl / 100)
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
                    pp = pos.entry_price - partial_tp_r * pos.initial_risk_dist
                    if low <= pp:
                        pos.partial_tp_hit = True
                        pos.stop_loss = pos.entry_price - 0.1 * pos.initial_risk_dist
                        partial_pnl = pos.risk_pct_effective * partial_tp_frac * partial_tp_r
                        equity *= (1 + partial_pnl / 100)
                        pos.risk_pct_effective *= (1 - partial_tp_frac)

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
                open_positions.append(EnhancedPosition(
                    asset=asset, side=side, strategy=str(row["strategy"]),
                    regime=str(row["regime"]), entry_price=entry,
                    stop_loss=sl_val, take_profit=tp_val, lots=0,
                    entry_time=ts, risk_pct_effective=self.risk_pct,
                    initial_risk_dist=risk_dist, entry_bar=bar_i))
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
    svc = MarketDataService(provider='yfinance')
    regime_svc = RegimeService()
    regime_svc.load(Path('models') / 'regime_model.pkl')
    param_store = ParameterStore()

    def make_params(self):
        return {"reference_hours": 8, "breakout_buffer": 0.0,
                "sl_atr_mult": 1.5, "tp_rr": 7.0,
                "max_entry_hours": 8, "session_open_hour": 7, "vol_mult": 1.2}
    sb.SessionBreakoutStrategy.default_params = make_params

    # NEW BEST 5: Replace USOIL with EURJPY
    new_best = ['XAUUSD', 'BTCUSD', 'XAGUSD', 'ETHUSD', 'EURJPY']
    old_best = ['XAGUSD', 'XAUUSD', 'ETHUSD', 'USOIL', 'BTCUSD']

    print("Loading data...")
    all_data = {}
    for asset in set(new_best + old_best):
        df = svc.get(asset, timeframe='1h', start='2024-06-01')
        df = add_regime_features(df)
        df = add_alpha_features(df)
        all_data[asset] = df
        print(f"  {asset}: {len(df)} bars")

    results = []

    # ============================================================
    # Compare old vs new asset mix at various risk levels
    # ============================================================
    for asset_set, label_prefix in [(old_best, "OLD"), (new_best, "NEW")]:
        md = {a: all_data[a] for a in asset_set}

        for partial_tp_r in [0.0, 3.0, 3.5]:
            for risk in [1.25, 1.5, 1.75, 2.0]:

                def make_sim(ptr, ptf=0.5):
                    def sim(self, asset_signals, initial_equity):
                        return enhanced_simulate(self, asset_signals, initial_equity,
                                                 partial_tp_r=ptr, partial_tp_frac=ptf)
                    return sim

                vpb.PortfolioBacktester._simulate_portfolio = make_sim(partial_tp_r)

                bt = PortfolioBacktester(
                    market_data=md, regime_service=regime_svc,
                    parameter_store=param_store, max_open_trades=12,
                    risk_pct=risk, trailing_stop_atr=0.0, trailing_activate_rr=0,
                    cooldown_bars=0, min_rr=1.5, max_positions_per_asset=2)
                r = bt.run(equity=10000)
                trades = r.trades_df
                if len(trades) == 0:
                    continue
                wr = (trades.pnl_pct > 0).mean() * 100
                eq = r.equity_curve
                dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
                pt_label = f"pt{partial_tp_r:.0f}R" if partial_tp_r > 0 else "noPT"
                label = f"{label_prefix}/{pt_label}/{risk:.2f}%"
                results.append((label, r.total_return_pct, r.n_trades, wr, dd))

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 72)
    print("RESULTS: Sorted by return (DD < 50% prioritized)")
    print("=" * 72)
    results.sort(key=lambda x: x[1] if x[4] > -50 else -99999, reverse=True)
    print("{:<30s} {:>10s} {:>6s} {:>5s} {:>7s}".format('Config', 'Return', 'Trades', 'WR', 'MaxDD'))
    print('-' * 72)
    for label, ret, nt, wr, dd in results:
        flag = " ***" if ret > 4000 and dd > -50 else (" **" if dd > -50 and ret > 3000 else "")
        print("{:<30s} {:>+9.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%{}".format(label, ret, nt, wr, dd, flag))

    print("\n=== BEST: MAX RETURN WITH DD < 50% ===")
    under50 = [(l, r, n, w, d) for l, r, n, w, d in results if d > -50]
    under50.sort(key=lambda x: x[1], reverse=True)
    for label, ret, nt, wr, dd in under50[:10]:
        print("{:<30s} {:>+9.1f}% {:>5d}  {:>4.1f}% {:>+6.1f}%".format(label, ret, nt, wr, dd))
