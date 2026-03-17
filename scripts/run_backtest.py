"""
scripts/run_backtest.py

Run a full vectorized backtest with regime breakdown and performance report.

Usage:
    python scripts/run_backtest.py --asset EURUSD --strategy trend_breakout
    python scripts/run_backtest.py --asset XAUUSD --strategy volatility_breakout --side long
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from utils.env_loader import load_env
from utils.logger import configure_logging, get_logger

load_env()
configure_logging()
log = get_logger("run_backtest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset",     default="EURUSD")
    parser.add_argument("--strategy",  default="trend_breakout")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start",     default="2022-01-01")
    parser.add_argument("--end",       default=None)
    parser.add_argument("--provider",  default="yfinance")
    parser.add_argument("--side",      choices=["long","short","both"], default="both")
    parser.add_argument("--save",      action="store_true", help="Save equity curve plot")
    args = parser.parse_args()

    from data.market_data import MarketDataService
    from features.regime_features import add_regime_features
    from features.alpha_features import add_alpha_features
    from regimes.regime_service import RegimeService
    from strategies.strategy_factory import get_strategy
    from optimization.optimizer import Optimizer
    from validation.metrics import (
        compute_trade_metrics, compute_equity_metrics, compute_metrics_by_group
    )

    log.info("Backtest: %s / %s / %s", args.asset, args.strategy, args.timeframe)

    svc = MarketDataService(provider=args.provider)
    df  = svc.get(args.asset, timeframe=args.timeframe, start=args.start, end=args.end)
    df  = add_regime_features(df)
    df  = add_alpha_features(df)

    # Attach regime labels
    regime_svc = RegimeService()
    regime_svc.fit(df)
    regime_result = regime_svc.detect(df, asset=args.asset)
    df["regime"] = regime_result["smoothed_regime"]

    strategy = get_strategy(args.strategy)
    sides = ["long", "short"] if args.side == "both" else [args.side]

    all_trades = []
    for side in sides:
        opt = Optimizer(strategy, df)
        signals = strategy.generate_signals(df, side=side)
        # Attach regime to signals
        signals["regime"] = df["regime"]
        trades, equity = opt._simulate(signals, side)
        if not trades.empty:
            trades["side"] = side
            # Attach regime at entry time
            trades["regime"] = signals.loc[trades["entry_time"].values, "regime"].values
            all_trades.append(trades)

    if not all_trades:
        log.warning("No trades generated")
        return

    trades_df = pd.concat(all_trades, ignore_index=True)
    pnl = trades_df["pnl_pct"]
    equity_curve = (1 + pnl / 100).cumprod() * 100

    print("\n" + "="*60)
    print(f"BACKTEST: {args.asset} | {args.strategy} | {args.timeframe}")
    print("="*60)

    tm = compute_trade_metrics(trades_df)
    em = compute_equity_metrics(equity_curve)
    for k, v in {**tm, **em}.items():
        print(f"  {k:30s}: {v}")

    print("\nPERFORMANCE BY REGIME:")
    by_regime = compute_metrics_by_group(trades_df, "regime")
    print(by_regime[["n_trades","win_rate","profit_factor","total_return_pct","max_drawdown_pct"]].to_string())

    print("\nPERFORMANCE BY SIDE:")
    by_side = compute_metrics_by_group(trades_df, "side")
    print(by_side[["n_trades","win_rate","profit_factor","total_return_pct"]].to_string())

    if args.save:
        from utils.plotting import plot_equity_curve, plot_regime_timeline
        fig = plot_equity_curve(equity_curve, title=f"{args.asset} {args.strategy}")
        fig.savefig(f"backtest_{args.asset}_{args.strategy}.png", dpi=120)
        log.info("Equity curve saved")


if __name__ == "__main__":
    main()
