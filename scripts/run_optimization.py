"""
scripts/run_optimization.py

Run walk-forward optimization and save best parameters to the parameter store.

Usage:
    python scripts/run_optimization.py --asset EURUSD --strategy trend_breakout
    python scripts/run_optimization.py --asset USDJPY --strategy mean_reversion --method optuna
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from utils.env_loader import load_env
from utils.logger import configure_logging, get_logger

load_env()
configure_logging()
log = get_logger("run_optimization")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset",      default="EURUSD")
    parser.add_argument("--strategy",   default="trend_breakout")
    parser.add_argument("--timeframe",  default="1h")
    parser.add_argument("--start",      default="2022-01-01")
    parser.add_argument("--end",        default=None)
    parser.add_argument("--provider",   default="yfinance")
    parser.add_argument("--method",     choices=["grid","random","optuna"], default="optuna")
    parser.add_argument("--trials",     type=int, default=150)
    parser.add_argument("--folds",      type=int, default=5)
    parser.add_argument("--objective",  default="calmar")
    parser.add_argument("--min_trades", type=int, default=15)
    parser.add_argument("--regime",     default="all", help="Regime to store params under")
    args = parser.parse_args()

    from data.market_data import MarketDataService
    from features.regime_features import add_regime_features
    from features.alpha_features import add_alpha_features
    from strategies.strategy_factory import get_strategy
    from validation.walk_forward import WalkForwardValidator
    from validation.robustness import RobustnessChecker
    from optimization.parameter_store import ParameterStore

    log.info(
        "Optimization: %s / %s / %s | method=%s trials=%d folds=%d",
        args.asset, args.strategy, args.timeframe, args.method, args.trials, args.folds,
    )

    svc = MarketDataService(provider=args.provider)
    df  = svc.get(args.asset, timeframe=args.timeframe, start=args.start, end=args.end)
    df  = add_regime_features(df)
    df  = add_alpha_features(df)

    strategy = get_strategy(args.strategy)
    store    = ParameterStore()
    wf       = WalkForwardValidator(n_folds=args.folds, purge_gap_bars=10)

    for side in ["long", "short"]:
        log.info("--- Optimizing %s side ---", side)
        result = wf.run(
            strategy   = strategy,
            df         = df,
            side       = side,
            objective  = args.objective,
            opt_method = args.method,
            opt_trials = args.trials,
            min_trades = args.min_trades,
        )

        if result.folds:
            best_params = result.folds[-1].best_params

            # Robustness check
            rc = RobustnessChecker(strategy, df)
            rb = rc.check(best_params, side=side)
            log.info("Robustness: is_robust=%s, sensitivity=%s",
                     rb["is_robust"], rb["sensitivity_scores"])

            store.set(
                asset      = args.asset,
                regime     = args.regime,
                strategy   = args.strategy,
                side       = side,
                params     = best_params,
                score      = result.avg_test_score,
                objective  = args.objective,
                n_trades   = result.overall_metrics.get("n_trades", 0),
                notes      = f"is_robust={rb['is_robust']}",
            )
            log.info("Saved params for %s/%s: %s", side, args.strategy, best_params)

    store.export_csv()
    log.info("Parameter store exported to CSV")
    log.info("Done — parameters saved to %s", store.path)


if __name__ == "__main__":
    main()
