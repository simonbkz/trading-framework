"""
main.py — entry point for the trading framework.

Modes:
    backtest  — run a backtest for a strategy/asset combo
    optimize  — optimise strategy parameters
    signal    — generate a live signal and write to JSON
    monitor   — print current performance metrics
    train     — fit and save the regime model

Usage:
    python main.py --mode backtest --asset EURUSD --strategy trend_breakout
    python main.py --mode optimize --asset EURUSD --strategy trend_breakout
    python main.py --mode signal   --assets EURUSD,USDJPY,XAUUSD
    python main.py --mode train    --assets EURUSD,USDJPY
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from utils.env_loader import load_env
from utils.logger import configure_logging, get_logger

load_env()
configure_logging()
log = get_logger("main")


def run_backtest(args):
    from data.market_data import MarketDataService
    from features.regime_features import add_regime_features
    from features.alpha_features import add_alpha_features
    from strategies.strategy_factory import get_strategy
    from optimization.optimizer import Optimizer
    from validation.metrics import compute_trade_metrics, compute_equity_metrics

    log.info("BACKTEST: %s | %s | %s -> %s", args.asset, args.strategy, args.start, args.end)

    svc = MarketDataService(provider=args.provider)
    df  = svc.get(args.asset, timeframe=args.timeframe, start=args.start, end=args.end)
    df  = add_regime_features(df)
    df  = add_alpha_features(df)

    strategy = get_strategy(args.strategy)
    opt = Optimizer(strategy, df, method=args.method, n_trials=args.trials,
                    min_trades=args.min_trades)

    for side in (["long"] if args.side == "long" else
                 ["short"] if args.side == "short" else
                 ["long", "short"]):
        log.info("Running %s side ...", side)
        result = opt.run(side=side, objective=args.objective)
        log.info("Best params: %s | score=%.4f", result.best_params, result.best_score)

        strategy.set_params(result.best_params, side=side)
        signals = strategy.generate_signals(df, side=side)
        trades, equity = opt._simulate(signals, side)
        tm = compute_trade_metrics(trades)
        em = compute_equity_metrics(equity)
        log.info("TRADE METRICS:  %s", tm)
        log.info("EQUITY METRICS: %s", em)


def run_optimize(args):
    from data.market_data import MarketDataService
    from features.regime_features import add_regime_features
    from features.alpha_features import add_alpha_features
    from strategies.strategy_factory import get_strategy
    from optimization.optimizer import Optimizer
    from optimization.parameter_store import ParameterStore
    from validation.walk_forward import WalkForwardValidator

    log.info("OPTIMIZE: %s | %s", args.asset, args.strategy)

    svc = MarketDataService(provider=args.provider)
    df  = svc.get(args.asset, timeframe=args.timeframe, start=args.start, end=args.end)
    df  = add_regime_features(df)
    df  = add_alpha_features(df)

    strategy  = get_strategy(args.strategy)
    store     = ParameterStore()

    for side in ["long", "short"]:
        log.info("Walk-forward optimizing %s ...", side)
        wf = WalkForwardValidator(n_folds=args.wf_folds, purge_gap_bars=10)
        result = wf.run(
            strategy   = strategy,
            df         = df,
            side       = side,
            objective  = args.objective,
            opt_method = args.method,
            opt_trials = args.trials,
            min_trades = args.min_trades,
        )
        # Save best params from the most recent fold
        best_fold = result.folds[-1]
        store.set(
            asset     = args.asset,
            regime    = "all",
            strategy  = args.strategy,
            side      = side,
            params    = best_fold.best_params,
            score     = result.avg_test_score,
            objective = args.objective,
            n_trades  = result.overall_metrics.get("n_trades", 0),
        )
        log.info(result.summary())


def run_signal(args):
    from data.market_data import MarketDataService
    from features.regime_features import add_regime_features
    from features.alpha_features import add_alpha_features
    from features.macro_features import add_all_macro_features
    from regimes.regime_service import RegimeService
    from filters.tradability_filter import TradabilityFilter
    from optimization.parameter_store import ParameterStore
    from selection.regime_strategy_router import RegimeStrategyRouter
    from execution.signal_engine import SignalEngine
    from portfolio.risk_engine import RiskEngine
    from data.sentiment_scraper import SentimentScraper

    assets = [a.strip() for a in args.assets.split(",")]
    log.info("SIGNAL GENERATION: %s", assets)

    svc = MarketDataService(provider=args.provider)
    market_data = {}
    for asset in assets:
        try:
            df = svc.get(asset, timeframe=args.timeframe, start=args.start)
            df = add_regime_features(df)
            df = add_alpha_features(df)
            market_data[asset] = df
        except Exception as exc:
            log.error("Failed to load %s: %s", asset, exc)

    if not market_data:
        log.error("No data loaded — aborting")
        return

    # Add macro features (VIX, DXY, yields, sentiment proxies)
    log.info("Adding macro features...")
    for asset in list(market_data.keys()):
        try:
            market_data[asset] = add_all_macro_features(
                market_data[asset], asset=asset, all_data=market_data, start=args.start,
            )
        except Exception as exc:
            log.warning("Macro features failed for %s: %s", asset, exc)

    # Scrape live news sentiment
    log.info("Scraping live sentiment...")
    scraper = SentimentScraper()
    portfolio_sentiment = scraper.get_portfolio_sentiment(assets)
    log.info(
        "Portfolio sentiment: %.2f (%s), risk_mult=%.2f",
        portfolio_sentiment["overall"],
        portfolio_sentiment["regime_suggestion"],
        portfolio_sentiment["risk_multiplier"],
    )
    for asset, sent in portfolio_sentiment.get("by_asset", {}).items():
        log.info(
            "  %s: score=%.2f conf=%.2f (%s) %s",
            asset, sent["score"], sent["confidence"],
            sent.get("regime_hint", "?"),
            sent.get("headlines", [""])[0][:60] if sent.get("headlines") else "",
        )

    # Use saved model if available, otherwise fit on primary asset
    regime_svc = RegimeService()
    regime_model_path = Path("models") / "regime_model.pkl"
    if regime_model_path.exists():
        regime_svc.load(regime_model_path)
        log.info("Loaded saved regime model from %s", regime_model_path)
    else:
        primary = list(market_data.values())[0]
        regime_svc.fit(primary)
        log.info("No saved model found -- fit regime model on %s", assets[0])

    tf_filter  = TradabilityFilter()
    param_store = ParameterStore()

    router = RegimeStrategyRouter(
        regime_service     = regime_svc,
        parameter_store    = param_store,
        asset_universe     = assets,
        tradability_filter = tf_filter,
    )

    proposals = router.route(market_data)

    risk_engine = RiskEngine()
    engine = SignalEngine(risk_engine=risk_engine, paper_mode=True)

    # Apply sentiment-based risk multiplier
    sentiment_equity = args.equity * portfolio_sentiment["risk_multiplier"]
    signals = engine.process(
        proposals, equity=sentiment_equity,
        regime_confidence=max(0.3, portfolio_sentiment["overall"] + 0.5),
    )

    if signals:
        engine.write_all(signals)
        log.info(
            "Wrote %d signal(s): %s",
            len(signals),
            ["%s %s %s" % (s.asset, s.side.upper(), s.strategy) for s in signals],
        )
    else:
        engine.write_no_signal()
        log.info("No signal generated")


def run_train(args):
    from data.market_data import MarketDataService
    from features.regime_features import add_regime_features
    from regimes.regime_service import RegimeService

    assets = [a.strip() for a in args.assets.split(",")]
    log.info("TRAINING regime model on: %s", assets)

    svc = MarketDataService(provider=args.provider)
    dfs = []
    for asset in assets:
        try:
            df = svc.get(asset, timeframe=args.timeframe, start=args.start, end=args.end)
            df = add_regime_features(df)
            dfs.append(df)
        except Exception as exc:
            log.error("Failed to load %s: %s", asset, exc)

    if not dfs:
        log.error("No data loaded — aborting")
        return

    import pandas as pd
    combined = pd.concat(dfs).sort_index()

    regime_svc = RegimeService(
        use_ensemble = True,
        hmm_states   = 4,
    )
    regime_svc.fit(combined)
    regime_svc.save(Path("models") / "regime_model.pkl")
    log.info("Regime model saved to models/regime_model.pkl")


def run_monitor(args):
    """Report model and strategy performance metrics."""
    from data.market_data import MarketDataService
    from features.regime_features import add_regime_features
    from features.alpha_features import add_alpha_features
    from regimes.regime_service import RegimeService
    from strategies.strategy_factory import get_strategy, list_strategies
    from optimization.optimizer import Optimizer
    from validation.metrics import compute_trade_metrics, compute_equity_metrics
    from config.regimes import ALL_REGIMES
    import pandas as pd
    from pathlib import Path

    assets = [a.strip() for a in args.assets.split(",")]
    log.info("PERFORMANCE REPORT for: %s", assets)

    svc = MarketDataService(provider=args.provider)

    # --- 1. Regime model accuracy report ---
    regime_model_path = Path("models") / "regime_model.pkl"
    if regime_model_path.exists():
        log.info("--- REGIME MODEL REPORT ---")
        regime_svc = RegimeService()
        regime_svc.load(regime_model_path)

        for asset in assets:
            try:
                df = svc.get(asset, timeframe=args.timeframe, start=args.start, end=args.end)
                df = add_regime_features(df)
                result = regime_svc.detect(df, asset=asset)

                regime_counts = result["smoothed_regime"].value_counts()
                tradable_pct = result["is_tradable"].mean() * 100
                avg_conf = result["confidence"].mean()

                log.info(
                    "%s: avg_confidence=%.2f tradable=%.1f%% regime_distribution:",
                    asset, avg_conf, tradable_pct,
                )
                for regime, count in regime_counts.items():
                    pct = count / len(result) * 100
                    log.info("  %-20s %5d bars (%5.1f%%)", regime, count, pct)
            except Exception as exc:
                log.error("Regime report failed for %s: %s", asset, exc)
    else:
        log.warning("No trained model found at %s. Run --mode train first.", regime_model_path)

    # --- 2. Strategy backtest performance report ---
    log.info("--- STRATEGY PERFORMANCE REPORT ---")
    for asset in assets:
        try:
            df = svc.get(asset, timeframe=args.timeframe, start=args.start, end=args.end)
            df = add_regime_features(df)
            df = add_alpha_features(df)
        except Exception as exc:
            log.error("Failed to load %s: %s", asset, exc)
            continue

        for strat_name in list_strategies():
            strategy = get_strategy(strat_name)
            opt = Optimizer(strategy, df, method="random", n_trials=30, min_trades=10)

            for side in ["long", "short"]:
                result = opt.run(side=side, objective=args.objective)
                if result.best_score <= -10:
                    continue  # no viable trades

                strategy.set_params(result.best_params, side=side)
                signals = strategy.generate_signals(df, side=side)
                trades, equity = opt._simulate(signals, side)
                tm = compute_trade_metrics(trades)

                # Only report strategies with enough trades
                if tm["n_trades"] < 10:
                    continue

                log.info(
                    "%s | %s | %s: trades=%d WR=%.1f%% PF=%.2f Sharpe=%.2f MaxDD=%.1f%% Return=%.1f%%",
                    asset, strat_name, side.upper(),
                    tm["n_trades"],
                    tm["win_rate"] * 100,
                    tm["profit_factor"],
                    compute_equity_metrics(equity).get("sharpe_ratio", 0),
                    tm["max_drawdown_pct"],
                    tm["total_return_pct"],
                )

    log.info("--- REPORT COMPLETE ---")


def run_portfolio_bt(args):
    """Full multi-asset portfolio backtest using the live pipeline."""
    from data.market_data import MarketDataService
    from features.regime_features import add_regime_features
    from features.alpha_features import add_alpha_features
    from features.macro_features import add_all_macro_features
    from regimes.regime_service import RegimeService
    from optimization.parameter_store import ParameterStore
    from validation.portfolio_backtest import PortfolioBacktester

    assets = [a.strip() for a in args.assets.split(",")]
    log.info("PORTFOLIO BACKTEST: %s | %s -> %s", assets, args.start, args.end or "now")

    svc = MarketDataService(provider=args.provider)
    market_data = {}
    for asset in assets:
        try:
            df = svc.get(asset, timeframe=args.timeframe, start=args.start, end=args.end)
            df = add_regime_features(df)
            df = add_alpha_features(df)
            market_data[asset] = df
            log.info("Loaded %s: %d bars", asset, len(df))
        except Exception as exc:
            log.error("Failed to load %s: %s", asset, exc)

    if not market_data:
        log.error("No data loaded -- aborting")
        return

    # Add macro features (VIX, DXY, yields, sentiment, events)
    log.info("Adding macro features...")
    for asset in list(market_data.keys()):
        try:
            market_data[asset] = add_all_macro_features(
                market_data[asset], asset=asset, all_data=market_data, start=args.start,
            )
        except Exception as exc:
            log.warning("Macro features failed for %s: %s", asset, exc)

    # Load or fit regime model
    regime_svc = RegimeService()
    regime_model_path = Path("models") / "regime_model.pkl"
    if regime_model_path.exists():
        regime_svc.load(regime_model_path)
        log.info("Loaded saved regime model from %s", regime_model_path)
    else:
        primary = list(market_data.values())[0]
        regime_svc.fit(primary)
        log.info("Fit regime model on %s (%d bars)", assets[0], len(primary))

    param_store = ParameterStore()

    bt = PortfolioBacktester(
        market_data=market_data,
        regime_service=regime_svc,
        parameter_store=param_store,
        max_open_trades=getattr(args, 'max_trades', 12),
        risk_pct=getattr(args, 'risk_pct', 2.0),
        warmup_bars=200,
        cooldown_bars=getattr(args, 'cooldown', 2),
        min_rr=getattr(args, 'min_rr', 1.2),
        max_positions_per_asset=getattr(args, 'max_per_asset', 3),
    )

    result = bt.run(equity=args.equity)

    # Save results
    trades_path = Path("signals") / "portfolio_bt_trades.csv"
    if result.n_trades > 0:
        result.trades_df.to_csv(trades_path, index=False)
        log.info("Trade log saved to %s", trades_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Regime-Aware Trading Framework")
    p.add_argument("--mode",      choices=["backtest","optimize","signal","train","monitor","portfolio_bt"],
                   default="signal")
    p.add_argument("--asset",     default="EURUSD")
    p.add_argument("--assets",    default="XAUUSD,BTCUSD,XAGUSD,ETHUSD,EURJPY")
    p.add_argument("--strategy",  default="trend_breakout")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--start",     default="2024-06-01")
    p.add_argument("--end",       default=None)
    p.add_argument("--provider",  default="yfinance")
    p.add_argument("--side",      choices=["long","short","both"], default="both")
    p.add_argument("--objective", default="calmar")
    p.add_argument("--method",    choices=["grid","random","optuna"], default="optuna")
    p.add_argument("--trials",    type=int, default=100)
    p.add_argument("--min_trades",type=int, default=20)
    p.add_argument("--wf_folds",  type=int, default=5)
    p.add_argument("--equity",    type=float, default=10000.0)
    p.add_argument("--max_trades",type=int, default=12)
    p.add_argument("--risk_pct",  type=float, default=2.0)
    p.add_argument("--cooldown",  type=int, default=0)
    p.add_argument("--min_rr",    type=float, default=1.5)
    p.add_argument("--max_per_asset", type=int, default=2)
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    mode_map = {
        "backtest":     run_backtest,
        "optimize":     run_optimize,
        "signal":       run_signal,
        "train":        run_train,
        "monitor":      run_monitor,
        "portfolio_bt": run_portfolio_bt,
    }

    fn = mode_map.get(args.mode)
    if fn is None:
        log.error("Mode '%s' not implemented yet", args.mode)
        sys.exit(1)

    fn(args)


if __name__ == "__main__":
    main()
