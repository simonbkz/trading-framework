"""
scripts/generate_signal.py

Standalone script for live signal generation.
Can be scheduled via cron / Task Scheduler to run periodically.

Usage:
    python scripts/generate_signal.py --assets EURUSD,USDJPY,XAUUSD --equity 10000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from utils.env_loader import load_env
from utils.logger import configure_logging, get_logger

load_env()
configure_logging()
log = get_logger("generate_signal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets",    default="EURUSD,USDJPY,XAUUSD,BTCUSD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start",     default="2022-01-01")
    parser.add_argument("--provider",  default="yfinance")
    parser.add_argument("--equity",    type=float, default=10000.0)
    parser.add_argument("--news",      action="store_true", help="Block on news events")
    args = parser.parse_args()

    from data.market_data import MarketDataService
    from features.regime_features import add_regime_features
    from features.alpha_features import add_alpha_features
    from regimes.regime_service import RegimeService
    from filters.tradability_filter import TradabilityFilter
    from optimization.parameter_store import ParameterStore
    from selection.regime_strategy_router import RegimeStrategyRouter
    from execution.signal_engine import SignalEngine
    from portfolio.risk_engine import RiskEngine

    assets = [a.strip() for a in args.assets.split(",")]
    log.info("Generating signals for: %s", assets)

    # Load data
    svc = MarketDataService(provider=args.provider)
    market_data = {}
    for asset in assets:
        try:
            df = svc.get(asset, timeframe=args.timeframe, start=args.start)
            df = add_regime_features(df)
            df = add_alpha_features(df)
            market_data[asset] = df
            latest = df.iloc[-1]
            log.info("%s: %d bars, last close=%.5f, ADX=%.1f",
                     asset, len(df), latest["close"], latest.get("adx", 0))
        except Exception as exc:
            log.error("Failed to load %s: %s", asset, exc)

    if not market_data:
        log.error("No data — exiting")
        sys.exit(1)

    # Fit regime model
    combined_df = list(market_data.values())[0]
    regime_svc = RegimeService()
    log.info("Fitting regime model...")
    regime_svc.fit(combined_df)

    # Detect regime for each asset
    for asset, df in market_data.items():
        result = regime_svc.latest(df, asset=asset)
        log.info(
            "Regime: %s | %s (smoothed: %s) | conf=%.2f",
            asset, result.predicted_regime, result.smoothed_regime, result.confidence,
        )

    # Route to signals
    param_store = ParameterStore()
    tf_filter   = TradabilityFilter()
    router = RegimeStrategyRouter(
        regime_service     = regime_svc,
        parameter_store    = param_store,
        asset_universe     = assets,
        tradability_filter = tf_filter,
    )
    proposals = router.route(market_data, news_blocked=args.news)

    # Risk check and output
    risk_engine = RiskEngine()
    engine = SignalEngine(risk_engine=risk_engine)
    signals = engine.process(proposals, equity=args.equity)

    if signals:
        best = signals[0]
        engine.write_latest(best)
        print("\n=== SIGNAL GENERATED ===")
        import json
        print(json.dumps(best.to_dict(), indent=2))
    else:
        engine.write_no_signal()
        print("\n=== NO SIGNAL ===")
        print("No tradeable opportunities found at this time.")


if __name__ == "__main__":
    main()
