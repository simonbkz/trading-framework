"""
Automated signal generation loop.

Runs continuously, generating signals at regular intervals and writing
them to both:
    1. signals/latest_signal.json (project)
    2. MQL5/Files/latest_signal.json (MT5 sandbox)

The MT5 SignalBridge EA reads the signal file and executes trades.

Usage:
    python scripts/run_signal_loop.py
    python scripts/run_signal_loop.py --interval 300 --assets EURUSD,USDJPY
    python scripts/run_signal_loop.py --once   # single run, no loop
"""
import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import SETTINGS
from utils.logger import get_logger

log = get_logger("signal_loop")

# Track emitted signals to prevent duplicate trades from the lookback window.
# Key: "asset|side|signal_bar_timestamp" → only emit once per breakout event.
# Persisted to disk so restarts don't re-emit.
_DEDUP_FILE = ROOT / "signals" / "_emitted_signals.json"
_emitted_signals: set = set()


def _load_emitted_signals():
    """Load dedup set from disk on startup."""
    global _emitted_signals
    if _DEDUP_FILE.exists():
        try:
            data = json.loads(_DEDUP_FILE.read_text())
            _emitted_signals = set(data)
            log.info("Loaded %d emitted signal keys from disk", len(_emitted_signals))
        except Exception as exc:
            log.warning("Failed to load dedup file: %s", exc)
            _emitted_signals = set()


def _save_emitted_signals():
    """Persist dedup set to disk."""
    try:
        _DEDUP_FILE.write_text(json.dumps(sorted(_emitted_signals)))
    except Exception as exc:
        log.warning("Failed to save dedup file: %s", exc)


def generate_signals(assets: list, timeframe: str, start: str, equity: float):
    """Run one cycle of signal generation."""
    from data.market_data import MarketDataService
    from features.regime_features import add_regime_features
    from features.alpha_features import add_alpha_features
    from regimes.regime_service import RegimeService
    from optimization.parameter_store import ParameterStore
    from filters.tradability_filter import TradabilityFilter
    from selection.regime_strategy_router import RegimeStrategyRouter
    from execution.signal_engine import SignalEngine
    from portfolio.risk_engine import RiskEngine

    # 1. Load market data
    svc = MarketDataService(provider="yfinance")
    market_data = {}
    for asset in assets:
        try:
            df = svc.get(asset, timeframe=timeframe, start=start)
            df = add_regime_features(df)
            df = add_alpha_features(df)
            market_data[asset] = df
        except Exception as exc:
            log.error("Failed to load %s: %s", asset, exc)

    if not market_data:
        log.error("No data loaded")
        return False

    # 2. Load or fit regime model
    regime_svc = RegimeService()
    model_path = ROOT / "models" / "regime_model.pkl"
    if model_path.exists():
        regime_svc.load(model_path)
    else:
        primary = list(market_data.values())[0]
        regime_svc.fit(primary)
        log.warning("No saved model -- fit on %s. Run --mode train first.", assets[0])

    # 3. Route signals
    param_store = ParameterStore()
    tf_filter = TradabilityFilter()

    router = RegimeStrategyRouter(
        regime_service=regime_svc,
        parameter_store=param_store,
        asset_universe=assets,
        tradability_filter=tf_filter,
    )
    proposals = router.route(market_data)

    # 4. Process through risk engine
    risk_engine = RiskEngine()
    # Initialize all equity baselines to actual account equity
    # (prevents false daily-loss / drawdown halts from hardcoded defaults)
    risk_engine.current_equity = equity
    risk_engine.peak_equity = equity
    risk_engine.day_start_equity = equity
    engine = SignalEngine(risk_engine=risk_engine, paper_mode=True)
    signals = engine.process(proposals, equity=equity)

    # 5. Deduplicate: only emit each breakout event once
    #    The lookback window catches the same signal bar across multiple cycles,
    #    but we must only send it to MT5 once to prevent duplicate trades.
    new_signals = []
    for sig in signals:
        # Use asset|side|entry as dedup key (entry price is stable per bar)
        # sig.timestamp is datetime.now() so it changes every cycle — unusable
        key = f"{sig.asset}|{sig.side}|{sig.entry}"
        if key in _emitted_signals:
            log.info("Skipping duplicate signal: %s %s (already emitted)", sig.asset, sig.side)
            continue
        _emitted_signals.add(key)
        new_signals.append(sig)

    # Prune old keys (keep last 200 to prevent memory growth)
    if len(_emitted_signals) > 200:
        # Convert to list, keep last 200
        items = sorted(_emitted_signals)
        _emitted_signals.clear()
        _emitted_signals.update(items[-200:])

    # Persist to disk so restarts don't re-emit
    _save_emitted_signals()

    # 6. Write signals (auto-mirrors to MQL5/Files/)
    if new_signals:
        engine.write_all(new_signals)
        log.info(
            "Wrote %d signal(s): %s",
            len(new_signals),
            ["%s %s %s" % (s.asset, s.side.upper(), s.strategy) for s in new_signals],
        )
    else:
        engine.write_no_signal()
        log.info("No signal generated")

    return True


def main():
    parser = argparse.ArgumentParser(description="Automated Signal Generation Loop")
    parser.add_argument("--assets", default="XAUUSD,BTCUSD,XAGUSD,ETHUSD,EURJPY",
                        help="Comma-separated asset list")
    parser.add_argument("--timeframe", default="1h", help="Data timeframe")
    parser.add_argument("--start", default="2025-06-01", help="Data start date")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between signal runs (default: 300 = 5 min)")
    parser.add_argument("--equity", type=float, default=10000.0,
                        help="Account equity for position sizing")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (no loop)")
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",")]

    # Load dedup state from previous run
    _load_emitted_signals()

    # Verify MT5 mirror path
    mt5_path = SETTINGS.execution.mt5_files_path
    log.info("=" * 50)
    log.info("SIGNAL LOOP STARTING")
    log.info("Assets:    %s", assets)
    log.info("Interval:  %d seconds", args.interval)
    log.info("Timeframe: %s", args.timeframe)
    log.info("Equity:    $%.0f", args.equity)
    log.info("Output:    %s", SETTINGS.execution.signal_output_path)
    log.info("MT5 path:  %s", mt5_path or "NOT DETECTED")
    log.info("=" * 50)

    if mt5_path is None:
        log.warning("MT5 Files path not detected. Signals will only write to project folder.")

    if args.once:
        generate_signals(assets, args.timeframe, args.start, args.equity)
        return

    # Continuous loop
    cycle = 0
    while True:
        cycle += 1
        now = datetime.now(tz=timezone.utc)
        log.info("--- Cycle %d at %s ---", cycle, now.strftime("%Y-%m-%d %H:%M UTC"))

        try:
            generate_signals(assets, args.timeframe, args.start, args.equity)
        except KeyboardInterrupt:
            log.info("Interrupted by user. Shutting down.")
            break
        except Exception:
            log.error("Signal generation failed:\n%s", traceback.format_exc())

        log.info("Next run in %d seconds...", args.interval)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            log.info("Interrupted by user. Shutting down.")
            break


if __name__ == "__main__":
    main()
