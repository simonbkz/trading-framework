# Regime-Aware Multi-Asset Trading Framework

A production-grade Python framework for regime-aware trading research, signal generation, and MT5 execution.

---

## Architecture Overview

```
Market Data ──► Feature Engineering ──► Regime Detection ──► Filter Layer
                                                                    │
                                                          Asset + Strategy Selection
                                                                    │
                                                     Parameter Loading (per asset/regime/side)
                                                                    │
                                                          Signal Generation
                                                                    │
                                                     Risk Engine + Position Sizing
                                                                    │
                                              MT5 Bridge (JSON file / API)
```

**Python is the brain.** MT5 handles only order execution and broker connectivity.

---

## Project Structure

```
trading_framework/
├── config/            # Settings, asset definitions, regime labels, providers
├── data/              # Market data loading, caching, preprocessing, news calendar
├── features/          # Regime features, alpha features, risk features
├── regimes/           # HMM, tree, ensemble regime detection
├── filters/           # News, session, liquidity, tradability filters
├── strategies/        # Trend breakout, mean reversion, vol breakout, session breakout
├── optimization/      # Grid/random/Optuna optimization, parameter store
├── validation/        # Walk-forward CV, purged CV, robustness checks, metrics
├── selection/         # Asset selector, strategy selector, side selector, router
├── portfolio/         # Risk engine, position sizer, exposure manager
├── execution/         # Signal engine, order schema, MT5 bridge schema, paper executor
├── monitoring/        # Performance monitor, equity tracking
├── utils/             # Env loader, logger, helpers, plotting
├── scripts/           # CLI scripts: backtest, optimize, signal generation
├── notebooks/         # Jupyter demo notebook
├── tests/             # pytest test suite
├── signals/           # Signal JSON output (MT5 picks up from here)
├── parameters/        # Persisted optimized parameter sets
├── models/            # Saved regime models
├── data_cache/        # Parquet data cache
├── main.py            # Main entry point
├── .env.example       # Environment variable template
└── requirements.txt   # Python dependencies
```

---

## Installation

```bash
git clone <repo>
cd trading_framework

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys (yfinance requires no key)
```

---

## Environment Variables

All secrets are loaded from `.env`. See `.env.example` for the full list.

| Variable | Required | Description |
|----------|----------|-------------|
| `MARKET_DATA_PROVIDER` | No | `yfinance` (default), `alpha_vantage`, `twelve_data`, `polygon`, `csv` |
| `NEWS_DATA_PROVIDER` | No | `csv` (default) or `trading_economics` |
| `ALPHA_VANTAGE_API_KEY` | If using Alpha Vantage | API key from alphavantage.co |
| `TWELVE_DATA_API_KEY` | If using Twelve Data | API key from twelvedata.com |
| `POLYGON_API_KEY` | If using Polygon | API key from polygon.io |
| `TRADING_ECONOMICS_API_KEY` | If using TE calendar | Trading Economics API key |
| `TRADING_ECONOMICS_API_SECRET` | If using TE calendar | Trading Economics secret |
| `DATA_CACHE_DIR` | No | Default: `./data_cache` |
| `MARKET_DATA_CSV_DIR` | If using CSV fallback | Default: `./data/market` |
| `NEWS_DATA_CSV_PATH` | If using CSV news | Default: `./data/news/events.csv` |
| `MT5_SIGNAL_OUTPUT_PATH` | No | Default: `./signals/latest_signal.json` |
| `TIMEZONE` | No | Default: `UTC` |
| `DEFAULT_RISK_PCT` | No | Default: `0.5` (0.5% per trade) |

---

## Data Provider Configuration

### yfinance (default — no API key required)
```
MARKET_DATA_PROVIDER=yfinance
```
Free, no registration. Used for EURUSD, USDJPY, XAUUSD, NAS100, BTCUSD, etc.

### Alpha Vantage
```
MARKET_DATA_PROVIDER=alpha_vantage
ALPHA_VANTAGE_API_KEY=your_key
```
Free tier: 25 requests/day, 5 req/min. Premium removes limits.

### Twelve Data
```
MARKET_DATA_PROVIDER=twelve_data
TWELVE_DATA_API_KEY=your_key
```
Free tier: 800 req/day. Good forex and crypto coverage.

### Polygon.io
```
MARKET_DATA_PROVIDER=polygon
POLYGON_API_KEY=your_key
```
Free tier covers forex, crypto, equities. No index futures on free tier.

---

## CSV Fallback Workflow

When no API key is available, supply your own OHLCV CSV files.

1. Set `MARKET_DATA_PROVIDER=csv` in your `.env`
2. Place CSV files in `MARKET_DATA_CSV_DIR` (default `./data/market/`)
3. Name files as `{SYMBOL}_{TIMEFRAME}.csv` (e.g. `EURUSD_1h.csv`)

Required CSV columns:
```
datetime,open,high,low,close,volume
2024-01-02 00:00:00,1.10234,1.10456,1.10123,1.10345,1234
```
The `datetime` column is detected automatically from: `datetime`, `date`, `time`, `timestamp`, or the first column.

For news events, place `events.csv` in `NEWS_DATA_CSV_PATH`:
```
datetime,currency,impact,event
2024-01-26 13:30:00,USD,high,US GDP Q4 Advance
```

---

## Training Workflow

Fit and save the regime detection model:

```bash
python main.py --mode train --assets EURUSD,USDJPY,XAUUSD --start 2022-01-01
```

Or via script:
```bash
python scripts/generate_signal.py --assets EURUSD,USDJPY
```

The ensemble model (HMM + LightGBM) is saved to `models/regime_model.pkl`.

---

## Backtesting Workflow

```bash
# Basic backtest
python main.py --mode backtest --asset EURUSD --strategy trend_breakout

# With specific side and date range
python scripts/run_backtest.py \
    --asset USDJPY \
    --strategy mean_reversion \
    --timeframe 1h \
    --start 2023-01-01 \
    --side long \
    --save

# Available strategies: trend_breakout | mean_reversion | volatility_breakout | session_breakout
```

Output includes:
- Trade-level metrics (N trades, PF, WR, expectancy)
- Equity curve metrics (CAGR, Sharpe, Sortino, Calmar, max DD)
- Breakdown by regime, side, and year

---

## Optimization Workflow

```bash
# Bayesian optimization (recommended)
python scripts/run_optimization.py \
    --asset EURUSD \
    --strategy trend_breakout \
    --method optuna \
    --trials 200 \
    --folds 5

# Grid search
python main.py --mode optimize --asset EURUSD --method grid
```

Results are saved to `parameters/parameter_store.json`.
A CSV export is also generated for easy inspection.

Re-optimize every 3 months or when rolling 20-trade profit factor drops below 0.80.

---

## Live Signal Workflow

```bash
# Generate signal for multiple assets
python scripts/generate_signal.py \
    --assets EURUSD,USDJPY,XAUUSD,BTCUSD \
    --equity 10000 \
    --timeframe 1h

# Schedule via Windows Task Scheduler or cron:
# Every hour: python /path/to/scripts/generate_signal.py
```

Output signal written to `signals/latest_signal.json`:
```json
{
  "timestamp": "2025-03-14T12:30:00+00:00",
  "asset": "EURUSD",
  "regime": "trend_up",
  "regime_confidence": 0.71,
  "strategy": "trend_breakout",
  "side": "long",
  "entry": 1.09234,
  "stop_loss": 1.08934,
  "take_profit": 1.09834,
  "lots": 0.10,
  "risk_pct": 0.5,
  "order_id": "a4f9c12b",
  "action": "open",
  "expiry_minutes": 60
}
```

---

## MT5 Integration Guide

Python remains the decision engine. MT5 handles execution only.

### File-Based Bridge (Recommended)

1. Set `MT5_SIGNAL_OUTPUT_PATH` to a path accessible by MT5:
   ```
   MT5_SIGNAL_OUTPUT_PATH=C:/Users/YourName/AppData/Roaming/MetaQuotes/Terminal/.../MQL5/Files/signals/latest_signal.json
   ```

2. Create an MT5 EA (`EA_Bridge.mq5`) that:
   - Reads `signals/latest_signal.json` on every tick
   - Checks `order_id` to avoid re-executing the same signal
   - Validates `timestamp` is within `expiry_minutes`
   - Places a market order with the specified SL/TP/lots
   - Writes result to `signals/last_result.json`

See `execution/mt5_bridge_schema.py` for the full schema documentation and MQL5 skeleton.

### API-Based Bridge (Windows Only)

```python
import MetaTrader5 as mt5
mt5.initialize()
request = {
    "action":   mt5.TRADE_ACTION_DEAL,
    "symbol":   signal.asset,
    "volume":   signal.lots,
    "type":     mt5.ORDER_TYPE_BUY,
    "price":    mt5.symbol_info_tick(signal.asset).ask,
    "sl":       signal.stop_loss,
    "tp":       signal.take_profit,
    "magic":    signal.magic_number,
    "comment":  signal.comment,
}
result = mt5.order_send(request)
```

---

## Running Tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ --cov=. --cov-report=html
```

---

## Supported Assets

| Symbol | Class | Provider Ticker |
|--------|-------|----------------|
| BTCUSD | Crypto | BTC-USD (yfinance) |
| XAUUSD | Commodity | GC=F (yfinance) |
| NAS100 | Index | NQ=F (yfinance) |
| EURUSD | Forex | EURUSD=X (yfinance) |
| GBPUSD | Forex | GBPUSD=X (yfinance) |
| USDJPY | Forex | USDJPY=X (yfinance) |
| AUDUSD | Forex | AUDUSD=X (yfinance) |

---

## Regime Labels

| Regime | Description | Best Strategies |
|--------|-------------|-----------------|
| `trend_up` | Sustained uptrend | trend_breakout, session_breakout |
| `trend_down` | Sustained downtrend | trend_breakout, session_breakout |
| `range_bound` | Oscillating price | mean_reversion |
| `mean_reverting` | Overshoots snap back | mean_reversion |
| `high_volatility` | Elevated ATR | volatility_breakout |
| `low_volatility` | Compressed range | volatility_breakout, session_breakout |
| `event_risk` | News imminent | No trading |

---

## Optional Agent Skills

The following agent skill categories could augment this development workflow:

| Skill Category | Use Case |
|---------------|----------|
| **Data provider integration** | Auto-fetch from new providers (e.g. Quandl, Bloomberg) |
| **Feature engineering templates** | Generate domain-specific features for new asset classes |
| **Walk-forward validation checks** | Automated regime stability testing across extended history |
| **Experiment tracking** | MLflow/Weights & Biases integration for regime model runs |
| **Documentation automation** | Auto-generate docstrings and API docs from code |
| **MT5 bridge packaging** | Package the bridge EA into a deployable MT5 `.ex5` file |

Rules:
- Skills are optional — the system runs without any external agent
- Core trading logic (regime detection, strategy selection, signal generation) remains native Python
- Do not outsource regime detection or strategy logic to external services
- All data processing is local — no signal data sent to third parties

---

## Known Limitations

1. **yfinance data quality**: Intraday data (< 1h) has limited history (60 days max). Use 1h or 1d for backtesting.
2. **HMM state mapping**: The heuristic state→regime mapping works for most market conditions but may misclassify during extreme events (COVID crash, flash crashes).
3. **Walk-forward speed**: 200 Optuna trials × 5 folds on 2 years of 1h data takes ~5–15 minutes per asset/strategy combination.
4. **MT5 file bridge latency**: File-based bridge has ~1-second latency. For HFT use the API bridge.
5. **No real-time streaming**: The framework polls data on each cycle. Real-time websocket feeds require a custom data adapter.

---

## Future Improvements

- [ ] Real-time data streaming adapter (Polygon WebSocket, Twelve Data streaming)
- [ ] Multi-asset portfolio correlation filter
- [ ] Reinforcement learning regime-aware position sizing
- [ ] Automated regime change notification (email/Telegram)
- [ ] Interactive Plotly dashboard for signal monitoring
- [ ] ONNX export for regime model → MT5 native inference
- [ ] Automatic re-optimization trigger when rolling PF < 0.80
