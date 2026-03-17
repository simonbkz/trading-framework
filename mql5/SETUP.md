# MT5 SignalBridge EA — Setup Guide

## Architecture

```
Python (main.py --mode signal)
   |
   v
signals/latest_signal.json   <-- batch JSON with 0..N signals
   |
   v
MT5 EA (SignalBridge.mq5)    <-- reads JSON, executes trades
```

## Files

| File | Purpose |
|---|---|
| `SignalBridge.mq5` | Expert Advisor — reads JSON, sends orders |
| `JsonParser.mqh` | Lightweight JSON parser (include file) |

## Installation

### Step 1: Copy files to MT5

Copy both files to your MT5 data folder:

```
<MT5 Data Folder>\MQL5\Experts\SignalBridge.mq5
<MT5 Data Folder>\MQL5\Experts\JsonParser.mqh
```

To find your data folder: in MT5, go to **File > Open Data Folder**.

### Step 2: Set file path

The EA needs to read the JSON file. There are two options:

**Option A: Direct path (recommended)**

Set `InpSignalPath` to the full path:
```
C:\Users\SIMON\trading_framework\signals\latest_signal.json
```

> **Important**: MT5's `FileOpen` can only read files inside its sandbox
> (`MQL5\Files\`) by default. To read external paths, enable
> **Tools > Options > Expert Advisors > Allow DLL imports**.

**Option B: Symlink into MQL5\Files**

Create a symbolic link so MT5 can read it from its sandbox:
```cmd
mklink "C:\...\MQL5\Files\latest_signal.json" "C:\Users\SIMON\trading_framework\signals\latest_signal.json"
```

Then set `InpSignalPath = "latest_signal.json"` and the EA will find it
in the common files folder.

### Step 3: Compile

1. Open MetaEditor (F4 from MT5)
2. Open `SignalBridge.mq5`
3. Press **Compile** (F7)
4. Should compile with 0 errors

### Step 4: Attach to chart

1. Open any chart (e.g. EURUSD H1)
2. Drag `SignalBridge` from Navigator > Expert Advisors onto the chart
3. In the Inputs tab, verify/adjust:
   - `InpSignalPath` — path to your JSON file
   - `InpPollSeconds` — how often to check (default: 5s)
   - `InpMaxOpenTrades` — max simultaneous trades (default: 3)
   - `InpMagicNumber` — must match Python's magic_number (default: 99999)
4. Enable **Allow Algo Trading** (button on toolbar)
5. Enable **Allow DLL imports** if using Option A path

## Testing

### Test 1: Paper mode (Python side only)

```bash
python main.py --mode signal --assets EURUSD,USDJPY,XAUUSD,BTCUSD
```

Check `signals/latest_signal.json` — should show the batch format.

### Test 2: MT5 Strategy Tester

1. Open Strategy Tester (Ctrl+R)
2. Select `SignalBridge`
3. Mode: **Every tick**
4. Set the signal path to a test file you pre-populate
5. Run — check the Journal tab for execution logs

### Test 3: Demo account (recommended first)

1. Open a **demo account** with your broker
2. Attach the EA to a chart
3. Run Python signal generation on a timer:
   ```bash
   # Run every 5 minutes via Task Scheduler or cron
   python main.py --mode signal --assets EURUSD,USDJPY,XAUUSD,BTCUSD
   ```
4. Monitor the MT5 Journal and Experts tabs for trade execution

### Test 4: Manual signal injection

Create a test signal file to verify the EA executes:

```json
{
  "timestamp": "2026-03-14T12:00:00+00:00",
  "count": 1,
  "action": "open",
  "signals": [
    {
      "timestamp": "2026-03-14T12:00:00+00:00",
      "asset": "EURUSD",
      "regime": "trend_up",
      "regime_confidence": 0.75,
      "strategy": "trend_breakout",
      "side": "long",
      "order_type": "market",
      "entry": 1.08500,
      "stop_loss": 1.08200,
      "take_profit": 1.09100,
      "lots": 0.01,
      "risk_pct": 0.5,
      "signal_strength": 0.80,
      "order_id": "test001",
      "action": "open",
      "magic_number": 99999,
      "comment": "Regime:trend_up|Strat:trend_breakout",
      "expiry_minutes": 60
    }
  ]
}
```

Save this to `signals/latest_signal.json` and the EA should pick it up
within `InpPollSeconds`.

## Symbol Mapping

The EA maps Python asset names to MT5 symbols. Edit `MapToMT5Symbol()`
in `SignalBridge.mq5` to match your broker:

| Python Name | Default MT5 | Common Alternatives |
|---|---|---|
| EURUSD | EURUSD | EURUSD.i, EURUSDm |
| GBPUSD | GBPUSD | GBPUSD.i, GBPUSDm |
| USDJPY | USDJPY | USDJPY.i, USDJPYm |
| XAUUSD | XAUUSD | GOLD, XAUUSDm |
| BTCUSD | BTCUSD | BTCUSDm, Bitcoin |
| NAS100 | USTEC | NAS100, US100 |

## Scheduling Python Signals

For live trading, run signal generation on a schedule:

**Windows Task Scheduler:**
```
Program: python
Arguments: main.py --mode signal --assets EURUSD,USDJPY,XAUUSD,BTCUSD
Start in: C:\Users\SIMON\trading_framework
Trigger: Every 5 minutes during trading hours
```

**Or use a simple loop script (`scripts/run_signals.bat`):**
```batch
@echo off
:loop
python main.py --mode signal --assets EURUSD,USDJPY,XAUUSD,BTCUSD
timeout /t 300
goto loop
```

## Troubleshooting

| Problem | Solution |
|---|---|
| EA doesn't read file | Check path, enable DLL imports, try symlink |
| "Unknown asset mapping" | Edit `MapToMT5Symbol()` for your broker |
| Orders rejected | Check lot size, SL/TP distance, account margin |
| Stale signals | Increase `InpSignalExpiry` or run Python more often |
| No signals generated | Run during market hours, check regime model exists |
