"""
MT5 Bridge Design Documentation and Schema.

Python is the decision engine.
MT5 EA reads JSON signal files and executes orders.

=============================================================================
BRIDGE DESIGN — FILE-BASED
=============================================================================

Flow:
    [Python] → writes  → signals/latest_signal.json
    [MT5 EA] → reads   → signals/latest_signal.json  (every tick via FileOpen)
    [MT5 EA] → executes → order, then writes result back
    [MT5 EA] → writes  → signals/last_result.json
    [Python] → reads   → signals/last_result.json (for monitoring)

Signal file location (shared between Python and MT5):
    Must be under MT5's Files/ directory OR a mapped path via symlink.
    Recommended: MT5_DATA_PATH/MQL5/Files/signals/latest_signal.json

MT5 EA responsibilities:
    - Poll signal file on every tick (or every N seconds)
    - Check signal timestamp + expiry_minutes
    - Check order_id to avoid re-executing the same signal
    - Place market order at broker price
    - Set SL and TP on the order
    - Write execution result to last_result.json
    - Handle partial fills and requotes

Python responsibilities:
    - All regime detection, strategy selection, signal generation
    - Risk sizing (passed to MT5 as lots)
    - Signal file management
    - Post-trade analysis and monitoring

=============================================================================
BRIDGE DESIGN — API-BASED (Alternative)
=============================================================================

If MT5 Python API is available (Windows, same machine):
    import MetaTrader5 as mt5
    mt5.initialize()
    mt5.order_send(request)

This is more reliable than file polling for live trading.
See scripts/live_signal.py for the API-based implementation.

=============================================================================
SIGNAL FILE FORMAT (latest_signal.json)
=============================================================================

{
    "timestamp":          "2025-03-14T12:30:00+00:00",  // UTC ISO
    "asset":              "EURUSD",
    "regime":             "trend_up",
    "regime_confidence":  0.71,
    "strategy":           "trend_breakout",
    "side":               "long",
    "order_type":         "market",
    "entry":              1.09234,     // reference entry price
    "stop_loss":          1.08934,
    "take_profit":        1.09834,
    "lots":               0.10,
    "risk_pct":           0.5,
    "signal_strength":    0.73,
    "order_id":           "a4f9c12b",  // unique — EA tracks to avoid re-entry
    "action":             "open",      // open | close | update | none
    "magic_number":       99999,
    "comment":            "Regime:trend_up|Strat:trend_breakout",
    "expiry_minutes":     60           // ignore signal if older than this
}

=============================================================================
RESULT FILE FORMAT (last_result.json) — written by MT5 EA
=============================================================================

{
    "order_id":     "a4f9c12b",
    "ticket":       12345678,
    "status":       "filled",       // filled | rejected | error
    "fill_price":   1.09236,
    "fill_time":    "2025-03-14T12:30:01+00:00",
    "lots":         0.10,
    "error_code":   0,
    "error_msg":    ""
}

=============================================================================
MT5 EA SKELETON (MQL5)
=============================================================================

// In EA_Bridge.mq5:
//   OnTick() {
//     string signal_path = "signals\\latest_signal.json";
//     int fh = FileOpen(signal_path, FILE_READ|FILE_TXT|FILE_COMMON);
//     if(fh == INVALID_HANDLE) return;
//     string json_str = "";
//     while(!FileIsEnding(fh)) json_str += FileReadString(fh);
//     FileClose(fh);
//     // Parse JSON, check timestamp + order_id, place order
//   }

=============================================================================
"""

# This module serves as documentation only.
# Actual bridge implementation is in execution/signal_engine.py
# and scripts/generate_signal.py

MT5_SIGNAL_SCHEMA = {
    "timestamp":         {"type": "string", "format": "ISO 8601 UTC"},
    "asset":             {"type": "string", "example": "EURUSD"},
    "regime":            {"type": "string", "enum": [
        "trend_up","trend_down","range_bound","mean_reverting",
        "high_volatility","low_volatility","event_risk"
    ]},
    "regime_confidence": {"type": "float",  "range": [0, 1]},
    "strategy":          {"type": "string", "enum": [
        "trend_breakout","mean_reversion","volatility_breakout","session_breakout"
    ]},
    "side":              {"type": "string",  "enum": ["long","short"]},
    "order_type":        {"type": "string",  "enum": ["market","limit","stop"]},
    "entry":             {"type": "float",   "description": "Reference entry price"},
    "stop_loss":         {"type": "float"},
    "take_profit":       {"type": "float"},
    "lots":              {"type": "float",   "description": "Position size in standard lots"},
    "risk_pct":          {"type": "float",   "description": "Risk as % of equity"},
    "signal_strength":   {"type": "float",   "range": [0, 1]},
    "order_id":          {"type": "string",  "description": "Unique signal ID"},
    "action":            {"type": "string",  "enum": ["open","close","update","none"]},
    "magic_number":      {"type": "int",     "description": "MT5 EA magic number"},
    "comment":           {"type": "string"},
    "expiry_minutes":    {"type": "int",     "description": "Signal TTL in minutes"},
}
