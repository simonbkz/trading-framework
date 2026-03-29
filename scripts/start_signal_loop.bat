@echo off
REM ============================================================
REM  Signal Loop Launcher (1H timeframe, 5 instruments)
REM  Runs the Python signal generator in a loop every 5 minutes.
REM  Output is logged to signals/signal_loop.log
REM ============================================================

cd /d "C:\Users\SIMON\trading_framework"

echo Starting Signal Loop at %date% %time%
echo Assets: XAUUSD,BTCUSD,XAGUSD,ETHUSD,XRPUSD
echo Interval: 300 seconds (5 minutes)
echo Timeframe: 1H
echo Log: signals\signal_loop.log
echo Press Ctrl+C to stop.
echo.

python scripts/run_signal_loop.py ^
    --assets XAUUSD,BTCUSD,XAGUSD,ETHUSD,XRPUSD ^
    --interval 300 ^
    --equity 500 ^
    --timeframe 1h ^
    --start 2025-06-01 ^
    >> signals\signal_loop.log 2>&1
