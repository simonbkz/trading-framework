' ============================================================
'  Runs the signal loop BAT file without a visible window.
'  Use this with Windows Task Scheduler for fully silent operation.
'
'  To use: right-click > Run, or add to Task Scheduler
' ============================================================
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c ""C:\Users\SIMON\trading_framework\scripts\start_signal_loop.bat""", 0, False
