@echo off
REM Stops the signal loop by killing the Python process
taskkill /F /FI "WINDOWTITLE eq Signal Loop*" >nul 2>&1
taskkill /F /IM python.exe /FI "MEMUSAGE gt 50000" >nul 2>&1
echo Signal loop stopped at %date% %time%
