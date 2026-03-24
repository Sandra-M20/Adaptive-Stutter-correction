@echo off
echo Starting Adaptive Stutter Correction Backend...
echo.
cd /d "%~dp0ui\backend"
python main.py
pause
