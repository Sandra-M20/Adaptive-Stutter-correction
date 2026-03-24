@echo off
echo.
echo ============================================================
echo  ADAPTIVEVOICE - DIRECT LAUNCHER (SAFE MODE)
echo ============================================================
echo.
echo Your computer is currently blocking local network ports.
echo Opening the dashboard DIRECTLY to bypass these issues...
echo.
start "" "%~dp0ui\frontend\public\stutter_ui.html"
echo SUCCESS: Dashboard opened.
echo.
pause
