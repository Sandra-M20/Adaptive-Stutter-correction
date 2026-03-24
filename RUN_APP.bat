@echo off
echo.
echo ============================================================
echo  Adaptive Stutter Correction System - Streamlit UI
echo ============================================================
echo.
echo Starting app... browser will open automatically at http://localhost:8501
echo Press Ctrl+C in this window to stop.
echo.
set STREAMLIT=%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\streamlit.exe
"%STREAMLIT%" run "%~dp0app.py"
pause
