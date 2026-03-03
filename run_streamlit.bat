@echo off
REM Quick Start Script for Streamlit UI
REM Run this file to automatically start the app

echo.
echo ============================================
echo   Oulu Pedestrian Traffic Predictor
echo   Streamlit UI - Quick Start
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/3] Checking Python version...
python --version
echo.

echo [2/3] Installing/updating dependencies...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

echo [3/3] Starting Streamlit app...
echo.
echo ============================================
echo Opening at: http://localhost:8501
echo============================================
echo.
echo Press Ctrl+C to stop the server
echo.

python -m streamlit run src/streamlit_app.py

pause
