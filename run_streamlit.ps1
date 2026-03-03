# Quick Start Script for Streamlit UI (PowerShell)
# Run this file to automatically start the app

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Oulu Pedestrian Traffic Predictor" -ForegroundColor Cyan
Write-Host "  Streamlit UI - Quick Start" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[✓] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[✗] ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[1/3] Checking Python version..." -ForegroundColor Yellow
python --version
Write-Host ""

Write-Host "[2/3] Installing/updating dependencies..." -ForegroundColor Yellow
python -m pip install -q -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "[✗] ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[✓] Dependencies installed successfully!" -ForegroundColor Green
Write-Host ""

Write-Host "[3/3] Starting Streamlit app..." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Opening at: http://localhost:8501" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C in the terminal to stop the server" -ForegroundColor Yellow
Write-Host ""

python -m streamlit run src/streamlit_app.py

Read-Host "Press Enter to exit"
