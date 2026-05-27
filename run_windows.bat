@echo off
echo ===================================================
echo       Starting Blurify AI (Face Blur Pipeline)
echo ===================================================

:: Check if Node is installed
where npm >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH!
    echo Please install Node.js to run the UI.
    pause
    exit /b
)

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.9+.
    pause
    exit /b
)

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Setting it up now...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [INFO] Installing Python requirements...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

:: Check if Node modules exist
if not exist "ui\node_modules\" (
    echo [INFO] Node modules not found. Installing now...
    cd ui
    call npm install
    cd ..
)

echo.
echo [1/2] Starting FastAPI AI Backend on port 8000...
start "Blurify AI - Backend" cmd /k "call .venv\Scripts\activate.bat && python -m src.api"

echo [2/2] Starting React Frontend on port 5173...
start "Blurify AI - Frontend" cmd /k "cd ui && npm run dev"

echo.
echo ===================================================
echo  All systems launched!
echo  Your browser should open shortly.
echo  If not, go to: http://localhost:5173
echo ===================================================
timeout /t 3 >nul
start http://localhost:5173
