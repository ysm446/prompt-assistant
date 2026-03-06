@echo off
echo ========================================
echo  Prompt Assistant
echo ========================================
echo.

REM Check Node.js
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found.
    echo Please install Node.js from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

REM Activate conda environment
call conda activate main
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'main'.
    pause
    exit /b 1
)

REM Set HF_HOME
set HF_HOME=%~dp0models

REM Start Python server in background
echo Starting Python server...
start /B python "%~dp0server.py" --port 8765

REM Start Electron
cd /d "%~dp0electron"

if not exist "node_modules" (
    echo [First run] Running npm install...
    npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
    echo.
)

echo Starting Electron...
npm start

pause
