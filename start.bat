@echo off
echo Starting Prompt Assistant...
echo Make sure SD WebUI Forge is running with --api option.
echo.

set HF_HOME=%~dp0models

call conda activate main
python "%~dp0app.py"

pause
