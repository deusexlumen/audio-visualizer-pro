@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  Audio Visualizer Pro - Debug-Start
echo ========================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [FEHLER] Python wurde nicht gefunden.
    pause
    exit /b 1
)

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

echo Starte mit faulthandler und schreibe Log nach debug.log...
python -X faulthandler -u gui.py > debug.log 2>&1

echo.
echo Fertig. Log wurde in debug.log geschrieben.
echo.
pause
