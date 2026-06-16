@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  Audio Visualizer Pro - Start
echo ========================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [FEHLER] Python wurde nicht gefunden.
    echo Bitte installiere Python und fuege es zur PATH-Variable hinzu.
    pause
    exit /b 1
)

if exist "venv\Scripts\activate.bat" (
    echo Virtuelle Umgebung wird aktiviert...
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    echo Virtuelle Umgebung wird aktiviert...
    call ".venv\Scripts\activate.bat"
)

echo Starte Audio Visualizer Pro GUI...
python gui.py

if %errorlevel% neq 0 (
    echo.
    echo [FEHLER] Die Anwendung wurde mit Fehlercode %errorlevel% beendet.
    pause
)
