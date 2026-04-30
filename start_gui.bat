@echo off
chcp 65001 >nul
echo ===================================
echo   Audio Visualizer Pro - GUI
echo ===================================
echo.
echo Starte grafische Oberfläche...
echo.

:: Prüfe ob Python installiert ist
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python ist nicht installiert oder nicht im PATH!
    echo Bitte installiere Python von https://python.org
    pause
    exit /b 1
)

:: Prüfe ob DearPyGui installiert ist
python -c "import dearpygui" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installiere benötigte Pakete...
    pip install dearpygui -q
)

echo ✅ Abhängigkeiten OK
echo 🚀 Starte GUI...
echo.

:: Starte DearPyGui
python gui.py

pause
