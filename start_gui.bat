@echo off
chcp 65001 >nul
echo ===================================
echo   Audio Visualizer Pro - GUI
echo ===================================
echo.
echo Starte grafische Oberflaeche...
echo.

:: Pruefe ob Python installiert ist
python --version >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python ist nicht installiert oder nicht im PATH!
    echo Bitte installiere Python von https://python.org
    pause
    exit /b 1
)

:: Pruefe ob DearPyGui installiert ist
python -c "import dearpygui" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installiere benoetigte Pakete...
    pip install dearpygui -q
)

echo [OK] Abhaengigkeiten OK
echo [INFO] Starte GUI...
echo.

:: Wechsle ins Verzeichnis der Batch-Datei (wichtig wenn von anderem Ort gestartet)
cd /d "%~dp0"

:: Starte DearPyGui ohne Bytecode-Cache (-B verhindert .pyc-Probleme)
python -B gui.py

echo.
echo [INFO] Python beendet mit Exit-Code: %errorlevel%
if %errorlevel% neq 0 (
    echo [FEHLER] Das Programm wurde unerwartet beendet!
)
pause
