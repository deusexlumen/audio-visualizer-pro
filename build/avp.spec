# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller-Spec fuer Audio Visualizer Pro (onedir-Build).

Aufruf: pyinstaller build/avp.spec --noconfirm --clean
(oder ueber build/build.py, das zusaetzlich die Version aus pyproject.toml zieht)

Onedir statt onefile: schnellerer Programmstart, numba/llvmlite und
librosa vertragen sich robuster mit einem echten Verzeichnis als mit der
onefile-Extraktion in ein temporaeres Verzeichnis bei jedem Start.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_dynamic_libs

ROOT = Path(SPECPATH).parent

block_cipher = None

# librosa nutzt lazy_loader -> PyInstallers statische Analyse findet nicht
# automatisch alle Submodule/Datenfiles. collect_all holt Code+Daten+Binaries.
librosa_datas, librosa_binaries, librosa_hidden = collect_all("librosa")

hiddenimports = (
    librosa_hidden
    + collect_submodules("src.gpu_visualizers")  # dynamisch via pkgutil entdeckte Module
    + ["numba", "llvmlite.binding", "soundfile", "moderngl", "glcontext"]
)

binaries = librosa_binaries + collect_dynamic_libs("soundfile")

datas = librosa_datas + [
    # Bundled Config-Presets (Rezepte/Farbprofile) + Rezept-Beispiele
    (str(ROOT / "config" / "*.json"), "config"),
    (str(ROOT / "config" / "recipes" / "*.json"), "config/recipes"),
    # GUI-Assets (Fonts, Icons, mitgelieferte Hintergruende)
    (str(ROOT / "assets" / "fonts"), "assets/fonts"),
    (str(ROOT / "assets" / "icons"), "assets/icons"),
    (str(ROOT / "assets" / "backgrounds"), "assets/backgrounds"),
]

# Ungenutzte Qt-Module ausschliessen (~100 MB Ersparnis) — die GUI verwendet
# nur QtCore/QtGui/QtWidgets/QtSvg.
excludes = [
    "PyQt6.QtQml", "PyQt6.QtQuick", "PyQt6.QtQuick3D",
    "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebChannel",
    "PyQt6.QtMultimedia", "PyQt6.QtMultimediaWidgets",
    "PyQt6.QtNetwork", "PyQt6.QtBluetooth", "PyQt6.QtPositioning",
    "PyQt6.QtSensors", "PyQt6.QtNfc", "PyQt6.QtSql", "PyQt6.QtTest",
    "PyQt6.QtDesigner", "PyQt6.QtHelp", "PyQt6.QtPdf", "PyQt6.QtCharts",
    "PyQt6.QtDataVisualization", "PyQt6.QtRemoteObjects", "PyQt6.QtSerialPort",
    "PyQt6.QtWebSockets", "PyQt6.QtOpenGL", "PyQt6.QtPrintSupport",
    "matplotlib", "IPython", "notebook", "tkinter",
]

a = Analysis(
    [str(ROOT / "gui.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "build" / "hooks" / "rthook_numba_cache.py")],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AudioVisualizerPro",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AudioVisualizerPro",
)
