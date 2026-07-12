"""PyInstaller-Runtime-Hook: leitet den numba-JIT-Cache in ein beschreibbares
Nutzerverzeichnis um, bevor numba (ueber librosa) importiert wird.

Der Install-Ordner ist im Frozen-Build read-only; ohne diese Umleitung
versucht numba, seinen Cache neben die .pyc-Dateien im Installationsordner
zu schreiben, und faellt (langsamer, aber funktional) auf In-Memory-JIT
zurueck. Schreibbarer Cache-Pfad vermeidet den Neu-Kompilierungs-Overhead
bei jedem Programmstart.

Notausgang, falls numba/llvmlite im Frozen-Build dennoch Probleme machen:
Umgebungsvariable NUMBA_DISABLE_JIT=1 setzen (Analyse ist NPZ-gecacht,
der reine Python-Fallback ist nur beim ersten unge-cachten Analyse-Lauf
spuerbar langsamer).
"""

import os
from pathlib import Path

_base = os.environ.get("LOCALAPPDATA")
_root = Path(_base) / "AudioVisualizerPro" if _base else Path.home() / ".audiovisualizerpro"
_cache_dir = _root / "numba_cache"

try:
    _cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache_dir))
except OSError:
    pass
