"""Baut die Windows-onedir-Distribution via PyInstaller.

Nutzung:
    python build/build.py

Erwartet ein venv mit den exakten Versionen aus requirements.lock (kritisch
fuer numba/llvmlite) plus PyInstaller. Ergebnis landet in dist/AudioVisualizerPro/.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _version() -> str:
    # Regex statt tomllib/tomli, damit das Skript auch unter Python 3.10 laeuft.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("Version nicht in pyproject.toml gefunden.")
    return match.group(1)


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller fehlt. Installieren mit: pip install pyinstaller")
        return 1

    version = _version()
    print(f"[Build] Audio Visualizer Pro v{version} — PyInstaller-Build (onedir)")

    dist_dir = ROOT / "dist"
    build_cache = ROOT / "build" / "_pyinstaller_cache"
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(ROOT / "build" / "avp.spec"),
        "--noconfirm", "--clean",
        "--distpath", str(dist_dir),
        "--workpath", str(build_cache),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print("[Build] PyInstaller-Lauf fehlgeschlagen.")
        return result.returncode

    exe_path = dist_dir / "AudioVisualizerPro" / "AudioVisualizerPro.exe"
    if exe_path.exists():
        print(f"[Build] Fertig: {exe_path}")
        print(
            "[Build] Installer bauen mit: "
            f'ISCC build\\installer.iss /DMyAppVersion={version}'
        )
    else:
        print("[Build] Warnung: erwartete EXE nicht gefunden — Ausgabe pruefen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
