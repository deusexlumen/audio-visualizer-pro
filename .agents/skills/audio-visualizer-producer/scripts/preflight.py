#!/usr/bin/env python3
"""Prueft Voraussetzungen fuer die Audio Visualizer Pro Python-CLI."""
from __future__ import annotations
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

NEEDED = ("main.py", "requirements.lock", "src/gpu_renderer.py", "src/gpu_visualizers/__init__.py")
MODULES = ("click", "numpy", "scipy", "librosa", "moderngl", "glcontext", "pydantic", "soundfile", "PIL", "dotenv")

def check_tool(name: str) -> dict:
    path = shutil.which(name)
    if not path:
        return {"available": False, "path": None}
    try:
        r = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=10)
        return {"available": r.returncode == 0, "path": path}
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "path": path}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    errors = []
    repo = args.repo.expanduser().resolve()
    audio = args.audio.expanduser().resolve()
    if sys.version_info < (3, 10):
        errors.append("Python >=3.10 erforderlich")
    if not repo.is_dir():
        errors.append(f"Repository nicht gefunden: {repo}")
    else:
        for path in NEEDED:
            if not (repo / path).is_file():
                errors.append(f"Repository-Datei fehlt: {path}")
    if not audio.is_file() or audio.stat().st_size == 0:
        errors.append(f"Audiodatei fehlt oder ist leer: {audio}")
    if args.background and not args.background.is_file():
        errors.append(f"Hintergrund fehlt: {args.background}")
    modules = {name: importlib.util.find_spec(name) is not None for name in MODULES}
    errors.extend(f"Python-Paket fehlt: {name}" for name, found in modules.items() if not found)
    tools = {name: check_tool(name) for name in ("ffmpeg", "ffprobe")}
    errors.extend(f"{name} fehlt oder laeuft nicht" for name, state in tools.items() if not state["available"])
    result = {"ok": not errors, "repo": str(repo), "audio": str(audio), "tools": tools,
              "python_dependencies": modules, "failures": errors,
              "warnings": ["GPU/OpenGL-Kontext durch Preflight nicht getestet. Preview-Render erforderlich."]}
    body = json.dumps(result, ensure_ascii=False, indent=2)
    print(body)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(body + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
