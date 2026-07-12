"""
Zentrale Pfad-Aufloesung fuer Entwicklung und PyInstaller-Frozen-Build.

Im Frozen-Build (onedir) liegt der Installationsordner typischerweise unter
Program Files bzw. einem Nutzerverzeichnis ohne Schreibrechte. Gebuendelte,
read-only Ressourcen (config-Defaults, Fonts, Icons) muessen relativ zu
sys._MEIPASS aufgeloest werden; beschreibbare Daten (Cache, Logs, generierte
Bilder, Nutzer-Rezepte) muessen zwingend ausserhalb des Install-Verzeichnisses
liegen, sonst schlaegt das Schreiben fehl oder ueberlebt eine Deinstallation
nicht sauber trennbar.
"""

import os
import sys
from pathlib import Path

APP_NAME = "AudioVisualizerPro"


def is_frozen() -> bool:
    """True, wenn als PyInstaller-Build (onedir/onefile) ausgefuehrt."""
    return getattr(sys, "frozen", False)


def resource_path(*parts: str) -> Path:
    """Pfad zu einer gebuendelten, read-only Ressource (Code, config-Defaults, assets)."""
    if is_frozen():
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent.parent
    return base.joinpath(*parts)


def user_data_dir(*parts: str) -> Path:
    """Beschreibbares, maschinenlokales Verzeichnis (Cache, Logs, generierte Bilder).

    Legt keine Verzeichnisse an — Aufrufer erledigt mkdir(), damit Schreibfehler
    (z.B. gesperrtes Profil) am gewohnten Ort abgefangen werden koennen.
    """
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) / APP_NAME if base else Path.home() / f".{APP_NAME.lower()}"
    return root.joinpath(*parts) if parts else root


def user_config_dir(*parts: str) -> Path:
    """Beschreibbares, roaming-faehiges Verzeichnis fuer Nutzerinhalte (Rezepte, Projekte).

    Legt keine Verzeichnisse an — Aufrufer erledigt mkdir().
    """
    base = os.environ.get("APPDATA")
    root = Path(base) / APP_NAME if base else Path.home() / f".{APP_NAME.lower()}"
    return root.joinpath(*parts) if parts else root
