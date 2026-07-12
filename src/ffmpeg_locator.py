"""
FFmpeg-Locator fuer Audio Visualizer Pro.

FFmpeg wird NICHT gebuendelt (Lizenzfragen je nach Build-Konfiguration,
~90 MB Bloat). Stattdessen wird bei Bedarf gesucht:
1. System-PATH (shutil.which) — der Normalfall bei Entwicklung/manueller Installation.
2. Lokales App-Datenverzeichnis (%LOCALAPPDATA%/AudioVisualizerPro/ffmpeg/) —
   Ziel eines vorherigen Downloads.
3. Download nach Nutzer-Zustimmung (GUI zeigt einen Dialog, CLI bricht mit
   klarer Fehlermeldung ab) — siehe download_ffmpeg().

Alle subprocess-Aufrufer im Projekt rufen get_ffmpeg_path() statt den
String "ffmpeg" hart zu codieren, damit Downloads aus dem lokalen
Verzeichnis automatisch gefunden werden.
"""

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from .app_logging import get_logger
from .paths import user_data_dir

logger = get_logger(__name__)

# gyan.dev "essentials"-Build: schlank, x64, regelmaessig aktualisiert.
FFMPEG_DOWNLOAD_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
# Der Hash aendert sich bei jedem Release, daher nicht hartcodiert — stattdessen
# wird (falls vorhanden) die Sidecar-Pruefsummendatei vom selben Server geholt.
FFMPEG_SHA256_URL = FFMPEG_DOWNLOAD_URL + ".sha256"

def _local_install_dir() -> Path:
    return user_data_dir("ffmpeg")


def _find_local_exe(binary: str) -> Optional[Path]:
    install_dir = _local_install_dir()
    if not install_dir.exists():
        return None
    exe_name = f"{binary}.exe" if os.name == "nt" else binary
    matches = list(install_dir.rglob(exe_name))
    return matches[0] if matches else None


def _find_binary(binary: str, cache: dict) -> Optional[str]:
    cached = cache.get(binary)
    if cached and Path(cached).exists():
        return cached

    on_path = shutil.which(binary)
    if on_path:
        cache[binary] = on_path
        return on_path

    local = _find_local_exe(binary)
    if local:
        cache[binary] = str(local)
        return cache[binary]

    return None


_cache: dict = {}


def find_ffmpeg() -> Optional[str]:
    """Sucht FFmpeg im PATH oder im lokalen App-Datenverzeichnis. Kein Download."""
    return _find_binary("ffmpeg", _cache)


def find_ffprobe() -> Optional[str]:
    """Sucht ffprobe (liegt im selben Download-Archiv wie ffmpeg). Kein Download."""
    return _find_binary("ffprobe", _cache)


def get_ffmpeg_path() -> str:
    """Liefert den FFmpeg-Pfad. Wirft FileNotFoundError mit deutscher Meldung,
    falls FFmpeg weder im PATH noch lokal installiert ist."""
    found = find_ffmpeg()
    if found:
        return found
    raise FileNotFoundError(
        "FFmpeg wurde nicht gefunden (weder im PATH noch unter "
        f"{_local_install_dir()}). Bitte FFmpeg installieren oder in der "
        "GUI herunterladen lassen."
    )


def get_ffprobe_path() -> str:
    """Liefert den ffprobe-Pfad. Wirft FileNotFoundError mit deutscher Meldung,
    falls ffprobe weder im PATH noch lokal installiert ist."""
    found = find_ffprobe()
    if found:
        return found
    raise FileNotFoundError(
        "ffprobe wurde nicht gefunden (weder im PATH noch unter "
        f"{_local_install_dir()}). Bitte FFmpeg (enthaelt ffprobe) installieren "
        "oder in der GUI herunterladen lassen."
    )


def download_ffmpeg(
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Laedt FFmpeg herunter, prueft die Pruefsumme (falls verfuegbar) und
    entpackt es in das lokale App-Datenverzeichnis.

    progress_callback(bytes_done, bytes_total) wird waehrend des Downloads
    wiederholt aufgerufen (total kann 0 sein, wenn der Server keine
    Content-Length liefert).

    Gibt den Pfad zur entpackten ffmpeg.exe zurueck. Wirft bei Netzwerk-,
    Pruefsummen- oder Archivfehlern — niemals stillschweigender Fallback.
    """
    install_dir = _local_install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)
    zip_path = install_dir / "ffmpeg_download.zip"

    logger.info(f"[FFmpeg] Lade herunter von {FFMPEG_DOWNLOAD_URL}")
    try:
        _download_with_progress(FFMPEG_DOWNLOAD_URL, zip_path, progress_callback)

        expected_sha256 = _fetch_expected_sha256()
        if expected_sha256:
            actual = _sha256_of(zip_path)
            if actual.lower() != expected_sha256.lower():
                raise ValueError(
                    f"Pruefsumme stimmt nicht ueberein (erwartet {expected_sha256}, "
                    f"erhalten {actual}). Download wurde verworfen."
                )
        else:
            logger.warning(
                "[FFmpeg] Keine Pruefsumme vom Server verfuegbar — "
                "nur Archiv-Integritaet wird geprueft."
            )

        with zipfile.ZipFile(zip_path) as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                raise ValueError(f"Korruptes Archiv (Datei defekt: {bad_file}).")
            zf.extractall(install_dir)
    finally:
        zip_path.unlink(missing_ok=True)

    # Cache invalidieren, damit find_ffmpeg()/find_ffprobe() den frischen Download sehen.
    _cache.pop("ffmpeg", None)
    _cache.pop("ffprobe", None)
    exe = _find_local_exe("ffmpeg")
    if exe is None:
        raise FileNotFoundError("FFmpeg-Archiv entpackt, aber keine ffmpeg.exe gefunden.")
    _cache["ffmpeg"] = str(exe)
    logger.info(f"[FFmpeg] Installiert unter {exe}")
    return _cache["ffmpeg"]


def _download_with_progress(url: str, dest: Path, progress_callback) -> None:
    with urllib.request.urlopen(url, timeout=30) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total)


def _fetch_expected_sha256() -> Optional[str]:
    try:
        with urllib.request.urlopen(FFMPEG_SHA256_URL, timeout=10) as response:
            text = response.read().decode("utf-8", errors="ignore").strip()
            return text.split()[0] if text else None
    except Exception:
        return None


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()
