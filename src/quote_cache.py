"""
Zitat-Persistenz & Cache fuer Audio Visualizer Pro.

Speichert extrahierte Zitate und Gemini Upload-IDs persistent,
so dass sie nicht bei jedem Neustart verloren gehen.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from .paths import user_data_dir
from .types import Quote


def _get_cache_dir() -> Path:
    """Gibt das Cache-Verzeichnis fuer Zitate zurueck."""
    cache_dir = user_data_dir("cache", "quotes")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_audio_hash(audio_path: str) -> str:
    """
    Berechnet einen Hash fuer eine Audio-Datei.
    Nutzt MD5 der ersten 1MB + Dateiname fuer Speed.
    """
    path = Path(audio_path)
    hasher = hashlib.md5()
    hasher.update(path.name.encode("utf-8"))
    hasher.update(str(path.stat().st_size).encode("utf-8"))
    hasher.update(str(path.stat().st_mtime).encode("utf-8"))
    # Erste 1MB fuer Content-Hash (verhindert Kollisionen bei Kopien)
    try:
        with open(path, "rb") as f:
            hasher.update(f.read(1024 * 1024))
    except Exception:
        pass
    return hasher.hexdigest()[:16]


def save_quotes(audio_path: str, quotes: List[Quote]) -> None:
    """Speichert Zitate als JSON-Cache."""
    if not quotes:
        return
    cache_dir = _get_cache_dir()
    h = get_audio_hash(audio_path)
    cache_file = cache_dir / f"{h}.json"
    data = [
        {
            "text": q.text,
            "start_time": q.start_time,
            "end_time": q.end_time,
            "confidence": q.confidence,
        }
        for q in quotes
    ]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_quotes(audio_path: str) -> Optional[List[Quote]]:
    """Laedt gecachte Zitate fuer eine Audio-Datei."""
    cache_dir = _get_cache_dir()
    h = get_audio_hash(audio_path)
    cache_file = cache_dir / f"{h}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            Quote(
                text=q.get("text", ""),
                start_time=float(q.get("start_time", 0.0)),
                end_time=float(q.get("end_time", 0.0)),
                confidence=float(q.get("confidence", 0.5)),
            )
            for q in data
        ]
    except Exception:
        return None


def save_transcript(audio_path: str, transcript: str) -> None:
    """Speichert ein vollstaendiges Transkript als Cache."""
    if not transcript or not transcript.strip():
        return
    cache_dir = _get_cache_dir()
    h = get_audio_hash(audio_path)
    cache_file = cache_dir / f"{h}_transcript.txt"
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(transcript)


def load_transcript(audio_path: str) -> Optional[str]:
    """Laedt ein gecachtes Transkript."""
    cache_dir = _get_cache_dir()
    h = get_audio_hash(audio_path)
    cache_file = cache_dir / f"{h}_transcript.txt"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def clear_quotes_cache(audio_path: str) -> None:
    """Loescht den Zitate-Cache fuer eine Audio-Datei."""
    cache_dir = _get_cache_dir()
    h = get_audio_hash(audio_path)
    for suffix in [".json", "_upload_id.txt", "_transcript.txt"]:
        f = cache_dir / f"{h}{suffix}"
        if f.exists():
            f.unlink()
    # Generische Result-Caches desselben Audios ebenfalls loeschen
    for f in cache_dir.glob(f"{h}_res_*.json"):
        try:
            f.unlink()
        except OSError:
            pass


def _result_key(*parts) -> str:
    """Baut einen kurzen, stabilen Schluessel aus beliebigen Bestandteilen."""
    hasher = hashlib.md5()
    for p in parts:
        hasher.update(str(p).encode("utf-8"))
    return hasher.hexdigest()[:16]


def save_json_result(audio_path: str, kind: str, signature: str, data) -> None:
    """Cacht ein beliebiges JSON-serialisierbares KI-Ergebnis.

    Der Schluessel bindet Audio-Hash, Ergebnis-Art (z.B. 'quotes', 'optimize')
    und eine Signatur (Modell + Prompt-Version + Parameter), damit ein Ergebnis
    nur bei identischen Eingaben wiederverwendet wird.
    """
    cache_dir = _get_cache_dir()
    h = get_audio_hash(audio_path) if audio_path else "noaudio"
    key = _result_key(kind, signature)
    cache_file = cache_dir / f"{h}_res_{kind}_{key}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except (OSError, TypeError) as e:  # nicht serialisierbar / kein Schreibzugriff
        return


def load_json_result(audio_path: str, kind: str, signature: str):
    """Laedt ein gecachtes KI-Ergebnis oder None, wenn keins existiert."""
    cache_dir = _get_cache_dir()
    h = get_audio_hash(audio_path) if audio_path else "noaudio"
    key = _result_key(kind, signature)
    cache_file = cache_dir / f"{h}_res_{kind}_{key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_upload_id(audio_path: str, upload_id: str) -> None:
    """Speichert eine Gemini Upload-ID mit Timestamp."""
    cache_dir = _get_cache_dir()
    h = get_audio_hash(audio_path)
    cache_file = cache_dir / f"{h}_upload_id.txt"
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(f"{upload_id}\n{time.time()}")


def load_upload_id(audio_path: str, max_age_hours: float = 24.0) -> Optional[str]:
    """
    Laedt eine gecachte Upload-ID, wenn sie nicht zu alt ist.
    """
    cache_dir = _get_cache_dir()
    h = get_audio_hash(audio_path)
    cache_file = cache_dir / f"{h}_upload_id.txt"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        upload_id = lines[0]
        if len(lines) >= 2:
            timestamp = float(lines[1])
            age_hours = (time.time() - timestamp) / 3600.0
            if age_hours > max_age_hours:
                cache_file.unlink()
                return None
        return upload_id
    except Exception:
        return None
