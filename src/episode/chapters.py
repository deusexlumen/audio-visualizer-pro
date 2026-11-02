"""Kapiteltext fuer die YouTube-Beschreibung und Folgentitel aus dem Dateinamen."""

from pathlib import Path
from typing import List, Tuple

from ..types import Chapter

# YouTube-Regeln fuer Kapitel in der Beschreibung
YOUTUBE_MIN_CHAPTERS = 3
YOUTUBE_MIN_CHAPTER_SECONDS = 10.0


def format_timestamp(seconds: float) -> str:
    """Sekunden als M:SS (unter einer Stunde) bzw. H:MM:SS."""
    total = int(max(0.0, seconds))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def youtube_chapter_text(chapters: List[Chapter], total_duration: float,
                         intro_offset: float = 0.0) -> Tuple[str, List[str]]:
    """Baut den Kapiteltext und meldet Verstoesse gegen YouTubes Regeln.

    Die erste Zeile steht immer bei 0:00. Alle weiteren Zeiten werden um
    intro_offset verschoben (Hauptvideo beginnt nach dem Intro).
    """
    ordered = sorted(chapters, key=lambda c: c.start)
    warnings: List[str] = []
    if not ordered:
        return "", ["Keine Kapitel: YouTube zeigt keine Kapitel."]

    starts = [0.0] + [c.start + intro_offset for c in ordered[1:]]
    end_total = total_duration + intro_offset
    lines = []
    for i, (chapter, start) in enumerate(zip(ordered, starts)):
        lines.append(f"{format_timestamp(start)} {chapter.title}")
        end = starts[i + 1] if i + 1 < len(starts) else end_total
        if end - start < YOUTUBE_MIN_CHAPTER_SECONDS:
            warnings.append(
                f"Kapitel '{chapter.title}' ist kürzer als 10 s: YouTube zeigt "
                "dann keine Kapitel."
            )
    if len(ordered) < YOUTUBE_MIN_CHAPTERS:
        warnings.append("Weniger als 3 Kapitel: YouTube zeigt keine Kapitel.")
    return "\n".join(lines), warnings


def write_chapter_file(video_path: str, chapters: List[Chapter], total_duration: float,
                       intro_offset: float = 0.0) -> Tuple[Path, List[str]]:
    """Schreibt <video>.kapitel.txt neben das Video (UTF-8)."""
    text, warnings = youtube_chapter_text(chapters, total_duration, intro_offset)
    path = Path(video_path).with_suffix(".kapitel.txt")
    path.write_text(text + "\n", encoding="utf-8")
    return path, warnings


def default_episode_title(audio_path: str) -> str:
    """Folgentitel aus dem Dateinamen: Unterstriche werden Leerzeichen."""
    return " ".join(Path(audio_path).stem.replace("_", " ").split())
