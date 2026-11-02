"""KI-Plan einer Folge: Kapitel, Buehnen-Saetze und Zitate pruefen und verorten.

Die KI liefert nur Text und Segmentnummern. Alle Zeiten kommen hier lokal
aus dem Segment-Transkript (locate_in_segments), nie aus Zeitangaben der KI.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..quote_timing import locate_in_segments
from ..types import Chapter, Quote
from .config import EpisodeLookConfig

MAX_STAGE = 3
MIN_CHAPTER_GAP = 10.0

# Antwort-Schema des Plans (Gemini response_schema)
EPISODE_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_index": {"type": "integer"},
                    "title": {"type": "string"},
                },
                "required": ["segment_index", "title"],
            },
        },
        "stage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "accent_word": {"type": "string"},
                },
                "required": ["text", "accent_word"],
            },
        },
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["text", "confidence"],
            },
        },
    },
    "required": ["chapters", "stage", "quotes"],
}


@dataclass
class EpisodePlan:
    """Ergebnis der KI-Vorbereitung; quotes enthaelt Buehne und Zitate."""
    chapters: List[Chapter] = field(default_factory=list)
    quotes: List[Quote] = field(default_factory=list)


def episode_plan_prompt(segments: List[dict], k_min: int, k_max: int, n_quotes: int) -> str:
    """Prompt fuer den Plan. Der Massstab fuer Buehnen-Saetze ist Spec-Vorgabe."""
    numbered = "\n".join(f"[{i}] {seg.get('text', '')}" for i, seg in enumerate(segments))
    lines = [
        "Hier ist das Transkript einer Podcast-Folge, in nummerierten Segmenten:",
        "",
        f"<<<{numbered}>>>",
        "",
        f"1. Teile die Folge in {k_min}–{k_max} Kapitel nach Thema. Gib für jedes "
        "Kapitel die Nummer des Segments, mit dem es beginnt, und einen kurzen "
        "Namen (höchstens 5 Wörter).",
        "2. Wähle 2–3 Bühnen-Sätze: die kühnsten Behauptungen der Folge, die Sätze, "
        "die Leute zum Nachdenken bringen. Nicht die beste Zusammenfassung. Nenne "
        "je Satz ein Schlüsselwort aus dem Satz.",
        f"3. Wähle etwa {n_quotes} weitere Zitate (prägnant, höchstens 25 Wörter).",
        "",
        "Harte Regeln:",
        "- Jeder Feldwert 'text' muss WOERTLICH im Transkript stehen: exakt",
        "  dieselbe Zeichenfolge, keine Umformulierung, keine Kuerzung",
        "  mitten im Satz, keine Auslassungspunkte.",
        "- Beginne moeglichst am Satzanfang und ende an einem Satzende.",
        "- Ein Zitat und ein Bühnen-Satz dürfen sich nicht überschneiden.",
        "- Floskeln wie 'Also', 'Ja genau', 'Stimmt' sind keine Zitate.",
        "- 'confidence' ist deine Einschaetzung von 0.0 bis 1.0.",
    ]
    return "\n".join(lines)


def _words(text: str) -> List[str]:
    return [w for w in (re.sub(r"^\W+|\W+$", "", t) for t in text.split()) if w]


def _accent_word(text: str, wanted: str) -> Optional[str]:
    words = _words(text)
    if not words:
        return None
    wanted = (wanted or "").strip().lower()
    for w in words:
        if w.lower() == wanted:
            return w
    return max(words, key=len)


def _chapters(raw_chapters, segments, cfg) -> List[Chapter]:
    found = []
    for item in raw_chapters or []:
        if not isinstance(item, dict):
            continue
        idx = item.get("segment_index")
        title = str(item.get("title", "")).strip()
        if not isinstance(idx, int) or not (0 <= idx < len(segments)) or not title:
            continue
        found.append(Chapter(start=float(segments[idx].get("start", 0.0)), title=title))
    found.sort(key=lambda c: c.start)
    kept: List[Chapter] = []
    for c in found:
        if kept and c.start - kept[-1].start < MIN_CHAPTER_GAP:
            continue
        kept.append(c)
    kept = kept[: cfg.max_chapters]
    if kept:
        # Das erste Kapitel beginnt immer bei 0 (YouTube: erste Zeile 0:00)
        kept[0] = kept[0].model_copy(update={"start": 0.0})
    return kept


def _locate(text: str, segments, duration: Optional[float]):
    window = locate_in_segments(text, segments)
    if window is None:
        return None
    start, end = window
    if duration:
        end = min(end, float(duration))
    return (start, end) if end > start else None


def build_episode_plan(raw: dict, segments: List[dict], duration: Optional[float],
                       cfg: EpisodeLookConfig = EpisodeLookConfig()) -> EpisodePlan:
    """Prueft die KI-Antwort und baut daraus Kapitel und Zitate mit Zeiten."""
    raw = raw if isinstance(raw, dict) else {}
    chapters = _chapters(raw.get("chapters"), segments, cfg)

    stages: List[Quote] = []
    for item in raw.get("stage") or []:
        if len(stages) >= MAX_STAGE or not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        window = _locate(text, segments, duration) if text else None
        if window is None:
            continue
        stages.append(Quote(text=text, start_time=window[0], end_time=window[1],
                            kind="stage",
                            accent_word=_accent_word(text, str(item.get("accent_word", "")))))

    quotes: List[Quote] = []
    for item in raw.get("quotes") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        window = _locate(text, segments, duration) if text else None
        if window is None:
            continue
        start, end = window
        # Zitate, die eine Buehne (inkl. Nachlauf) schneiden, fallen weg
        if any(start < s.end_time + cfg.min_hold and s.start_time < end for s in stages):
            continue
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        quotes.append(Quote(text=text, start_time=start, end_time=end, confidence=confidence))

    items = sorted(stages + quotes, key=lambda q: q.start_time)
    return EpisodePlan(chapters=chapters, quotes=items)
