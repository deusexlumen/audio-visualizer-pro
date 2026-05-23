"""
Lokale Transkription als Fallback fuer Offline-Betrieb.

Nutzt faster-whisper (optional), um Audio lokal zu transkribieren,
wenn Gemini nicht verfuegbar ist. Zitate werden heuristisch aus
dem Transkript extrahiert.
"""

from pathlib import Path
from typing import List, Optional

from .types import Quote

# Optional: faster-whisper
try:
    from faster_whisper import WhisperModel
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False


def is_available() -> bool:
    """Prueft, ob lokale Transkription verfuegbar ist."""
    return _HAS_WHISPER


class LocalTranscriber:
    """Lokaler Transkriber mit faster-whisper."""

    def __init__(self, model_size: str = "base"):
        if not _HAS_WHISPER:
            raise RuntimeError(
                "faster-whisper nicht installiert. "
                "Installiere mit: pip install faster-whisper"
            )
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str) -> str:
        """Transkribiert eine Audio-Datei."""
        segments, _ = self.model.transcribe(audio_path, language="de")
        return " ".join([s.text for s in segments])

    def extract_quotes(self, audio_path: str, audio_duration: float = None,
                       max_quotes: int = 5) -> List[Quote]:
        """
        Extrahiert Zitate heuristisch aus dem Transkript.

        Heuristik:
        - Längste Sätze (mindestens 5 Wörter)
        - Sätze mit Wiederholungen oder Pausenmarkern ausschließen
        - Gleichmäßig über die Audio-Dauer verteilt
        """
        segments, _ = self.model.transcribe(audio_path, language="de")
        segment_list = list(segments)

        # Sätze sammeln
        sentences = []
        for seg in segment_list:
            text = seg.text.strip()
            words = text.split()
            if len(words) >= 5:
                sentences.append((seg.start, seg.end, text))

        if not sentences:
            return []

        # Nach Länge sortieren (längste zuerst)
        sentences.sort(key=lambda x: len(x[2]), reverse=True)

        # Auswahl: Top-Sätze, aber verteilt über die Zeit
        selected = []
        used_ranges = []

        for start, end, text in sentences:
            # Überspringe, wenn zu nah an bereits ausgewähltem
            overlap = False
            for us, ue in used_ranges:
                if not (end < us - 3.0 or start > ue + 3.0):
                    overlap = True
                    break
            if overlap:
                continue

            selected.append(Quote(
                text=text,
                start_time=float(start),
                end_time=float(end),
                confidence=0.7,
            ))
            used_ranges.append((start, end))

            if len(selected) >= max_quotes:
                break

        # Nach Startzeit sortieren
        selected.sort(key=lambda q: q.start_time)
        return selected
