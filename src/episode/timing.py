"""Lesezeit-Regel und geschaetzte Wortzeiten fuer Zitate im Folgen-Look."""

import bisect
import re
from typing import List, Optional, Tuple

from ..types import Quote
from .config import EpisodeLookConfig

# Kuerzeste sichtbare Dauer, wenn das naechste Ereignis sehr nah folgt
_MIN_VISIBLE = 0.5


def display_window(quote: Quote, next_event_start: Optional[float],
                   cfg: EpisodeLookConfig) -> Tuple[float, float]:
    """Sichtbares Fenster eines Zitats (Spec §5.4).

    Sichtbar ab Sprechbeginn; steht danach so lange, bis die Lesezeit
    erreicht ist, mindestens aber min_hold nach Sprechende. Endet
    spaetestens gap_before_next vor dem naechsten Ereignis und ueberlappt
    es nie.
    """
    start = quote.start_time
    wanted = max(quote.end_time + cfg.min_hold,
                 start + len(quote.text) / cfg.chars_per_second)
    if next_event_start is None:
        return start, wanted
    n = next_event_start
    end = max(min(wanted, n - cfg.gap_before_next), min(start + _MIN_VISIBLE, n))
    return start, end


def word_reveal_times(quote: Quote) -> List[float]:
    """Einblendzeit je Wort, linear ueber die Zeichenposition geschaetzt.

    Gemini liefert nur Segmentzeiten; innerhalb des Zitats ist das eine
    Naeherung, keine wortgenaue Synchronitaet.
    """
    text = quote.text
    if not text:
        return []
    span = quote.end_time - quote.start_time
    return [quote.start_time + (m.start() / len(text)) * span
            for m in re.finditer(r"\S+", text)]


def visible_word_count(reveal_times: List[float], t: float) -> int:
    """Anzahl der Woerter, die zur Zeit t schon erschienen sind."""
    return bisect.bisect_right(reveal_times, t)
