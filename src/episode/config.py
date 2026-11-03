"""Konfiguration des Folgen-Looks und Render-Auftrag fuer eine Folge."""

from dataclasses import dataclass, field
from typing import List, Optional

from ..types import Chapter, Quote


@dataclass(frozen=True)
class EpisodeLookConfig:
    """Feste Werte des Looks (Spec 2026-11-02, §5 und §10)."""
    # Lesezeit-Regel (§5.4)
    chars_per_second: float = 15.0
    min_hold: float = 1.5
    gap_before_next: float = 0.3
    # Kapitel-Karte (§5.3)
    card_duration: float = 4.0
    card_fade: float = 0.4
    # Zitat-Buehne (§5.2)
    stage_fade: float = 0.8
    # Text
    text_fade_out: float = 0.35
    word_fade: float = 0.25
    # Kamerafahrt: hoechstens 8 % Zoom, damit das Bild scharf bleibt
    max_zoom: float = 1.08
    accent_color: str = "#F2B866"
    brand_text: str = "DEUS EX LUMEN"
    font_path: Optional[str] = None
    # KI-Plan
    quote_interval: float = 90.0
    min_chapters: int = 3
    max_chapters: int = 8


@dataclass
class EpisodeRenderSpec:
    """Alles, was der Renderer fuer eine Folge im neuen Look braucht."""
    image_path: str
    chapters: List[Chapter]
    quotes: List[Quote]
    title: str
    number: Optional[str] = None
    config: EpisodeLookConfig = field(default_factory=EpisodeLookConfig)


# Im Folgen-Look wirkt das Grading nur auf den Lichtstrich, der per Screen
# ueber das ganze Bild gelegt wird. Jede Anhebung von Schwarz (Helligkeit,
# Waerme, Kontrast < 1, LUT, Korn) wuerde das Bild selbst einfaerben.
_NEUTRAL_GRADING = {
    "contrast": 1.0, "saturation": 1.0, "brightness": 0.0, "warmth": 0.0,
    "film_grain": 0.0, "grain": 0.0, "vignette": 0.0, "chromatic_aberration": 0.0,
    "lut": None,
}


def episode_postprocess(postprocess: Optional[dict]) -> dict:
    """Postprocess-Werte fuer den Folgen-Look: Grading neutral, Bloom bleibt."""
    return {**(postprocess or {}), **_NEUTRAL_GRADING}
