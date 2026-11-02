"""Zeitachse des Folgen-Looks: was zu jedem Zeitpunkt zu sehen ist.

Reine Logik ohne GPU. Renderer, Text-Renderer und Vorschau fragen nur
state_at(t) ab; so sehen Vorschau und Export bei gleicher Zeit gleich aus.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..types import Chapter, Quote
from .config import EpisodeLookConfig
from .timing import display_window


@dataclass(frozen=True)
class CameraPose:
    """Ausschnitt des Bildes: Zoom >= 1 und Versatz in UV-Einheiten."""
    scale: float
    offset_x: float
    offset_y: float


@dataclass(frozen=True)
class EpisodeState:
    """Zustand eines Frames. -1 bei chapter_index/text_index = keins."""
    camera: CameraPose
    stage_amount: float
    card_amount: float
    chapter_index: int
    text_index: int
    text_alpha: float


# Schluesselposen als (Zoom-Anteil, Versatz-Anteil x, Versatz-Anteil y).
# Zoom-Anteil 0..1 von max_zoom, Versatz-Anteil -1..1 vom erlaubten Versatz.
# Folge: rein, links, rechts, raus — danach wieder von vorn.
_KEY_POSES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.4, 0.2),
    (0.6, -0.6, 0.0),
    (1.0, 0.6, -0.3),
]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _smoothstep(x: float) -> float:
    x = _clamp01(x)
    return x * x * (3.0 - 2.0 * x)


class EpisodeTimeline:
    """Berechnet Fenster, Karten und Kamerafahrt einer Folge einmal vorab."""

    def __init__(self, chapters: List[Chapter], quotes: List[Quote], duration: float,
                 cfg: Optional[EpisodeLookConfig] = None):
        self.cfg = cfg or EpisodeLookConfig()
        self.duration = float(duration)
        self.chapters = sorted(chapters, key=lambda c: c.start)
        self.items: List[Quote] = sorted(quotes, key=lambda q: q.start_time)

        fade = self.cfg.card_fade
        self.card_windows: List[Tuple[float, float]] = [
            (max(0.0, c.start - fade), c.start + self.cfg.card_duration)
            for c in self.chapters
        ]
        self.windows: List[Tuple[float, float]] = [
            display_window(q, self._next_event(i, q.start_time), self.cfg)
            for i, q in enumerate(self.items)
        ]
        self._key_times, self._key_poses = self._camera_keys()

    # --- Aufbau ---

    def _next_event(self, i: int, start: float) -> Optional[float]:
        """Fruehester Start eines folgenden Zitats oder einer folgenden Karte."""
        candidates = []
        if i + 1 < len(self.items):
            candidates.append(self.items[i + 1].start_time)
        candidates += [a for a, _ in self.card_windows if a > start]
        return min(candidates) if candidates else None

    def _camera_keys(self):
        """Schluesselzeiten an jeder Kapitelgrenze, plus Anfang und Ende."""
        times = [0.0]
        times += [c.start for c in self.chapters if 0.0 < c.start < self.duration]
        times.append(max(self.duration, times[-1] + 1e-3))
        poses = [self._pose(_KEY_POSES[k % len(_KEY_POSES)]) for k in range(len(times))]
        return times, poses

    def _pose(self, key) -> CameraPose:
        z, fx, fy = key
        scale = 1.0 + z * (self.cfg.max_zoom - 1.0)
        # Der sichtbare Ausschnitt hat die halbe Breite 0.5/scale; damit kein
        # Bildrand sichtbar wird, darf der Versatz (scale-1)/(2*scale) nicht
        # ueberschreiten.
        limit = (scale - 1.0) / (2.0 * scale)
        return CameraPose(scale, fx * limit, fy * limit)

    # --- Abfragen ---

    def _camera_at(self, t: float) -> CameraPose:
        times, poses = self._key_times, self._key_poses
        t = min(max(t, times[0]), times[-1])
        k = 0
        while k + 2 < len(times) and t >= times[k + 1]:
            k += 1
        a, b = poses[k], poses[k + 1]
        u = _smoothstep((t - times[k]) / (times[k + 1] - times[k]))
        return CameraPose(a.scale + (b.scale - a.scale) * u,
                          a.offset_x + (b.offset_x - a.offset_x) * u,
                          a.offset_y + (b.offset_y - a.offset_y) * u)

    def _card_amount(self, t: float) -> float:
        fade = self.cfg.card_fade
        best = 0.0
        for chapter, (a, b) in zip(self.chapters, self.card_windows):
            if a <= t <= b:
                ramp_in = 1.0 if chapter.start <= 0.0 else _clamp01((t - (chapter.start - fade)) / fade)
                ramp_out = _clamp01((b - t) / fade)
                best = max(best, min(ramp_in, ramp_out))
        return best

    def _chapter_index(self, t: float) -> int:
        idx = -1
        for i, c in enumerate(self.chapters):
            if c.start <= t:
                idx = i
        return idx

    def _stage_amount(self, t: float) -> float:
        fade = self.cfg.stage_fade
        best = 0.0
        for q, (start, end) in zip(self.items, self.windows):
            if q.kind != "stage":
                continue
            if start - fade <= t <= end + fade:
                amount = min(_clamp01((t - (start - fade)) / fade),
                             _clamp01((end + fade - t) / fade))
                best = max(best, amount)
        return best

    def _text_at(self, t: float) -> Tuple[int, float]:
        for i, (start, end) in enumerate(self.windows):
            if start <= t < end:
                alpha = _clamp01((end - t) / self.cfg.text_fade_out)
                return i, alpha
        return -1, 0.0

    def state_at(self, t: float) -> EpisodeState:
        card = self._card_amount(t)
        text_index, text_alpha = self._text_at(t)
        if card > 0.0:
            # Die Karte gewinnt immer gegen ein Zitat
            text_index, text_alpha = -1, 0.0
        return EpisodeState(
            camera=self._camera_at(t),
            stage_amount=self._stage_amount(t),
            card_amount=card,
            chapter_index=self._chapter_index(t),
            text_index=text_index,
            text_alpha=text_alpha,
        )

    def card_conflicts(self) -> List[int]:
        """Indizes der Zitate, deren Fenster eine Kapitel-Karte schneidet."""
        return [i for i, (s, e) in enumerate(self.windows)
                if any(s < b and a < e for a, b in self.card_windows)]
