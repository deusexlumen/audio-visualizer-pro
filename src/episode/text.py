"""Text des Folgen-Looks: Marke, Zitat unten links, Buehnen-Satz, Kapitel-Karte.

Gezeichnet wird auf der CPU mit PIL. Jede Ebene wird einmal pro
(Element, Anzahl sichtbarer Woerter) bzw. pro Kapitel gerendert und
gecacht; pro Frame bleibt nur ein NumPy-Alpha-Blend im Textbereich.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import EpisodeLookConfig
from .timeline import EpisodeTimeline
from .timing import visible_word_count, word_reveal_times

_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
SERIF_FONT_PATH = _FONT_DIR / "SourceSerif4-Variable.ttf"
SANS_FONT_PATH = _FONT_DIR / "Inter-Variable.ttf"

TEXT_COLOR = (245, 241, 234)
BRAND_ALPHA = 0.75
CARD_COLOR = (7, 8, 11)


def _hex_rgb(value: str) -> Tuple[int, int, int]:
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def _font(path: Path, size: int, weight: str) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(path), max(8, int(round(size))))
    font.set_variation_by_name(weight)
    return font


def _greedy(words: List[str], measure: Callable[[str], int], width: int) -> List[List[str]]:
    lines: List[List[str]] = []
    cur: List[str] = []
    for w in words:
        if cur and measure(" ".join(cur + [w])) > width:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def wrap_balanced(words: List[str], measure: Callable[[str], int], max_width: int) -> List[str]:
    """Umbruch mit moeglichst gleich langen Zeilen und ohne Einzelwort am Ende."""
    if not words:
        return []
    count = len(_greedy(words, measure, max_width))
    # Kleinste Breite, die noch dieselbe Zeilenzahl ergibt: gleicht die Zeilen an
    lo, hi = max(measure(w) for w in words), max_width
    while lo < hi:
        mid = (lo + hi) // 2
        if len(_greedy(words, measure, mid)) <= count:
            hi = mid
        else:
            lo = mid + 1
    lines = _greedy(words, measure, max(lo, 1))
    if len(words) >= 4 and len(lines) >= 2 and len(lines[-1]) == 1 and len(lines[-2]) >= 3:
        moved = [lines[-2][-1]] + lines[-1]
        if measure(" ".join(moved)) <= max_width:
            lines[-2] = lines[-2][:-1]
            lines[-1] = moved
    return [" ".join(line) for line in lines]


@dataclass
class _Layer:
    """Gerenderte Ebene: Farbe (vormultipliziert) und Deckkraft an Position x, y."""
    rgb: np.ndarray    # float32 (h, w, 3), schon mit alpha multipliziert
    alpha: np.ndarray  # float32 (h, w, 1)
    x: int
    y: int


def _to_layer(img: Image.Image, x: int, y: int) -> _Layer:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    alpha = arr[:, :, 3:4]
    return _Layer(rgb=arr[:, :, :3] * 255.0 * alpha, alpha=alpha, x=x, y=y)


def _blend(frame: np.ndarray, layer: _Layer, amount: float) -> None:
    """Mischt die Ebene mit Deckkraft amount direkt in frame (uint8).

    Nur der Bereich der Ebene wird nach float gewandelt; das spart bei
    1080p den Grossteil der Zeit gegenueber einem Vollbild-Blend.
    """
    h, w = layer.alpha.shape[:2]
    x0, y0 = max(0, layer.x), max(0, layer.y)
    x1, y1 = min(frame.shape[1], layer.x + w), min(frame.shape[0], layer.y + h)
    if x1 <= x0 or y1 <= y0 or amount <= 0.0:
        return
    sx, sy = x0 - layer.x, y0 - layer.y
    a = layer.alpha[sy:sy + y1 - y0, sx:sx + x1 - x0] * amount
    c = layer.rgb[sy:sy + y1 - y0, sx:sx + x1 - x0] * amount
    region = frame[y0:y1, x0:x1].astype(np.float32)
    blended = region * (1.0 - a) + c + 0.5
    frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


@dataclass
class _WordBox:
    text: str
    x: float
    baseline: float
    color: Tuple[int, int, int]


class _TextBlock:
    """Feste Anordnung aller Woerter eines Elements; Ebenen je Wortanzahl."""

    def __init__(self, words: List[_WordBox], font, origin: Tuple[int, int],
                 size: Tuple[int, int], shadow_radius: float, shadow_offset: float,
                 shadow_opacity: float):
        self.words = words
        self.font = font
        self.origin = origin
        self.size = size
        self.shadow_radius = shadow_radius
        self.shadow_offset = shadow_offset
        self.shadow_opacity = shadow_opacity
        self._layers: Dict[int, _Layer] = {}

    def layer(self, n: int) -> _Layer:
        n = max(0, min(n, len(self.words)))
        if n not in self._layers:
            self._layers[n] = self._render(n)
        return self._layers[n]

    def _render(self, n: int) -> _Layer:
        w, h = self.size
        mask = Image.new("L", (w, h), 0)
        text = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dm, dt = ImageDraw.Draw(mask), ImageDraw.Draw(text)
        for word in self.words[:n]:
            dm.text((word.x, word.baseline + self.shadow_offset), word.text,
                    font=self.font, fill=255, anchor="ls")
            dt.text((word.x, word.baseline), word.text, font=self.font,
                    fill=word.color + (255,), anchor="ls")
        shadow = mask.filter(ImageFilter.GaussianBlur(self.shadow_radius))
        shadow = shadow.point(lambda v: int(v * self.shadow_opacity))
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        img.putalpha(shadow)
        img.alpha_composite(text)
        return _to_layer(img, *self.origin)


def _clean(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word).lower()


class EpisodeTextRenderer:
    """Zeichnet den Text des Folgen-Looks auf fertige RGB-Frames."""

    def __init__(self, timeline: EpisodeTimeline, width: int, height: int,
                 title: str, number: Optional[str],
                 cfg: Optional[EpisodeLookConfig] = None):
        self.timeline = timeline
        self.width, self.height = int(width), int(height)
        self.title = title or ""
        self.number = (number or "").strip() or None
        self.cfg = cfg or timeline.cfg
        self.accent = _hex_rgb(self.cfg.accent_color)
        serif = Path(self.cfg.font_path) if self.cfg.font_path else SERIF_FONT_PATH
        self._serif_path = serif
        self._reveal = [word_reveal_times(q) for q in timeline.items]
        self._blocks: Dict[int, _TextBlock] = {}
        self._cards: Dict[int, List[_Layer]] = {}
        self._brand = self._brand_layer()

    # --- Aufbau der Ebenen ---

    def _layout(self, words: List[str], font, max_width: int, line_height: float,
                colors: List[Tuple[int, int, int]], align: str,
                anchor_xy: Tuple[float, float], valign: str, shadow: Tuple[float, float, float]) -> _TextBlock:
        """Ordnet Woerter in Zeilen an; anchor_xy ist links unten bzw. Mitte."""
        measure = lambda s: int(round(font.getlength(s)))  # noqa: E731
        lines = wrap_balanced(words, measure, max_width)
        space = font.getlength(" ")
        ascent, descent = font.getmetrics()
        radius, offset, opacity = shadow
        pad = int(radius * 3 + offset + 2)
        block_w = max(measure(line) for line in lines)
        block_h = int(round(line_height * (len(lines) - 1) + ascent + descent))
        ax, ay = anchor_xy
        if align == "center":
            left = ax - block_w / 2
        else:
            left = ax
        top = ay - block_h / 2 if valign == "middle" else ay - block_h
        origin = (int(left) - pad, int(top) - pad)

        boxes: List[_WordBox] = []
        i = 0
        for k, line in enumerate(lines):
            lw = measure(line)
            x = pad + ((block_w - lw) / 2 if align == "center" else 0)
            baseline = pad + ascent + k * line_height
            for word in line.split():
                boxes.append(_WordBox(word, x, baseline, colors[i]))
                x += font.getlength(word) + space
                i += 1
        size = (block_w + 2 * pad, block_h + 2 * pad)
        return _TextBlock(boxes, font, origin, size, radius, offset, opacity)

    def _block(self, index: int) -> _TextBlock:
        if index in self._blocks:
            return self._blocks[index]
        # Nur das aktuelle Element im Cache halten
        self._blocks.clear()
        q = self.timeline.items[index]
        words = q.text.split()
        h, w = self.height, self.width
        if q.kind == "stage":
            size = 0.062 * h
            font = _font(SANS_FONT_PATH, size, "Bold")
            accent = _clean(q.accent_word or "")
            colors = [TEXT_COLOR] * len(words)
            for i, word in enumerate(words):
                if accent and _clean(word) == accent:
                    colors[i] = self.accent
                    break
            block = self._layout(words, font, int(0.72 * w), 1.2 * size, colors,
                                 "center", (w / 2, h / 2), "middle",
                                 (0.10 * size, 0.03 * size, 0.45))
        else:
            size = 0.042 * h
            font = _font(self._serif_path, size, "Regular")
            block = self._layout(words, font, int(0.64 * w), 1.35 * size,
                                 [TEXT_COLOR] * len(words), "left",
                                 (0.06 * w, 0.86 * h), "bottom",
                                 (0.15 * size, 0.04 * size, 0.40))
        self._blocks[index] = block
        return block

    def _brand_layer(self) -> _Layer:
        size = 0.02 * self.height
        font = _font(SANS_FONT_PATH, size, "SemiBold")
        words = self.cfg.brand_text.split()
        block = self._layout(words, font, self.width, 1.2 * size, [TEXT_COLOR] * len(words),
                             "left", (0.04 * self.width, 0.045 * self.height + size), "bottom",
                             (0.12 * size, 0.04 * size, 0.40))
        return block.layer(len(words))

    def _card_label(self) -> str:
        if self.number:
            return f"{self.cfg.brand_text} · FOLGE {self.number}"
        return self.cfg.brand_text

    def _card_layers(self, chapter_index: int) -> List[_Layer]:
        if chapter_index in self._cards:
            return self._cards[chapter_index]
        w, h = self.width, self.height
        # Dunkle Spalte links: deckend bis 28 %, Verlauf auf 0 bis 40 % der Breite
        col = int(np.ceil(0.40 * w))
        ramp = np.clip((0.40 * w - np.arange(col)) / (0.12 * w), 0.0, 1.0) * 0.88
        alpha = np.ascontiguousarray(
            np.broadcast_to(ramp[None, :, None], (h, col, 1)), dtype=np.float32)
        column = _Layer(rgb=np.array(CARD_COLOR, np.float32) * alpha, alpha=alpha, x=0, y=0)

        chapter = self.timeline.chapters[chapter_index]
        x = 0.06 * w
        col_w = int(0.30 * w)
        label_size, title_size, chap_size = 0.022 * h, 0.058 * h, 0.026 * h
        label_font = _font(SANS_FONT_PATH, label_size, "SemiBold")
        title_font = _font(self._serif_path, title_size, "Bold")
        chap_font = _font(SANS_FONT_PATH, chap_size, "Regular")
        shadow = (0.08 * title_size, 0.0, 0.3)

        title_words = (self.title or self.cfg.brand_text).split()
        title = self._layout(title_words, title_font, col_w, 1.15 * title_size,
                             [TEXT_COLOR] * len(title_words), "left",
                             (x, 0.5 * h), "middle", shadow)
        title_top = title.origin[1]
        title_bottom = title.origin[1] + title.size[1]
        label_words = self._card_label().split()
        label = self._layout(label_words, label_font, col_w, 1.2 * label_size,
                             [self.accent] * len(label_words), "left",
                             (x, title_top), "bottom", shadow)
        chap_words = f"Kapitel {chapter_index + 1} · {chapter.title}".split()
        chap = self._layout(chap_words, chap_font, col_w, 1.3 * chap_size,
                            [TEXT_COLOR] * len(chap_words), "left",
                            (x, title_bottom + 1.6 * chap_size), "bottom", shadow)
        layers = [column, label.layer(len(label_words)), title.layer(len(title_words)),
                  chap.layer(len(chap_words))]
        self._cards = {chapter_index: layers}
        return layers

    # --- Pro Frame ---

    def _text_amount(self, index: int, t: float) -> Tuple[int, float]:
        """Anzahl voll sichtbarer Woerter und Einblendanteil des neuesten Wortes."""
        times = self._reveal[index]
        n = visible_word_count(times, t)
        if n == 0:
            return 0, 0.0
        fade = self.cfg.word_fade
        f = 1.0 if fade <= 0 else min(1.0, max(0.0, (t - times[n - 1]) / fade))
        return n, f

    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        state = self.timeline.state_at(t)
        out = frame.copy()

        if state.card_amount > 0.0:
            # Karte des Kapitels, das gerade beginnt (auch waehrend des Einblendens)
            idx = self._card_chapter(t, max(0, state.chapter_index))
            for layer in self._card_layers(idx):
                _blend(out, layer, state.card_amount)

        _blend(out, self._brand, BRAND_ALPHA * (1.0 - state.card_amount))

        if state.text_index >= 0 and state.text_alpha > 0.0:
            block = self._block(state.text_index)
            n, f = self._text_amount(state.text_index, t)
            if n > 0:
                # Neuestes Wort weich dazu: Ebene n-1 und Ebene n mischen.
                # Beide haben dieselbe Groesse und Position.
                prev, new = block.layer(n - 1), block.layer(n)
                mixed = _Layer(rgb=prev.rgb + (new.rgb - prev.rgb) * f,
                               alpha=prev.alpha + (new.alpha - prev.alpha) * f,
                               x=new.x, y=new.y)
                _blend(out, mixed, state.text_alpha)

        return out

    def _card_chapter(self, t: float, fallback: int) -> int:
        """Index des Kapitels, dessen Kartenfenster t enthaelt."""
        for i, (a, b) in enumerate(self.timeline.card_windows):
            if a <= t <= b:
                return i
        return fallback
