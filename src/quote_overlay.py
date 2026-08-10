"""
Quote Overlay Renderer fuer Audio Visualizer Pro.

Zeichnet Key-Zitate als elegante Text-Overlays auf gerenderte Frames.
Zeitbasiert mit Fade-In/Out Animationen.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import textwrap
import threading

from pathlib import Path
from .gemini_integration import Quote


def _normalize_color(value):
    """Normalisiert eine Farbe zu einem RGB(A)-Tupel.

    JSON-Configs liefern Farben als Hex-String ("#RRGGBB"/"#RRGGBBAA")
    oder als Liste; die Runtime erwartet Tupel von Ints.
    """
    if isinstance(value, str):
        v = value.lstrip("#")
        if len(v) not in (6, 8):
            raise ValueError(f"Ungueltige Hex-Farbe: {value!r}")
        comps = [int(v[i:i + 2], 16) for i in range(0, len(v), 2)]
        return tuple(comps)
    return tuple(value)


@dataclass
class QuoteOverlayConfig:
    """Konfiguration fuer Quote Overlays."""
    enabled: bool = True
    font_size: int = 52
    font_color: tuple = (255, 255, 255)  # RGB - Weiss
    box_color: tuple = (26, 26, 46, 200)  # RGBA - Dunkelblau, halbtransparent
    box_alpha: int = 200  # Alternative Alpha-Steuerung fuer box_color
    box_padding: int = 32
    box_radius: int = 16
    box_margin_bottom: int = 100  # Abstand vom unteren Rand
    max_width_ratio: float = 0.75  # Max 75% der Bildbreite
    fade_duration: float = 0.6    # Sekunden fuer Fade-In/Out
    shadow_color: tuple = (0, 0, 0, 120)
    shadow_offset: tuple = (3, 3)
    line_spacing: int = 10
    max_chars_per_line: int = 40
    display_duration: float = 8.0  # Maximale Anzeigedauer pro Zitat in Sekunden
    position: str = "bottom"      # 'bottom', 'center', 'top'
    font_path: Optional[str] = None  # Benutzerdefinierte Schriftart
    text_align: str = "center"    # 'left', 'center', 'right'
    
    # Auto-Scaling
    auto_scale_font: bool = True
    min_font_size: int = 16
    max_font_size: int = 72
    
    # Text-Schatten fuer bessere Lesbarkeit
    text_shadow_enabled: bool = True
    text_shadow_color: tuple = (0, 0, 0, 180)
    text_shadow_offset: tuple = (2, 2)
    text_shadow_blur: float = 2.0
    
    # Box-Design
    box_gradient: bool = True       # Subtiler vertikaler Gradient
    accent_line: bool = True        # Dünne Accent-Linie oben
    accent_line_color: tuple = (255, 200, 100, 255)  # Warmes Gold
    accent_line_height: int = 3
    
    # Spatial Frequency Compensation (Hintergrund-Blur unter dem Text)
    spatial_compensation: bool = True
    compensation_blur: float = 12.0
    compensation_darken: float = 0.55
    
    # Audio-Visual Sync & Latenz-Kompensation
    latency_offset: float = 0.0       # Sekunden (negativ = frueher, positiv = spaeter)
    buffer_lookahead: float = 2.0     # Sekunden Prefetch fuer asynchrone Streams
    
    # Position & Skalierung
    offset_x: int = 0       # Horizontaler Offset in Pixeln (negativ = links, positiv = rechts)
    offset_y: int = 0       # Vertikaler Offset in Pixeln (negativ = oben, positiv = unten)
    scale: float = 1.0      # Skalierungsfaktor (0.5 = halbe Groesse, 2.0 = doppelte Groesse)

    def __post_init__(self):
        # Farben aus JSON-Configs (Hex-String/Liste) in RGB(A)-Tupel normalisieren
        self.font_color = _normalize_color(self.font_color)
        self.box_color = _normalize_color(self.box_color)
        self.shadow_color = _normalize_color(self.shadow_color)
        self.text_shadow_color = _normalize_color(self.text_shadow_color)
        self.accent_line_color = _normalize_color(self.accent_line_color)


class QuoteOverlayRenderer:
    """
    Rendert elegante Quote-Overlays auf Video-Frames.
    
    Features:
    - Zeitbasierte Anzeige (Start/End Zeit aus Quote)
    - Sanftes Fade-In/Out
    - Abgerundete Hintergrund-Box mit Schatten
    - Automatischer Zeilenumbruch
    - Zentrierte Position unten im Bild
    """
    
    def __init__(self, quotes: Optional[List[Quote]] = None, 
                 config: Optional[QuoteOverlayConfig] = None):
        self.quotes = quotes or []
        self.config = config or QuoteOverlayConfig()
        self._font = None
        self._font_path = None
        self._load_font()
        
        # Frame-synchroner Buffer fuer asynchrone KI-Datenstroeme
        self._lock = threading.Lock()
        self._frame_index = None
        self._frame_count = 0
        self._fps = 30
        self._dirty = True

        # Overlay-Cache: Das gerenderte Text-Overlay ist fuer ein Zitat ueber
        # hunderte Frames identisch (nur der Fade-Alpha aendert sich).
        # Key: (text, bildgroesse, config-fingerprint) -> vorgerendertes Overlay.
        self._overlay_cache: Dict[Any, dict] = {}
        self._overlay_cache_max = 8
    
    def _load_font(self, size: int = None):
        """Laedt eine Schriftart mit Fallback."""
        font_size = size or self.config.font_size
        # Benutzerdefinierte Schriftart zuerst probieren
        if self.config.font_path and Path(self.config.font_path).exists():
            try:
                self._font = ImageFont.truetype(self.config.font_path, font_size)
                return
            except (OSError, IOError):
                pass
        
        font_paths = [
            # Windows
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            # macOS
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        
        for path in font_paths:
            try:
                self._font = ImageFont.truetype(path, font_size)
                self._font_path = path
                return
            except (OSError, IOError):
                continue
        
        # Fallback auf Default-Schrift
        self._font = ImageFont.load_default()
        self._font_path = None
    
    def _get_font(self):
        """Gibt den Font zurueck, laedt bei Bedarf neu wenn sich Groesse oder Skalierung aendern."""
        config_scale = getattr(self.config, 'scale', 1.0)
        effective_size = max(1, int(self.config.font_size * config_scale))
        current_key = (effective_size, self.config.font_path)
        if not hasattr(self, '_font_cache_key') or self._font_cache_key != current_key:
            self._font_cache_key = current_key
            self._load_font(size=effective_size)
        return self._font
    
    def set_latency_offset(self, offset: float):
        """Setzt die Latenz-Kompensation (Sekunden). Negativ = frueher, Positiv = spaeter."""
        self.config.latency_offset = offset
        self._dirty = True
    
    def add_quote(self, quote: Quote):
        """Thread-safes Hinzufuegen eines Zitats waehrend des Renderns."""
        with self._lock:
            self.quotes.append(quote)
            self._dirty = True
    
    def _effective_end_time(self, quote: Quote) -> float:
        """
        Berechnet das effektive Ende eines Zitats.
        Respektiert quote.end_time, aber display_duration als Obergrenze.
        """
        display_end = quote.start_time + self.config.display_duration
        return min(quote.end_time, display_end)

    def build_frame_index(self, frame_count: int, fps: int):
        """
        Baut einen vorberechneten Frame-Index fuer O(1) Lookups.
        Muss vor dem Render-Loop einmalig aufgerufen werden.
        """
        with self._lock:
            self._frame_count = frame_count
            self._fps = fps
            self._frame_index = [[] for _ in range(frame_count)]

            for quote in self.quotes:
                adj_start = quote.start_time + self.config.latency_offset
                effective_end = self._effective_end_time(quote)
                adj_end = effective_end + self.config.latency_offset
                start_frame = max(0, min(int(adj_start * fps), frame_count - 1))
                end_frame = max(0, min(int(adj_end * fps), frame_count - 1))

                for f in range(start_frame, end_frame + 1):
                    self._frame_index[f].append(quote)

            self._dirty = False
    
    def _get_active_quote(self, time_seconds: float, frame_idx: int = None) -> Optional[Quote]:
        """
        Findet das aktuell aktive Zitat fuer eine gegebene Zeit.
        
        Args:
            time_seconds: Aktuelle Zeit im Video
            frame_idx: Optionaler Frame-Index fuer schnellen O(1) Lookup
            
        Beruecksichtigt Latenz-Kompensation und maximale Anzeigedauer.
        """
        # Schneller Frame-Index-Pfad (O(1))
        if frame_idx is not None and self._frame_index is not None and not self._dirty:
            if 0 <= frame_idx < self._frame_count:
                candidates = self._frame_index[frame_idx]
                for quote in candidates:
                    effective_end = self._effective_end_time(quote)
                    adj_start = quote.start_time + self.config.latency_offset
                    adj_end = effective_end + self.config.latency_offset
                    if adj_start <= time_seconds <= adj_end:
                        return quote
                return None

        # Fallback: lineare Suche (kompatibel mit alten Aufrufen)
        for quote in self.quotes:
            effective_end = self._effective_end_time(quote)
            adj_start = quote.start_time + self.config.latency_offset
            adj_end = effective_end + self.config.latency_offset
            if adj_start <= time_seconds <= adj_end:
                return quote
        return None
    
    def _calculate_fade_alpha(self, time_seconds: float,
                             quote: Quote) -> float:
        """
        Berechnet den Fade-Alpha-Wert (0.0 - 1.0) fuer ein Zitat.

        Fade-In am Anfang, Fade-Out am Ende.
        Beruecksichtigt Latenz-Kompensation und maximale Anzeigedauer.
        """
        fade = self.config.fade_duration
        effective_end = self._effective_end_time(quote)
        latency = self.config.latency_offset
        adj_start = quote.start_time + latency
        adj_end = effective_end + latency
        
        # Am Anfang: Fade-In (mit Latenz-Kompensation)
        if time_seconds < adj_start + fade:
            progress = (time_seconds - adj_start) / fade
            return max(0.0, min(1.0, progress))
        
        # Am Ende: Fade-Out (mit Latenz-Kompensation)
        elif time_seconds > adj_end - fade:
            progress = (adj_end - time_seconds) / fade
            return max(0.0, min(1.0, progress))
        
        # In der Mitte: Voll sichtbar
        return 1.0
    
    def _wrap_text(self, text: str) -> List[str]:
        """Bricht Text in mehrere Zeilen um."""
        return textwrap.wrap(
            text, 
            width=self.config.max_chars_per_line,
            break_long_words=False,
            replace_whitespace=False
        )
    
    def _calculate_text_size(self, lines: List[str]) -> tuple:
        """Berechnet die Groesse des Text-Blocks."""
        if not lines:
            return (0, 0)
        
        font = self._get_font()
        # PIL 10.0+ nutzt getbbox, aeltere getsize
        if hasattr(font, 'getbbox'):
            max_width = 0
            total_height = 0
            for line in lines:
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                max_width = max(max_width, line_width)
                total_height += line_height
            # Zeilenabstand hinzufuegen
            total_height += (len(lines) - 1) * self.config.line_spacing
            return (max_width, total_height)
        else:
            # Fallback fuer aeltere PIL Versionen
            max_width = max(font.getsize(line)[0] for line in lines)
            line_height = font.getsize(lines[0])[1]
            total_height = len(lines) * line_height + (len(lines) - 1) * self.config.line_spacing
            return (max_width, total_height)
    
    def _config_fingerprint(self) -> tuple:
        """Fingerprint aller Config-Felder, die das gerenderte Overlay beeinflussen."""
        c = self.config
        return (
            c.font_size, tuple(c.font_color), tuple(c.box_color), c.box_alpha,
            c.box_padding, c.box_radius, c.box_margin_bottom, c.max_width_ratio,
            tuple(c.shadow_color), tuple(c.shadow_offset), c.line_spacing,
            c.max_chars_per_line, c.position, c.font_path, c.text_align,
            c.text_shadow_enabled, tuple(c.text_shadow_color),
            c.box_gradient, c.accent_line, tuple(c.accent_line_color),
            c.accent_line_height,
            getattr(c, 'offset_x', 0), getattr(c, 'offset_y', 0),
            getattr(c, 'scale', 1.0),
        )

    def _get_cached_overlay(self, text: str, size: tuple) -> Optional[dict]:
        """Liefert das vorgerenderte Overlay fuer ein Zitat (mit Cache).

        Returns:
            Dict mit 'rgb' (float32 HxWx3), 'alpha' (float32 HxW, 0-1),
            'pos' (x1, y1) und 'comp_rect' (Bereich fuer die
            Hintergrund-Kompensation) — oder None, wenn kein Text vorhanden.
        """
        key = (text, size, self._config_fingerprint())
        cached = self._overlay_cache.get(key)
        if cached is not None:
            return cached

        entry = self._build_overlay(text, size)
        if entry is None:
            return None
        # Einfache Groessenbegrenzung: aeltesten Eintrag verwerfen
        if len(self._overlay_cache) >= self._overlay_cache_max:
            self._overlay_cache.pop(next(iter(self._overlay_cache)))
        self._overlay_cache[key] = entry
        return entry

    def _build_overlay(self, text: str, size: tuple) -> Optional[dict]:
        """Rendert das komplette Overlay (Schatten, Box, Text) EINMAL bei Alpha=1.

        Der Rueckgabewert wird im Render-Loop nur noch per NumPy mit dem
        aktuellen Fade-Alpha auf den Frame gemischt — das erspart die beiden
        GaussianBlurs und die Gradient-Schleife pro Frame.
        """
        img_width, img_height = size
        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Text umbrechen
        lines = self._wrap_text(text)
        if not lines:
            return None

        # Groessen berechnen
        text_width, text_height = self._calculate_text_size(lines)
        padding = self.config.box_padding
        box_width = min(text_width + 2 * padding, int(img_width * self.config.max_width_ratio))
        box_height = text_height + 2 * padding

        # Offset und Skalierung anwenden
        config_scale = getattr(self.config, 'scale', 1.0)
        offset_x = getattr(self.config, 'offset_x', 0)
        offset_y = getattr(self.config, 'offset_y', 0)

        box_width = int(box_width * config_scale)
        box_height = int(box_height * config_scale)
        padding = int(padding * config_scale)
        box_radius = int(self.config.box_radius * config_scale)

        # Box positionieren je nach Einstellung (mit Offset)
        if self.config.position == "bottom":
            box_x = (img_width - box_width) // 2 + offset_x
            box_y = img_height - box_height - self.config.box_margin_bottom + offset_y
        elif self.config.position == "top":
            box_x = (img_width - box_width) // 2 + offset_x
            box_y = self.config.box_margin_bottom + offset_y
        else:  # center
            box_x = (img_width - box_width) // 2 + offset_x
            box_y = (img_height - box_height) // 2 + offset_y

        alpha = 1.0  # Cache wird bei voller Deckkraft gebaut, Fade kommt in apply()

        # === WEICHER BOX-SCHATTEN ===
        shadow = self.config.shadow_offset
        scaled_shadow = (int(shadow[0] * config_scale), int(shadow[1] * config_scale))
        shadow_rect = [
            box_x + scaled_shadow[0], box_y + scaled_shadow[1],
            box_x + box_width + scaled_shadow[0], box_y + box_height + scaled_shadow[1]
        ]
        # Shadow-Layer erstellen und weichzeichnen
        shadow_layer = Image.new('RGBA', size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        shadow_draw.rounded_rectangle(
            shadow_rect,
            radius=box_radius,
            fill=self.config.shadow_color
        )
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=max(2, int(6 * config_scale))))
        overlay = Image.alpha_composite(overlay, shadow_layer)
        draw = ImageDraw.Draw(overlay)  # Neuen Draw nach Shadow-Composite
        
        # === HINTERGRUND-BOX MIT GRADIENT ===
        base_color = list(self.config.box_color)
        if len(base_color) < 4:
            base_color = list(self.config.box_color) + [self.config.box_alpha]
        base_color[3] = int(base_color[3] * alpha)
        
        if self.config.box_gradient:
            # Subtiler vertikaler Gradient: oben 10% heller
            grad_layer = Image.new('RGBA', (box_width, box_height), (0, 0, 0, 0))
            grad_draw = ImageDraw.Draw(grad_layer)
            for y in range(box_height):
                t = y / box_height
                factor = 1.0 + (1.0 - t) * 0.10
                r = min(255, int(base_color[0] * factor))
                g = min(255, int(base_color[1] * factor))
                b = min(255, int(base_color[2] * factor))
                grad_draw.line([(0, y), (box_width, y)], fill=(r, g, b, 255))
            # Abgerundete Maske mit Fade-Alpha skalieren
            mask = Image.new('L', (box_width, box_height), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rounded_rectangle([0, 0, box_width, box_height], radius=box_radius, fill=255)
            mask = mask.point(lambda a: int(a * base_color[3] / 255.0))
            grad_layer.putalpha(mask)
            overlay.paste(grad_layer, (int(box_x), int(box_y)), grad_layer)
        else:
            draw.rounded_rectangle(
                [box_x, box_y, box_x + box_width, box_y + box_height],
                radius=box_radius,
                fill=tuple(base_color)
            )
        
        # === ACCENT-LINIE OBEN ===
        if self.config.accent_line:
            accent_color = list(self.config.accent_line_color)
            if len(accent_color) < 4:
                accent_color = accent_color + [255]
            accent_color[3] = int(accent_color[3] * alpha)
            line_h = max(1, int(self.config.accent_line_height * config_scale))
            line_pad = int(padding * 0.6)
            # Runde die Ecken der Accent-Linie
            draw.rounded_rectangle(
                [box_x + line_pad, box_y + 3,
                 box_x + box_width - line_pad, box_y + 3 + line_h],
                radius=line_h // 2,
                fill=tuple(accent_color)
            )
        
        # === TEXT MIT SCHATTEN ===
        font_color = list(self.config.font_color)[:3] + [int(255 * alpha)]
        font = self._get_font()
        
        # Text-Schatten vorbereiten
        text_shadow_enabled = self.config.text_shadow_enabled
        ts_color = None
        if text_shadow_enabled:
            ts_color = list(self.config.text_shadow_color)
            if len(ts_color) < 4:
                ts_color = ts_color + [180]
            ts_color[3] = int(ts_color[3] * alpha)
        
        # Vertikale Zentrierung des Text-Blocks in der Box
        if hasattr(font, 'getbbox'):
            line_heights = []
            for line in lines:
                bbox = font.getbbox(line)
                line_heights.append(bbox[3] - bbox[1])
            total_text_height = sum(line_heights) + (len(lines) - 1) * self.config.line_spacing
        else:
            line_height = font.getsize(lines[0])[1]
            total_text_height = len(lines) * line_height + (len(lines) - 1) * self.config.line_spacing
        
        text_start_y = box_y + (box_height - total_text_height) // 2
        current_y = text_start_y
        
        for i, line in enumerate(lines):
            # Horizontale Zentrierung jeder Zeile
            if hasattr(font, 'getbbox'):
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height_actual = bbox[3] - bbox[1]
            else:
                line_width, line_height_actual = font.getsize(line)
            
            if self.config.text_align == "center":
                line_x = box_x + (box_width - line_width) // 2
            elif self.config.text_align == "right":
                line_x = box_x + box_width - line_width - padding
            else:  # left
                line_x = box_x + padding
            
            # Text-Schatten: mehrere leicht versetzte Passen fuer weichen Schatten
            if text_shadow_enabled and ts_color is not None:
                for dx, dy in [(1, 1), (2, 2), (1, 2), (2, 1)]:
                    draw.text(
                        (line_x + dx, current_y + dy),
                        line, font=font, fill=tuple(ts_color)
                    )
            
            draw.text((line_x, current_y), line, font=font, fill=tuple(font_color))
            current_y += line_height_actual + self.config.line_spacing

        # Auf den relevanten Ausschnitt zuschneiden (Box + Schatten + Blur-Rand),
        # damit das Per-Frame-Blending nur eine kleine Region beruehrt.
        margin = int(16 * config_scale) + max(abs(scaled_shadow[0]), abs(scaled_shadow[1]))
        x1 = max(0, int(box_x) - margin)
        y1 = max(0, int(box_y) - margin)
        x2 = min(img_width, int(box_x + box_width) + margin)
        y2 = min(img_height, int(box_y + box_height) + margin)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = np.asarray(overlay.crop((x1, y1, x2, y2)), dtype=np.float32)

        # Bereich fuer die Hintergrund-Kompensation (nur die Box selbst)
        cx1 = max(0, int(box_x))
        cy1 = max(0, int(box_y))
        cx2 = min(img_width, int(box_x + box_width))
        cy2 = min(img_height, int(box_y + box_height))

        return {
            "rgb": crop[:, :, :3],
            "alpha": crop[:, :, 3] / 255.0,
            "pos": (x1, y1),
            "comp_rect": (cx1, cy1, cx2, cy2),
        }

    def apply(self, frame: np.ndarray, time_seconds: float, frame_idx: int = None) -> np.ndarray:
        """
        Wendet Quote-Overlays auf einen Frame an.

        Das Overlay selbst kommt aus dem Cache (einmal pro Zitat gerendert);
        pro Frame passieren nur noch die Hintergrund-Kompensation in der
        Box-Region und ein NumPy-Alpha-Blend mit dem Fade-Wert.

        Args:
            frame: RGB numpy array (H, W, 3)
            time_seconds: Aktuelle Zeit im Video in Sekunden
            frame_idx: Optionaler Frame-Index fuer schnellen O(1) Buffer-Lookup

        Returns:
            Frame mit Overlay (falls ein Zitat aktiv ist)
        """
        if not self.config.enabled or not self.quotes:
            return frame

        quote = self._get_active_quote(time_seconds, frame_idx)
        if quote is None:
            return frame

        # Fade-Alpha berechnen
        fade = self._calculate_fade_alpha(time_seconds, quote)
        if fade <= 0.01:
            return frame

        height, width = frame.shape[:2]
        entry = self._get_cached_overlay(quote.text, (width, height))
        if entry is None:
            return frame

        result = frame.copy()

        # === SPATIAL FREQUENCY COMPENSATION ===
        # Hintergrund im Text-Bereich weichzeichnen und abdunkeln
        # (haengt vom Frame-Inhalt ab und bleibt daher pro Frame)
        if getattr(self.config, 'spatial_compensation', False):
            comp_blur = getattr(self.config, 'compensation_blur', 12.0)
            comp_darken = getattr(self.config, 'compensation_darken', 0.55)
            cx1, cy1, cx2, cy2 = entry["comp_rect"]
            if cx2 > cx1 and cy2 > cy1:
                region = Image.fromarray(result[cy1:cy2, cx1:cx2])
                region = region.filter(ImageFilter.GaussianBlur(radius=comp_blur))
                region = ImageEnhance.Brightness(region).enhance(comp_darken)
                result[cy1:cy2, cx1:cx2] = np.asarray(region)

        # === Cached Overlay mit Fade-Alpha einblenden (reines NumPy) ===
        x1, y1 = entry["pos"]
        rgb = entry["rgb"]
        h, w = rgb.shape[:2]
        alpha = (entry["alpha"] * fade)[:, :, np.newaxis]
        region = result[y1:y1 + h, x1:x1 + w].astype(np.float32)
        blended = region * (1.0 - alpha) + rgb * alpha
        result[y1:y1 + h, x1:x1 + w] = np.clip(blended, 0, 255).astype(np.uint8)

        return result
