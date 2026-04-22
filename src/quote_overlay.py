"""
Quote Overlay Renderer fuer Audio Visualizer Pro.

Zeichnet Key-Zitate als elegante Text-Overlays auf gerenderte Frames.
Zeitbasiert mit Fade-In/Out Animationen.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import textwrap

from pathlib import Path
from .gemini_integration import Quote


@dataclass
class QuoteOverlayConfig:
    """Konfiguration fuer Quote Overlays."""
    enabled: bool = True
    font_size: int = 52
    font_color: tuple = (255, 255, 255)  # RGB - Weiss
    box_color: tuple = (26, 26, 46, 200)  # RGBA - Dunkelblau, halbtransparent
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
    
    # Animationen
    slide_animation: str = "none"   # 'none', 'up', 'down', 'left', 'right'
    slide_distance: float = 100.0   # Pixel Slide-Distanz
    slide_out_animation: str = "none"  # 'none', 'up', 'down', 'left', 'right'
    slide_out_distance: float = 100.0  # Pixel Slide-Out Distanz
    scale_in: bool = False          # Box zoomt von 0.8 -> 1.0 rein
    typewriter: bool = False
    typewriter_speed: float = 15.0  # Buchstaben pro Sekunde
    typewriter_mode: str = "char"   # 'char' oder 'word'
    glow_pulse: bool = False        # Glow pulsiert beim Erscheinen
    glow_pulse_intensity: float = 0.5  # Max Glow waehrend Pulse
    
    # Position & Skalierung
    offset_x: int = 0       # Horizontaler Offset in Pixeln (negativ = links, positiv = rechts)
    offset_y: int = 0       # Vertikaler Offset in Pixeln (negativ = oben, positiv = unten)
    scale: float = 1.0      # Skalierungsfaktor (0.5 = halbe Groesse, 2.0 = doppelte Groesse)


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
    
    def _load_font(self):
        """Laedt eine Schriftart mit Fallback."""
        # Benutzerdefinierte Schriftart zuerst probieren
        if self.config.font_path and Path(self.config.font_path).exists():
            try:
                self._font = ImageFont.truetype(self.config.font_path, self.config.font_size)
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
                self._font = ImageFont.truetype(path, self.config.font_size)
                self._font_path = path
                return
            except (OSError, IOError):
                continue
        
        # Fallback auf Default-Schrift
        self._font = ImageFont.load_default()
        self._font_path = None
    
    def _get_active_quote(self, time_seconds: float) -> Optional[Quote]:
        """
        Findet das aktuell aktive Zitat fuer eine gegebene Zeit.
        
        Beruecksichtigt die maximale Anzeigedauer (display_duration).
        """
        for quote in self.quotes:
            # Begrenze die Anzeigedauer auf display_duration
            effective_end = min(quote.end_time, quote.start_time + self.config.display_duration)
            if quote.start_time <= time_seconds <= effective_end:
                return quote
        return None
    
    def _calculate_fade_alpha(self, time_seconds: float, 
                             quote: Quote) -> float:
        """
        Berechnet den Fade-Alpha-Wert (0.0 - 1.0) fuer ein Zitat.
        
        Fade-In am Anfang, Fade-Out am Ende.
        Beruecksichtigt die maximale Anzeigedauer.
        """
        fade = self.config.fade_duration
        effective_end = min(quote.end_time, quote.start_time + self.config.display_duration)
        duration = effective_end - quote.start_time
        
        # Am Anfang: Fade-In
        if time_seconds < quote.start_time + fade:
            progress = (time_seconds - quote.start_time) / fade
            return max(0.0, min(1.0, progress))
        
        # Am Ende: Fade-Out
        elif time_seconds > effective_end - fade:
            progress = (effective_end - time_seconds) / fade
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
        
        # PIL 10.0+ nutzt getbbox, aeltere getsize
        if hasattr(self._font, 'getbbox'):
            max_width = 0
            total_height = 0
            for line in lines:
                bbox = self._font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                max_width = max(max_width, line_width)
                total_height += line_height
            # Zeilenabstand hinzufuegen
            total_height += (len(lines) - 1) * self.config.line_spacing
            return (max_width, total_height)
        else:
            # Fallback fuer aeltere PIL Versionen
            max_width = max(self._font.getsize(line)[0] for line in lines)
            line_height = self._font.getsize(lines[0])[1]
            total_height = len(lines) * line_height + (len(lines) - 1) * self.config.line_spacing
            return (max_width, total_height)
    
    def apply(self, frame: np.ndarray, time_seconds: float) -> np.ndarray:
        """
        Wendet Quote-Overlays auf einen Frame an.
        
        Args:
            frame: RGB numpy array (H, W, 3)
            time_seconds: Aktuelle Zeit im Video in Sekunden
            
        Returns:
            Frame mit Overlay (falls ein Zitat aktiv ist)
        """
        if not self.config.enabled or not self.quotes:
            return frame
        
        quote = self._get_active_quote(time_seconds)
        if quote is None:
            return frame
        
        # Fade-Alpha berechnen
        alpha = self._calculate_fade_alpha(time_seconds, quote)
        if alpha <= 0.01:
            return frame
        
        # Konvertiere zu PIL Image (mit Alpha-Kanal fuer Overlay)
        img = Image.fromarray(frame)
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Text umbrechen
        lines = self._wrap_text(quote.text)
        if not lines:
            return frame
        
        # Groessen berechnen
        text_width, text_height = self._calculate_text_size(lines)
        padding = self.config.box_padding
        box_width = min(text_width + 2 * padding, int(img.width * self.config.max_width_ratio))
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
            box_x = (img.width - box_width) // 2 + offset_x
            box_y = img.height - box_height - self.config.box_margin_bottom + offset_y
        elif self.config.position == "top":
            box_x = (img.width - box_width) // 2 + offset_x
            box_y = self.config.box_margin_bottom + offset_y
        else:  # center
            box_x = (img.width - box_width) // 2 + offset_x
            box_y = (img.height - box_height) // 2 + offset_y
        
        # Schatten zeichnen
        shadow = self.config.shadow_offset
        scaled_shadow = (int(shadow[0] * config_scale), int(shadow[1] * config_scale))
        shadow_rect = [
            box_x + scaled_shadow[0], box_y + scaled_shadow[1],
            box_x + box_width + scaled_shadow[0], box_y + box_height + scaled_shadow[1]
        ]
        draw.rounded_rectangle(
            shadow_rect, 
            radius=box_radius,
            fill=self.config.shadow_color
        )
        
        # Hintergrund-Box zeichnen (mit Fade-Alpha)
        box_color = list(self.config.box_color)
        if len(box_color) < 4:
            box_color = list(self.config.box_color) + [160]
        box_color[3] = int(box_color[3] * alpha)
        draw.rounded_rectangle(
            [box_x, box_y, box_x + box_width, box_y + box_height],
            radius=box_radius,
            fill=tuple(box_color)
        )
        
        # Text zeichnen
        font_color = list(self.config.font_color) + [int(255 * alpha)]
        
        # Vertikale Zentrierung des Text-Blocks in der Box
        if hasattr(self._font, 'getbbox'):
            line_heights = []
            for line in lines:
                bbox = self._font.getbbox(line)
                line_heights.append(bbox[3] - bbox[1])
            total_text_height = sum(line_heights) + (len(lines) - 1) * self.config.line_spacing
        else:
            line_height = self._font.getsize(lines[0])[1]
            total_text_height = len(lines) * line_height + (len(lines) - 1) * self.config.line_spacing
        
        text_start_y = box_y + (box_height - total_text_height) // 2
        current_y = text_start_y
        
        for i, line in enumerate(lines):
            # Horizontale Zentrierung jeder Zeile
            if hasattr(self._font, 'getbbox'):
                bbox = self._font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height_actual = bbox[3] - bbox[1]
            else:
                line_width, line_height_actual = self._font.getsize(line)
            
            if self.config.text_align == "center":
                line_x = box_x + (box_width - line_width) // 2
            elif self.config.text_align == "right":
                line_x = box_x + box_width - line_width - padding
            else:  # left
                line_x = box_x + padding
            draw.text((line_x, current_y), line, font=self._font, fill=tuple(font_color))
            current_y += line_height_actual + self.config.line_spacing
        
        # Overlay auf Originalbild komponieren
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        
        # Zurueck zu RGB
        return np.array(img.convert('RGB'))
