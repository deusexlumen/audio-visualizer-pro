"""
Audio Visualizer Pro – DearPyGui Frontend v3.0

Premium Dark UI mit kategorien-basierten Farbakzenten,
eleganter Typografie und durchdachtem Spacing.
"""

from __future__ import annotations

import os
import re
import sys
import time
import threading
import queue
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# .env laden fuer API Keys
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# src zum Pfad hinzufuegen
sys.path.insert(0, str(Path(__file__).parent))

import dearpygui.dearpygui as dpg

from src.analyzer import AudioAnalyzer
from src.gpu_preview import render_gpu_preview
from src.gpu_renderer import GPUBatchRenderer
from src.gpu_visualizers import get_visualizer, list_visualizers
from src.quote_overlay import QuoteOverlayConfig
from src.types import Quote
from src.gemini_integration import GeminiIntegration
from src.quote_cache import load_quotes, save_quotes
from src.quote_refiner import refine_quote_timestamps


# =============================================================================
# DESIGN SYSTEM – Farben & Konstanten
# =============================================================================

class Theme:
    """Premium Dark Design-System mit kategorien-basierten Akzenten."""

    # Basis — tiefschwarz mit leichtem Blau-Stich
    BG_DEEPEST     = (8, 10, 16)
    BG_WINDOW      = (12, 15, 24)
    BG_CARD        = (18, 22, 34)
    BG_CARD_HOVER  = (26, 32, 48)
    BG_INPUT       = (14, 18, 28)
    BG_INPUT_HOVER = (20, 26, 40)
    BORDER         = (40, 50, 72)
    BORDER_ACTIVE  = (90, 140, 220)
    BORDER_GLOW    = (90, 140, 220, 60)

    TEXT_PRIMARY   = (230, 235, 245)
    TEXT_SECONDARY = (150, 165, 185)
    TEXT_MUTED     = (95, 110, 130)
    TEXT_DISABLED  = (65, 75, 90)

    # Kategorie-Akzente — jeder Bereich hat seine eigene Farbe
    ACCENT_PRIMARY = (96, 176, 255)    # Cyan-Blau — Hauptaktionen
    ACCENT_AUDIO   = (80, 200, 160)    # Mint-Grün — Audio
    ACCENT_VIZ     = (200, 120, 255)   # Violett — Visualizer
    ACCENT_KI      = (255, 180, 80)    # Amber — KI
    ACCENT_QUOTES  = (255, 110, 140)   # Rose — Zitate
    ACCENT_BG      = (120, 160, 255)   # Hellblau — Hintergrund
    ACCENT_PP      = (180, 140, 255)   # Lavendel — Post-Process
    ACCENT_EXPORT  = (80, 220, 120)    # Grün — Export

    # Status
    STATUS_OK      = (80, 210, 140)
    STATUS_WARN    = (240, 200, 90)
    STATUS_ERR     = (230, 90, 90)
    STATUS_INFO    = (100, 170, 240)

    # Slider
    SLIDER_GRAB       = ACCENT_PRIMARY
    SLIDER_GRAB_HOVER = (140, 200, 255)
    SLIDER_BG         = (26, 34, 52)
    SLIDER_BG_HOVER   = (36, 46, 68)

    @classmethod
    def dim(cls, color: tuple, factor: float = 0.6) -> tuple:
        """Dunklere Version einer Farbe."""
        return tuple(int(c * factor) for c in color[:3])

    @classmethod
    def alpha(cls, color: tuple, a: int) -> tuple:
        """Fügt Alpha-Kanal hinzu."""
        return color[:3] + (a,)

    @classmethod
    def hex_to_rgb(cls, hex_str: str) -> tuple[int, int, int]:
        hex_str = hex_str.lstrip("#")
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


# =============================================================================
# ASSET STORAGE
# =============================================================================

ASSET_DIRS = {
    "audio": Path("assets/user_uploads"),
    "backgrounds": Path("assets/backgrounds"),
    "fonts": Path("assets/fonts"),
}


def _ensure_asset_dirs():
    for d in ASSET_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


_ensure_asset_dirs()


# =============================================================================
# APP STATE
# =============================================================================

class AppState:
    """Zentraler Zustand für alle UI-Parameter und Dateipfade."""

    def __init__(self):
        self.audio_path: str | None = None
        self.background_path: str | None = None
        self.font_path: str | None = None
        self.output_dir: str = "output"

        self.features = None
        self.audio_duration: float = 0.0

        self.visualizer_type: str = "voice_flow"
        self.available_visualizers = list_visualizers()

        self.viz_offset_x: float = 0.0
        self.viz_offset_y: float = 0.0
        self.viz_scale: float = 1.0
        self.viz_extra_params: dict = {}

        self.bg_blur: float = 0.0
        self.bg_vignette: float = 0.0
        self.bg_opacity: float = 0.3

        self.pp_contrast: float = 1.0
        self.pp_saturation: float = 1.0
        self.pp_brightness: float = 0.0
        self.pp_warmth: float = 0.0
        self.pp_grain: float = 0.0

        self.preview_time_percent: float = 0.3
        self.preview_fps: int = 30
        self.preview_width: int = 854
        self.preview_height: int = 480

        self.resolution: tuple[int, int] = (1920, 1080)
        self.render_fps: int = 30
        self.codec: str = "h264"
        self.quality: str = "high"
        self.gpu_encode: bool = False
        
        # Farb-Parameter
        self.color_mode: str = "chroma"
        self.base_hue: float = 0.55
        self.color_saturation: float = 0.7

        self.is_analyzing: bool = False
        self.is_rendering: bool = False
        self.is_ki_optimizing: bool = False
        self.is_extracting_quotes: bool = False
        self.status_message: str = "Bereit."
        self.last_audio_dir: str = str(Path.home() / "Downloads")
        self.last_bg_dir: str = str(Path.home() / "Downloads")
        self.ki_status: str = ""
        self.ki_suggested_colors: dict = {}
        self.ki_prompt: str = ""

        # Quotes
        self.quotes: list = []
        self.quotes_enabled: bool = False
        self.quote_config: QuoteOverlayConfig = QuoteOverlayConfig(enabled=True)

        self._preview_params_hash: str = ""
        self._preview_image: Image.Image | None = None

        self._recent_files: list = self._load_recent_files()
        self._card_states: dict = self._load_card_states()

    def get_params(self) -> dict:
        base = {
            "offset_x": self.viz_offset_x,
            "offset_y": self.viz_offset_y,
            "scale": self.viz_scale,
            "color_mode": self.color_mode,
            "base_hue": self.base_hue,
            "color_saturation": self.color_saturation,
        }
        base.update(self.viz_extra_params)
        return base

    def get_postprocess(self) -> dict:
        return {
            "contrast": self.pp_contrast,
            "saturation": self.pp_saturation,
            "brightness": self.pp_brightness,
            "warmth": self.pp_warmth,
            "film_grain": self.pp_grain,
        }

    def preview_params_hash(self) -> str:
        qc = self.quote_config
        return (
            f"{self.visualizer_type}_{self.audio_path}_{self.background_path}_"
            f"{self.viz_offset_x:.3f}_{self.viz_offset_y:.3f}_{self.viz_scale:.3f}_"
            f"{self.bg_blur:.1f}_{self.bg_vignette:.2f}_{self.bg_opacity:.2f}_"
            f"{self.pp_contrast:.2f}_{self.pp_saturation:.2f}_{self.pp_brightness:.2f}_"
            f"{self.pp_warmth:.2f}_{self.pp_grain:.2f}_{self.preview_time_percent:.2f}_"
            f"{self.color_mode}_{self.base_hue:.2f}_{self.color_saturation:.2f}_"
            f"{self.quotes_enabled}_{len(self.quotes)}_{qc.position}_{qc.font_size}_{qc.display_duration}_"
            f"{qc.fade_duration:.1f}_{qc.max_chars_per_line}_{qc.line_spacing}_{qc.slide_animation}_"
            f"{qc.scale_in}_{qc.glow_pulse}_{qc.compensation_blur:.1f}_{qc.latency_offset:.1f}_"
            f"{hash(frozenset(self.viz_extra_params.items())) if self.viz_extra_params else 0}"
        )

    def to_dict(self) -> dict:
        """Serialisiert den AppState als Dictionary."""
        qc = self.quote_config
        return {
            "version": 1,
            "audio_path": self.audio_path,
            "background_path": self.background_path,
            "output_dir": self.output_dir,
            "visualizer_type": self.visualizer_type,
            "viz_offset_x": self.viz_offset_x,
            "viz_offset_y": self.viz_offset_y,
            "viz_scale": self.viz_scale,
            "viz_extra_params": self.viz_extra_params,
            "bg_blur": self.bg_blur,
            "bg_vignette": self.bg_vignette,
            "bg_opacity": self.bg_opacity,
            "pp_contrast": self.pp_contrast,
            "pp_saturation": self.pp_saturation,
            "pp_brightness": self.pp_brightness,
            "pp_warmth": self.pp_warmth,
            "pp_grain": self.pp_grain,
            "color_mode": self.color_mode,
            "base_hue": self.base_hue,
            "color_saturation": self.color_saturation,
            "resolution": list(self.resolution),
            "render_fps": self.render_fps,
            "gpu_encode": self.gpu_encode,
            "quotes_enabled": self.quotes_enabled,
            "quote_config": {
                "position": qc.position,
                "font_size": qc.font_size,
                "display_duration": qc.display_duration,
                "fade_duration": qc.fade_duration,
                "max_chars_per_line": qc.max_chars_per_line,
                "line_spacing": qc.line_spacing,
                "slide_animation": qc.slide_animation,
                "scale_in": qc.scale_in,
                "glow_pulse": qc.glow_pulse,
                "compensation_blur": qc.compensation_blur,
                "latency_offset": qc.latency_offset,
            },
            "quotes": [
                {
                    "text": q.text,
                    "start_time": q.start_time,
                    "end_time": q.end_time,
                    "confidence": q.confidence,
                }
                for q in self.quotes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppState":
        """Erstellt einen AppState aus einem Dictionary."""
        state = cls()
        state.audio_path = data.get("audio_path")
        state.background_path = data.get("background_path")
        state.output_dir = data.get("output_dir", "output")
        state.visualizer_type = data.get("visualizer_type", "voice_flow")
        state.viz_offset_x = data.get("viz_offset_x", 0.0)
        state.viz_offset_y = data.get("viz_offset_y", 0.0)
        state.viz_scale = data.get("viz_scale", 1.0)
        state.viz_extra_params = data.get("viz_extra_params", {})
        state.bg_blur = data.get("bg_blur", 0.0)
        state.bg_vignette = data.get("bg_vignette", 0.0)
        state.bg_opacity = data.get("bg_opacity", 0.3)
        state.pp_contrast = data.get("pp_contrast", 1.0)
        state.pp_saturation = data.get("pp_saturation", 1.0)
        state.pp_brightness = data.get("pp_brightness", 0.0)
        state.pp_warmth = data.get("pp_warmth", 0.0)
        state.pp_grain = data.get("pp_grain", 0.0)
        state.color_mode = data.get("color_mode", "chroma")
        state.base_hue = data.get("base_hue", 0.55)
        state.color_saturation = data.get("color_saturation", 0.7)
        state.resolution = tuple(data.get("resolution", [1920, 1080]))
        state.render_fps = data.get("render_fps", 30)
        state.gpu_encode = data.get("gpu_encode", False)
        state.quotes_enabled = data.get("quotes_enabled", False)

        # Quote-Config
        qc_data = data.get("quote_config", {})
        state.quote_config = QuoteOverlayConfig(
            position=qc_data.get("position", "bottom"),
            font_size=qc_data.get("font_size", 52),
            display_duration=qc_data.get("display_duration", 8.0),
            fade_duration=qc_data.get("fade_duration", 0.6),
            max_chars_per_line=qc_data.get("max_chars_per_line", 40),
            line_spacing=qc_data.get("line_spacing", 10),
            slide_animation=qc_data.get("slide_animation", "none"),
            scale_in=qc_data.get("scale_in", False),
            glow_pulse=qc_data.get("glow_pulse", False),
            compensation_blur=qc_data.get("compensation_blur", 12.0),
            latency_offset=qc_data.get("latency_offset", 0.0),
        )

        # Quotes
        quotes_data = data.get("quotes", [])
        state.quotes = [
            Quote(
                text=q.get("text", ""),
                start_time=float(q.get("start_time", 0.0)),
                end_time=float(q.get("end_time", 0.0)),
                confidence=float(q.get("confidence", 1.0)),
            )
            for q in quotes_data
        ]
        return state

    @staticmethod
    def _load_recent_files() -> list:
        try:
            path = Path(".cache") / "gui_recent_files.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return [p for p in data.get("recent", []) if os.path.exists(p)][:8]
        except Exception:
            pass
        return []

    @staticmethod
    def _save_recent_files(files: list):
        try:
            Path(".cache").mkdir(parents=True, exist_ok=True)
            path = Path(".cache") / "gui_recent_files.json"
            path.write_text(json.dumps({"recent": files[:8]}, indent=2), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _load_card_states() -> dict:
        try:
            path = Path(".cache") / "gui_card_states.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    @staticmethod
    def _save_card_states(states: dict):
        try:
            Path(".cache").mkdir(parents=True, exist_ok=True)
            path = Path(".cache") / "gui_card_states.json"
            path.write_text(json.dumps(states, indent=2), encoding="utf-8")
        except Exception:
            pass

    def add_recent_file(self, path: str):
        # Defensiv: Attribute initialisieren falls nicht vorhanden
        if not hasattr(self, '_recent_files'):
            self._recent_files = []
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._recent_files = self._recent_files[:8]
        self._save_recent_files(self._recent_files)


# =============================================================================
# DEARPYGUI FRONTEND
# =============================================================================

class AudioVisualizerGUI:
    """Hauptklasse für die DearPyGui-Oberfläche mit professionellem Design."""

    def __init__(self):
        self.state = AppState()
        self._preview_texture_tag = "preview_texture"
        self._preview_raw_data = np.zeros(
            (self.state.preview_width * self.state.preview_height * 4,),
            dtype=np.float32
        )
        self._init_preview_placeholder()
        self._last_preview_update = 0.0
        self._preview_min_interval = 0.15
        self._ki_future = None
        self.gemini = None
        try:
            self.gemini = GeminiIntegration()
        except Exception as e:
            print(f"[GUI] Gemini-Integration nicht verfuegbar: {e}")

        self._render_queue = queue.Queue()
        self._ki_queue = queue.Queue()
        self._quotes_queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._analyze_result = None
        self._preview_debounce_time = 0.0
        self._preview_debounce_delay = 0.25
        self._preview_last_error_time = 0.0
        self._preview_error_params_hash = ""
        self._preview_error_cooldown = 3.0

    def _init_preview_placeholder(self):
        """Erzeugt ein Placeholder-Bild für den Preview-Bereich vor dem Laden."""
        w, h = self.state.preview_width, self.state.preview_height
        img = Image.new("RGBA", (w, h), Theme.BG_DEEPEST + (255,))
        draw = ImageDraw.Draw(img)

        # Dezenter Rahmen
        draw.rectangle([1, 1, w - 2, h - 2], outline=Theme.BORDER + (255,), width=1)

        text = "Audio laden für Preview"
        sub = "MP3 · WAV · FLAC · AAC · OGG · M4A"
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 26)
            font_sub = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
            font_sub = ImageFont.load_default()

        # Haupttext mittig
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((w - tw) // 2, (h - th) // 2 - 18), text, font=font, fill=Theme.TEXT_SECONDARY + (255,))

        # Subtext mittig
        bbox2 = draw.textbbox((0, 0), sub, font=font_sub)
        tw2 = bbox2[2] - bbox2[0]
        draw.text(((w - tw2) // 2, (h - th) // 2 + 18), sub, font=font_sub, fill=Theme.TEXT_MUTED + (255,))

        arr = np.array(img, dtype=np.float32) / 255.0
        self._preview_raw_data[:] = arr.flatten()

    # -------------------------------------------------------------------------
    # Theme Setup
    # -------------------------------------------------------------------------

    def _setup_shortcuts(self):
        """Registriert globale Tastenkürzel."""
        with dpg.handler_registry():
            dpg.add_key_press_handler(key=dpg.mvKey_O, callback=self._on_ctrl_o)
            dpg.add_key_press_handler(key=dpg.mvKey_B, callback=self._on_ctrl_b)
            dpg.add_key_press_handler(key=dpg.mvKey_E, callback=self._on_ctrl_e)
            dpg.add_key_press_handler(key=dpg.mvKey_Escape, callback=self._on_escape)

    def _on_ctrl_o(self, sender, app_data):
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            self._show_audio_dialog()

    def _on_ctrl_b(self, sender, app_data):
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            self._show_bg_dialog()

    def _on_ctrl_e(self, sender, app_data):
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            self._on_render_clicked()

    def _on_escape(self, sender, app_data):
        if self.state.is_rendering:
            self._on_cancel_render_clicked()

    def _setup_theme(self):
        """Premium Dark Theme — kategorien-basierte Akzente, elegante Typografie."""
        with dpg.theme(tag="global_theme"):
            with dpg.theme_component(dpg.mvAll):
                # Fenster
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, Theme.BG_WINDOW)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, Theme.BG_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, Theme.BG_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_Text, Theme.TEXT_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_Border, Theme.BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, Theme.BG_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, Theme.BG_CARD)

                # Buttons — Primary-Stil
                dpg.add_theme_color(dpg.mvThemeCol_Button, Theme.ACCENT_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, Theme.SLIDER_GRAB_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, Theme.dim(Theme.ACCENT_PRIMARY, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, Theme.TEXT_DISABLED)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5)

                # Inputs / Slider
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, Theme.BG_INPUT)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, Theme.BG_INPUT_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (32, 42, 62))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, Theme.SLIDER_GRAB)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, Theme.SLIDER_GRAB_HOVER)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 12)

                # Combo / Dropdown
                dpg.add_theme_color(dpg.mvThemeCol_Header, Theme.BG_CARD_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (40, 55, 85))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (50, 70, 110))

                # Child Window (Cards)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, Theme.BG_CARD)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)

                # Separator
                dpg.add_theme_color(dpg.mvThemeCol_Separator, Theme.BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, Theme.BORDER_ACTIVE)
                dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, Theme.BORDER_ACTIVE)

                # Progress Bar
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, Theme.ACCENT_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered, Theme.SLIDER_GRAB_HOVER)

                # Scrollbar — modern & dünn
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, Theme.BG_DEEPEST)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, Theme.BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, Theme.BORDER_ACTIVE)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, Theme.ACCENT_PRIMARY)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 3)

                # Tree Node (Collapsible Cards)
                dpg.add_theme_color(dpg.mvThemeCol_Header, Theme.BG_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, Theme.BG_CARD_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (35, 45, 65))

                # Checkbox
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, Theme.ACCENT_PRIMARY)

                # Window-Rahmen
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 16)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 6, 4)

        dpg.bind_theme("global_theme")

    def _apply_card_theme(self, tag: str, accent: tuple = Theme.ACCENT_PRIMARY):
        """Wendet ein Card-Theme mit kategorien-spezifischem Akzent an."""
        theme_tag = f"card_theme_{tag}"
        if dpg.does_item_exist(theme_tag):
            dpg.bind_item_theme(tag, theme_tag)
            return
        with dpg.theme(tag=theme_tag):
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_Border, Theme.alpha(accent, 80))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, Theme.BG_CARD)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
        dpg.bind_item_theme(tag, theme_tag)

    def _make_secondary_button_theme(self) -> str:
        """Erstellt ein Theme für sekundäre Buttons."""
        tag = "btn_secondary_theme"
        if dpg.does_item_exist(tag):
            return tag
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, Theme.BG_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, Theme.BG_CARD_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, Theme.BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_Text, Theme.TEXT_SECONDARY)
        return tag

    def _make_accent_button_theme(self, accent: tuple) -> str:
        """Erstellt ein Theme für Buttons mit spezifischer Akzentfarbe."""
        tag = f"btn_accent_{accent[0]}_{accent[1]}_{accent[2]}"
        if dpg.does_item_exist(tag):
            return tag
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, accent)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (
                    min(255, accent[0] + 30),
                    min(255, accent[1] + 30),
                    min(255, accent[2] + 30)
                ))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, Theme.dim(accent, 0.8))
        return tag

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def setup_ui(self):
        dpg.create_context()
        dpg.configure_app(docking=False, init_file="dpg_layout.ini")
        self._setup_theme()
        self._setup_shortcuts()

        with dpg.font_registry():
            # Hauptfont
            default_font = dpg.add_font("C:/Windows/Fonts/segoeui.ttf", 15)
            dpg.bind_font(default_font)
            # Kleinerer Font für Labels
            self._font_small = dpg.add_font("C:/Windows/Fonts/segoeui.ttf", 13)
            # Größerer Font für Überschriften
            self._font_header = dpg.add_font("C:/Windows/Fonts/segoeuib.ttf", 18)
            # Card-Titel
            self._font_card_title = dpg.add_font("C:/Windows/Fonts/segoeuib.ttf", 15)
            # Status/Mono
            self._font_mono = dpg.add_font("C:/Windows/Fonts/consola.ttf", 14)

        # Texture fuer Preview
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=self.state.preview_width,
                height=self.state.preview_height,
                default_value=self._preview_raw_data,
                format=dpg.mvFormat_Float_rgba,
                tag=self._preview_texture_tag,
            )

        with dpg.window(
            label="Audio Visualizer Pro",
            tag="main_window",
            width=1280,
            height=900,
            no_close=True,
            no_collapse=True,
            no_resize=False,
            horizontal_scrollbar=True,
        ):
            self._build_app_header()
            self._build_menu_bar()

            # --- OBEN: Preview (immer sichtbar) ---
            self._build_preview_panel()

            # --- UNTEN: Tabs für alle Einstellungen ---
            self._build_settings_tabs()
            # Dynamische Visualizer-Parameter fuer den initialen Visualizer aufbauen
            self._rebuild_viz_param_controls()

            # --- UNTEN: Status-Bar ---
            self._build_status_bar()

        self._setup_file_dialogs()

        viewport_kwargs = {
            "title": "Audio Visualizer Pro",
            "width": 1280,
            "height": 900,
            "vsync": True,
        }
        icon_path = Path(__file__).parent / "assets" / "icon.ico"
        if icon_path.exists():
            viewport_kwargs["small_icon"] = str(icon_path)
            viewport_kwargs["large_icon"] = str(icon_path)
        dpg.create_viewport(**viewport_kwargs)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    # -------------------------------------------------------------------------
    # App Header
    # -------------------------------------------------------------------------

    def _build_app_header(self):
        """Eleganter App-Header mit Titel, Untertitel und schnellen Status-Indikatoren."""
        with dpg.group(horizontal=True, tag="app_header"):
            # Logo-Placeholder (farbiges Quadrat)
            dpg.add_color_button(
                default_value=Theme.ACCENT_PRIMARY + (255,),
                width=32, height=32,
                no_border=True, no_drag_drop=True,
            )
            dpg.add_spacer(width=10)

            with dpg.group():
                dpg.add_text("Audio Visualizer Pro", color=Theme.TEXT_PRIMARY)
                dpg.bind_item_font(dpg.last_item(), self._font_header)
                dpg.add_text("GPU-beschleunigte Audio-Visualisierung", color=Theme.TEXT_MUTED)
                dpg.bind_item_font(dpg.last_item(), self._font_small)

            dpg.add_spacer(width=20)

            # Schnell-Status als kleine Chips
            self._build_status_chip("audio_chip", "Audio", Theme.STATUS_ERR, "Audio-Datei geladen")
            dpg.add_spacer(width=8)
            self._build_status_chip("analyze_chip", "Analyse", Theme.STATUS_ERR, "Audio-Analyse abgeschlossen")
            dpg.add_spacer(width=8)
            self._build_status_chip("render_chip", "Render", Theme.STATUS_ERR, "Video-Rendering aktiv")
            dpg.add_spacer(width=8)
            self._build_status_chip("ki_chip", "KI", Theme.STATUS_ERR, "Gemini KI verfügbar")

    def _build_status_chip(self, tag: str, label: str, color: tuple, tooltip: str = ""):
        """Kleiner farbiger Status-Chip."""
        with dpg.group(horizontal=True, tag=tag):
            dpg.add_color_button(
                default_value=color + (255,),
                width=10, height=10,
                no_border=True, no_drag_drop=True,
            )
            dpg.add_text(label, color=Theme.TEXT_SECONDARY)
            dpg.bind_item_font(dpg.last_item(), self._font_small)
        if tooltip:
            with dpg.tooltip(parent=tag):
                dpg.add_text(tooltip, color=Theme.TEXT_MUTED)
                dpg.bind_item_font(dpg.last_item(), self._font_small)

    # -------------------------------------------------------------------------
    # Menu Bar
    # -------------------------------------------------------------------------

    def _build_menu_bar(self):
        """Menü-Leiste mit klaren Trennungen."""
        with dpg.menu_bar():
            with dpg.menu(label="Datei"):
                dpg.add_menu_item(label="Audio laden...        Ctrl+O", callback=self._show_audio_dialog)
                dpg.add_menu_item(label="Hintergrund laden...  Ctrl+B", callback=self._show_bg_dialog)
                dpg.add_separator()
                dpg.add_menu_item(label="Beenden              Alt+F4", callback=lambda: dpg.stop_dearpygui())

            with dpg.menu(label="Hilfe"):
                dpg.add_menu_item(label="Tastenkürzel", callback=self._show_shortcuts)
                dpg.add_separator()
                dpg.add_menu_item(label="Über Audio Visualizer Pro", callback=self._show_about)

    # -------------------------------------------------------------------------
    # Control Panel
    # -------------------------------------------------------------------------

    def _build_settings_tabs(self):
        """Tab-Bar für alle Einstellungen unter der Preview."""
        with dpg.child_window(width=-1, height=-120, border=False, horizontal_scrollbar=True):
            with dpg.tab_bar(tag="settings_tab_bar"):
                # --- AUDIO TAB ---
                with dpg.tab(label="🎵 Audio"):
                    self._build_audio_section()
                # --- VISUALIZER TAB ---
                with dpg.tab(label="🎨 Visualizer"):
                    self._build_visualizer_section()
                # --- HINTERGRUND TAB ---
                with dpg.tab(label="🖼️ Hintergrund"):
                    self._build_background_section()
                # --- POST-PROCESS TAB ---
                with dpg.tab(label="⚙️ Post-Process"):
                    self._build_postprocess_section()
                # --- KI TAB ---
                with dpg.tab(label="🤖 KI"):
                    self._build_ki_section()
                # --- ZITATE TAB ---
                with dpg.tab(label="💬 Zitate"):
                    self._build_quotes_section()
                # --- EXPORT TAB ---
                with dpg.tab(label="📤 Export"):
                    self._build_preview_section()
                    dpg.add_separator()
                    self._build_export_section()

    def _build_audio_section(self):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Audio laden",
                callback=self._show_audio_dialog,
                width=140,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_accent_button_theme(Theme.ACCENT_AUDIO))
            dpg.add_spacer(width=8)
            dpg.add_text("Keine Datei geladen", tag="audio_status", wrap=300, color=Theme.TEXT_MUTED)

        self._add_tooltip("Unterstützt: MP3, WAV, FLAC, AAC, OGG, M4A", parent="audio_status")

        # Recent files (dynamisch aktualisierbar)
        with dpg.group(tag="recent_files_group"):
            self._refresh_recent_files_ui()

    def _refresh_recent_files_ui(self):
        """Baut die Recent-Files-Liste neu auf."""
        parent = "recent_files_group"
        if not dpg.does_item_exist(parent):
            return
        # Bestehende Children löschen
        children = dpg.get_item_children(parent, 1)
        if children:
            for c in children:
                dpg.delete_item(c)
        if not hasattr(self.state, '_recent_files') or not self.state._recent_files:
            return
        dpg.add_spacer(height=6, parent=parent)
        dpg.add_text("Zuletzt verwendet", color=Theme.TEXT_MUTED, parent=parent)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        for i, path in enumerate(self.state._recent_files[:5]):
            fname = Path(path).name
            btn_tag = f"recent_audio_{i}"
            if dpg.does_item_exist(btn_tag):
                dpg.delete_item(btn_tag)
            dpg.add_button(
                label=f"📄 {fname[:35]}{'...' if len(fname) > 35 else ''}",
                callback=lambda s, a, p=path: self._load_recent_audio(p),
                width=-1,
                tag=btn_tag,
                parent=parent,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())
            self._add_tooltip(path, parent=btn_tag)

    def _load_recent_audio(self, path: str):
        if not os.path.exists(path):
            self._set_status("Datei nicht mehr vorhanden.", "warn")
            if hasattr(self.state, '_recent_files') and path in self.state._recent_files:
                self.state._recent_files.remove(path)
                AppState._save_recent_files(self.state._recent_files)
            self._refresh_recent_files_ui()
            return
        ext = Path(path).suffix.lower()
        valid_exts = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        if ext not in valid_exts:
            self._show_error_modal(f"Ungültiges Dateiformat: {ext}\n\nErlaubt: MP3, WAV, FLAC, AAC, OGG, M4A")
            return
        self.state.audio_path = path
        self.state.last_audio_dir = str(Path(path).parent)
        self.state.add_recent_file(path)
        self._refresh_recent_files_ui()
        dpg.set_value("audio_status", f"{Path(path).name}")
        dpg.configure_item("audio_status", color=Theme.TEXT_PRIMARY)
        self._analyze_audio()
        self._request_preview_update()
        self._update_status_indicators()
        self._set_status(f"Audio geladen: {Path(path).name}", "ok")

    def _build_visualizer_section(self):
        with dpg.group(horizontal=True):
            dpg.add_text("Typ", color=Theme.TEXT_SECONDARY)
            dpg.bind_item_font(dpg.last_item(), self._font_small)
            dpg.add_button(
                label="↺ Reset",
                callback=lambda s, a: self._reset_visualizer_params(),
                width=60,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())
        dpg.add_combo(
            items=self.state.available_visualizers,
            default_value=self.state.visualizer_type,
            callback=self._on_visualizer_changed,
            width=-1,
            tag="viz_combo",
        )
        self._add_tooltip("Wähle einen GPU-beschleunigten Visualizer")
        dpg.add_spacer(height=8)

        dpg.add_text("Position & Größe", color=Theme.TEXT_SECONDARY)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        self._styled_slider(
            label="Offset X",
            tag="param_offset_x",
            min_val=-1.0, max_val=1.0, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Horizontale Verschiebung",
        )
        self._styled_slider(
            label="Offset Y",
            tag="param_offset_y",
            min_val=-1.0, max_val=1.0, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Vertikale Verschiebung",
        )
        self._styled_slider(
            label="Skalierung",
            tag="param_scale",
            min_val=0.5, max_val=2.0, default_val=1.0,
            callback=self._on_param_changed,
            tooltip="Gesamtgröße des Visualizers",
        )
        dpg.add_spacer(height=8)

        dpg.add_text("Farbpalette", color=Theme.TEXT_SECONDARY)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        dpg.add_combo(
            items=["chroma (dynamisch)", "fixed (eine Farbe)", "monochrome", "warm", "cool"],
            default_value="chroma (dynamisch)",
            callback=self._on_color_mode_changed,
            width=-1,
            tag="param_color_mode",
        )
        self._styled_slider(
            label="Farbton",
            tag="param_base_hue",
            min_val=0.0, max_val=1.0, default_val=0.55,
            callback=self._on_param_changed,
            tooltip="Grundfarbton (0=rot, 0.33=grün, 0.66=blau)",
        )
        self._styled_slider(
            label="Sättigung",
            tag="param_color_saturation",
            min_val=0.0, max_val=1.0, default_val=0.7,
            callback=self._on_param_changed,
            tooltip="Farbintensität (0=grau, 1=knallig)",
        )
        dpg.add_spacer(height=8)

        # Container fuer dynamische Visualizer-Parameter
        dpg.add_text("Effekte", tag="viz_params_label", color=Theme.TEXT_SECONDARY, show=False)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        dpg.add_group(tag="viz_params_container")

    def _rebuild_viz_param_controls(self):
        """Erzeugt dynamische Slider fuer die PARAMS des aktuellen Visualizers."""
        container = "viz_params_container"
        if not dpg.does_item_exist(container):
            return

        # Alte Controls entfernen
        dpg.delete_item(container, children_only=True)

        viz_name = self.state.visualizer_type
        try:
            from src.gpu_visualizers import get_visualizer
            viz_cls = get_visualizer(viz_name)
        except Exception:
            dpg.configure_item("viz_params_label", show=False)
            return

        # Sammle alle Parameter: EFFECTS + COLOR_PARAMS + PARAMS
        all_params = {}
        # Universelle Effekte
        for k, v in getattr(viz_cls, 'EFFECTS', {}).items():
            all_params[k] = v
        # Visualizer-spezifische PARAMS (ueberschreiben EFFECTS bei Doppelung)
        for k, v in getattr(viz_cls, 'PARAMS', {}).items():
            all_params[k] = v

        if not all_params:
            dpg.configure_item("viz_params_label", show=False)
            return

        dpg.configure_item("viz_params_label", show=True)

        # Param-Namen schoen formatieren
        def _fmt_name(name):
            return name.replace('_', ' ').title()

        dpg.push_container_stack(container)
        try:
            for param_name, (default, min_val, max_val, step) in all_params.items():
                tag = f"viz_param_{param_name}"
                current_val = self.state.viz_extra_params.get(param_name, default)
                # Schrittweite fuer Format-String bestimmen
                if step >= 1:
                    fmt = "%.0f"
                elif step >= 0.1:
                    fmt = "%.1f"
                elif step >= 0.01:
                    fmt = "%.2f"
                else:
                    fmt = "%.3f"
                self._styled_slider(
                    label=_fmt_name(param_name),
                    tag=tag,
                    min_val=min_val, max_val=max_val,
                    default_val=current_val,
                    callback=lambda s, a, pname=param_name: self._on_viz_extra_param_changed(pname, s, a),
                    tooltip=f"{_fmt_name(param_name)}: {min_val} - {max_val}",
                    format=fmt,
                )
        finally:
            dpg.pop_container_stack()

    def _on_viz_extra_param_changed(self, param_name: str, sender, app_data):
        """Callback fuer dynamische Visualizer-Parameter."""
        self.state.viz_extra_params[param_name] = app_data
        self._request_preview_update()

    def _reset_visualizer_params(self):
        """Setzt Visualizer-Parameter auf Standard zurück."""
        defaults = {
            "param_offset_x": (0.0, "viz_offset_x"),
            "param_offset_y": (0.0, "viz_offset_y"),
            "param_scale": (1.0, "viz_scale"),
            "param_base_hue": (0.55, "base_hue"),
            "param_color_saturation": (0.7, "color_saturation"),
        }
        for tag, (val, attr) in defaults.items():
            dpg.set_value(tag, val)
            setattr(self.state, attr, val)
            val_tag = f"{tag}_value"
            if dpg.does_item_exist(val_tag):
                fmt = dpg.get_item_configuration(tag).get("format", "%.2f")
                try:
                    dpg.set_value(val_tag, fmt % val)
                except Exception:
                    dpg.set_value(val_tag, str(round(val, 2)))
        self.state.viz_extra_params = {}
        self._request_preview_update()
        self._set_status("Visualizer-Parameter zurückgesetzt.", "info")

    def _build_ki_section(self):
        dpg.add_text("Dein Wunsch (optional)", color=Theme.TEXT_SECONDARY)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        dpg.add_input_text(
            hint="z.B. 'dunkler, mehr Kontrast, cyberpunk-Stil'",
            default_value="",
            callback=self._on_ki_prompt_changed,
            width=-1,
            tag="ki_prompt_input",
        )
        dpg.add_spacer(height=8)
        dpg.add_button(
            label="Parameter optimieren",
            callback=self._on_ki_optimize_clicked,
            width=-1,
            tag="btn_ki_optimize",
        )
        dpg.bind_item_theme(dpg.last_item(), self._make_accent_button_theme(Theme.ACCENT_KI))
        self._add_tooltip("Nutzt Gemini KI für intelligente Parameter-Anpassung")
        dpg.add_text("", tag="ki_status_text", wrap=340, color=Theme.TEXT_SECONDARY)
        dpg.add_text("", tag="ki_colors_text", wrap=340, color=Theme.STATUS_OK)

    def _build_quotes_section(self):
        dpg.add_checkbox(
            label="Zitate aktivieren",
            default_value=self.state.quotes_enabled,
            callback=self._on_quotes_enabled_changed,
            tag="chk_quotes_enabled",
        )
        dpg.add_spacer(height=8)
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Key-Zitate extrahieren",
                callback=self._on_extract_quotes_clicked,
                width=180,
                tag="btn_extract_quotes",
            )
            dpg.add_button(
                label="Demo-Zitate",
                callback=self._on_demo_quotes_clicked,
                width=120,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())

        dpg.add_text("", tag="quotes_status_text", wrap=340, color=Theme.TEXT_SECONDARY)
        dpg.add_spacer(height=8)

        # --- ZITAT-LISTE (CRUD) ---
        dpg.add_text("Zitate", color=Theme.TEXT_SECONDARY)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        dpg.add_child_window(
            tag="quotes_list_container",
            width=-1, height=180,
            border=True,
            horizontal_scrollbar=False,
        )
        dpg.add_spacer(height=6)
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="➕ Hinzufügen",
                callback=self._on_add_quote_clicked,
                width=100,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_accent_button_theme(Theme.ACCENT_PRIMARY))
            dpg.add_button(
                label="💾 Speichern",
                callback=self._on_save_quotes_clicked,
                width=100,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())
            dpg.add_button(
                label="🔄 Zurücksetzen",
                callback=self._on_reset_quotes_clicked,
                width=110,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())
        dpg.add_spacer(height=8)

        dpg.add_text("Erscheinungsbild", color=Theme.TEXT_SECONDARY)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        dpg.add_combo(
            label="Position",
            items=["bottom", "center", "top"],
            default_value=self.state.quote_config.position,
            callback=self._on_quote_config_changed,
            width=-1,
            tag="quote_position",
        )
        self._styled_slider(
            label="Schriftgröße",
            tag="quote_font_size",
            min_val=16, max_val=96, default_val=self.state.quote_config.font_size,
            callback=self._on_quote_config_changed,
            tooltip="Schriftgröße der Zitate",
            format="%.0f",
        )
        self._styled_slider(
            label="Anzeigedauer (s)",
            tag="quote_display_duration",
            min_val=2.0, max_val=20.0, default_val=self.state.quote_config.display_duration,
            callback=self._on_quote_config_changed,
            tooltip="Wie lange ein Zitat angezeigt wird",
        )
        dpg.add_spacer(height=4)
        self._styled_slider(
            label="Fade-Dauer (s)",
            tag="quote_fade_duration",
            min_val=0.1, max_val=2.0, default_val=self.state.quote_config.fade_duration,
            callback=self._on_quote_config_changed,
            tooltip="Ein-/Ausblende-Geschwindigkeit",
            format="%.1f",
        )
        self._styled_slider(
            label="Max. Zeichen/Zeile",
            tag="quote_max_chars",
            min_val=20, max_val=80, default_val=self.state.quote_config.max_chars_per_line,
            callback=self._on_quote_config_changed,
            tooltip="Automatischer Zeilenumbruch",
            format="%.0f",
        )
        self._styled_slider(
            label="Zeilenabstand (px)",
            tag="quote_line_spacing",
            min_val=0, max_val=30, default_val=self.state.quote_config.line_spacing,
            callback=self._on_quote_config_changed,
            tooltip="Abstand zwischen Textzeilen",
            format="%.0f",
        )
        dpg.add_combo(
            label="Slide-Animation",
            items=["none", "up", "down", "left", "right"],
            default_value=self.state.quote_config.slide_animation,
            callback=self._on_quote_config_changed,
            width=-1,
            tag="quote_slide_animation",
        )
        with dpg.group(horizontal=True):
            dpg.add_checkbox(
                label="Scale-In-Effekt",
                default_value=self.state.quote_config.scale_in,
                callback=self._on_quote_config_changed,
                tag="quote_scale_in",
            )
            dpg.add_checkbox(
                label="Glow-Pulse",
                default_value=self.state.quote_config.glow_pulse,
                callback=self._on_quote_config_changed,
                tag="quote_glow_pulse",
            )
        self._styled_slider(
            label="Hintergrund-Blur (px)",
            tag="quote_compensation_blur",
            min_val=0.0, max_val=30.0, default_val=self.state.quote_config.compensation_blur,
            callback=self._on_quote_config_changed,
            tooltip="Weichzeichnung unter dem Text",
            format="%.1f",
        )
        self._styled_slider(
            label="Latenz-Offset (s)",
            tag="quote_latency_offset",
            min_val=-2.0, max_val=2.0, default_val=self.state.quote_config.latency_offset,
            callback=self._on_quote_config_changed,
            tooltip="Positiv = spaeter, Negativ = frueher",
            format="%.1f",
        )

    def _build_background_section(self):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Hintergrund laden",
                callback=self._show_bg_dialog,
                width=160,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_accent_button_theme(Theme.ACCENT_BG))
            dpg.add_spacer(width=8)
            dpg.add_text("Kein Bild", tag="bg_status_text", color=Theme.TEXT_MUTED)
            dpg.add_button(
                label="↺",
                callback=lambda s, a: self._reset_background_params(),
                width=30,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())
            self._add_tooltip("Hintergrund-Parameter zurücksetzen")

        dpg.add_spacer(height=8)
        dpg.add_text("Effekte", color=Theme.TEXT_SECONDARY)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        self._styled_slider(
            label="Blur",
            tag="param_bg_blur",
            min_val=0.0, max_val=20.0, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Weichzeichnung des Hintergrunds",
        )
        self._styled_slider(
            label="Vignette",
            tag="param_bg_vignette",
            min_val=0.0, max_val=1.0, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Dunkle Ränder für dramatischen Look",
        )
        self._styled_slider(
            label="Deckkraft",
            tag="param_bg_opacity",
            min_val=0.0, max_val=1.0, default_val=0.3,
            callback=self._on_param_changed,
            tooltip="Sichtbarkeit des Hintergrundbilds",
        )

    def _reset_background_params(self):
        defaults = {
            "param_bg_blur": (0.0, "bg_blur"),
            "param_bg_vignette": (0.0, "bg_vignette"),
            "param_bg_opacity": (0.3, "bg_opacity"),
        }
        for tag, (val, attr) in defaults.items():
            dpg.set_value(tag, val)
            setattr(self.state, attr, val)
            val_tag = f"{tag}_value"
            if dpg.does_item_exist(val_tag):
                fmt = dpg.get_item_configuration(tag).get("format", "%.2f")
                try:
                    dpg.set_value(val_tag, fmt % val)
                except Exception:
                    dpg.set_value(val_tag, str(round(val, 2)))
        self._request_preview_update()
        self._set_status("Hintergrund-Parameter zurückgesetzt.", "info")

    def _build_postprocess_section(self):
        with dpg.group(horizontal=True):
            dpg.add_text("Color Grading", color=Theme.TEXT_SECONDARY)
            dpg.bind_item_font(dpg.last_item(), self._font_small)
            dpg.add_button(
                label="↺",
                callback=lambda s, a: self._reset_postprocess_params(),
                width=30,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())
            self._add_tooltip("Post-Process zurücksetzen")
        self._styled_slider(
            label="Kontrast",
            tag="param_pp_contrast",
            min_val=0.5, max_val=2.0, default_val=1.0,
            callback=self._on_param_changed,
            tooltip="1.0 = neutral, >1 = mehr Kontrast",
        )
        self._styled_slider(
            label="Sättigung",
            tag="param_pp_saturation",
            min_val=0.0, max_val=2.0, default_val=1.0,
            callback=self._on_param_changed,
            tooltip="0 = Schwarzweiß, >1 = hyper-bunt",
        )
        self._styled_slider(
            label="Helligkeit",
            tag="param_pp_brightness",
            min_val=-0.5, max_val=0.5, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Globale Helligkeitsanpassung",
        )
        dpg.add_spacer(height=8)
        dpg.add_text("Effekte", color=Theme.TEXT_SECONDARY)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        self._styled_slider(
            label="Warmth",
            tag="param_pp_warmth",
            min_val=-1.0, max_val=1.0, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Warme (orange) oder kalte (blau) Farbtemperatur",
        )
        self._styled_slider(
            label="Film Grain",
            tag="param_pp_grain",
            min_val=0.0, max_val=1.0, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Analoger Filmkorn-Effekt",
        )

    def _reset_postprocess_params(self):
        defaults = {
            "param_pp_contrast": (1.0, "pp_contrast"),
            "param_pp_saturation": (1.0, "pp_saturation"),
            "param_pp_brightness": (0.0, "pp_brightness"),
            "param_pp_warmth": (0.0, "pp_warmth"),
            "param_pp_grain": (0.0, "pp_grain"),
        }
        for tag, (val, attr) in defaults.items():
            dpg.set_value(tag, val)
            setattr(self.state, attr, val)
            val_tag = f"{tag}_value"
            if dpg.does_item_exist(val_tag):
                fmt = dpg.get_item_configuration(tag).get("format", "%.2f")
                try:
                    dpg.set_value(val_tag, fmt % val)
                except Exception:
                    dpg.set_value(val_tag, str(round(val, 2)))
        self._request_preview_update()
        self._set_status("Post-Process zurückgesetzt.", "info")

    def _build_preview_section(self):
        self._styled_slider(
            label="Position (%)",
            tag="param_preview_time",
            min_val=0.0, max_val=1.0, default_val=0.3,
            callback=self._on_param_changed,
            tooltip="Zeitpunkt im Audio für die Preview",
        )
        dpg.add_text(
            "Preview aktualisiert automatisch beim Loslassen des Sliders.",
            wrap=340, color=Theme.TEXT_MUTED,
        )
        dpg.bind_item_font(dpg.last_item(), self._font_small)

    def _build_export_section(self):
        # --- PROJEKT SAVE/LOAD ---
        dpg.add_text("Projekt", color=Theme.TEXT_SECONDARY)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="💾 Speichern",
                callback=self._on_save_project_clicked,
                width=100,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_accent_button_theme(Theme.ACCENT_PRIMARY))
            dpg.add_button(
                label="📂 Laden",
                callback=self._on_load_project_clicked,
                width=80,
            )
            dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())
        dpg.add_combo(
            label="Letzte Projekte",
            items=[],
            default_value="",
            callback=self._on_quick_load_project,
            width=-1,
            tag="project_quick_load",
        )
        self._refresh_project_list()
        dpg.add_spacer(height=8)

        dpg.add_text("Auflösung", color=Theme.TEXT_SECONDARY)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        dpg.add_combo(
            label="",
            items=["1920x1080 (Full HD)", "1280x720 (HD)", "854x480 (SD)"],
            default_value="1920x1080 (Full HD)",
            callback=self._on_resolution_changed,
            width=-1,
            tag="res_combo",
        )
        dpg.add_spacer(height=4)
        dpg.add_combo(
            label="FPS",
            items=["24", "30", "60"],
            default_value="30",
            callback=self._on_fps_changed,
            width=-1,
            tag="fps_combo",
        )
        dpg.add_spacer(height=4)
        dpg.add_combo(
            label="Codec",
            items=["h264 (kompatibel)", "h265 (klein)", "prores (Editing)"],
            default_value="h264 (kompatibel)",
            callback=self._on_codec_changed,
            width=-1,
            tag="codec_combo",
        )
        dpg.add_spacer(height=4)
        dpg.add_combo(
            label="Qualität",
            items=["Draft (schnell)", "Standard", "High", "Lossless"],
            default_value="High",
            callback=self._on_quality_changed,
            width=-1,
            tag="quality_combo",
        )
        dpg.add_spacer(height=8)
        dpg.add_checkbox(
            label="GPU-Encoding (NVENC/AMF/QSV)",
            default_value=self.state.gpu_encode,
            callback=self._on_gpu_encode_changed,
            tag="chk_gpu_encode",
        )
        self._add_tooltip("~5-10x schneller mit Grafikkarte")
        dpg.add_spacer(height=8)
        dpg.add_text("Output", color=Theme.TEXT_SECONDARY)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        dpg.add_input_text(
            label="",
            default_value=self.state.output_dir,
            callback=self._on_output_dir_changed,
            width=-1,
        )
        dpg.add_spacer(height=10)
        dpg.add_button(
            label="▶ Video exportieren",
            callback=self._on_render_clicked,
            width=-1,
            tag="btn_render",
        )
        dpg.bind_item_theme(dpg.last_item(), self._make_accent_button_theme(Theme.ACCENT_EXPORT))
        dpg.add_button(
            label="Abbrechen",
            callback=self._on_cancel_render_clicked,
            width=-1,
            tag="btn_cancel_render",
            enabled=False,
            show=True,
        )
        dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())
        dpg.add_progress_bar(
            default_value=0.0,
            width=-1,
            tag="render_progress",
        )
        dpg.add_text("", tag="render_status_text", wrap=340, color=Theme.TEXT_MUTED)
        dpg.bind_item_font(dpg.last_item(), self._font_small)
        dpg.add_button(
            label="📁 Ordner öffnen",
            callback=self._on_open_output_folder,
            width=-1,
            tag="btn_open_folder",
            show=False,
        )
        dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())

    def _styled_slider(self, label: str, tag: str, min_val: float, max_val: float,
                       default_val: float, callback, tooltip: str = "",
                       format: str = "%.2f"):
        """Slider mit Label links, Wert-Anzeige rechts und Tooltip."""
        value_tag = f"{tag}_value"
        val_str = format % default_val if format.startswith("%") else str(default_val)
        with dpg.group(horizontal=True):
            dpg.add_text(label, color=Theme.TEXT_SECONDARY)
            dpg.bind_item_font(dpg.last_item(), self._font_small)
            dpg.add_slider_float(
                min_value=min_val, max_value=max_val,
                default_value=default_val,
                callback=lambda s, a: self._on_slider_changed(s, a, callback, value_tag),
                width=-1, tag=tag,
                format=format,
            )
            dpg.add_text(val_str, color=Theme.TEXT_MUTED, tag=value_tag)
            dpg.bind_item_font(dpg.last_item(), self._font_mono)
        if tooltip:
            self._add_tooltip(tooltip, parent=tag)

    def _on_slider_changed(self, sender, app_data, original_callback, value_tag: str):
        """Wrapper für Slider-Callbacks, der den Wert-Text aktualisiert."""
        val = dpg.get_value(sender)
        fmt = dpg.get_item_configuration(sender).get("format", "%.2f")
        try:
            val_str = fmt % val
        except Exception:
            val_str = str(round(val, 2))
        dpg.set_value(value_tag, val_str)
        if original_callback:
            original_callback(sender, app_data)

    def _add_tooltip(self, text: str, parent: str | None = None):
        """Tooltip mit dezentem Styling."""
        with dpg.tooltip(parent=parent if parent else dpg.last_item()):
            dpg.add_text(text, color=Theme.TEXT_MUTED)
            dpg.bind_item_font(dpg.last_item(), self._font_small)

    # -------------------------------------------------------------------------
    # Preview Panel
    # -------------------------------------------------------------------------

    def _build_preview_panel(self):
        """Preview-Panel oben mit festem Platzbedarf."""
        with dpg.child_window(
            width=-1,
            height=480,
            border=False,
            tag="preview_panel",
        ):
            preview_card = "preview_card"
            with dpg.child_window(
                width=-1,
                height=-1,
                border=True,
                tag=preview_card,
            ):
                # Header mit Info
                with dpg.group(horizontal=True):
                    dpg.add_text("Preview", color=Theme.ACCENT_PRIMARY)
                    dpg.bind_item_font(dpg.last_item(), self._font_card_title)
                    dpg.add_spacer(width=12)
                    dpg.add_text(
                        f"{self.state.preview_width}x{self.state.preview_height} @ {self.state.preview_fps}fps",
                        color=Theme.TEXT_MUTED,
                        tag="preview_resolution_text",
                    )
                    dpg.bind_item_font(dpg.last_item(), self._font_small)

                dpg.add_separator()
                dpg.add_spacer(height=6)

                # Preview-Bild oder Welcome-Screen
                with dpg.group(horizontal=True, tag="preview_container"):
                    dpg.add_spacer(width=1)

                    # Welcome overlay (wird ausgeblendet wenn Audio geladen)
                    with dpg.group(tag="welcome_overlay"):
                        dpg.add_spacer(height=50)
                        dpg.add_text("🎵 Audio Visualizer Pro", color=Theme.TEXT_PRIMARY, tag="welcome_title")
                        dpg.bind_item_font(dpg.last_item(), self._font_header)
                        dpg.add_spacer(height=12)
                        dpg.add_text(
                            "So geht's:",
                            color=Theme.ACCENT_PRIMARY, tag="welcome_subtitle",
                        )
                        dpg.bind_item_font(dpg.last_item(), self._font_card_title)
                        dpg.add_spacer(height=8)
                        dpg.add_text(
                            "1. Audio-Datei laden (MP3, WAV, FLAC...)\n"
                            "2. Visualizer und Farben einstellen\n"
                            "3. Auf 'Video exportieren' klicken",
                            color=Theme.TEXT_SECONDARY, wrap=400, tag="welcome_steps",
                        )
                        dpg.add_spacer(height=6)
                        dpg.add_text(
                            "💡 Tipp: Nutze die Tastenkürzel Ctrl+O und Ctrl+B für schnelles Laden.",
                            color=Theme.TEXT_MUTED, wrap=400, tag="welcome_drag_hint",
                        )
                        dpg.bind_item_font(dpg.last_item(), self._font_small)
                        dpg.add_spacer(height=8)
                        dpg.add_text(
                            "",
                            color=Theme.ACCENT_AUDIO, wrap=400, tag="welcome_loading",
                        )
                        dpg.bind_item_font(dpg.last_item(), self._font_mono)
                        dpg.add_spacer(height=20)
                        dpg.add_button(
                            label="📁 Audio laden",
                            callback=self._show_audio_dialog,
                            width=180,
                            height=36,
                            tag="welcome_load_btn",
                        )
                        dpg.bind_item_theme(dpg.last_item(), self._make_accent_button_theme(Theme.ACCENT_AUDIO))

                    # Lade-Indikator (über dem Bild)
                    dpg.add_text(
                        "⏳ Preview wird generiert...",
                        color=Theme.ACCENT_PRIMARY,
                        tag="preview_loading_text",
                        show=False,
                    )
                    dpg.bind_item_font(dpg.last_item(), self._font_mono)

                    # Eigentliches Bild (initial versteckt)
                    dpg.add_image(
                        self._preview_texture_tag,
                        width=self.state.preview_width,
                        height=self.state.preview_height,
                        tag="preview_image",
                        show=False,
                    )
                    dpg.add_spacer(width=1)

                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True, tag="audio_info_group"):
                    dpg.add_text(
                        "--:-- / --:--",
                        color=Theme.TEXT_MUTED,
                        tag="preview_time_text",
                    )
                    dpg.bind_item_font(dpg.last_item(), self._font_mono)
                    dpg.add_spacer(width=12)
                    dpg.add_text(
                        "",
                        color=Theme.TEXT_MUTED,
                        tag="audio_info_text",
                    )
                    dpg.bind_item_font(dpg.last_item(), self._font_small)

            self._apply_card_theme(preview_card, Theme.ACCENT_PRIMARY)

    # -------------------------------------------------------------------------
    # Status Bar
    # -------------------------------------------------------------------------

    def _build_status_bar(self):
        """Elegante Status-Leiste mit farbigen Indikatoren."""
        with dpg.group(horizontal=True, tag="status_bar"):
            # Status-Icon
            dpg.add_color_button(
                default_value=Theme.STATUS_OK + (255,),
                width=8, height=8,
                no_border=True, no_drag_drop=True,
                tag="status_dot",
            )
            dpg.add_spacer(width=8)
            dpg.add_text(
                self.state.status_message,
                color=Theme.TEXT_SECONDARY,
                tag="status_text",
            )
            dpg.bind_item_font(dpg.last_item(), self._font_small)

    def _set_status(self, msg: str, level: str = "info"):
        """Aktualisiert die Status-Zeile."""
        self.state.status_message = msg
        dpg.set_value("status_text", msg)

        color = Theme.TEXT_SECONDARY
        dot_color = Theme.STATUS_INFO
        if level == "ok":
            color = Theme.STATUS_OK
            dot_color = Theme.STATUS_OK
        elif level == "warn":
            color = Theme.STATUS_WARN
            dot_color = Theme.STATUS_WARN
        elif level == "error":
            color = Theme.STATUS_ERR
            dot_color = Theme.STATUS_ERR
        dpg.configure_item("status_text", color=color)
        dpg.set_value("status_dot", dot_color + (255,))

    def _update_status_indicators(self):
        """Aktualisiert die Status-Chips im Header und Button-States."""
        has_audio = self.state.audio_path is not None and os.path.exists(self.state.audio_path)
        self._update_chip("audio_chip", Theme.STATUS_OK if has_audio else Theme.STATUS_ERR)

        analyzed = self.state.features is not None
        self._update_chip("analyze_chip",
            Theme.STATUS_OK if analyzed else (Theme.STATUS_WARN if has_audio else Theme.STATUS_ERR))

        rendering = self.state.is_rendering
        self._update_chip("render_chip",
            Theme.STATUS_WARN if rendering else (Theme.STATUS_OK if analyzed else Theme.STATUS_ERR))

        ki_ok = self.gemini is not None
        self._update_chip("ki_chip", Theme.STATUS_OK if ki_ok else Theme.STATUS_ERR)

        # Welcome-Screen togglen
        if has_audio and analyzed:
            dpg.configure_item("welcome_overlay", show=False)
            dpg.configure_item("preview_image", show=True)
            dpg.set_value("welcome_loading", "")
        elif has_audio and self.state.is_analyzing:
            dpg.configure_item("welcome_overlay", show=True)
            dpg.configure_item("preview_image", show=False)
            dpg.set_value("welcome_title", "⏳ Analysiere Audio...")
            dpg.set_value("welcome_subtitle", "Features werden extrahiert. Das kann einen Moment dauern.")
            dpg.configure_item("welcome_steps", show=False)
            dpg.configure_item("welcome_drag_hint", show=False)
            dpg.set_value("welcome_loading", "RMS · Onset · Chroma · Tempo · BPM")
            dpg.configure_item("welcome_load_btn", show=False)
        else:
            # Kein Audio geladen → zeige Welcome-Screen, Preview ist versteckt
            dpg.configure_item("welcome_overlay", show=True)
            dpg.configure_item("preview_image", show=False)
            dpg.set_value("welcome_title", "🎵 Audio Visualizer Pro")
            dpg.set_value("welcome_subtitle", "So geht's:")
            dpg.configure_item("welcome_steps", show=True)
            dpg.configure_item("welcome_drag_hint", show=True)
            dpg.set_value("welcome_loading", "")
            dpg.configure_item("welcome_load_btn", show=True)

        # Hintergrund-Status
        has_bg = self.state.background_path is not None and os.path.exists(self.state.background_path)
        if has_bg:
            dpg.set_value("bg_status_text", Path(self.state.background_path).name)
            dpg.configure_item("bg_status_text", color=Theme.TEXT_PRIMARY)
        else:
            dpg.set_value("bg_status_text", "Kein Bild")
            dpg.configure_item("bg_status_text", color=Theme.TEXT_MUTED)

    def _update_chip(self, chip_tag: str, color: tuple):
        """Aktualisiert die Farbe eines Status-Chips."""
        # Der Chip hat eine color_button als erstes Kind
        children = dpg.get_item_children(chip_tag, 1)
        if children:
            dpg.set_value(children[0], color + (255,))

    # -------------------------------------------------------------------------
    # File Dialogs
    # -------------------------------------------------------------------------

    def _setup_file_dialogs(self):
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_audio_selected,
            id="audio_file_dialog",
            width=700,
            height=500,
        ):
            dpg.add_file_extension("Audio (.mp3 .wav .flac .aac .ogg .m4a){.mp3,.wav,.flac,.aac,.ogg,.m4a}")
            dpg.add_file_extension(".*")

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_background_selected,
            id="bg_file_dialog",
            width=700,
            height=500,
        ):
            dpg.add_file_extension("Bilder (.png .jpg .jpeg .webp){.png,.jpg,.jpeg,.webp}")
            dpg.add_file_extension(".*")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def _show_audio_dialog(self, sender=None, app_data=None):
        if self.state.last_audio_dir and os.path.isdir(self.state.last_audio_dir):
            dpg.configure_item("audio_file_dialog", default_path=self.state.last_audio_dir)
        dpg.show_item("audio_file_dialog")

    def _show_bg_dialog(self, sender=None, app_data=None):
        if self.state.last_bg_dir and os.path.isdir(self.state.last_bg_dir):
            dpg.configure_item("bg_file_dialog", default_path=self.state.last_bg_dir)
        dpg.show_item("bg_file_dialog")

    def _show_about(self):
        tag = "about_window"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        with dpg.window(label="Über", tag=tag, modal=True, width=440, height=320, no_resize=True):
            dpg.add_spacer(height=12)
            dpg.add_text("Audio Visualizer Pro", color=Theme.TEXT_PRIMARY)
            dpg.bind_item_font(dpg.last_item(), self._font_header)
            dpg.add_text("Version 3.0", color=Theme.TEXT_MUTED)
            dpg.add_separator()
            dpg.add_spacer(height=8)
            dpg.add_text(
                "GPU-beschleunigte Audio-Visualisierung mit KI-gestützter\n"
                "Parameter-Optimierung und professionellem Post-Processing.",
                color=Theme.TEXT_SECONDARY,
            )
            dpg.add_spacer(height=12)
            with dpg.table(header_row=False):
                dpg.add_table_column()
                dpg.add_table_column()
                techs = [
                    ("ModernGL", "GPU-Rendering"),
                    ("FFmpeg", "Video-Encoding"),
                    ("Gemini", "KI-Analyse"),
                    ("DearPyGui", "Native UI"),
                ]
                for name, desc in techs:
                    with dpg.table_row():
                        dpg.add_text(f"• {name}", color=Theme.ACCENT_PRIMARY)
                        dpg.add_text(desc, color=Theme.TEXT_MUTED)
            dpg.add_spacer(height=16)
            dpg.add_button(label="Schließen", callback=lambda: dpg.delete_item(tag), width=-1)

    def _show_shortcuts(self):
        tag = "shortcuts_window"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        with dpg.window(label="Tastenkürzel", tag=tag, modal=True, width=380, height=220, no_resize=True):
            dpg.add_text("Steuerung", color=Theme.TEXT_PRIMARY)
            dpg.bind_item_font(dpg.last_item(), self._font_card_title)
            dpg.add_separator()
            dpg.add_spacer(height=8)
            with dpg.table(header_row=False):
                dpg.add_table_column()
                dpg.add_table_column()
                for key, desc in [
                    ("Ctrl + O", "Audio laden"),
                    ("Ctrl + B", "Hintergrund laden"),
                    ("Ctrl + E", "Export starten"),
                    ("Escape", "Abbrechen"),
                ]:
                    with dpg.table_row():
                        dpg.add_text(key, color=Theme.ACCENT_PRIMARY)
                        dpg.add_text(desc, color=Theme.TEXT_SECONDARY)
            dpg.add_spacer(height=12)
            dpg.add_button(label="Schließen", callback=lambda: dpg.delete_item(tag), width=-1)

    def _on_audio_selected(self, sender, app_data):
        if app_data.get("selections"):
            selections = list(app_data["selections"].values())
            if selections:
                path = selections[0]
                ext = Path(path).suffix.lower()
                valid_exts = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
                if ext not in valid_exts:
                    self._show_error_modal(f"Ungültiges Dateiformat: {ext}\n\nErlaubt: MP3, WAV, FLAC, AAC, OGG, M4A")
                    return
                self.state.audio_path = path
                self.state.last_audio_dir = str(Path(path).parent)
                self.state.add_recent_file(path)
                self._refresh_recent_files_ui()
                dpg.set_value("audio_status", f"{Path(path).name}")
                dpg.configure_item("audio_status", color=Theme.TEXT_PRIMARY)
                self._analyze_audio()
                self._request_preview_update()
                self._update_status_indicators()
                self._set_status(f"Audio geladen: {Path(path).name}", "ok")
                # Automatisch gecachte Zitate laden
                self._load_cached_quotes(path)

    def _on_background_selected(self, sender, app_data):
        if app_data.get("selections"):
            selections = list(app_data["selections"].values())
            if selections:
                path = selections[0]
                self.state.background_path = path
                self.state.last_bg_dir = str(Path(path).parent)
                self._request_preview_update()
                self._update_status_indicators()
                self._set_status(f"Hintergrund geladen: {Path(path).name}", "ok")

    def _on_visualizer_changed(self, sender, app_data):
        self.state.visualizer_type = dpg.get_value(sender)
        self.state.viz_extra_params = {}
        self.state.ki_suggested_colors = {}
        dpg.set_value("ki_colors_text", "")
        self._rebuild_viz_param_controls()
        self._request_preview_update()

    def _on_param_changed(self, sender, app_data):
        tag = dpg.get_item_alias(sender) or sender
        mapping = {
            "param_offset_x": ("viz_offset_x", float),
            "param_offset_y": ("viz_offset_y", float),
            "param_scale": ("viz_scale", float),
            "param_bg_blur": ("bg_blur", float),
            "param_bg_vignette": ("bg_vignette", float),
            "param_bg_opacity": ("bg_opacity", float),
            "param_pp_contrast": ("pp_contrast", float),
            "param_pp_saturation": ("pp_saturation", float),
            "param_pp_brightness": ("pp_brightness", float),
            "param_pp_warmth": ("pp_warmth", float),
            "param_pp_grain": ("pp_grain", float),
            "param_preview_time": ("preview_time_percent", float),
            "param_base_hue": ("base_hue", float),
            "param_color_saturation": ("color_saturation", float),
        }
        if tag in mapping:
            attr, typ = mapping[tag]
            setattr(self.state, attr, typ(dpg.get_value(sender)))
            # Zeit-Anzeige aktualisieren
            if tag == "param_preview_time":
                self._update_preview_time_text()
            self._request_preview_update()

    def _update_preview_time_text(self):
        """Aktualisiert die Zeit-Anzeige unter der Preview."""
        if self.state.audio_duration > 0:
            total = self.state.audio_duration
            pos = self.state.preview_time_percent * total
            dpg.set_value("preview_time_text", f"{pos:.1f}s / {total:.1f}s")
        else:
            dpg.set_value("preview_time_text", "--:-- / --:--")

    def _update_audio_info_text(self):
        """Aktualisiert die Audio-Info-Anzeige."""
        if self.state.features is None:
            dpg.set_value("audio_info_text", "")
            return
        f = self.state.features
        parts = []
        if hasattr(f, 'tempo') and f.tempo > 0:
            parts.append(f"🎼 {f.tempo:.0f} BPM")
        if hasattr(f, 'mode') and f.mode:
            parts.append(f"📢 {f.mode}")
        if hasattr(f, 'sample_rate') and f.sample_rate:
            parts.append(f"🔊 {f.sample_rate/1000:.1f} kHz")
        dpg.set_value("audio_info_text", "  ·  ".join(parts))

    def _on_color_mode_changed(self, sender, app_data):
        val = dpg.get_value(sender)
        # Map Anzeige-Name zu internem Wert
        mode_map = {
            "chroma (dynamisch)": "chroma",
            "fixed (eine Farbe)": "fixed",
            "monochrome": "monochrome",
            "warm": "warm",
            "cool": "cool",
        }
        self.state.color_mode = mode_map.get(val, "chroma")
        self._request_preview_update()

    def _on_resolution_changed(self, sender, app_data):
        res_str = dpg.get_value(sender)
        match = re.search(r"(\d+)[x×](\d+)", res_str)
        if match:
            self.state.resolution = (int(match.group(1)), int(match.group(2)))
        else:
            self.state.resolution = (1920, 1080)

    def _on_output_dir_changed(self, sender, app_data):
        self.state.output_dir = dpg.get_value(sender)

    def _on_gpu_encode_changed(self, sender, app_data):
        self.state.gpu_encode = dpg.get_value(sender)

    def _on_fps_changed(self, sender, app_data):
        val = dpg.get_value(sender)
        self.state.render_fps = int(val)
        self.state.preview_fps = int(val)

    def _on_codec_changed(self, sender, app_data):
        val = dpg.get_value(sender)
        codec_map = {
            "h264 (kompatibel)": "h264",
            "h265 (klein)": "h265",
            "prores (Editing)": "prores",
        }
        self.state.codec = codec_map.get(val, "h264")

    def _on_quality_changed(self, sender, app_data):
        val = dpg.get_value(sender)
        quality_map = {
            "Draft (schnell)": "draft",
            "Standard": "standard",
            "High": "high",
            "Lossless": "lossless",
        }
        self.state.quality = quality_map.get(val, "high")

    def _on_quotes_enabled_changed(self, sender, app_data):
        self.state.quotes_enabled = dpg.get_value(sender)
        self._request_preview_update()

    def _on_extract_quotes_clicked(self, sender, app_data):
        if self.state.is_extracting_quotes:
            return
        if not self.gemini:
            dpg.set_value("quotes_status_text", "KI nicht verfügbar. Prüfe API-Key.")
            dpg.configure_item("quotes_status_text", color=Theme.STATUS_ERR)
            return
        if not self.state.audio_path or not os.path.exists(self.state.audio_path):
            dpg.set_value("quotes_status_text", "Lade zuerst eine Audio-Datei.")
            dpg.configure_item("quotes_status_text", color=Theme.STATUS_ERR)
            return
        if self.state.features is None:
            dpg.set_value("quotes_status_text", "Audio wird noch analysiert...")
            dpg.configure_item("quotes_status_text", color=Theme.STATUS_WARN)
            return

        self.state.is_extracting_quotes = True
        dpg.configure_item("btn_extract_quotes", label="⏳ Extrahiere...")
        dpg.set_value("quotes_status_text", "Gemini analysiert Audio...")
        dpg.configure_item("quotes_status_text", color=Theme.TEXT_SECONDARY)

        audio_path = self.state.audio_path
        audio_duration = self.state.features.duration

        def _extract():
            def _progress(msg):
                self._quotes_queue.put({"type": "progress", "message": msg})
            try:
                quotes = self.gemini.extract_quotes(
                    audio_path,
                    audio_duration=audio_duration,
                    progress_callback=_progress
                )
                # Zeit begrenzen
                for q in quotes:
                    q.start_time = max(0.0, min(q.start_time, audio_duration - 1.0))
                    q.end_time = max(q.start_time + 1.0, min(q.end_time, audio_duration))
                self._quotes_queue.put({"type": "done", "quotes": quotes})
            except Exception as e:
                self._quotes_queue.put({"type": "error", "message": str(e)})

        threading.Thread(target=_extract, daemon=True).start()

    def _on_demo_quotes_clicked(self, sender, app_data):
        duration = self.state.audio_duration if self.state.features else 60.0
        quotes = [
            Quote(text="Das Abenteuer beginnt jetzt.", start_time=min(5.0, duration * 0.05), end_time=min(13.0, duration * 0.1), confidence=0.95),
            Quote(text="Jeder Moment ist eine Chance.", start_time=min(duration * 0.3, duration - 15.0), end_time=min(duration * 0.3 + 8.0, duration - 5.0), confidence=0.90),
            Quote(text="Bleib dran, es lohnt sich.", start_time=min(duration * 0.6, duration - 10.0), end_time=min(duration * 0.6 + 8.0, duration - 2.0), confidence=0.88),
        ]
        if self.state.features:
            quotes = refine_quote_timestamps(quotes, self.state.features)
        self.state.quotes = quotes
        dpg.set_value("quotes_status_text", f"{len(self.state.quotes)} Demo-Zitate erstellt!")
        dpg.configure_item("quotes_status_text", color=Theme.STATUS_OK)
        self._refresh_quotes_list()
        self._request_preview_update()

    def _process_quotes_queue(self):
        if self._quotes_queue.empty():
            return
        # Sammle alle Messages, behandle Progress sofort, merke das finale Ergebnis
        final_msg = None
        while True:
            try:
                msg = self._quotes_queue.get_nowait()
                if msg.get("type") == "progress":
                    # Progress sofort anzeigen
                    status = msg.get("message", "")
                    dpg.set_value("quotes_status_text", f"⏳ {status}")
                    dpg.configure_item("quotes_status_text", color=Theme.TEXT_SECONDARY)
                else:
                    final_msg = msg
            except queue.Empty:
                break
        if final_msg is None:
            return
        msg_type = final_msg.get("type")
        self.state.is_extracting_quotes = False
        dpg.configure_item("btn_extract_quotes", label="Key-Zitate extrahieren")
        if msg_type == "done":
            quotes = final_msg.get("quotes", [])
            # Zeitstempel mit Audio-Analyse verfeinern
            if self.state.features:
                quotes = refine_quote_timestamps(quotes, self.state.features)
            self.state.quotes = quotes
            # Zitate automatisch cachen
            if self.state.audio_path:
                save_quotes(self.state.audio_path, quotes)
            dpg.set_value("quotes_status_text", f"✅ {len(quotes)} Zitate extrahiert!")
            dpg.configure_item("quotes_status_text", color=Theme.STATUS_OK)
            self._refresh_quotes_list()
            self._request_preview_update()
            self._set_status(f"{len(quotes)} Zitate extrahiert", "ok")
        elif msg_type == "local_done":
            quotes = final_msg.get("quotes", [])
            if self.state.features:
                quotes = refine_quote_timestamps(quotes, self.state.features)
            self.state.quotes = quotes
            if self.state.audio_path:
                save_quotes(self.state.audio_path, quotes)
            dpg.set_value("quotes_status_text", f"💻 {len(quotes)} Zitate lokal extrahiert")
            dpg.configure_item("quotes_status_text", color=Theme.STATUS_OK)
            self._refresh_quotes_list()
            self._request_preview_update()
            self._set_status(f"{len(quotes)} Zitate lokal extrahiert", "ok")
        elif msg_type == "local_error":
            # Lokale Extraktion fehlgeschlagen -> Demo-Zitate
            self.state.quotes = [
                Quote(text="Das Abenteuer beginnt jetzt.", start_time=0.0, end_time=5.0, confidence=0.95),
                Quote(text="Jeder Moment ist eine Chance.", start_time=10.0, end_time=15.0, confidence=0.90),
            ]
            self._refresh_quotes_list()
            self._request_preview_update()
            dpg.set_value("quotes_status_text", "⚠️ Lokale Extraktion fehlgeschlagen, Demo-Zitate geladen")
            dpg.configure_item("quotes_status_text", color=Theme.STATUS_WARN)
        elif msg_type == "error":
            err_msg = final_msg.get('message', 'Unbekannter Fehler')
            dpg.set_value("quotes_status_text", f"❌ Fehler: {err_msg}")
            dpg.configure_item("quotes_status_text", color=Theme.STATUS_ERR)
            self._set_status(f"Zitat-Extraktion fehlgeschlagen: {err_msg}", "err")
            # Versuche lokalen Fallback (faster-whisper)
            self._try_local_quote_fallback()

    def _try_local_quote_fallback(self):
        """Versucht lokale Transkription als Fallback, wenn Gemini fehlschlägt."""
        try:
            from src.local_transcription import is_available, LocalTranscriber
            if not is_available():
                # Kein lokaler Fallback verfuegbar -> Demo-Zitate
                self.state.quotes = [
                    Quote(text="Das Abenteuer beginnt jetzt.", start_time=0.0, end_time=5.0, confidence=0.95),
                    Quote(text="Jeder Moment ist eine Chance.", start_time=10.0, end_time=15.0, confidence=0.90),
                ]
                self._refresh_quotes_list()
                self._request_preview_update()
                return
            # Lokale Transkription im Hintergrund starten
            dpg.set_value("quotes_status_text", "💻 Lokale Transkription wird gestartet...")
            dpg.configure_item("quotes_status_text", color=Theme.TEXT_SECONDARY)
            audio_path = self.state.audio_path
            audio_duration = self.state.audio_duration

            def _local_extract():
                try:
                    transcriber = LocalTranscriber(model_size="base")
                    quotes = transcriber.extract_quotes(
                        audio_path,
                        audio_duration=audio_duration,
                        max_quotes=min(5, max(2, int(audio_duration / 90))) if audio_duration else 5,
                    )
                    self._quotes_queue.put({"type": "local_done", "quotes": quotes})
                except Exception as e:
                    self._quotes_queue.put({"type": "local_error", "message": str(e)})

            threading.Thread(target=_local_extract, daemon=True).start()
        except Exception:
            # Fallback auf Demo-Zitate
            self.state.quotes = [
                Quote(text="Das Abenteuer beginnt jetzt.", start_time=0.0, end_time=5.0, confidence=0.95),
                Quote(text="Jeder Moment ist eine Chance.", start_time=10.0, end_time=15.0, confidence=0.90),
            ]
            self._refresh_quotes_list()
            self._request_preview_update()

    def _load_cached_quotes(self, audio_path: str):
        """Laedt gecachte Zitate fuer die Audio-Datei."""
        cached = load_quotes(audio_path)
        if cached:
            self.state.quotes = cached
            dpg.set_value("quotes_status_text", f"{len(cached)} gecachte Zitate geladen")
            dpg.configure_item("quotes_status_text", color=Theme.STATUS_OK)
            self._refresh_quotes_list()
            self._request_preview_update()
            self._set_status(f"{len(cached)} gecachte Zitate geladen", "ok")
        else:
            self.state.quotes = []
            dpg.set_value("quotes_status_text", "Keine gecachten Zitate")
            dpg.configure_item("quotes_status_text", color=Theme.TEXT_MUTED)
            self._refresh_quotes_list()
            self._request_preview_update()

    def _refresh_quotes_list(self):
        """Aktualisiert die Anzeige der Quote-Liste mit editierbaren Zeilen."""
        container = "quotes_list_container"
        if not dpg.does_item_exist(container):
            return

        # Container leeren
        dpg.delete_item(container, children_only=True)

        if not self.state.quotes:
            dpg.push_container_stack(container)
            try:
                dpg.add_text("Noch keine Zitate. Extrahiere oder füge manuell hinzu.",
                             color=Theme.TEXT_MUTED, wrap=320)
            finally:
                dpg.pop_container_stack()
            dpg.set_value("quotes_status_text", "")
            return

        dpg.push_container_stack(container)
        try:
            for i, q in enumerate(self.state.quotes):
                with dpg.group(horizontal=True):
                    # Index
                    dpg.add_text(f"{i+1}.", color=Theme.TEXT_MUTED)
                    # Text (editierbar)
                    dpg.add_input_text(
                        default_value=q.text,
                        width=180,
                        callback=lambda s, a, idx=i: self._on_quote_text_changed(idx, s, a),
                        tag=f"quote_txt_{i}",
                    )
                    # Start-Zeit
                    dpg.add_input_float(
                        default_value=q.start_time,
                        width=60,
                        format="%.1f",
                        step=0.5,
                        callback=lambda s, a, idx=i: self._on_quote_start_changed(idx, s, a),
                        tag=f"quote_start_{i}",
                    )
                    dpg.add_text("s", color=Theme.TEXT_MUTED)
                    # End-Zeit
                    dpg.add_input_float(
                        default_value=q.end_time,
                        width=60,
                        format="%.1f",
                        step=0.5,
                        callback=lambda s, a, idx=i: self._on_quote_end_changed(idx, s, a),
                        tag=f"quote_end_{i}",
                    )
                    dpg.add_text("s", color=Theme.TEXT_MUTED)
                    # Confidence (read-only)
                    dpg.add_text(f"({q.confidence:.0%})", color=Theme.TEXT_MUTED)
                    # Löschen-Button
                    dpg.add_button(
                        label="🗑️",
                        callback=lambda s, a, idx=i: self._on_delete_quote(idx),
                        width=30,
                        tag=f"quote_del_{i}",
                    )
                    dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())
                dpg.add_separator()
        finally:
            dpg.pop_container_stack()

        dpg.set_value("quotes_status_text", f"{len(self.state.quotes)} Zitate")
        dpg.configure_item("quotes_status_text", color=Theme.STATUS_OK)

    def _on_quote_config_changed(self, sender, app_data):
        tag = dpg.get_item_alias(sender) or sender
        if tag == "quote_position":
            self.state.quote_config.position = dpg.get_value(sender)
        elif tag == "quote_font_size":
            self.state.quote_config.font_size = int(dpg.get_value(sender))
        elif tag == "quote_display_duration":
            self.state.quote_config.display_duration = float(dpg.get_value(sender))
        elif tag == "quote_fade_duration":
            self.state.quote_config.fade_duration = float(dpg.get_value(sender))
        elif tag == "quote_max_chars":
            self.state.quote_config.max_chars_per_line = int(dpg.get_value(sender))
        elif tag == "quote_line_spacing":
            self.state.quote_config.line_spacing = int(dpg.get_value(sender))
        elif tag == "quote_slide_animation":
            self.state.quote_config.slide_animation = dpg.get_value(sender)
        elif tag == "quote_scale_in":
            self.state.quote_config.scale_in = bool(dpg.get_value(sender))
        elif tag == "quote_glow_pulse":
            self.state.quote_config.glow_pulse = bool(dpg.get_value(sender))
        elif tag == "quote_compensation_blur":
            self.state.quote_config.compensation_blur = float(dpg.get_value(sender))
        elif tag == "quote_latency_offset":
            self.state.quote_config.latency_offset = float(dpg.get_value(sender))
        self._request_preview_update()

    # -------------------------------------------------------------------------
    # Quote CRUD
    # -------------------------------------------------------------------------

    def _on_quote_text_changed(self, idx: int, sender, app_data):
        """Aktualisiert den Text eines Zitats."""
        if 0 <= idx < len(self.state.quotes):
            self.state.quotes[idx].text = str(app_data).strip()
            self._request_preview_update()

    def _on_quote_start_changed(self, idx: int, sender, app_data):
        """Aktualisiert die Startzeit eines Zitats."""
        if 0 <= idx < len(self.state.quotes):
            self.state.quotes[idx].start_time = float(app_data)
            self._request_preview_update()

    def _on_quote_end_changed(self, idx: int, sender, app_data):
        """Aktualisiert die Endzeit eines Zitats."""
        if 0 <= idx < len(self.state.quotes):
            self.state.quotes[idx].end_time = float(app_data)
            self._request_preview_update()

    def _on_delete_quote(self, idx: int):
        """Loescht ein Zitat aus der Liste."""
        if 0 <= idx < len(self.state.quotes):
            del self.state.quotes[idx]
            self._refresh_quotes_list()
            self._request_preview_update()
            self._set_status(f"Zitat {idx+1} geloescht", "info")

    def _on_add_quote_clicked(self):
        """Oeffnet ein Modal zum Hinzufuegen eines neuen Zitats."""
        tag = "add_quote_modal"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        with dpg.window(label="Zitat hinzufuegen", tag=tag, modal=True, width=400, height=220, no_resize=True):
            dpg.add_spacer(height=8)
            dpg.add_text("Text:", color=Theme.TEXT_SECONDARY)
            dpg.add_input_text(tag="add_quote_text", width=-1, hint="Zitat-Text eingeben...")
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text("Start:", color=Theme.TEXT_SECONDARY)
                dpg.add_input_float(tag="add_quote_start", default_value=0.0, width=80, format="%.1f")
                dpg.add_text("s", color=Theme.TEXT_MUTED)
                dpg.add_spacer(width=20)
                dpg.add_text("Ende:", color=Theme.TEXT_SECONDARY)
                dpg.add_input_float(tag="add_quote_end", default_value=5.0, width=80, format="%.1f")
                dpg.add_text("s", color=Theme.TEXT_MUTED)
            dpg.add_spacer(height=12)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Hinzufuegen", callback=self._on_add_quote_confirmed, width=120)
                dpg.bind_item_theme(dpg.last_item(), self._make_accent_button_theme(Theme.ACCENT_PRIMARY))
                dpg.add_button(label="Abbrechen", callback=lambda: dpg.delete_item(tag), width=100)
                dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())

    def _on_add_quote_confirmed(self, sender, app_data):
        """Bestaetigt das Hinzufuegen eines neuen Zitats."""
        text = dpg.get_value("add_quote_text").strip()
        start = dpg.get_value("add_quote_start")
        end = dpg.get_value("add_quote_end")
        if not text:
            self._show_error_modal("Bitte einen Text eingeben.")
            return
        if end <= start:
            self._show_error_modal("Endzeit muss nach der Startzeit liegen.")
            return
        self.state.quotes.append(Quote(
            text=text,
            start_time=float(start),
            end_time=float(end),
            confidence=1.0
        ))
        dpg.delete_item("add_quote_modal")
        self._refresh_quotes_list()
        self._request_preview_update()
        self._set_status("Neues Zitat hinzugefuegt", "ok")

    def _on_save_quotes_clicked(self):
        """Speichert die aktuellen Zitate in den Cache."""
        if not self.state.audio_path:
            self._show_error_modal("Keine Audio-Datei geladen. Zitate koennen nicht gespeichert werden.")
            return
        save_quotes(self.state.audio_path, self.state.quotes)
        dpg.set_value("quotes_status_text", f"{len(self.state.quotes)} Zitate gespeichert")
        dpg.configure_item("quotes_status_text", color=Theme.STATUS_OK)
        self._set_status("Zitate gespeichert", "ok")

    def _on_reset_quotes_clicked(self):
        """Laedt Zitate aus dem Cache zurueck (verwirft Aenderungen)."""
        if not self.state.audio_path:
            self.state.quotes = []
            self._refresh_quotes_list()
            self._request_preview_update()
            return
        cached = load_quotes(self.state.audio_path)
        if cached is not None:
            self.state.quotes = cached
            self._refresh_quotes_list()
            self._request_preview_update()
            self._set_status("Zitate aus Cache zurueckgesetzt", "info")
        else:
            self.state.quotes = []
            self._refresh_quotes_list()
            self._request_preview_update()
            self._set_status("Keine gecachten Zitate gefunden", "warn")

    def _request_preview_update(self):
        self._preview_debounce_time = time.time()

    def _update_analysis_spinner(self):
        """Animiert den Analyse-Lade-Text."""
        if not self.state.is_analyzing:
            return
        frames = ["⏳", "⌛", "⏳", "⌛"]
        idx = int(time.time() * 3) % len(frames)
        dpg.set_value("welcome_loading", f"{frames[idx]} RMS · Onset · Chroma · Tempo · BPM")

    # -------------------------------------------------------------------------
    # Audio-Analyse
    # -------------------------------------------------------------------------

    def _analyze_audio(self):
        if not self.state.audio_path or not os.path.exists(self.state.audio_path):
            return
        if self.state.is_analyzing:
            return
        self.state.is_analyzing = True
        self.state.features = None
        self.state.audio_duration = 0.0
        self._set_status("Audio wird analysiert...", "warn")
        self._update_status_indicators()

        target_path = self.state.audio_path

        def _analyze():
            try:
                analyzer = AudioAnalyzer()
                features = analyzer.analyze(target_path, fps=self.state.preview_fps)
                self._analyze_result = ("ok", target_path, features)
            except Exception as e:
                self._analyze_result = ("error", target_path, str(e))

        self._analyze_result = None
        threading.Thread(target=_analyze, daemon=True).start()

    def _process_analyze_result(self):
        """Wird im Main Thread aufgerufen, um Analyse-Ergebnisse sicher zu verarbeiten."""
        if self._analyze_result is None:
            return
        status, path, data = self._analyze_result
        self._analyze_result = None
        self.state.is_analyzing = False

        # Falls der Nutzer zwischenzeitlich eine andere Datei geladen hat
        if self.state.audio_path != path:
            if self.state.audio_path and os.path.exists(self.state.audio_path):
                self._analyze_audio()
            self._update_status_indicators()
            return

        if status == "ok":
            self.state.features = data
            self.state.audio_duration = data.duration
            self._set_status(f"Analyse fertig: {data.duration:.1f}s @ {data.tempo:.0f} BPM", "ok")
            self._update_preview_time_text()
            self._update_audio_info_text()
            self._request_preview_update()
        else:
            self._set_status(f"Analyse-Fehler: {data}", "error")
            self._show_error_modal(f"Audio-Analyse fehlgeschlagen:\n\n{data}\n\nPrüfe ob die Datei korrupt ist oder FFmpeg installiert ist.")
        self._update_status_indicators()

    # -------------------------------------------------------------------------
    # Preview Rendering
    # -------------------------------------------------------------------------

    def _update_preview(self):
        now = time.time()
        # Debounce: warte bis Slider-Bewegungen aufgehört haben
        if now - self._preview_debounce_time < self._preview_debounce_delay:
            return
        if now - self._last_preview_update < self._preview_min_interval:
            return
        self._last_preview_update = now

        has_audio = self.state.audio_path and os.path.exists(self.state.audio_path)
        has_bg = self.state.background_path and os.path.exists(self.state.background_path)

        if not has_audio and not has_bg:
            return

        params_hash = self.state.preview_params_hash()
        if params_hash == self.state._preview_params_hash and self.state._preview_image is not None:
            return

        # Retry-Cooldown: Nach jedem Fehler mindestens 3s warten (verhindert Spam)
        if now - self._preview_last_error_time < self._preview_error_cooldown:
            return

        # Fallback: nur Hintergrundbild anzeigen wenn kein Audio da ist
        if not has_audio and has_bg:
            self._render_background_only()
            self.state._preview_params_hash = params_hash
            return

        if not has_audio:
            return
        if self.state.is_analyzing:
            return
        if self.state.features is None:
            return

        dpg.configure_item("preview_loading_text", show=True)
        try:
            preview_quotes = self.state.quotes if self.state.quotes_enabled else None
            preview_quote_cfg = self.state.quote_config if self.state.quotes_enabled else None
            img = render_gpu_preview(
                audio_path=self.state.audio_path,
                visualizer_type=self.state.visualizer_type,
                params=self.state.get_params(),
                width=self.state.preview_width,
                height=self.state.preview_height,
                fps=self.state.preview_fps,
                preview_time_percent=self.state.preview_time_percent,
                background_image=self.state.background_path,
                background_blur=self.state.bg_blur,
                background_vignette=self.state.bg_vignette,
                background_opacity=self.state.bg_opacity,
                postprocess=self.state.get_postprocess(),
                quotes=preview_quotes,
                quote_config=preview_quote_cfg,
                viz_offset_x=self.state.viz_offset_x,
                viz_offset_y=self.state.viz_offset_y,
                viz_scale=self.state.viz_scale,
                features=self.state.features,
            )
            if img is not None:
                self.state._preview_image = img
                self.state._preview_params_hash = params_hash
                self._upload_texture(img)
            else:
                # render_gpu_preview hat intern einen Fehler gefangen und None zurückgegeben
                self._preview_last_error_time = time.time()
                self._preview_error_params_hash = params_hash
        except Exception as e:
            print(f"[Preview] Fehler: {e}")
            self._preview_last_error_time = time.time()
            self._preview_error_params_hash = params_hash
        finally:
            dpg.configure_item("preview_loading_text", show=False)

    def _render_background_only(self):
        """Zeigt das Hintergrundbild allein an, wenn kein Audio geladen ist."""
        w, h = self.state.preview_width, self.state.preview_height
        try:
            img = Image.open(self.state.background_path).convert("RGB")
            img = img.resize((w, h), Image.LANCZOS)

            # Blur anwenden falls konfiguriert
            if self.state.bg_blur > 0.01:
                from PIL import ImageFilter
                img = img.filter(ImageFilter.GaussianBlur(radius=self.state.bg_blur))

            # Vignette + Opacity via NumPy (schneller als Python-Loop)
            if self.state.bg_vignette > 0.01 or self.state.bg_opacity < 0.99:
                overlay = Image.new("RGB", (w, h), (0, 0, 0))
                # Radialer Gradient mit NumPy
                y, x = np.ogrid[:h, :w]
                cx, cy = w / 2, h / 2
                max_dist = np.sqrt(cx**2 + cy**2)
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                norm = dist / max_dist
                # Vignette: dunkler nach außen
                vignette_mask = 1.0 - (norm * self.state.bg_vignette)
                vignette_mask = np.clip(vignette_mask, 0, 1)
                # Opacity mischen
                opacity_mask = self.state.bg_opacity * vignette_mask
                opacity_mask_u8 = (opacity_mask * 255).astype(np.uint8)
                mask_img = Image.fromarray(opacity_mask_u8, mode="L")
                img = Image.composite(img, overlay, mask_img)

            self.state._preview_image = img
            self._upload_texture(img)
        except Exception as e:
            print(f"[Preview] Hintergrund-Fehler: {e}")

    def _upload_texture(self, img: Image.Image):
        img_rgba = img.convert("RGBA")
        arr = np.array(img_rgba, dtype=np.float32) / 255.0
        flat = arr.flatten()
        expected_size = self.state.preview_width * self.state.preview_height * 4
        if flat.size != expected_size:
            img_rgba = img_rgba.resize(
                (self.state.preview_width, self.state.preview_height),
                Image.LANCZOS
            )
            arr = np.array(img_rgba, dtype=np.float32) / 255.0
            flat = arr.flatten()
        self._preview_raw_data[:] = flat[:]
        dpg.set_value(self._preview_texture_tag, self._preview_raw_data)

    # -------------------------------------------------------------------------
    # Video Export
    # -------------------------------------------------------------------------

    def _on_render_clicked(self, sender, app_data):
        if self.state.is_rendering:
            self._show_error_modal("Rendering läuft bereits. Warte bis es fertig ist.")
            return
        if not self.state.audio_path or not os.path.exists(self.state.audio_path):
            self._show_error_modal("Keine Audio-Datei geladen.\n\nKlicke auf 'Audio laden' oder drücke Ctrl+O.")
            return
        if self.state.features is None:
            if self.state.is_analyzing:
                self._show_error_modal("Audio wird noch analysiert.\n\nWarte einen Moment, bis die Analyse abgeschlossen ist.")
            else:
                self._show_error_modal("Audio wurde noch nicht analysiert.\n\nDie Analyse startet automatisch, sobald du eine Datei lädst.")
            return

        self.state.is_rendering = True
        self._cancel_event.clear()
        self._auto_save_project()
        dpg.configure_item("btn_render", label="⏳ Render läuft...")
        dpg.configure_item("btn_cancel_render", enabled=True)
        dpg.configure_item("btn_open_folder", show=False)
        dpg.set_value("render_progress", 0.0)
        self._set_status("Starte Rendering...", "warn")
        self._update_status_indicators()

        while not self._render_queue.empty():
            try:
                self._render_queue.get_nowait()
            except queue.Empty:
                break

        def _render():
            output_path = None
            renderer = None
            try:
                w, h = self.state.resolution
                fps = self.state.render_fps
                out_dir = Path(self.state.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(out_dir / f"visualization_{ts}.mp4")

                def _progress_callback(frame, total):
                    self._render_queue.put({"type": "progress", "frame": frame, "total": total})

                renderer = GPUBatchRenderer(width=w, height=h, fps=fps)
                render_quotes = self.state.quotes if self.state.quotes_enabled else None
                render_quote_cfg = self.state.quote_config if self.state.quotes_enabled else None
                renderer.render(
                    audio_path=self.state.audio_path,
                    visualizer_type=self.state.visualizer_type,
                    output_path=output_path,
                    features=self.state.features,
                    params=self.state.get_params(),
                    background_image=self.state.background_path,
                    background_blur=self.state.bg_blur,
                    background_vignette=self.state.bg_vignette,
                    background_opacity=self.state.bg_opacity,
                    postprocess=self.state.get_postprocess(),
                    quotes=render_quotes,
                    quote_config=render_quote_cfg,
                    codec=self.state.codec,
                    quality=self.state.quality,
                    gpu_encode=self.state.gpu_encode,
                    viz_offset_x=self.state.viz_offset_x,
                    viz_offset_y=self.state.viz_offset_y,
                    viz_scale=self.state.viz_scale,
                    progress_callback=_progress_callback,
                    cancel_event=self._cancel_event,
                )

                if self._cancel_event.is_set():
                    self._render_queue.put({"type": "cancelled"})
                else:
                    self._render_queue.put({"type": "done", "path": output_path})
            except Exception as e:
                self._render_queue.put({"type": "error", "message": str(e)})
            finally:
                if renderer is not None:
                    try:
                        renderer.release()
                    except Exception:
                        pass

        threading.Thread(target=_render, daemon=True).start()

    def _on_cancel_render_clicked(self, sender, app_data):
        if self.state.is_rendering:
            self._cancel_event.set()
            self._set_status("Abbruch angefordert...", "warn")

    def _show_error_modal(self, message: str):
        """Zeigt ein modales Fehler-Fenster mit klarem Text."""
        tag = "error_modal"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        lines = message.count('\n') + 1
        height = min(400, max(160, 60 + lines * 20))
        with dpg.window(label="Hinweis", tag=tag, modal=True, width=440, height=height, no_resize=True):
            dpg.add_spacer(height=12)
            dpg.add_text(message, color=Theme.TEXT_PRIMARY, wrap=400)
            dpg.add_spacer(height=16)
            dpg.add_button(label="Verstanden", callback=lambda: dpg.delete_item(tag), width=-1)

    def _on_open_output_folder(self, sender, app_data):
        """Öffnet den Output-Ordner im Explorer."""
        import subprocess
        path = os.path.abspath(self.state.output_dir)
        if os.path.exists(path):
            subprocess.Popen(["explorer", path])
        else:
            self._set_status("Output-Ordner nicht gefunden.", "warn")

    # -------------------------------------------------------------------------
    # Projekt-Preset-System
    # -------------------------------------------------------------------------

    def _projects_dir(self) -> Path:
        """Gibt das Projekte-Verzeichnis zurueck."""
        p = Path("projects")
        p.mkdir(exist_ok=True)
        return p

    def _list_project_files(self) -> list:
        """Listet alle gespeicherten Projekte auf (neueste zuerst)."""
        pdir = self._projects_dir()
        files = sorted(pdir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        return [f.stem for f in files]

    def _refresh_project_list(self):
        """Aktualisiert die Quick-Load-Combo."""
        projects = self._list_project_files()
        items = [""] + projects[:5]
        dpg.configure_item("project_quick_load", items=items)

    def _on_save_project_clicked(self, sender, app_data):
        """Speichert das aktuelle Projekt."""
        tag = "save_project_modal"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        with dpg.window(label="Projekt speichern", tag=tag, modal=True, width=360, height=140, no_resize=True):
            dpg.add_spacer(height=8)
            dpg.add_text("Name:", color=Theme.TEXT_SECONDARY)
            dpg.add_input_text(tag="save_project_name", width=-1, hint="z.B. podcast_ep12")
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Speichern", callback=self._on_save_project_confirmed, width=100)
                dpg.bind_item_theme(dpg.last_item(), self._make_accent_button_theme(Theme.ACCENT_PRIMARY))
                dpg.add_button(label="Abbrechen", callback=lambda: dpg.delete_item(tag), width=100)
                dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())

    def _on_save_project_confirmed(self, sender, app_data):
        """Bestaetigt das Speichern eines Projekts."""
        name = dpg.get_value("save_project_name").strip()
        if not name:
            self._show_error_modal("Bitte einen Projektnamen eingeben.")
            return
        name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        pdir = self._projects_dir()
        file_path = pdir / f"{name}.json"
        data = self.state.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            import json
            json.dump(data, f, ensure_ascii=False, indent=2)
        dpg.delete_item("save_project_modal")
        self._refresh_project_list()
        self._set_status(f"Projekt '{name}' gespeichert", "ok")

    def _on_load_project_clicked(self, sender, app_data):
        """Oeffnet den Projekt-Laden-Dialog."""
        tag = "load_project_modal"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        projects = self._list_project_files()
        with dpg.window(label="Projekt laden", tag=tag, modal=True, width=360, height=200, no_resize=True):
            dpg.add_spacer(height=8)
            if not projects:
                dpg.add_text("Keine gespeicherten Projekte.", color=Theme.TEXT_MUTED)
            else:
                dpg.add_text("Projekt auswaehlen:", color=Theme.TEXT_SECONDARY)
                for pname in projects[:10]:
                    dpg.add_button(
                        label=pname,
                        callback=lambda s, a, n=pname: self._load_project_by_name(n, tag),
                        width=-1,
                    )
                    dpg.bind_item_theme(dpg.last_item(), self._make_secondary_button_theme())
            dpg.add_spacer(height=8)
            dpg.add_button(label="Schliessen", callback=lambda: dpg.delete_item(tag), width=-1)

    def _on_quick_load_project(self, sender, app_data):
        """Laedt ein Projekt aus der Quick-Load-Combo."""
        name = dpg.get_value(sender).strip()
        if name:
            self._load_project_by_name(name)

    def _load_project_by_name(self, name: str, modal_tag: str = None):
        """Laedt ein Projekt anhand des Namens."""
        pdir = self._projects_dir()
        file_path = pdir / f"{name}.json"
        if not file_path.exists():
            self._show_error_modal(f"Projekt '{name}' nicht gefunden.")
            return
        try:
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state = AppState.from_dict(data)
            self._sync_ui_to_state()
            if modal_tag and dpg.does_item_exist(modal_tag):
                dpg.delete_item(modal_tag)
            self._refresh_project_list()
            self._set_status(f"Projekt '{name}' geladen", "ok")
        except Exception as e:
            self._show_error_modal(f"Fehler beim Laden: {e}")

    def _sync_ui_to_state(self):
        """Synchronisiert alle UI-Controls mit dem aktuellen AppState."""
        # Audio
        if self.state.audio_path:
            dpg.set_value("audio_status", Path(self.state.audio_path).name)
        # Visualizer
        dpg.set_value("viz_combo", self.state.visualizer_type)
        dpg.set_value("param_offset_x", self.state.viz_offset_x)
        dpg.set_value("param_offset_y", self.state.viz_offset_y)
        dpg.set_value("param_scale", self.state.viz_scale)
        dpg.set_value("param_color_mode", self.state.color_mode)
        dpg.set_value("param_base_hue", self.state.base_hue)
        dpg.set_value("param_color_saturation", self.state.color_saturation)
        # Background
        dpg.set_value("param_bg_blur", self.state.bg_blur)
        dpg.set_value("param_bg_vignette", self.state.bg_vignette)
        dpg.set_value("param_bg_opacity", self.state.bg_opacity)
        # Post-Process
        dpg.set_value("param_pp_contrast", self.state.pp_contrast)
        dpg.set_value("param_pp_saturation", self.state.pp_saturation)
        dpg.set_value("param_pp_brightness", self.state.pp_brightness)
        dpg.set_value("param_pp_warmth", self.state.pp_warmth)
        dpg.set_value("param_pp_grain", self.state.pp_grain)
        # Export
        dpg.set_value("chk_gpu_encode", self.state.gpu_encode)
        fps_val = str(self.state.render_fps)
        dpg.set_value("fps_combo", fps_val if fps_val in ["24", "30", "60"] else "30")
        codec_display = {
            "h264": "h264 (kompatibel)",
            "h265": "h265 (klein)",
            "prores": "prores (Editing)",
        }.get(self.state.codec, "h264 (kompatibel)")
        dpg.set_value("codec_combo", codec_display)
        quality_display = {
            "draft": "Draft (schnell)",
            "standard": "Standard",
            "high": "High",
            "lossless": "Lossless",
        }.get(self.state.quality, "High")
        dpg.set_value("quality_combo", quality_display)
        # Quotes
        dpg.set_value("chk_quotes_enabled", self.state.quotes_enabled)
        dpg.set_value("quote_position", self.state.quote_config.position)
        dpg.set_value("quote_font_size", self.state.quote_config.font_size)
        dpg.set_value("quote_display_duration", self.state.quote_config.display_duration)
        self._rebuild_viz_param_controls()
        self._refresh_quotes_list()
        self._request_preview_update()

    def _auto_save_project(self):
        """Auto-Save der aktuellen Config fuer naechsten Start."""
        try:
            cache_file = Path(".cache") / "last_project.json"
            cache_file.parent.mkdir(exist_ok=True)
            import json
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _try_load_auto_save(self):
        """Versucht, den Auto-Save beim Start zu laden."""
        try:
            cache_file = Path(".cache") / "last_project.json"
            if cache_file.exists():
                import json
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.state = AppState.from_dict(data)
                self._sync_ui_to_state()
                self._set_status("Letztes Projekt automatisch geladen", "ok")
        except Exception:
            pass

    def _poll_render_queue(self):
        if not self.state.is_rendering and self._render_queue.empty():
            return

        final_msg = None
        while True:
            try:
                msg = self._render_queue.get_nowait()
                final_msg = msg
            except queue.Empty:
                break

        if final_msg is None:
            return

        msg_type = final_msg.get("type")
        if msg_type == "progress":
            frame = final_msg["frame"]
            total = final_msg["total"]
            progress = frame / total
            dpg.set_value("render_progress", progress)
            pct = progress * 100
            self._set_status(f"Rendering... {pct:.1f}% ({frame}/{total})", "warn")
            dpg.set_value("render_status_text", f"Geschätzte Restzeit: {self._estimate_time_remaining(frame, total)}")
        elif msg_type == "done":
            self.state.is_rendering = False
            dpg.configure_item("btn_cancel_render", enabled=False)
            output_path = final_msg.get("path", "")
            dpg.set_value("render_progress", 1.0)
            dpg.configure_item("btn_render", label="▶ Video exportieren")
            dpg.set_value("render_status_text", f"Gespeichert: {output_path}")
            dpg.configure_item("btn_open_folder", show=True)
            self._set_status(f"Fertig: {Path(output_path).name}", "ok")
            self._update_status_indicators()
        elif msg_type == "cancelled":
            self.state.is_rendering = False
            dpg.configure_item("btn_cancel_render", enabled=False)
            dpg.configure_item("btn_open_folder", show=False)
            dpg.set_value("render_progress", 0.0)
            dpg.configure_item("btn_render", label="▶ Video exportieren")
            dpg.set_value("render_status_text", "")
            self._set_status("Rendering abgebrochen.", "warn")
            self._update_status_indicators()
        elif msg_type == "error":
            self.state.is_rendering = False
            dpg.configure_item("btn_cancel_render", enabled=False)
            dpg.configure_item("btn_open_folder", show=False)
            error_msg = final_msg.get("message", "Unbekannter Fehler")
            dpg.set_value("render_progress", 0.0)
            dpg.configure_item("btn_render", label="▶ Video exportieren")
            dpg.set_value("render_status_text", "")
            self._set_status(f"Render-Fehler: {error_msg}", "error")
            self._update_status_indicators()

    def _estimate_time_remaining(self, frame: int, total: int) -> str:
        """Schätzt die verbleibende Render-Zeit."""
        # Einfache Heuristik: ~0.05s pro Frame
        remaining = (total - frame) * 0.05
        if remaining < 60:
            return f"{remaining:.0f}s"
        return f"{remaining/60:.1f}min"

    # -------------------------------------------------------------------------
    # KI Optimierung
    # -------------------------------------------------------------------------

    def _get_param_specs(self) -> dict:
        try:
            viz_class = get_visualizer(self.state.visualizer_type)
            specs = {}
            if hasattr(viz_class, 'EFFECTS'):
                specs.update(viz_class.EFFECTS)
            if hasattr(viz_class, 'PARAMS'):
                specs.update(viz_class.PARAMS)
            return specs
        except Exception:
            return {}

    def _on_ki_prompt_changed(self, sender, app_data):
        self.state.ki_prompt = app_data

    def _on_ki_optimize_clicked(self, sender, app_data):
        if self.state.is_ki_optimizing:
            return
        if not self.gemini:
            self._set_ki_status("KI nicht verfügbar. Prüfe API-Key.", error=True)
            return
        if not self.state.audio_path or not os.path.exists(self.state.audio_path):
            self._set_ki_status("Lade zuerst eine Audio-Datei.", error=False)
            return
        if self.state.features is None:
            self._set_ki_status("Audio wird noch analysiert...", error=False)
            return

        self.state.is_ki_optimizing = True
        dpg.configure_item("btn_ki_optimize", label="⏳ KI denkt nach...")
        self._set_ki_status("Sende Anfrage an Gemini...")
        self._update_status_indicators()

        features_dict = self._features_to_dict(self.state.features)
        param_specs = self._get_param_specs()
        current_params = self.state.get_params()
        colors = self.state.ki_suggested_colors or {"primary": "#FFFFFF", "secondary": "#888888", "background": "#000000"}
        user_prompt = getattr(self.state, 'ki_prompt', '')

        try:
            self._ki_future = self.gemini.optimize_all_settings_async(
                visualizer_type=self.state.visualizer_type,
                current_params=current_params,
                audio_features=features_dict,
                colors=colors,
                param_specs=param_specs,
                user_prompt=user_prompt if user_prompt else None,
            )
            threading.Thread(target=self._poll_ki_result, daemon=True).start()
        except Exception as e:
            self.state.is_ki_optimizing = False
            self._ki_queue.put({"type": "error", "message": str(e)})
            self._update_status_indicators()

    def _features_to_dict(self, features) -> dict:
        return {
            'duration': float(getattr(features, 'duration', 0)),
            'tempo': float(getattr(features, 'tempo', 120)),
            'mode': str(getattr(features, 'mode', 'music')),
            'rms_mean': float(getattr(features, 'rms_mean', 0.5)),
            'rms_std': float(getattr(features, 'rms_std', 0.1)),
            'onset_mean': float(getattr(features, 'onset_mean', 0.3)),
            'onset_std': float(getattr(features, 'onset_std', 0.1)),
            'spectral_mean': float(getattr(features, 'spectral_mean', 0.5)),
            'transient_mean': float(getattr(features, 'transient_mean', 0.0)),
            'voice_clarity_mean': float(getattr(features, 'voice_clarity_mean', 0.0)),
        }

    def _poll_ki_result(self):
        try:
            result = self._ki_future.result(timeout=60)
            self._ki_queue.put({"type": "done", "result": result})
        except Exception as e:
            self._ki_queue.put({"type": "error", "message": str(e)})

    def _process_ki_queue(self):
        if self._ki_queue.empty():
            return
        final_msg = None
        while True:
            try:
                msg = self._ki_queue.get_nowait()
                final_msg = msg
            except queue.Empty:
                break
        if final_msg is None:
            return
        msg_type = final_msg.get("type")
        if msg_type == "done":
            dpg.configure_item("btn_ki_optimize", label="Parameter optimieren")
            self._apply_ki_result(final_msg.get("result", {}))
        elif msg_type == "error":
            dpg.configure_item("btn_ki_optimize", label="Parameter optimieren")
            self._set_ki_status(f"KI-Fehler: {final_msg.get('message', 'Unbekannter Fehler')}", error=True)
        self.state.is_ki_optimizing = False
        self._update_status_indicators()

    def _apply_ki_result(self, result: dict):
        if not isinstance(result, dict):
            self._set_ki_status("KI-Antwort ungültig.", error=True)
            return

        def _update_slider_and_text(tag: str, val: float):
            """Setzt Slider-Wert und aktualisiert das zugehörige Text-Label."""
            dpg.set_value(tag, val)
            val_tag = f"{tag}_value"
            if dpg.does_item_exist(val_tag):
                fmt = dpg.get_item_configuration(tag).get("format", "%.2f")
                try:
                    dpg.set_value(val_tag, fmt % val)
                except Exception:
                    dpg.set_value(val_tag, str(round(val, 2)))

        params = result.get("params", {})
        ui_param_map = {
            "offset_x": ("param_offset_x", "viz_offset_x"),
            "offset_y": ("param_offset_y", "viz_offset_y"),
            "scale": ("param_scale", "viz_scale"),
        }
        extra_params = {}
        for name, val in params.items():
            if name in ui_param_map:
                tag, attr = ui_param_map[name]
                _update_slider_and_text(tag, float(val))
                setattr(self.state, attr, float(val))
            else:
                extra_params[name] = val
        self.state.viz_extra_params = extra_params

        pp = result.get("postprocess", {})
        pp_map = {
            "contrast": ("param_pp_contrast", "pp_contrast"),
            "saturation": ("param_pp_saturation", "pp_saturation"),
            "brightness": ("param_pp_brightness", "pp_brightness"),
            "warmth": ("param_pp_warmth", "pp_warmth"),
            "film_grain": ("param_pp_grain", "pp_grain"),
        }
        for key, (tag, attr) in pp_map.items():
            if key in pp:
                val = float(pp[key])
                _update_slider_and_text(tag, val)
                setattr(self.state, attr, val)

        bg = result.get("background", {})
        bg_map = {
            "blur": ("param_bg_blur", "bg_blur"),
            "vignette": ("param_bg_vignette", "bg_vignette"),
            "opacity": ("param_bg_opacity", "bg_opacity"),
        }
        for key, (tag, attr) in bg_map.items():
            if key in bg:
                val = float(bg[key])
                _update_slider_and_text(tag, val)
                setattr(self.state, attr, val)

        colors = result.get("colors", {})
        if colors:
            self.state.ki_suggested_colors = colors
            color_text = (
                f"KI-Farben:  Primary={colors.get('primary','-')}  "
                f"Secondary={colors.get('secondary','-')}  "
                f"BG={colors.get('background','-')}"
            )
            dpg.set_value("ki_colors_text", color_text)
        else:
            dpg.set_value("ki_colors_text", "")

        self._request_preview_update()
        self._set_ki_status("Parameter optimiert!")
        self._set_status("KI-Optimierung abgeschlossen.", "ok")

    def _set_ki_status(self, msg: str, error: bool = False):
        self.state.ki_status = msg
        color = Theme.STATUS_ERR if error else Theme.TEXT_SECONDARY
        dpg.set_value("ki_status_text", msg)
        dpg.configure_item("ki_status_text", color=color)

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    def run(self):
        self.setup_ui()
        self._try_load_auto_save()
        self._update_status_indicators()

        try:
            while dpg.is_dearpygui_running():
                self._process_analyze_result()
                self._process_ki_queue()
                self._process_quotes_queue()
                self._update_analysis_spinner()
                self._update_preview()
                self._poll_render_queue()
                dpg.render_dearpygui_frame()
        except KeyboardInterrupt:
            pass
        finally:
            dpg.destroy_context()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = AudioVisualizerGUI()
    app.run()
