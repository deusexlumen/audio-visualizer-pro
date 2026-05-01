"""
Audio Visualizer Pro – DearPyGui Frontend v2.1

Komplett überarbeitete GUI mit professionellem Dark-Neon-Design.
- Visuelle Cards mit farbigen Akzenten pro Kategorie
- Tooltips für alle Parameter
- Schicke Live-Preview mit Info-Overlay
- Status-Bar mit farbigen Indikatoren
- Scrollbares Control-Panel
"""

from __future__ import annotations

import os
import sys
import time
import threading
import queue
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

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


# =============================================================================
# DESIGN SYSTEM – Farben & Konstanten
# =============================================================================

class Theme:
    """Zentrales Design-System fuer alle visuellen Elemente."""

    # Basis-Farben
    BG_DEEPEST    = (10, 14, 23)       # Fenster-Hintergrund
    BG_CARD       = (18, 24, 38)       # Card-Hintergrund
    BG_CARD_HOVER = (24, 32, 52)       # Card-Hover
    BG_INPUT      = (14, 20, 34)       # Input-Felder
    BORDER        = (36, 48, 72)       # Rahmen
    BORDER_ACTIVE = (60, 80, 120)      # Aktiver Rahmen
    TEXT_PRIMARY  = (230, 235, 245)    # Haupttext
    TEXT_SECONDARY= (150, 160, 180)    # Sekundaertext
    TEXT_MUTED    = (100, 110, 130)    # Gedämpfter Text

    # Akzentfarben pro Kategorie (RGB)
    AUDIO         = (79, 195, 247)     # Cyan
    VISUALIZER    = (124, 77, 255)     # Violett
    KI            = (0, 230, 118)      # Grün
    BACKGROUND    = (255, 145, 0)      # Orange
    POSTPROCESS   = (255, 82, 82)      # Rot/Rose
    EXPORT        = (255, 215, 64)     # Amber
    PREVIEW       = (105, 240, 174)    # Mint
    STATUS_OK     = (0, 230, 118)      # Grün
    STATUS_WARN   = (255, 215, 64)     # Gelb
    STATUS_ERR    = (255, 82, 82)      # Rot

    # Slider-Farben (RGB)
    SLIDER_GRAB      = (100, 180, 255)
    SLIDER_GRAB_HOVER= (130, 200, 255)
    SLIDER_BG        = (30, 40, 60)
    SLIDER_BG_HOVER  = (40, 55, 80)

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

        self.is_analyzing: bool = False
        self.is_rendering: bool = False
        self.is_ki_optimizing: bool = False
        self.is_extracting_quotes: bool = False
        self.status_message: str = "Bereit."
        self.ki_status: str = ""
        self.ki_suggested_colors: dict = {}
        self.ki_prompt: str = ""

        # Quotes
        self.quotes: list = []
        self.quotes_enabled: bool = False
        self.quote_config: QuoteOverlayConfig = QuoteOverlayConfig(enabled=True)

        self._preview_params_hash: str = ""
        self._preview_image: Image.Image | None = None

    def get_gpu_encode(self) -> bool:
        return self.gpu_encode

    def get_params(self) -> dict:
        base = {
            "offset_x": self.viz_offset_x,
            "offset_y": self.viz_offset_y,
            "scale": self.viz_scale,
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
        return (
            f"{self.visualizer_type}_{self.audio_path}_{self.background_path}_"
            f"{self.viz_offset_x:.3f}_{self.viz_offset_y:.3f}_{self.viz_scale:.3f}_"
            f"{self.bg_blur:.1f}_{self.bg_vignette:.2f}_{self.bg_opacity:.2f}_"
            f"{self.pp_contrast:.2f}_{self.pp_saturation:.2f}_{self.pp_brightness:.2f}_"
            f"{self.pp_warmth:.2f}_{self.pp_grain:.2f}_{self.preview_time_percent:.2f}_"
            f"{self.quotes_enabled}_{len(self.quotes)}"
        )


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
        self._last_preview_update = 0.0
        self._preview_min_interval = 0.15
        self._ki_future = None
        self.gemini = None
        try:
            self.gemini = GeminiIntegration()
        except Exception as e:
            print(f"[GUI] Gemini-Integration nicht verfuegbar: {e}")

        self._render_queue = queue.Queue()
        self._cancel_event = threading.Event()

    # -------------------------------------------------------------------------
    # Theme Setup
    # -------------------------------------------------------------------------

    def _setup_theme(self):
        """Erstellt ein professionelles Dark-Neon-Theme."""
        with dpg.theme(tag="global_theme"):
            with dpg.theme_component(dpg.mvAll):
                # Fenster
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, Theme.BG_DEEPEST)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, Theme.BG_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, Theme.BG_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_Text, Theme.TEXT_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_Border, Theme.BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, Theme.BG_CARD)

                # Buttons
                dpg.add_theme_color(dpg.mvThemeCol_Button, Theme.BG_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, Theme.BG_CARD_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 55, 85))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)

                # Inputs / Slider
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, Theme.BG_INPUT)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (24, 34, 52))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (30, 45, 70))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, Theme.SLIDER_GRAB)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, Theme.SLIDER_GRAB_HOVER)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 12)

                # Combo / Dropdown
                dpg.add_theme_color(dpg.mvThemeCol_Header, Theme.BG_CARD_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (35, 50, 80))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (45, 65, 100))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, Theme.BG_CARD)

                # Child Window (Cards)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, Theme.BG_CARD)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)

                # Separator
                dpg.add_theme_color(dpg.mvThemeCol_Separator, Theme.BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, Theme.BORDER_ACTIVE)
                dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, Theme.BORDER_ACTIVE)

                # Progress Bar
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, Theme.SLIDER_GRAB)
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered, Theme.SLIDER_GRAB_HOVER)

                # Scrollbar
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, Theme.BG_DEEPEST)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, Theme.BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, Theme.BORDER_ACTIVE)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, Theme.SLIDER_GRAB)

                # Allgemeine Spacing
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 6, 4)

        dpg.bind_theme("global_theme")

    def _apply_card_theme(self, tag: str, accent_color: tuple[int, int, int]):
        """Wendet ein Card-Theme mit farbigem Akzent auf ein Child-Window an."""
        with dpg.theme(tag=f"card_theme_{tag}"):
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_Border, accent_color + (120,))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, Theme.BG_CARD)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
        dpg.bind_item_theme(tag, f"card_theme_{tag}")

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def setup_ui(self):
        dpg.create_context()
        dpg.configure_app(docking=False, init_file="dpg_layout.ini")
        self._setup_theme()

        with dpg.font_registry():
            default_font = dpg.add_font("C:/Windows/Fonts/segoeui.ttf", 16)
            dpg.bind_font(default_font)

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
            width=1500,
            height=980,
            no_close=True,
            no_collapse=True,
        ):
            self._build_menu_bar()

            with dpg.group(horizontal=True):
                # --- LINKS: Control-Panel ---
                self._build_control_panel()
                # --- RECHTS: Preview ---
                self._build_preview_panel()

            # --- UNTEN: Status-Bar ---
            self._build_status_bar()

        self._setup_file_dialogs()

        viewport_kwargs = {
            "title": "Audio Visualizer Pro",
            "width": 1500,
            "height": 980,
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

    def _build_menu_bar(self):
        """Baut die Menü-Leiste."""
        with dpg.menu_bar():
            with dpg.menu(label="Datei"):
                dpg.add_menu_item(label="Audio laden...", callback=self._show_audio_dialog)
                dpg.add_menu_item(label="Hintergrundbild laden...", callback=self._show_bg_dialog)
                dpg.add_separator()
                dpg.add_menu_item(label="Beenden", callback=lambda: dpg.stop_dearpygui())

            with dpg.menu(label="Hilfe"):
                dpg.add_menu_item(label="Tastenkürzel", callback=self._show_shortcuts)
                dpg.add_separator()
                dpg.add_menu_item(label="Über", callback=self._show_about)

    # -------------------------------------------------------------------------
    # Control Panel
    # -------------------------------------------------------------------------

    def _build_control_panel(self):
        """Baut das scrollbare linke Control-Panel mit Cards."""
        with dpg.child_window(
            width=400,
            height=-42,  # Platz fuer Status-Bar
            border=False,
            tag="control_panel",
        ):
            # --- AUDIO CARD ---
            self._build_card(
                title="Audio",
                accent=Theme.AUDIO,
                icon="🎵",
                content_tag="audio_card_content",
                build_fn=self._build_audio_section,
            )
            dpg.add_spacer(height=10)

            # --- VISUALIZER CARD ---
            self._build_card(
                title="Visualizer",
                accent=Theme.VISUALIZER,
                icon="🎨",
                content_tag="viz_card_content",
                build_fn=self._build_visualizer_section,
            )
            dpg.add_spacer(height=10)

            # --- KI CARD ---
            self._build_card(
                title="KI Optimierung",
                accent=Theme.KI,
                icon="🤖",
                content_tag="ki_card_content",
                build_fn=self._build_ki_section,
            )
            dpg.add_spacer(height=10)

            # --- QUOTES CARD ---
            self._build_card(
                title="Zitate",
                accent=(255, 105, 180),  # Hot Pink
                icon="💬",
                content_tag="quotes_card_content",
                build_fn=self._build_quotes_section,
            )
            dpg.add_spacer(height=10)

            # --- HINTERGRUND CARD ---
            self._build_card(
                title="Hintergrund",
                accent=Theme.BACKGROUND,
                icon="🖼️",
                content_tag="bg_card_content",
                build_fn=self._build_background_section,
            )
            dpg.add_spacer(height=10)

            # --- POST-PROCESS CARD ---
            self._build_card(
                title="Post-Process",
                accent=Theme.POSTPROCESS,
                icon="✨",
                content_tag="pp_card_content",
                build_fn=self._build_postprocess_section,
            )
            dpg.add_spacer(height=10)

            # --- PREVIEW-ZEIT CARD ---
            self._build_card(
                title="Preview",
                accent=Theme.PREVIEW,
                icon="👁️",
                content_tag="preview_card_content",
                build_fn=self._build_preview_section,
            )
            dpg.add_spacer(height=10)

            # --- EXPORT CARD ---
            self._build_card(
                title="Export",
                accent=Theme.EXPORT,
                icon="🎬",
                content_tag="export_card_content",
                build_fn=self._build_export_section,
            )

    def _build_card(self, title: str, accent: tuple, icon: str, content_tag: str, build_fn):
        """Erstellt eine visuelle Card mit farbigem Akzent."""
        card_tag = f"card_{content_tag}"
        with dpg.child_window(
            width=-1,
            height=0,  # Auto-height
            border=True,
            tag=card_tag,
        ):
            # Akzent-Leiste oben
            dpg.add_color_button(
                default_value=accent + (255,),
                width=-1,
                height=3,
                no_border=True,
                no_drag_drop=True,
            )
            dpg.add_spacer(height=4)

            # Titel mit Icon
            with dpg.group(horizontal=True):
                dpg.add_text(icon, color=accent)
                dpg.add_text(title, color=accent)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Inhalt
            build_fn()

        self._apply_card_theme(card_tag, accent)

    def _build_audio_section(self):
        dpg.add_button(
            label="Audio laden...",
            callback=self._show_audio_dialog,
            width=-1,
        )
        dpg.add_text("Keine Audio-Datei geladen", tag="audio_status", wrap=360, color=Theme.TEXT_MUTED)
        self._add_tooltip("Unterstützt: MP3, WAV, FLAC, AAC, OGG, M4A")

    def _build_visualizer_section(self):
        dpg.add_text("Typ", color=Theme.TEXT_SECONDARY)
        dpg.add_combo(
            items=self.state.available_visualizers,
            default_value=self.state.visualizer_type,
            callback=self._on_visualizer_changed,
            width=-1,
            tag="viz_combo",
        )
        self._add_tooltip("Wähle einen der GPU-beschleunigten Visualizer")
        dpg.add_spacer(height=6)

        dpg.add_text("Position & Größe", color=Theme.TEXT_SECONDARY)
        self._styled_slider(
            label="Offset X",
            tag="param_offset_x",
            min_val=-1.0, max_val=1.0, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Horizontale Verschiebung des Visualizers",
        )
        self._styled_slider(
            label="Offset Y",
            tag="param_offset_y",
            min_val=-1.0, max_val=1.0, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Vertikale Verschiebung des Visualizers",
        )
        self._styled_slider(
            label="Skalierung",
            tag="param_scale",
            min_val=0.5, max_val=2.0, default_val=1.0,
            callback=self._on_param_changed,
            tooltip="Gesamtgröße des Visualizers",
        )

    def _build_ki_section(self):
        dpg.add_text("Dein Wunsch (optional)", color=Theme.TEXT_SECONDARY)
        dpg.add_input_text(
            hint="z.B. 'dunkler, mehr Kontrast, cyberpunk-Stil'",
            default_value="",
            callback=self._on_ki_prompt_changed,
            width=-1,
            tag="ki_prompt_input",
        )
        dpg.add_spacer(height=6)
        dpg.add_button(
            label="✨ Parameter optimieren",
            callback=self._on_ki_optimize_clicked,
            width=-1,
            tag="btn_ki_optimize",
        )
        self._add_tooltip("Nutzt Gemini KI, um Parameter an das Audio anzupassen")
        dpg.add_text("", tag="ki_status_text", wrap=360, color=Theme.TEXT_SECONDARY)
        dpg.add_text("", tag="ki_colors_text", wrap=360, color=Theme.STATUS_OK)

    def _build_quotes_section(self):
        dpg.add_checkbox(
            label="Zitate aktivieren",
            default_value=self.state.quotes_enabled,
            callback=self._on_quotes_enabled_changed,
            tag="chk_quotes_enabled",
        )
        dpg.add_spacer(height=6)
        dpg.add_button(
            label="🔮 Key-Zitate extrahieren",
            callback=self._on_extract_quotes_clicked,
            width=-1,
            tag="btn_extract_quotes",
        )
        dpg.add_button(
            label="🎲 Demo-Zitate erstellen",
            callback=self._on_demo_quotes_clicked,
            width=-1,
        )
        dpg.add_text("", tag="quotes_status_text", wrap=360, color=Theme.TEXT_SECONDARY)
        dpg.add_spacer(height=6)

        # Quote-Liste (wird dynamisch aktualisiert)
        dpg.add_text("Zitate:", color=Theme.TEXT_SECONDARY)
        dpg.add_child_window(
            width=-1, height=120, border=True, tag="quotes_list_window"
        )
        dpg.add_spacer(height=6)

        # Grundlegende Quote-Config
        dpg.add_text("Einstellungen", color=Theme.TEXT_SECONDARY)
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
        )
        self._styled_slider(
            label="Anzeigedauer (s)",
            tag="quote_display_duration",
            min_val=2.0, max_val=20.0, default_val=self.state.quote_config.display_duration,
            callback=self._on_quote_config_changed,
            tooltip="Wie lange ein Zitat angezeigt wird",
        )

    def _build_background_section(self):
        dpg.add_button(
            label="Hintergrundbild laden...",
            callback=self._show_bg_dialog,
            width=-1,
        )
        dpg.add_spacer(height=6)
        dpg.add_text("Effekte", color=Theme.TEXT_SECONDARY)
        self._styled_slider(
            label="Blur",
            tag="param_bg_blur",
            min_val=0.0, max_val=20.0, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Weichzeichnung des Hintergrundbilds",
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
            tooltip="Wie stark das Hintergrundbild sichtbar ist",
        )

    def _build_postprocess_section(self):
        dpg.add_text("Color Grading", color=Theme.TEXT_SECONDARY)
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
            tooltip="1.0 = neutral, 0 = Schwarzweiß, >1 = hyper-bunt",
        )
        self._styled_slider(
            label="Helligkeit",
            tag="param_pp_brightness",
            min_val=-0.5, max_val=0.5, default_val=0.0,
            callback=self._on_param_changed,
            tooltip="Globale Helligkeitsanpassung",
        )
        dpg.add_spacer(height=6)
        dpg.add_text("Effekte", color=Theme.TEXT_SECONDARY)
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

    def _build_preview_section(self):
        self._styled_slider(
            label="Position (%)",
            tag="param_preview_time",
            min_val=0.0, max_val=1.0, default_val=0.3,
            callback=self._on_param_changed,
            tooltip="Zeitpunkt im Audio, an dem die Preview gerendert wird",
        )
        dpg.add_text(
            "Preview aktualisiert automatisch beim Loslassen des Sliders.",
            wrap=360, color=Theme.TEXT_MUTED,
        )

    def _build_export_section(self):
        dpg.add_text("Auflösung", color=Theme.TEXT_SECONDARY)
        dpg.add_combo(
            label="",
            items=["1920x1080 (Full HD)", "1280x720 (HD)", "854x480 (SD)"],
            default_value="1920x1080 (Full HD)",
            callback=self._on_resolution_changed,
            width=-1,
            tag="res_combo",
        )
        dpg.add_spacer(height=6)
        dpg.add_checkbox(
            label="⚡ GPU-Encoding (NVENC/AMF/QSV)",
            default_value=self.state.gpu_encode,
            callback=self._on_gpu_encode_changed,
            tag="chk_gpu_encode",
        )
        self._add_tooltip("Nutzt die Grafikkarte fuer Video-Encoding (~5-10x schneller)")
        dpg.add_spacer(height=6)
        dpg.add_text("Output", color=Theme.TEXT_SECONDARY)
        dpg.add_input_text(
            label="",
            default_value=self.state.output_dir,
            callback=self._on_output_dir_changed,
            width=-1,
        )
        dpg.add_spacer(height=8)
        dpg.add_button(
            label="▶ Video exportieren",
            callback=self._on_render_clicked,
            width=-1,
            tag="btn_render",
        )
        dpg.add_button(
            label="⏹ Abbrechen",
            callback=self._on_cancel_render_clicked,
            width=-1,
            tag="btn_cancel_render",
            enabled=False,
            show=True,
        )
        dpg.add_progress_bar(
            default_value=0.0,
            width=-1,
            tag="render_progress",
        )
        dpg.add_text("", tag="render_status_text", wrap=360, color=Theme.TEXT_MUTED)

    def _styled_slider(self, label: str, tag: str, min_val: float, max_val: float,
                       default_val: float, callback, tooltip: str = ""):
        """Erstellt einen Slider mit Label und Tooltip."""
        with dpg.group(horizontal=True):
            dpg.add_text(label, color=Theme.TEXT_SECONDARY)
            dpg.add_slider_float(
                min_value=min_val, max_value=max_val,
                default_value=default_val, callback=callback,
                width=-1, tag=tag,
            )
        if tooltip:
            self._add_tooltip(tooltip, parent=tag)

    def _add_tooltip(self, text: str, parent: str | None = None):
        """Fügt einen Tooltip hinzu."""
        with dpg.tooltip(parent=parent if parent else dpg.last_item()):
            dpg.add_text(text, color=Theme.TEXT_MUTED)

    # -------------------------------------------------------------------------
    # Preview Panel
    # -------------------------------------------------------------------------

    def _build_preview_panel(self):
        """Baut das rechte Preview-Panel mit Rahmen und Info-Overlay."""
        with dpg.child_window(
            width=-1,
            height=-42,
            border=False,
            tag="preview_panel",
        ):
            # Preview-Container als Card
            preview_card = "preview_card"
            with dpg.child_window(
                width=-1,
                height=-1,
                border=True,
                tag=preview_card,
            ):
                # Akzent oben
                dpg.add_color_button(
                    default_value=Theme.PREVIEW + (255,),
                    width=-1,
                    height=3,
                    no_border=True,
                    no_drag_drop=True,
                )
                dpg.add_spacer(height=6)

                # Header mit Info
                with dpg.group(horizontal=True):
                    dpg.add_text("👁️", color=Theme.PREVIEW)
                    dpg.add_text("Live Preview", color=Theme.PREVIEW)
                    dpg.add_spacer(width=20)
                    dpg.add_text(
                        f"{self.state.preview_width}×{self.state.preview_height} @ {self.state.preview_fps}fps",
                        color=Theme.TEXT_MUTED,
                        tag="preview_resolution_text",
                    )

                dpg.add_separator()
                dpg.add_spacer(height=6)

                # Das eigentliche Bild (zentriert)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=1)  # Flex spacer workaround
                    dpg.add_image(
                        self._preview_texture_tag,
                        width=self.state.preview_width,
                        height=self.state.preview_height,
                        tag="preview_image",
                    )
                    dpg.add_spacer(width=1)

                dpg.add_spacer(height=6)
                # Zeit-Anzeige
                dpg.add_text(
                    "--:-- / --:--",
                    color=Theme.TEXT_MUTED,
                    tag="preview_time_text",
                )

            self._apply_card_theme(preview_card, Theme.PREVIEW)

    # -------------------------------------------------------------------------
    # Status Bar
    # -------------------------------------------------------------------------

    def _build_status_bar(self):
        """Baut die untere Status-Leiste mit farbigen Indikatoren."""
        with dpg.group(horizontal=True, tag="status_bar"):
            # Audio-Indikator
            dpg.add_text("●", color=Theme.STATUS_ERR, tag="status_dot_audio")
            dpg.add_text("Audio", color=Theme.TEXT_MUTED, tag="status_label_audio")
            dpg.add_spacer(width=12)

            # Analyse-Indikator
            dpg.add_text("●", color=Theme.STATUS_ERR, tag="status_dot_analyze")
            dpg.add_text("Analyse", color=Theme.TEXT_MUTED, tag="status_label_analyze")
            dpg.add_spacer(width=12)

            # Render-Indikator
            dpg.add_text("●", color=Theme.STATUS_ERR, tag="status_dot_render")
            dpg.add_text("Render", color=Theme.TEXT_MUTED, tag="status_label_render")
            dpg.add_spacer(width=12)

            # KI-Indikator
            dpg.add_text("●", color=Theme.STATUS_ERR, tag="status_dot_ki")
            dpg.add_text("KI", color=Theme.TEXT_MUTED, tag="status_label_ki")
            dpg.add_spacer(width=20)

            # Trenner
            dpg.add_text("|", color=Theme.BORDER)
            dpg.add_spacer(width=12)

            # Status-Text
            dpg.add_text(
                self.state.status_message,
                color=Theme.TEXT_SECONDARY,
                tag="status_text",
            )

    def _set_status(self, msg: str, level: str = "info"):
        """Aktualisiert die Status-Zeile und Indikatoren."""
        self.state.status_message = msg
        dpg.set_value("status_text", msg)

        color = Theme.TEXT_SECONDARY
        if level == "ok":
            color = Theme.STATUS_OK
        elif level == "warn":
            color = Theme.STATUS_WARN
        elif level == "error":
            color = Theme.STATUS_ERR
        dpg.configure_item("status_text", color=color)

    def _update_status_indicators(self):
        """Aktualisiert die farbigen Status-Indikatoren."""
        # Audio
        has_audio = self.state.audio_path is not None and os.path.exists(self.state.audio_path)
        dpg.configure_item("status_dot_audio", color=Theme.STATUS_OK if has_audio else Theme.STATUS_ERR)
        dpg.configure_item("status_label_audio", color=Theme.TEXT_PRIMARY if has_audio else Theme.TEXT_MUTED)

        # Analyse
        analyzed = self.state.features is not None
        dpg.configure_item("status_dot_analyze", color=Theme.STATUS_OK if analyzed else (Theme.STATUS_WARN if has_audio else Theme.STATUS_ERR))
        dpg.configure_item("status_label_analyze", color=Theme.TEXT_PRIMARY if analyzed else Theme.TEXT_MUTED)

        # Render
        rendering = self.state.is_rendering
        dpg.configure_item("status_dot_render", color=Theme.STATUS_WARN if rendering else (Theme.STATUS_OK if analyzed else Theme.STATUS_ERR))
        dpg.configure_item("status_label_render", color=Theme.TEXT_PRIMARY if rendering or analyzed else Theme.TEXT_MUTED)

        # KI
        ki_ok = self.gemini is not None
        dpg.configure_item("status_dot_ki", color=Theme.STATUS_OK if ki_ok else Theme.STATUS_ERR)
        dpg.configure_item("status_label_ki", color=Theme.TEXT_PRIMARY if ki_ok else Theme.TEXT_MUTED)

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
        dpg.show_item("audio_file_dialog")

    def _show_bg_dialog(self, sender=None, app_data=None):
        dpg.show_item("bg_file_dialog")

    def _show_about(self):
        with dpg.window(label="Über Audio Visualizer Pro", modal=True, width=420, height=280, no_resize=True):
            dpg.add_spacer(height=8)
            dpg.add_text("Audio Visualizer Pro", color=Theme.TEXT_PRIMARY)
            dpg.add_text("Version 2.1.0", color=Theme.TEXT_MUTED)
            dpg.add_separator()
            dpg.add_spacer(height=4)
            dpg.add_text(
                "GPU-beschleunigte Audio-Visualisierung mit KI-gestützter\n"
                "Parameter-Optimierung und professionellem Post-Processing.",
                color=Theme.TEXT_SECONDARY,
            )
            dpg.add_spacer(height=8)
            dpg.add_text("Technologien:", color=Theme.TEXT_SECONDARY)
            dpg.add_text("• ModernGL (OpenGL) für GPU-Rendering", color=Theme.TEXT_MUTED)
            dpg.add_text("• FFmpeg für professionelles Video-Encoding", color=Theme.TEXT_MUTED)
            dpg.add_text("• Gemini KI für intelligente Analyse", color=Theme.TEXT_MUTED)
            dpg.add_text("• DearPyGui für native Desktop-Oberfläche", color=Theme.TEXT_MUTED)
            dpg.add_spacer(height=12)
            dpg.add_button(label="Schließen", callback=lambda: dpg.delete_item(dpg.last_container()), width=-1)

    def _show_shortcuts(self):
        with dpg.window(label="Tastenkürzel", modal=True, width=350, height=200, no_resize=True):
            dpg.add_text("Steuerung", color=Theme.TEXT_PRIMARY)
            dpg.add_separator()
            with dpg.table(header_row=False):
                dpg.add_table_column()
                dpg.add_table_column()
                for key, desc in [
                    ("Ctrl+O", "Audio laden"),
                    ("Ctrl+B", "Hintergrund laden"),
                    ("Ctrl+E", "Export starten"),
                    ("Escape", "Abbrechen / Schließen"),
                ]:
                    with dpg.table_row():
                        dpg.add_text(key, color=Theme.AUDIO)
                        dpg.add_text(desc, color=Theme.TEXT_SECONDARY)
            dpg.add_spacer(height=12)
            dpg.add_button(label="Schließen", callback=lambda: dpg.delete_item(dpg.last_container()), width=-1)

    def _on_audio_selected(self, sender, app_data):
        if app_data.get("selections"):
            selections = list(app_data["selections"].values())
            if selections:
                path = selections[0]
                self.state.audio_path = path
                dpg.set_value("audio_status", f"🎵 {Path(path).name}")
                dpg.configure_item("audio_status", color=Theme.TEXT_PRIMARY)
                self._analyze_audio()
                self._request_preview_update()
                self._update_status_indicators()

    def _on_background_selected(self, sender, app_data):
        if app_data.get("selections"):
            selections = list(app_data["selections"].values())
            if selections:
                path = selections[0]
                self.state.background_path = path
                self._request_preview_update()

    def _on_visualizer_changed(self, sender, app_data):
        self.state.visualizer_type = dpg.get_value(sender)
        self.state.viz_extra_params = {}
        self.state.ki_suggested_colors = {}
        dpg.set_value("ki_colors_text", "")
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

    def _on_resolution_changed(self, sender, app_data):
        res_str = dpg.get_value(sender)
        # Extrahiert die erste WIDTHxHEIGHT Kombination aus dem String
        import re
        match = re.search(r"(\d+)[x×](\d+)", res_str)
        if match:
            self.state.resolution = (int(match.group(1)), int(match.group(2)))
        else:
            self.state.resolution = (1920, 1080)

    def _on_output_dir_changed(self, sender, app_data):
        self.state.output_dir = dpg.get_value(sender)

    def _on_gpu_encode_changed(self, sender, app_data):
        self.state.gpu_encode = dpg.get_value(sender)

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

        def _extract():
            try:
                quotes = self.gemini.extract_quotes(
                    self.state.audio_path,
                    audio_duration=self.state.features.duration
                )
                # Zeit begrenzen
                for q in quotes:
                    q.start_time = max(0.0, min(q.start_time, self.state.features.duration - 1.0))
                    q.end_time = max(q.start_time + 1.0, min(q.end_time, self.state.features.duration))
                self.state.quotes = quotes
                self.state.is_extracting_quotes = False
                dpg.configure_item("btn_extract_quotes", label="🔮 Key-Zitate extrahieren")
                dpg.set_value("quotes_status_text", f"{len(quotes)} Zitate extrahiert!")
                dpg.configure_item("quotes_status_text", color=Theme.STATUS_OK)
                self._refresh_quotes_list()
                self._request_preview_update()
            except Exception as e:
                self.state.is_extracting_quotes = False
                dpg.configure_item("btn_extract_quotes", label="🔮 Key-Zitate extrahieren")
                dpg.set_value("quotes_status_text", f"Fehler: {e}")
                dpg.configure_item("quotes_status_text", color=Theme.STATUS_ERR)

        threading.Thread(target=_extract, daemon=True).start()

    def _on_demo_quotes_clicked(self, sender, app_data):
        duration = self.state.audio_duration if self.state.features else 60.0
        self.state.quotes = [
            Quote(text="Das Abenteuer beginnt jetzt.", start_time=min(5.0, duration * 0.05), end_time=min(13.0, duration * 0.1), confidence=0.95),
            Quote(text="Jeder Moment ist eine Chance.", start_time=min(duration * 0.3, duration - 15.0), end_time=min(duration * 0.3 + 8.0, duration - 5.0), confidence=0.90),
            Quote(text="Bleib dran, es lohnt sich.", start_time=min(duration * 0.6, duration - 10.0), end_time=min(duration * 0.6 + 8.0, duration - 2.0), confidence=0.88),
        ]
        dpg.set_value("quotes_status_text", f"{len(self.state.quotes)} Demo-Zitate erstellt!")
        dpg.configure_item("quotes_status_text", color=Theme.STATUS_OK)
        self._refresh_quotes_list()
        self._request_preview_update()

    def _refresh_quotes_list(self):
        """Aktualisiert die Anzeige der Quote-Liste."""
        # Lösche alte Items im Quotes-List-Window
        # DPG hat keine einfache Möglichkeit, alle Children zu löschen,
        # also ersetzen wir den Inhalt durch neues Setzen
        if not self.state.quotes:
            dpg.set_value("quotes_status_text", "Noch keine Zitate.")
            return

        # Da DPG nicht einfach Children löschen kann ohne tags,
        # aktualisieren wir nur den Status-Text
        lines = []
        for i, q in enumerate(self.state.quotes):
            start_m = int(q.start_time // 60)
            start_s = int(q.start_time % 60)
            lines.append(f"{i+1}. [{start_m}:{start_s:02d}] {q.text[:40]}{'...' if len(q.text) > 40 else ''}")
        dpg.set_value("quotes_status_text", "\n".join(lines))

    def _on_quote_config_changed(self, sender, app_data):
        tag = dpg.get_item_alias(sender) or sender
        if tag == "quote_position":
            self.state.quote_config.position = dpg.get_value(sender)
        elif tag == "quote_font_size":
            self.state.quote_config.font_size = int(dpg.get_value(sender))
        elif tag == "quote_display_duration":
            self.state.quote_config.display_duration = float(dpg.get_value(sender))
        self._request_preview_update()

    def _request_preview_update(self):
        self._last_preview_update = 0.0

    # -------------------------------------------------------------------------
    # Audio-Analyse
    # -------------------------------------------------------------------------

    def _analyze_audio(self):
        if not self.state.audio_path or not os.path.exists(self.state.audio_path):
            return
        self.state.is_analyzing = True
        self._set_status("Audio wird analysiert...", "warn")
        self._update_status_indicators()

        def _analyze():
            try:
                analyzer = AudioAnalyzer()
                features = analyzer.analyze(self.state.audio_path, fps=self.state.preview_fps)
                self.state.features = features
                self.state.audio_duration = features.duration
                self.state.is_analyzing = False
                self._set_status(f"Analyse: {features.duration:.1f}s @ {features.tempo:.0f} BPM", "ok")
                self._update_preview_time_text()
                self._update_status_indicators()
            except Exception as e:
                self.state.is_analyzing = False
                self._set_status(f"Analyse-Fehler: {e}", "error")
                self._update_status_indicators()

        threading.Thread(target=_analyze, daemon=True).start()

    # -------------------------------------------------------------------------
    # Preview Rendering
    # -------------------------------------------------------------------------

    def _update_preview(self):
        now = time.time()
        if now - self._last_preview_update < self._preview_min_interval:
            return
        self._last_preview_update = now

        if not self.state.audio_path or not os.path.exists(self.state.audio_path):
            return
        if self.state.is_analyzing:
            return
        if self.state.features is None:
            return

        params_hash = self.state.preview_params_hash()
        if params_hash == self.state._preview_params_hash and self.state._preview_image is not None:
            return

        self.state._preview_params_hash = params_hash

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
            )
            if img is not None:
                self.state._preview_image = img
                self._upload_texture(img)
        except Exception as e:
            print(f"[Preview] Fehler: {e}")

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
            self._set_status("Rendering läuft bereits...", "warn")
            return
        if not self.state.audio_path or not os.path.exists(self.state.audio_path):
            self._set_status("Keine Audio-Datei geladen!", "error")
            return
        if self.state.features is None:
            self._set_status("Audio wird noch analysiert...", "warn")
            return

        self.state.is_rendering = True
        self._cancel_event.clear()
        dpg.configure_item("btn_render", label="⏳ Render läuft...")
        dpg.configure_item("btn_cancel_render", enabled=True)
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
                renderer.release()

                if self._cancel_event.is_set():
                    self._render_queue.put({"type": "cancelled"})
                else:
                    self._render_queue.put({"type": "done", "path": output_path})
            except Exception as e:
                self._render_queue.put({"type": "error", "message": str(e)})

        threading.Thread(target=_render, daemon=True).start()

    def _on_cancel_render_clicked(self, sender, app_data):
        if self.state.is_rendering:
            self._cancel_event.set()
            self._set_status("Abbruch angefordert...", "warn")

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
            self._set_status(f"Fertig: {Path(output_path).name}", "ok")
            self._update_status_indicators()
        elif msg_type == "cancelled":
            self.state.is_rendering = False
            dpg.configure_item("btn_cancel_render", enabled=False)
            dpg.set_value("render_progress", 0.0)
            dpg.configure_item("btn_render", label="▶ Video exportieren")
            dpg.set_value("render_status_text", "")
            self._set_status("Rendering abgebrochen.", "warn")
            self._update_status_indicators()
        elif msg_type == "error":
            self.state.is_rendering = False
            dpg.configure_item("btn_cancel_render", enabled=False)
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
            self._set_ki_status("Lade zuerst eine Audio-Datei.", error=True)
            return
        if self.state.features is None:
            self._set_ki_status("Audio wird noch analysiert...", error=True)
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
            dpg.configure_item("btn_ki_optimize", label="✨ Parameter optimieren")
            self._set_ki_status(f"Fehler: {e}", error=True)
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
            dpg.configure_item("btn_ki_optimize", label="✨ Parameter optimieren")
            self._apply_ki_result(result)
        except Exception as e:
            dpg.configure_item("btn_ki_optimize", label="✨ Parameter optimieren")
            self._set_ki_status(f"KI-Fehler: {e}", error=True)
        finally:
            self.state.is_ki_optimizing = False
            self._update_status_indicators()

    def _apply_ki_result(self, result: dict):
        if not isinstance(result, dict):
            self._set_ki_status("KI-Antwort ungültig.", error=True)
            return

        params = result.get("params", {})
        ui_param_map = {
            "offset_x": "param_offset_x",
            "offset_y": "param_offset_y",
            "scale": "param_scale",
        }
        extra_params = {}
        for name, val in params.items():
            if name in ui_param_map:
                tag = ui_param_map[name]
                dpg.set_value(tag, float(val))
                attr_map = {
                    "offset_x": "viz_offset_x",
                    "offset_y": "viz_offset_y",
                    "scale": "viz_scale",
                }
                setattr(self.state, attr_map[name], float(val))
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
                dpg.set_value(tag, val)
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
                dpg.set_value(tag, val)
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
        self._update_status_indicators()

        try:
            while dpg.is_dearpygui_running():
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
