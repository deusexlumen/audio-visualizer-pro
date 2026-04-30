"""
Audio Visualizer Pro – DearPyGui Frontend

Komplette Neuimplementierung der GUI mit DearPyGui (DPG).
- Zwei-Spalten-Layout: Control-Panel links, Live-Preview rechts
- Persistente Asset-Pfade (kein OS-Temp)
- Live-Preview via ModernGL → DPG Raw Texture
"""

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
        # Dateipfade (persistente Assets)
        self.audio_path: str | None = None
        self.background_path: str | None = None
        self.font_path: str | None = None
        self.output_dir: str = "output"

        # Audio-Analyse (gecached)
        self.features = None
        self.audio_duration: float = 0.0

        # Visualizer-Auswahl
        self.visualizer_type: str = "voice_flow"
        self.available_visualizers = list_visualizers()

        # Parameter
        self.viz_offset_x: float = 0.0
        self.viz_offset_y: float = 0.0
        self.viz_scale: float = 1.0
        self.viz_extra_params: dict = {}  # KI-optimierte Parameter ohne UI-Slider

        # Hintergrund
        self.bg_blur: float = 0.0
        self.bg_vignette: float = 0.0
        self.bg_opacity: float = 0.3

        # Post-Process
        self.pp_contrast: float = 1.0
        self.pp_saturation: float = 1.0
        self.pp_brightness: float = 0.0
        self.pp_warmth: float = 0.0
        self.pp_grain: float = 0.0

        # Preview
        self.preview_time_percent: float = 0.3
        self.preview_fps: int = 30
        self.preview_width: int = 854
        self.preview_height: int = 480

        # Render-Config
        self.resolution: tuple[int, int] = (1920, 1080)
        self.render_fps: int = 30
        self.codec: str = "h264"
        self.quality: str = "high"

        # Status
        self.is_analyzing: bool = False
        self.is_rendering: bool = False
        self.is_ki_optimizing: bool = False
        self.status_message: str = "Bereit."
        self.ki_status: str = ""
        self.ki_suggested_colors: dict = {}
        self.ki_prompt: str = ""

        # Preview-Cache
        self._preview_params_hash: str = ""
        self._preview_image: Image.Image | None = None

    def get_params(self) -> dict:
        """Gibt die aktuellen Visualizer-Parameter zurück."""
        base = {
            "offset_x": self.viz_offset_x,
            "offset_y": self.viz_offset_y,
            "scale": self.viz_scale,
        }
        base.update(self.viz_extra_params)
        return base

    def get_postprocess(self) -> dict:
        """Gibt die aktuellen Post-Process-Parameter zurück."""
        return {
            "contrast": self.pp_contrast,
            "saturation": self.pp_saturation,
            "brightness": self.pp_brightness,
            "warmth": self.pp_warmth,
            "film_grain": self.pp_grain,
        }

    def preview_params_hash(self) -> str:
        """Erzeugt einen Hash über alle Preview-relevanten Parameter."""
        return (
            f"{self.visualizer_type}_{self.audio_path}_{self.background_path}_"
            f"{self.viz_offset_x:.3f}_{self.viz_offset_y:.3f}_{self.viz_scale:.3f}_"
            f"{self.bg_blur:.1f}_{self.bg_vignette:.2f}_{self.bg_opacity:.2f}_"
            f"{self.pp_contrast:.2f}_{self.pp_saturation:.2f}_{self.pp_brightness:.2f}_"
            f"{self.pp_warmth:.2f}_{self.pp_grain:.2f}_{self.preview_time_percent:.2f}"
        )


# =============================================================================
# DEARPYGUI FRONTEND
# =============================================================================

class AudioVisualizerGUI:
    """Hauptklasse für die DearPyGui-Oberfläche."""

    def __init__(self):
        self.state = AppState()
        self._preview_texture_tag = "preview_texture"
        self._preview_raw_data = np.zeros(
            (self.state.preview_width * self.state.preview_height * 4,),
            dtype=np.float32
        )
        self._last_preview_update = 0.0
        self._preview_min_interval = 0.15  # Sekunden zwischen Preview-Updates
        self._ki_future = None
        self.gemini = None
        try:
            self.gemini = GeminiIntegration()
        except Exception as e:
            print(f"[GUI] Gemini-Integration nicht verfuegbar: {e}")

        # Thread-sichere Queue fuer Render-Fortschritt
        self._render_queue = queue.Queue()

        # Cancel-Flag fuer Render-Abbruch (thread-safe)
        self._cancel_event = threading.Event()

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def setup_ui(self):
        dpg.create_context()
        dpg.configure_app(docking=False, init_file="dpg_layout.ini")

        with dpg.font_registry():
            # Default-Font skalieren
            default_font = dpg.add_font("C:/Windows/Fonts/segoeui.ttf", 16)
            dpg.bind_font(default_font)

        # Texture fuer Preview MUSS vor add_image existieren
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
            width=1400,
            height=900,
            no_close=True,
            no_collapse=True,
        ):
            dpg.add_menu_bar()
            with dpg.menu(label="Datei"):
                dpg.add_menu_item(label="Audio laden...", callback=self._show_audio_dialog)
                dpg.add_menu_item(label="Hintergrundbild laden...", callback=self._show_bg_dialog)
                dpg.add_separator()
                dpg.add_menu_item(label="Beenden", callback=lambda: dpg.stop_dearpygui())

            with dpg.menu(label="Hilfe"):
                dpg.add_menu_item(label="Über", callback=self._show_about)

            # Haupt-Layout: Zwei Spalten
            with dpg.group(horizontal=True):
                # --- LINKS: Control-Panel ---
                self._build_control_panel()

                # --- RECHTS: Preview ---
                self._build_preview_panel()

        # File Dialogs
        self._setup_file_dialogs()

        dpg.create_viewport(
            title="Audio Visualizer Pro",
            width=1400,
            height=900,
            vsync=True,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    def _build_control_panel(self):
        """Baut das linke Control-Panel."""
        with dpg.child_window(
            width=380,
            height=-1,
            border=True,
            tag="control_panel",
        ):
            # --- Audio ---
            dpg.add_text("🎵 Audio", color=(100, 180, 255))
            dpg.add_separator()
            dpg.add_button(
                label="Audio laden...",
                callback=self._show_audio_dialog,
                width=-1,
            )
            dpg.add_text("Keine Audio-Datei geladen", tag="audio_status", wrap=360)
            dpg.add_spacer(height=8)

            # --- Visualizer ---
            dpg.add_text("🎨 Visualizer", color=(100, 180, 255))
            dpg.add_separator()
            dpg.add_combo(
                items=self.state.available_visualizers,
                default_value=self.state.visualizer_type,
                callback=self._on_visualizer_changed,
                width=-1,
                tag="viz_combo",
            )
            dpg.add_spacer(height=8)

            # --- Parameter ---
            dpg.add_text("🔧 Parameter", color=(100, 180, 255))
            dpg.add_separator()
            dpg.add_slider_float(
                label="Offset X", min_value=-1.0, max_value=1.0,
                default_value=0.0, callback=self._on_param_changed, tag="param_offset_x",
            )
            dpg.add_slider_float(
                label="Offset Y", min_value=-1.0, max_value=1.0,
                default_value=0.0, callback=self._on_param_changed, tag="param_offset_y",
            )
            dpg.add_slider_float(
                label="Skalierung", min_value=0.5, max_value=2.0,
                default_value=1.0, callback=self._on_param_changed, tag="param_scale",
            )
            dpg.add_spacer(height=8)

            # --- KI Optimierung ---
            dpg.add_text("🤖 KI Optimierung", color=(100, 180, 255))
            dpg.add_separator()
            dpg.add_input_text(
                label="Wunsch (optional)",
                hint="z.B. 'dunkler, mehr Kontrast'",
                default_value="",
                callback=self._on_ki_prompt_changed,
                width=-1,
                tag="ki_prompt_input",
            )
            dpg.add_button(
                label="✨ Parameter optimieren",
                callback=self._on_ki_optimize_clicked,
                width=-1,
                tag="btn_ki_optimize",
            )
            dpg.add_text("", tag="ki_status_text", wrap=360, color=(180, 180, 180))
            dpg.add_text("", tag="ki_colors_text", wrap=360, color=(150, 220, 150))
            dpg.add_spacer(height=8)

            # --- Hintergrund ---
            dpg.add_text("🖼️ Hintergrund", color=(100, 180, 255))
            dpg.add_separator()
            dpg.add_button(
                label="Hintergrundbild laden...",
                callback=self._show_bg_dialog,
                width=-1,
            )
            dpg.add_slider_float(
                label="Blur", min_value=0.0, max_value=20.0,
                default_value=0.0, callback=self._on_param_changed, tag="param_bg_blur",
            )
            dpg.add_slider_float(
                label="Vignette", min_value=0.0, max_value=1.0,
                default_value=0.0, callback=self._on_param_changed, tag="param_bg_vignette",
            )
            dpg.add_slider_float(
                label="Opacity", min_value=0.0, max_value=1.0,
                default_value=0.3, callback=self._on_param_changed, tag="param_bg_opacity",
            )
            dpg.add_spacer(height=8)

            # --- Post-Process ---
            dpg.add_text("✨ Post-Process", color=(100, 180, 255))
            dpg.add_separator()
            dpg.add_slider_float(
                label="Kontrast", min_value=0.5, max_value=2.0,
                default_value=1.0, callback=self._on_param_changed, tag="param_pp_contrast",
            )
            dpg.add_slider_float(
                label="Sättigung", min_value=0.0, max_value=2.0,
                default_value=1.0, callback=self._on_param_changed, tag="param_pp_saturation",
            )
            dpg.add_slider_float(
                label="Helligkeit", min_value=-0.5, max_value=0.5,
                default_value=0.0, callback=self._on_param_changed, tag="param_pp_brightness",
            )
            dpg.add_slider_float(
                label="Warmth", min_value=-1.0, max_value=1.0,
                default_value=0.0, callback=self._on_param_changed, tag="param_pp_warmth",
            )
            dpg.add_slider_float(
                label="Film Grain", min_value=0.0, max_value=1.0,
                default_value=0.0, callback=self._on_param_changed, tag="param_pp_grain",
            )
            dpg.add_spacer(height=8)

            # --- Preview Zeit ---
            dpg.add_text("⏱️ Preview-Zeit", color=(100, 180, 255))
            dpg.add_separator()
            dpg.add_slider_float(
                label="Position (%)", min_value=0.0, max_value=1.0,
                default_value=0.3, callback=self._on_param_changed, tag="param_preview_time",
            )
            dpg.add_spacer(height=8)

            # --- Render ---
            dpg.add_text("🎬 Export", color=(100, 180, 255))
            dpg.add_separator()
            dpg.add_combo(
                label="Auflösung",
                items=["1920x1080", "1280x720", "854x480"],
                default_value="1920x1080",
                callback=self._on_resolution_changed,
                width=-1,
                tag="res_combo",
            )
            dpg.add_input_text(
                label="Output-Ordner",
                default_value=self.state.output_dir,
                callback=self._on_output_dir_changed,
                width=-1,
            )
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
            dpg.add_spacer(height=8)

            # --- Status ---
            dpg.add_text("Status", color=(100, 180, 255))
            dpg.add_separator()
            dpg.add_text(self.state.status_message, tag="status_text", wrap=360)

    def _build_preview_panel(self):
        """Baut das rechte Preview-Panel."""
        with dpg.child_window(
            width=-1,
            height=-1,
            border=True,
            tag="preview_panel",
        ):
            dpg.add_text("👁️ Live Preview", color=(100, 180, 255))
            dpg.add_separator()
            dpg.add_image(
                self._preview_texture_tag,
                width=self.state.preview_width,
                height=self.state.preview_height,
                tag="preview_image",
            )

    def _setup_file_dialogs(self):
        """Richtet die Datei-Dialoge ein."""
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

    def _show_audio_dialog(self, sender, app_data):
        dpg.show_item("audio_file_dialog")

    def _show_bg_dialog(self, sender, app_data):
        dpg.show_item("bg_file_dialog")

    def _show_about(self):
        with dpg.window(label="Über", modal=True, width=400, height=200):
            dpg.add_text("Audio Visualizer Pro v2.0")
            dpg.add_text("GPU-beschleunigte Audio-Visualisierung")
            dpg.add_separator()
            dpg.add_button(label="Schließen", callback=lambda: dpg.delete_item(dpg.last_container()))

    def _on_audio_selected(self, sender, app_data):
        """Callback wenn eine Audio-Datei ausgewählt wurde."""
        if app_data.get("selections"):
            selections = list(app_data["selections"].values())
            if selections:
                path = selections[0]
                self.state.audio_path = path
                dpg.set_value("audio_status", f"Audio: {Path(path).name}")
                self._analyze_audio()
                self._request_preview_update()

    def _on_background_selected(self, sender, app_data):
        """Callback wenn ein Hintergrundbild ausgewählt wurde."""
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
        """Wird aufgerufen wenn ein Slider bewegt wird."""
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
            self._request_preview_update()

    def _on_resolution_changed(self, sender, app_data):
        res_str = dpg.get_value(sender)
        w, h = map(int, res_str.split("x"))
        self.state.resolution = (w, h)

    def _on_output_dir_changed(self, sender, app_data):
        self.state.output_dir = dpg.get_value(sender)

    def _request_preview_update(self):
        """Markiert die Preview für ein Update."""
        self._last_preview_update = 0.0

    # -------------------------------------------------------------------------
    # Audio-Analyse
    # -------------------------------------------------------------------------

    def _analyze_audio(self):
        """Analysiert die Audio-Datei im Hintergrund."""
        if not self.state.audio_path or not os.path.exists(self.state.audio_path):
            return

        self.state.is_analyzing = True
        self._set_status("Audio wird analysiert...")

        def _analyze():
            try:
                analyzer = AudioAnalyzer()
                features = analyzer.analyze(self.state.audio_path, fps=self.state.preview_fps)
                self.state.features = features
                self.state.audio_duration = features.duration
                self.state.is_analyzing = False
                self._set_status(f"Audio analysiert: {features.duration:.1f}s @ {features.tempo:.0f} BPM")
            except Exception as e:
                self.state.is_analyzing = False
                self._set_status(f"Analyse-Fehler: {e}")

        threading.Thread(target=_analyze, daemon=True).start()

    # -------------------------------------------------------------------------
    # Preview Rendering
    # -------------------------------------------------------------------------

    def _update_preview(self):
        """Aktualisiert die Live-Preview."""
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
            return  # Keine Änderung, nicht neu rendern

        self.state._preview_params_hash = params_hash

        try:
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
            )

            if img is not None:
                self.state._preview_image = img
                self._upload_texture(img)

        except Exception as e:
            print(f"[Preview] Fehler: {e}")

    def _upload_texture(self, img: Image.Image):
        """Lädt ein PIL-Image in die DPG-Texture."""
        # Konvertiere zu RGBA und flachem float32-Array
        img_rgba = img.convert("RGBA")
        arr = np.array(img_rgba, dtype=np.float32) / 255.0
        flat = arr.flatten()

        # Stelle sicher, dass das Array die richtige Größe hat
        expected_size = self.state.preview_width * self.state.preview_height * 4
        if flat.size != expected_size:
            # Resize falls nötig
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
        """Startet den Video-Export im Hintergrund mit Fortschritts-Updates."""
        if self.state.is_rendering:
            self._set_status("Rendering läuft bereits...")
            return
        if not self.state.audio_path or not os.path.exists(self.state.audio_path):
            self._set_status("Keine Audio-Datei geladen!")
            return
        if self.state.features is None:
            self._set_status("Audio wird noch analysiert...")
            return

        self.state.is_rendering = True
        self._cancel_event.clear()
        dpg.configure_item("btn_render", label="⏳ Render läuft...")
        dpg.configure_item("btn_cancel_render", enabled=True)
        dpg.set_value("render_progress", 0.0)
        self._set_status("Starte Rendering...")

        # Queue leeren vor neuem Render
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
                    codec=self.state.codec,
                    quality=self.state.quality,
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
        """Setzt das Cancel-Flag, um den laufenden Render abzubrechen."""
        if self.state.is_rendering:
            self._cancel_event.set()
            self._set_status("Abbruch angefordert...")

    def _poll_render_queue(self):
        """Verarbeitet Render-Fortschritts-Updates aus der Queue (im Main-Thread)."""
        if not self.state.is_rendering and self._render_queue.empty():
            return

        # Alle pending Messages verarbeiten (nicht blockieren)
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
            self._set_status(f"Rendering... {pct:.1f}% ({frame}/{total})")

        elif msg_type == "done":
            self.state.is_rendering = False
            dpg.configure_item("btn_cancel_render", enabled=False)
            output_path = final_msg.get("path", "")
            dpg.set_value("render_progress", 1.0)
            dpg.configure_item("btn_render", label="▶ Video exportieren")
            self._set_status(f"Fertig: {output_path}")

        elif msg_type == "cancelled":
            self.state.is_rendering = False
            dpg.configure_item("btn_cancel_render", enabled=False)
            dpg.set_value("render_progress", 0.0)
            dpg.configure_item("btn_render", label="▶ Video exportieren")
            self._set_status("Rendering abgebrochen.")

        elif msg_type == "error":
            self.state.is_rendering = False
            dpg.configure_item("btn_cancel_render", enabled=False)
            error_msg = final_msg.get("message", "Unbekannter Fehler")
            dpg.set_value("render_progress", 0.0)
            dpg.configure_item("btn_render", label="▶ Video exportieren")
            self._set_status(f"Render-Fehler: {error_msg}")

    # -------------------------------------------------------------------------
    # KI Optimierung
    # -------------------------------------------------------------------------

    def _get_param_specs(self) -> dict:
        """Holt die Parameter-Spezifikationen vom aktuellen Visualizer."""
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
        """Speichert den optionalen KI-Prompt."""
        self.state.ki_prompt = app_data

    def _on_ki_optimize_clicked(self, sender, app_data):
        """Startet die KI-gestuetzte Parameter-Optimierung."""
        if self.state.is_ki_optimizing:
            return
        if not self.gemini:
            self._set_ki_status("KI nicht verfuegbar. Pruefe API-Key.", error=True)
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

        # Audio-Features als Dict serialisierbar machen
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
            # Thread startet, der auf das Future wartet
            threading.Thread(target=self._poll_ki_result, daemon=True).start()
        except Exception as e:
            self.state.is_ki_optimizing = False
            dpg.configure_item("btn_ki_optimize", label="✨ Parameter optimieren")
            self._set_ki_status(f"Fehler: {e}", error=True)

    def _features_to_dict(self, features) -> dict:
        """Konvertiert AudioFeatures zu einem serialisierbaren Dict."""
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
        """Wartet im Hintergrund auf das KI-Future und aktualisiert die UI."""
        try:
            result = self._ki_future.result(timeout=60)
            # UI-Update im Main Thread via DPG (thread-safe fuer set_value)
            dpg.set_value("btn_ki_optimize", "✨ Parameter optimieren")
            dpg.configure_item("btn_ki_optimize", label="✨ Parameter optimieren")
            self._apply_ki_result(result)
        except Exception as e:
            dpg.configure_item("btn_ki_optimize", label="✨ Parameter optimieren")
            self._set_ki_status(f"KI-Fehler: {e}", error=True)
        finally:
            self.state.is_ki_optimizing = False

    def _apply_ki_result(self, result: dict):
        """Wendet die KI-optimierten Werte auf die UI an."""
        if not isinstance(result, dict):
            self._set_ki_status("KI-Antwort ungueltig.", error=True)
            return

        # === Parameter ===
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
                # State aktualisieren via _on_param_changed Logik
                attr_map = {
                    "offset_x": "viz_offset_x",
                    "offset_y": "viz_offset_y",
                    "scale": "viz_scale",
                }
                setattr(self.state, attr_map[name], float(val))
            else:
                extra_params[name] = val

        self.state.viz_extra_params = extra_params

        # === Post-Process ===
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

        # === Background ===
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

        # === Farben ===
        colors = result.get("colors", {})
        if colors:
            self.state.ki_suggested_colors = colors
            color_text = (
                f"KI-Farben: Primary={colors.get('primary','-')} "
                f"Secondary={colors.get('secondary','-')} "
                f"BG={colors.get('background','-')}"
            )
            dpg.set_value("ki_colors_text", color_text)
        else:
            dpg.set_value("ki_colors_text", "")

        # Preview neu rendern
        self._request_preview_update()
        self._set_ki_status("Parameter optimiert!")
        self._set_status("KI-Optimierung abgeschlossen.")

    def _set_ki_status(self, msg: str, error: bool = False):
        """Aktualisiert die KI-Status-Zeile."""
        self.state.ki_status = msg
        color = (255, 100, 100) if error else (180, 180, 180)
        dpg.set_value("ki_status_text", msg)
        dpg.configure_item("ki_status_text", color=color)

    # -------------------------------------------------------------------------
    # Hilfsmethoden
    # -------------------------------------------------------------------------

    def _set_status(self, msg: str):
        """Aktualisiert die Status-Zeile."""
        self.state.status_message = msg
        dpg.set_value("status_text", msg)

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    def run(self):
        self.setup_ui()

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
