"""Hauptfenster der neuen Audio Visualizer Pro GUI."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, QSettings, QUrl
from PyQt6.QtGui import QAction, QKeySequence, QDesktopServices
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QMessageBox, QTabWidget,
    QScrollArea, QProgressBar, QFileDialog,
)

from src.app_logging import get_logger
from src.gui.assets_panel import AssetsPanel
from src.gui.icons import get_icon, get_app_icon
from src.gui.ki_panel import KIPanel
from src.gui.params_panel import ParamsPanel
from src.gui.preview_widget import PreviewWidget
from src.gui.quotes_panel import QuotesPanel
from src.gui.state import AppState
from src.gui.styles import build_app_stylesheet, Theme
from src.gui.timeline_widget import TimelineWidget

logger = get_logger(__name__)

# Datei-Endungen fuer Drag & Drop
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}
MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".gif", ".mkv", ".webm"}
PROJECT_EXTENSION = ".avproj"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer Pro")
        self.setWindowIcon(get_app_icon())
        self.setMinimumSize(1200, 750)

        self.state = AppState()
        self.gemini = None
        self.gemini_error = None
        try:
            from src.gemini_integration import GeminiIntegration
            self.gemini = GeminiIntegration()
        except Exception as e:
            self.gemini_error = str(e)
            logger.warning(f"[GUI] Gemini nicht verfügbar: {e}")

        self._preview_worker: PreviewWorker | None = None
        self._analyze_worker: AnalyzeWorker | None = None
        self._ai_optimize_worker: AIOptimizeWorker | None = None
        self._quote_extract_worker: QuoteExtractWorker | None = None
        self._transcribe_worker = None
        self._render_worker: RenderWorker | None = None
        self._intro_worker: IntroWorker | None = None

        self._settings = QSettings("AudioVisualizerPro", "AudioVisualizerPro")
        self._project_path: str | None = None
        self._dirty = False

        self._setup_ui()
        self._setup_menus()
        self._setup_signals()
        self._restore_window_state()
        self.setAcceptDrops(True)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # QSplitter fuer Panel-Layout
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.assets_panel = AssetsPanel(self.state)
        splitter.addWidget(self.assets_panel)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(8)

        self.preview_widget = PreviewWidget()
        center_layout.addWidget(self.preview_widget, stretch=1)

        self.timeline = TimelineWidget()
        center_layout.addWidget(self.timeline)

        splitter.addWidget(center)

        self.right_tabs = QTabWidget()
        self.params_panel = ParamsPanel(self.state)
        self.ki_panel = KIPanel(self.state, gemini=self.gemini, gemini_error=self.gemini_error)
        self.quotes_panel = QuotesPanel(self.state, gemini=self.gemini)

        self.right_tabs.addTab(self._make_scrollable(self.params_panel), "Parameter")
        self.right_tabs.addTab(self._make_scrollable(self.ki_panel), "KI")
        self.right_tabs.addTab(self._make_scrollable(self.quotes_panel), "Zitate")
        splitter.addWidget(self.right_tabs)

        splitter.setSizes([260, 620, 320])
        self._splitter = splitter
        layout.addWidget(splitter, stretch=1)

        # Untere Statusleiste: Statustext + Fortschritt + Aktions-Buttons
        bottom = QHBoxLayout()
        bottom.setContentsMargins(12, 8, 12, 8)
        bottom.setSpacing(10)
        self.status_label = QLabel("Bereit.")
        bottom.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        bottom.addWidget(self.progress_bar)

        bottom.addStretch()

        self.btn_preview = QPushButton(" Vorschau")
        self.btn_preview.setIcon(get_icon("refresh"))
        self.btn_preview.setToolTip("Vorschau manuell neu rendern")
        self.btn_preview.setFixedWidth(120)
        bottom.addWidget(self.btn_preview)

        self.btn_cancel = QPushButton(" Abbrechen")
        self.btn_cancel.setIcon(get_icon("stop", Theme.ERROR))
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setToolTip("Laufendes Rendering abbrechen")
        self.btn_cancel.setFixedWidth(120)
        self.btn_cancel.hide()
        bottom.addWidget(self.btn_cancel)

        self.btn_render = QPushButton(" Rendern")
        self.btn_render.setIcon(get_icon("play", Theme.BACKGROUND))
        self.btn_render.setObjectName("primary")
        self.btn_render.setToolTip("Video in voller Aufloesung rendern")
        self.btn_render.setFixedWidth(120)
        bottom.addWidget(self.btn_render)

        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom)
        layout.addWidget(bottom_widget)

    def _setup_menus(self):
        """Erstellt die Menueleiste mit Tastenkuerzeln."""
        menubar = self.menuBar()

        # --- Datei ---
        file_menu = menubar.addMenu("&Datei")

        act_open_audio = QAction(get_icon("music"), "Audio öffnen…", self)
        act_open_audio.setShortcut(QKeySequence("Ctrl+O"))
        act_open_audio.triggered.connect(self.assets_panel._load_audio)
        file_menu.addAction(act_open_audio)

        file_menu.addSeparator()

        act_open_project = QAction(get_icon("folder-open"), "Projekt öffnen…", self)
        act_open_project.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_open_project.triggered.connect(self._open_project_dialog)
        file_menu.addAction(act_open_project)

        act_save_project = QAction(get_icon("save"), "Projekt speichern", self)
        act_save_project.setShortcut(QKeySequence("Ctrl+S"))
        act_save_project.triggered.connect(self._save_project)
        file_menu.addAction(act_save_project)

        act_save_project_as = QAction("Projekt speichern unter…", self)
        act_save_project_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_project_as.triggered.connect(self._save_project_as)
        file_menu.addAction(act_save_project_as)

        self._recent_menu = file_menu.addMenu("Zuletzt verwendet")
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        act_quit = QAction("Beenden", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # --- Rendern ---
        render_menu = menubar.addMenu("&Rendern")

        act_render = QAction(get_icon("play"), "Rendern starten", self)
        act_render.setShortcut(QKeySequence("F5"))
        act_render.triggered.connect(self._on_render_clicked)
        render_menu.addAction(act_render)

        act_cancel = QAction(get_icon("stop"), "Abbrechen", self)
        act_cancel.setShortcut(QKeySequence("Esc"))
        act_cancel.triggered.connect(self._on_cancel_clicked)
        render_menu.addAction(act_cancel)

        render_menu.addSeparator()

        act_output = QAction(get_icon("folder-open"), "Ausgabeordner öffnen", self)
        act_output.triggered.connect(self._open_output_dir)
        render_menu.addAction(act_output)

        # --- Hilfe ---
        help_menu = menubar.addMenu("&Hilfe")

        act_log = QAction("Log-Datei öffnen", self)
        act_log.triggered.connect(self._open_log_file)
        help_menu.addAction(act_log)

        act_about = QAction("Über Audio Visualizer Pro", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # === Projekt speichern/laden ===

    def _save_project(self):
        if not self._project_path:
            self._save_project_as()
            return
        self._write_project(self._project_path)

    def _save_project_as(self):
        start_dir = self._settings.value("last_project_dir", "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Projekt speichern", start_dir,
            f"Audio Visualizer Projekt (*{PROJECT_EXTENSION})",
        )
        if not path:
            return
        if not path.endswith(PROJECT_EXTENSION):
            path += PROJECT_EXTENSION
        self._write_project(path)

    def _write_project(self, path: str):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "Fehler beim Speichern", str(e))
            return
        self._project_path = path
        self._settings.setValue("last_project_dir", str(Path(path).parent))
        self._add_recent_project(path)
        self._set_dirty(False)
        self._set_status(f"Projekt gespeichert: {Path(path).name}", "ok")

    def _open_project_dialog(self):
        start_dir = self._settings.value("last_project_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Projekt öffnen", start_dir,
            f"Audio Visualizer Projekt (*{PROJECT_EXTENSION})",
        )
        if path:
            self._load_project(path)

    def _load_project(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state.apply_dict(data)
        except Exception as e:
            QMessageBox.critical(self, "Fehler beim Laden", f"Projekt konnte nicht geladen werden:\n{e}")
            return
        self._project_path = path
        self._settings.setValue("last_project_dir", str(Path(path).parent))
        self._add_recent_project(path)

        # Panels aktualisieren, die nicht ueber State-Signale gebunden sind
        if self.state.audio_path and Path(self.state.audio_path).exists():
            self.assets_panel.audio_info.setText(Path(self.state.audio_path).name)
            self._start_analysis()
        elif self.state.audio_path:
            self.assets_panel.audio_info.setText("Audio-Datei nicht gefunden")
            self._set_status(f"Audio-Datei nicht gefunden: {self.state.audio_path}", "warn")
        if self.state.background_path:
            self.assets_panel.bg_path_label.setText(Path(self.state.background_path).name)

        self._set_dirty(False)
        self._set_status(f"Projekt geladen: {Path(path).name}", "ok")

    def _add_recent_project(self, path: str):
        recent = self._settings.value("recent_projects", [], type=list)
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._settings.setValue("recent_projects", recent[:8])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        self._recent_menu.clear()
        recent = self._settings.value("recent_projects", [], type=list)
        existing = [p for p in recent if Path(p).exists()]
        if not existing:
            empty = QAction("(leer)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for p in existing:
            act = QAction(Path(p).name, self)
            act.setToolTip(p)
            act.triggered.connect(lambda checked, path=p: self._load_project(path))
            self._recent_menu.addAction(act)

    def _set_dirty(self, dirty: bool):
        self._dirty = dirty
        title = "Audio Visualizer Pro"
        if self._project_path:
            title += f" — {Path(self._project_path).name}"
        if dirty:
            title += " *"
        self.setWindowTitle(title)

    # === Hilfe / Ordner ===

    def _open_output_dir(self):
        out_dir = Path(self.state.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir.resolve())))

    def _open_log_file(self):
        from src.app_logging import LOG_FILE
        if LOG_FILE.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_FILE.resolve())))
        else:
            QMessageBox.information(self, "Log", "Noch keine Log-Datei vorhanden.")

    def _show_about(self):
        QMessageBox.about(
            self,
            "Über Audio Visualizer Pro",
            "<b>Audio Visualizer Pro</b><br>"
            "GPU-beschleunigte Audio-Visualisierungen mit KI-Unterstützung.<br><br>"
            "16 Visualizer · HDR-Rendering · Bloom · Gemini-Integration<br>"
            "Lizenz: MIT",
        )

    # === Fenster-Zustand (QSettings) ===

    def _restore_window_state(self):
        geometry = self._settings.value("window_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter_state = self._settings.value("splitter_state")
        if splitter_state is not None:
            self._splitter.restoreState(splitter_state)

    def _save_window_state(self):
        self._settings.setValue("window_geometry", self.saveGeometry())
        self._settings.setValue("splitter_state", self._splitter.saveState())

    # === Drag & Drop ===

    def dragEnterEvent(self, event):
        if self._first_supported_drop(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._first_supported_drop(event)
        if path is None:
            return
        event.acceptProposedAction()
        suffix = Path(path).suffix.lower()
        if suffix == PROJECT_EXTENSION:
            self._load_project(path)
        elif suffix in AUDIO_EXTENSIONS:
            self.state.set("audio_path", path)
            self.assets_panel.audio_info.setText(Path(path).name)
            self._start_analysis()
        elif suffix in MEDIA_EXTENSIONS:
            self.state.set("background_path", path)
            self.assets_panel.bg_path_label.setText(Path(path).name)

    @staticmethod
    def _first_supported_drop(event) -> str | None:
        """Liefert den ersten unterstuetzten Dateipfad eines Drop-Events."""
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        supported = AUDIO_EXTENSIONS | MEDIA_EXTENSIONS | {PROJECT_EXTENSION}
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if Path(path).suffix.lower() in supported:
                return path
        return None

    @staticmethod
    def _make_scrollable(widget: QWidget) -> QScrollArea:
        """Hilfsmethode: Widget in scrollbaren Bereich verpacken."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(widget)
        return scroll

    def _setup_signals(self):
        self.assets_panel.analyze_requested.connect(self._start_analysis)
        self.timeline.time_changed.connect(self._on_time_changed)
        self.state.changed.connect(self._on_state_changed)
        self.ki_panel.optimize_requested.connect(self._start_ai_optimize)
        self.quotes_panel.btn_extract.clicked.connect(self._start_quote_extract)
        self.quotes_panel.btn_transcribe.clicked.connect(self._start_transcribe)
        self.btn_render.clicked.connect(self._on_render_clicked)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.btn_preview.clicked.connect(self._start_preview)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._start_preview)

    # State-Keys, die als "Projekt-Aenderung" gelten (fuer den *-Marker)
    _PROJECT_KEYS = frozenset({
        "audio_path", "background_path", "visualizer_type", "viz_params",
        "viz_offset_x", "viz_offset_y", "viz_scale",
        "color_mode", "base_hue", "color_saturation", "viz_brightness",
        "primary_color", "secondary_color", "background_color",
        "bg_blur", "bg_vignette", "bg_opacity",
        "pp_contrast", "pp_saturation", "pp_brightness", "pp_warmth", "pp_grain",
        "pp_exposure", "pp_bloom", "pp_bloom_threshold", "pp_vignette", "pp_chromatic",
        "resolution", "render_fps", "codec", "quality", "gpu_encode",
        "intro_enabled", "intro_path", "intro_fade_duration",
        "quotes", "quotes_enabled", "quote_config",
    })

    def _on_state_changed(self, key: str):
        if key in self._PROJECT_KEYS and not self._dirty:
            self._set_dirty(True)
        if key in {
            "visualizer_type", "viz_params", "viz_offset_x", "viz_offset_y", "viz_scale",
            "bg_blur", "bg_vignette", "bg_opacity",
            "pp_contrast", "pp_saturation", "pp_brightness", "pp_warmth", "pp_grain",
            "pp_exposure", "pp_bloom", "pp_bloom_threshold", "pp_vignette", "pp_chromatic",
            "background_path", "preview_time_percent",
            "quotes", "quotes_enabled", "quote_config", "ki_suggested_colors",
            "color_mode", "base_hue", "color_saturation", "brightness",
        }:
            self._preview_timer.start(150)

    def _on_time_changed(self, percent: float):
        self.state.preview_time_percent = percent
        self._preview_timer.start(150)

    def _start_analysis(self):
        from src.gui.workers import AnalyzeWorker

        if self._analyze_worker and self._analyze_worker.isRunning():
            return
        path = self.state.audio_path
        if not path:
            return
        self._set_status("Analysiere Audio...", "warn")
        self._analyze_worker = AnalyzeWorker(path, fps=self.state.preview_fps)
        self._analyze_worker.analysis_ready.connect(self._on_analysis_ready)
        self._analyze_worker.analysis_error.connect(self._on_analysis_error)
        self._analyze_worker.finished.connect(lambda: self._cleanup_worker("_analyze_worker"))
        self._analyze_worker.start()

    def _on_analysis_ready(self, features):
        if self.sender() is not self._analyze_worker:
            return
        self.state.features = features
        self.state.audio_duration = features.duration
        self.timeline.set_duration(features.duration)
        self.timeline.set_features(features)
        self.assets_panel.audio_info.setText(
            f"{features.duration:.1f}s · {features.tempo:.0f} BPM · {features.mode}"
        )
        self._set_status("Analyse fertig.", "ok")
        self._start_preview()

    def _on_analysis_error(self, msg: str, tb: str = ""):
        if self.sender() is not self._analyze_worker:
            return
        if tb:
            logger.error(f"[GUI] Analyse-Fehler:\n{tb}")
        self._set_status(f"Analyse-Fehler: {msg}", "error")
        QMessageBox.critical(self, "Analyse-Fehler", msg)

    def _start_preview(self):
        from src.gui.workers import PreviewWorker

        if not self.state.audio_path or self.state.features is None:
            return
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.requestInterruption()
        self._preview_worker = PreviewWorker(
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
            background_color=self.state.background_color,
            postprocess=self.state.get_postprocess(),
            viz_offset_x=self.state.viz_offset_x,
            viz_offset_y=self.state.viz_offset_y,
            viz_scale=self.state.viz_scale,
            features=self.state.features,
            quotes=self.state.quotes if self.state.quotes_enabled else None,
            quote_config=self.state.quote_config if self.state.quotes_enabled else None,
        )
        self._preview_worker.preview_ready.connect(self._on_preview_ready)
        self._preview_worker.preview_error.connect(self._on_preview_error)
        self._preview_worker.finished.connect(lambda: self._cleanup_worker("_preview_worker"))
        self.preview_widget.set_busy(True)
        self._preview_worker.start()

    def _start_ai_optimize(self):
        from src.gui.workers import AIOptimizeWorker

        if self._ai_optimize_worker and self._ai_optimize_worker.isRunning():
            return
        req = self.ki_panel.get_optimize_request()
        if not req:
            return
        self._ai_optimize_worker = AIOptimizeWorker(
            gemini=req["gemini"],
            visualizer_type=req["visualizer_type"],
            current_params=req["current_params"],
            audio_features=req["audio_features"],
            colors=req["colors"],
            param_specs=req["param_specs"],
            user_prompt=req["user_prompt"],
            recommendation=req.get("recommendation"),
            parent=self,
        )
        self._ai_optimize_worker.optimize_ready.connect(self.ki_panel.on_optimize_finished)
        self._ai_optimize_worker.optimize_error.connect(self.ki_panel.on_optimize_error)
        self._ai_optimize_worker.finished.connect(lambda: self._cleanup_worker("_ai_optimize_worker"))
        self._ai_optimize_worker.start()

    def _start_quote_extract(self):
        from src.gui.workers import QuoteExtractWorker

        if self._quote_extract_worker and self._quote_extract_worker.isRunning():
            return
        req = self.quotes_panel.get_extract_request()
        if not req or not req.get("audio_path"):
            return
        self._quote_extract_worker = QuoteExtractWorker(
            gemini=req["gemini"],
            audio_path=req["audio_path"],
            audio_duration=req["audio_duration"],
            max_quotes=req.get("max_quotes"),
            use_cache=req.get("use_cache", True),
            parent=self,
        )
        self._quote_extract_worker.quotes_ready.connect(self.quotes_panel.on_extract_finished)
        self._quote_extract_worker.quotes_error.connect(self.quotes_panel.on_extract_error)
        self._quote_extract_worker.finished.connect(lambda: self._cleanup_worker("_quote_extract_worker"))
        self._quote_extract_worker.start()

    def _start_transcribe(self):
        from src.gui.workers import TranscribeWorker

        if self._transcribe_worker and self._transcribe_worker.isRunning():
            return
        req = self.quotes_panel.get_transcribe_request()
        if not req or not req.get("audio_path") or not req.get("gemini"):
            return
        self._transcribe_worker = TranscribeWorker(
            gemini=req["gemini"],
            audio_path=req["audio_path"],
            parent=self,
        )
        self._transcribe_worker.transcribe_ready.connect(self.quotes_panel.on_transcribe_finished)
        self._transcribe_worker.transcribe_error.connect(self.quotes_panel.on_transcribe_error)
        self._transcribe_worker.finished.connect(lambda: self._cleanup_worker("_transcribe_worker"))
        self._transcribe_worker.start()

    def _on_preview_ready(self, img):
        if self.sender() is not self._preview_worker:
            return
        self.preview_widget.set_busy(False)
        self.preview_widget.set_image(img)
        self._set_status("Vorschau aktualisiert.", "ok")

    def _on_preview_error(self, msg: str, tb: str = ""):
        if self.sender() is not self._preview_worker:
            return
        self.preview_widget.set_busy(False)
        self._set_status(f"Vorschau-Fehler: {msg}", "error")
        logger.error(f"[GUI] Vorschau-Fehler: {msg}\n{tb}" if tb else f"[GUI] Vorschau-Fehler: {msg}")
        QMessageBox.warning(
            self,
            "Vorschau fehlgeschlagen",
            f"Die Vorschau konnte nicht erstellt werden:\n\n{msg}\n\n"
            f"Details stehen in logs/app.log.",
        )

    def _set_render_ui(self, running: bool):
        """Schaltet die Aktions-Buttons und den Fortschrittsbalken um."""
        self.btn_render.setEnabled(not running)
        self.btn_cancel.setVisible(running)
        self.progress_bar.setVisible(running)
        if not running:
            self.progress_bar.setValue(0)

    def _on_cancel_clicked(self):
        if self._render_worker and self._render_worker.isRunning():
            self._render_worker.cancel()
            self._set_status("Rendering wird abgebrochen...", "warn")
        if self._intro_worker and self._intro_worker.isRunning():
            self._intro_worker.cancel()
            self._set_render_ui(False)
            self._set_status("Intro abgebrochen.", "warn")

    def _on_render_clicked(self):
        from src.gui.workers import RenderWorker

        if (self._render_worker and self._render_worker.isRunning()) or \
                (self._intro_worker and self._intro_worker.isRunning()):
            return

        if not self.state.audio_path or not Path(self.state.audio_path).exists():
            QMessageBox.critical(self, "Fehler", "Keine Audio-Datei geladen.")
            return
        if self.state.features is None:
            QMessageBox.critical(self, "Fehler", "Audio wurde noch nicht analysiert.")
            return

        out_dir = Path(self.state.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(out_dir / f"visualization_{ts}.mp4")

        w, h = self.state.resolution
        config = {
            "audio_path": self.state.audio_path,
            "visualizer_type": self.state.visualizer_type,
            "output_path": output_path,
            "features": self.state.features,
            "params": self.state.get_params(),
            "background_image": self.state.background_path,
            "background_blur": self.state.bg_blur,
            "background_vignette": self.state.bg_vignette,
            "background_opacity": self.state.bg_opacity,
            "background_color": self.state.background_color,
            "postprocess": self.state.get_postprocess(),
            "quotes": self.state.quotes if self.state.quotes_enabled else None,
            "quote_config": self.state.quote_config if self.state.quotes_enabled else None,
            "width": w,
            "height": h,
            "fps": self.state.render_fps,
            "codec": self.state.codec,
            "quality": self.state.quality,
            "gpu_encode": self.state.gpu_encode,
            "viz_offset_x": self.state.viz_offset_x,
            "viz_offset_y": self.state.viz_offset_y,
            "viz_scale": self.state.viz_scale,
        }

        self._set_render_ui(True)
        self._set_status("Starte Rendering...", "warn")

        self._render_worker = RenderWorker(config, parent=self)
        self._render_worker.render_progress.connect(self._on_render_progress)
        self._render_worker.render_finished.connect(self._on_render_finished)
        self._render_worker.render_error.connect(self._on_render_error)
        self._render_worker.finished.connect(lambda: self._cleanup_worker("_render_worker"))
        self._render_worker.start()

    def _on_render_progress(self, progress: float):
        if self.sender() is not self._render_worker:
            return
        pct = int(progress * 100)
        self.progress_bar.setValue(pct)
        self._set_status(f"Rendere Video... {pct}%", "warn")

    def _on_render_finished(self, output_path: str):
        if self.sender() is not self._render_worker:
            return
        if self.state.intro_enabled and self.state.intro_path:
            if Path(self.state.intro_path).exists():
                self._start_intro_merge(output_path)
                return
            self._set_status("Intro-Datei nicht gefunden, überspringe Intro.", "warn")
        self._finish_render(output_path)

    def _start_intro_merge(self, main_video_path: str):
        from src.gui.workers import IntroWorker

        if self._intro_worker and self._intro_worker.isRunning():
            return
        # Original-Pfad merken, damit Dateien mit mehreren Punkten im Namen korrekt ersetzt werden
        self._intro_main_path = main_video_path
        tmp_path = str(Path(main_video_path).with_suffix(".intro_tmp.mp4"))
        self._intro_worker = IntroWorker(
            intro_path=self.state.intro_path,
            main_video_path=main_video_path,
            output_path=tmp_path,
            fade_duration=self.state.intro_fade_duration,
            parent=self,
        )
        self._intro_worker.intro_progress.connect(self._on_intro_progress)
        self._intro_worker.intro_finished.connect(self._on_intro_finished)
        self._intro_worker.intro_error.connect(self._on_intro_error)
        self._intro_worker.finished.connect(lambda: self._cleanup_worker("_intro_worker"))
        self.progress_bar.setValue(0)
        self._set_status("Füge Intro hinzu...", "warn")
        self._intro_worker.start()

    def _on_intro_progress(self, progress: float):
        if self.sender() is not self._intro_worker:
            return
        pct = int(progress * 100)
        self.progress_bar.setValue(pct)
        self._set_status(f"Füge Intro hinzu... {pct}%", "warn")

    def _on_intro_finished(self, tmp_path: str):
        if self.sender() is not self._intro_worker:
            return
        main_path = getattr(self, "_intro_main_path", None) or str(Path(tmp_path).with_suffix(".mp4"))
        try:
            if Path(main_path).exists():
                Path(main_path).unlink()
            os.replace(tmp_path, main_path)
        except Exception as e:
            self._set_status(f"Intro-Fehler: {e}", "error")
            self._set_render_ui(False)
            QMessageBox.critical(self, "Intro-Fehler", f"Konnte Intro-Datei nicht übernehmen:\n{e}")
            return
        self._finish_render(main_path)

    def _on_intro_error(self, msg: str, tb: str = ""):
        if self.sender() is not self._intro_worker:
            return
        if tb:
            logger.error(f"[GUI] Intro-Fehler:\n{tb}")
        self._set_render_ui(False)
        self._set_status(f"Intro-Fehler: {msg}", "error")
        QMessageBox.critical(self, "Intro-Fehler", msg)
        tmp_path = getattr(self._intro_worker, "output_path", None)
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    def _finish_render(self, output_path: str):
        self._set_render_ui(False)
        self._set_status(f"Fertig: {output_path}", "ok")

        box = QMessageBox(self)
        box.setWindowTitle("Rendering abgeschlossen")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(f"Das Video wurde gespeichert:\n{output_path}")
        open_btn = box.addButton("Ordner öffnen", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            self._open_output_folder(output_path)

    def _open_output_folder(self, output_path: str):
        """Oeffnet den Ausgabeordner im Datei-Explorer."""
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        folder = str(Path(output_path).resolve().parent)
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_render_error(self, msg: str, tb: str = ""):
        if self.sender() is not self._render_worker:
            return
        if tb:
            logger.error(f"[GUI] Render-Fehler:\n{tb}")
        self._set_render_ui(False)
        self._set_status(f"Render-Fehler: {msg}", "error")
        QMessageBox.critical(self, "Render-Fehler", msg)

    def _set_status(self, msg: str, kind: str = "info"):
        self.status_label.setText(msg)
        color_map = {
            "info": Theme.TEXT_SECONDARY,
            "ok": Theme.SUCCESS,
            "warn": Theme.WARNING,
            "error": Theme.ERROR,
        }
        rgb = color_map.get(kind, Theme.TEXT_SECONDARY)
        self.status_label.setStyleSheet(f"color: rgb{rgb};")

    def _cleanup_worker(self, worker_attr: str):
        worker = getattr(self, worker_attr, None)
        if worker is None or self.sender() is not worker:
            return
        try:
            worker.deleteLater()
        except Exception:
            pass
        setattr(self, worker_attr, None)

    def closeEvent(self, event):
        workers = [
            self._preview_worker,
            self._analyze_worker,
            self._ai_optimize_worker,
            self._quote_extract_worker,
            self._render_worker,
            self._intro_worker,
        ]
        for worker in workers:
            if worker and worker.isRunning():
                worker.requestInterruption()
        if self._render_worker and self._render_worker.isRunning():
            self._render_worker.cancel()
        if self._intro_worker and self._intro_worker.isRunning():
            self._intro_worker.cancel()
        for worker in workers:
            if worker and worker.isRunning():
                worker.wait(2000)
        self._save_window_state()
        event.accept()
