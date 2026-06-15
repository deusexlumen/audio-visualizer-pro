"""Hauptfenster der neuen Audio Visualizer Pro GUI."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QStatusBar, QLabel, QMessageBox,
)

from src.gui.assets_panel import AssetsPanel
from src.gui.params_panel import ParamsPanel
from src.gui.preview_widget import PreviewWidget
from src.gui.state import AppState
from src.gui.styles import build_app_stylesheet, Theme
from src.gui.timeline_widget import TimelineWidget
from src.gui.workers import AnalyzeWorker, PreviewWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer Pro")
        self.setMinimumSize(1200, 750)

        self.state = AppState()
        self._preview_worker: PreviewWorker | None = None
        self._analyze_worker: AnalyzeWorker | None = None

        self._setup_ui()
        self._setup_signals()

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

        self.params_panel = ParamsPanel(self.state)
        splitter.addWidget(self.params_panel)

        splitter.setSizes([260, 620, 320])
        layout.addWidget(splitter)

        # Bottom Bar
        bottom = QHBoxLayout()
        bottom.setContentsMargins(12, 8, 12, 8)
        self.status_label = QLabel("Bereit.")
        bottom.addWidget(self.status_label)

        bottom.addStretch()

        self.btn_render = QPushButton("▶ Render")
        self.btn_render.setObjectName("primary")
        self.btn_render.setFixedWidth(120)
        bottom.addWidget(self.btn_render)

        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom)
        layout.addWidget(bottom_widget)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _setup_signals(self):
        self.assets_panel.analyze_requested.connect(self._start_analysis)
        self.timeline.time_changed.connect(self._on_time_changed)
        self.state.changed.connect(self._on_state_changed)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._start_preview)

    def _on_state_changed(self, key: str):
        if key in {
            "visualizer_type", "viz_params", "viz_offset_x", "viz_offset_y", "viz_scale",
            "bg_blur", "bg_vignette", "bg_opacity",
            "pp_contrast", "pp_saturation", "pp_brightness", "pp_warmth", "pp_grain",
            "background_path", "preview_time_percent",
        }:
            self._preview_timer.start(50)

    def _on_time_changed(self, percent: float):
        self.state.preview_time_percent = percent
        self._preview_timer.start(50)

    def _start_analysis(self):
        path = self.state.audio_path
        if not path:
            return
        self._set_status("Analysiere Audio...", "warn")
        self._analyze_worker = AnalyzeWorker(path, fps=self.state.preview_fps)
        self._analyze_worker.analysis_ready.connect(self._on_analysis_ready)
        self._analyze_worker.analysis_error.connect(self._on_analysis_error)
        self._analyze_worker.start()

    def _on_analysis_ready(self, features):
        self.state.features = features
        self.state.audio_duration = features.duration
        self.timeline.set_duration(features.duration)
        self.assets_panel.audio_info.setText(
            f"{features.duration:.1f}s · {features.tempo:.0f} BPM · {features.mode}"
        )
        self._set_status("Analyse fertig.", "ok")
        self._start_preview()

    def _on_analysis_error(self, msg: str):
        self._set_status(f"Analyse-Fehler: {msg}", "error")
        QMessageBox.critical(self, "Analyse-Fehler", msg)

    def _start_preview(self):
        if not self.state.audio_path or self.state.features is None:
            return

        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.requestInterruption()
            self._preview_worker.wait(100)

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
        self._preview_worker.start()

    def _on_preview_ready(self, img):
        self.preview_widget.set_image(img)
        self._set_status("Preview aktualisiert.", "ok")

    def _on_preview_error(self, msg: str):
        self._set_status(f"Preview-Fehler: {msg}", "error")

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

    def closeEvent(self, event):
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.requestInterruption()
            self._preview_worker.wait(500)
        if self._analyze_worker and self._analyze_worker.isRunning():
            self._analyze_worker.wait(500)
        event.accept()
