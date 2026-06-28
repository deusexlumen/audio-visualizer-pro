"""Hauptfenster der neuen Audio Visualizer Pro GUI."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QStatusBar, QLabel, QMessageBox, QTabWidget,
)

from src.gui.assets_panel import AssetsPanel
from src.gui.ki_panel import KIPanel
from src.gui.params_panel import ParamsPanel
from src.gui.preview_widget import PreviewWidget
from src.gui.quotes_panel import QuotesPanel
from src.gui.state import AppState
from src.gui.styles import build_app_stylesheet, Theme
from src.gui.timeline_widget import TimelineWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer Pro")
        self.setMinimumSize(1200, 750)

        self.state = AppState()
        self.gemini = None
        try:
            from src.gemini_integration import GeminiIntegration
            self.gemini = GeminiIntegration()
        except Exception as e:
            print(f"[GUI] Gemini nicht verfügbar: {e}")

        self._preview_worker: PreviewWorker | None = None
        self._analyze_worker: AnalyzeWorker | None = None
        self._ai_optimize_worker: AIOptimizeWorker | None = None
        self._quote_extract_worker: QuoteExtractWorker | None = None
        self._render_worker: RenderWorker | None = None
        self._intro_worker: IntroWorker | None = None

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

        self.right_tabs = QTabWidget()
        self.params_panel = ParamsPanel(self.state)
        self.ki_panel = KIPanel(self.state, gemini=self.gemini)
        self.quotes_panel = QuotesPanel(self.state, gemini=self.gemini)

        self.right_tabs.addTab(self.params_panel, "Params")
        self.right_tabs.addTab(self.ki_panel, "KI")
        self.right_tabs.addTab(self.quotes_panel, "Quotes")
        splitter.addWidget(self.right_tabs)

        splitter.setSizes([260, 620, 320])
        layout.addWidget(splitter)

        # Bottom Bar
        bottom = QHBoxLayout()
        bottom.setContentsMargins(12, 8, 12, 8)
        self.status_label = QLabel("Bereit.")
        bottom.addWidget(self.status_label)

        bottom.addStretch()

        self.btn_preview = QPushButton("🔄 Preview")
        self.btn_preview.setToolTip("Preview manuell neu rendern")
        self.btn_preview.setFixedWidth(120)
        bottom.addWidget(self.btn_preview)

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
        self.ki_panel.optimize_requested.connect(self._start_ai_optimize)
        self.quotes_panel.btn_extract.clicked.connect(self._start_quote_extract)
        self.btn_render.clicked.connect(self._on_render_clicked)
        self.btn_preview.clicked.connect(self._start_preview)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._start_preview)

    def _on_state_changed(self, key: str):
        if key in {
            "visualizer_type", "viz_params", "viz_offset_x", "viz_offset_y", "viz_scale",
            "bg_blur", "bg_vignette", "bg_opacity",
            "pp_contrast", "pp_saturation", "pp_brightness", "pp_warmth", "pp_grain",
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
        self.assets_panel.audio_info.setText(
            f"{features.duration:.1f}s · {features.tempo:.0f} BPM · {features.mode}"
        )
        self._set_status("Analyse fertig.", "ok")
        self._start_preview()

    def _on_analysis_error(self, msg: str):
        if self.sender() is not self._analyze_worker:
            return
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
            parent=self,
        )
        self._quote_extract_worker.quotes_ready.connect(self.quotes_panel.on_extract_finished)
        self._quote_extract_worker.quotes_error.connect(self.quotes_panel.on_extract_error)
        self._quote_extract_worker.finished.connect(lambda: self._cleanup_worker("_quote_extract_worker"))
        self._quote_extract_worker.start()

    def _on_preview_ready(self, img):
        if self.sender() is not self._preview_worker:
            return
        self.preview_widget.set_image(img)
        self._set_status("Preview aktualisiert.", "ok")

    def _on_preview_error(self, msg: str):
        if self.sender() is not self._preview_worker:
            return
        self._set_status(f"Preview-Fehler: {msg}", "error")

    def _on_render_clicked(self):
        from src.gui.workers import RenderWorker

        if self._render_worker and self._render_worker.isRunning():
            self._render_worker.cancel()
            return
        if self._intro_worker and self._intro_worker.isRunning():
            self._intro_worker.cancel()
            self.btn_render.setText("▶ Render")
            self._set_status("Intro abgebrochen.", "warn")
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

        self.btn_render.setText("⏳ Render...")
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
        self._set_status(f"Rendering... {pct}%", "warn")

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
        self.btn_render.setText("⏳ Intro...")
        self._set_status("Füge Intro hinzu...", "warn")
        self._intro_worker.start()

    def _on_intro_progress(self, progress: float):
        if self.sender() is not self._intro_worker:
            return
        pct = int(progress * 100)
        self._set_status(f"Intro... {pct}%", "warn")

    def _on_intro_finished(self, tmp_path: str):
        if self.sender() is not self._intro_worker:
            return
        main_path = str(Path(tmp_path).with_suffix(".mp4"))
        try:
            os.replace(tmp_path, main_path)
        except Exception as e:
            self._set_status(f"Intro-Fehler: {e}", "error")
            self.btn_render.setText("▶ Render")
            QMessageBox.critical(self, "Intro-Fehler", f"Konnte Intro-Datei nicht übernehmen:\n{e}")
            return
        self._finish_render(main_path)

    def _on_intro_error(self, msg: str):
        if self.sender() is not self._intro_worker:
            return
        self.btn_render.setText("▶ Render")
        self._set_status(f"Intro-Fehler: {msg}", "error")
        QMessageBox.critical(self, "Intro-Fehler", msg)
        tmp_path = getattr(self._intro_worker, "output_path", None)
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    def _finish_render(self, output_path: str):
        self.btn_render.setText("▶ Render")
        self._set_status(f"Fertig: {output_path}", "ok")
        QMessageBox.information(self, "Render fertig", f"Video gespeichert:\n{output_path}")

    def _on_render_error(self, msg: str):
        if self.sender() is not self._render_worker:
            return
        self.btn_render.setText("▶ Render")
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
        event.accept()
