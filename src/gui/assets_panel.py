"""Panel zum Laden von Audio, Hintergrund und Quotes."""

from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QGroupBox,
    QSlider, QHBoxLayout,
)

from src.gui.state import AppState


class AssetsPanel(QWidget):
    analyze_requested = pyqtSignal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Audio
        audio_box = QGroupBox("Audio")
        audio_layout = QVBoxLayout(audio_box)
        self.btn_load_audio = QPushButton("Audio laden")
        self.btn_load_audio.clicked.connect(self._load_audio)
        audio_layout.addWidget(self.btn_load_audio)
        self.audio_info = QLabel("Kein Audio geladen")
        self.audio_info.setWordWrap(True)
        audio_layout.addWidget(self.audio_info)
        layout.addWidget(audio_box)

        # Background
        bg_box = QGroupBox("Hintergrund")
        bg_layout = QVBoxLayout(bg_box)
        self.btn_load_bg = QPushButton("Bild/Video laden")
        self.btn_load_bg.clicked.connect(self._load_background)
        bg_layout.addWidget(self.btn_load_bg)

        self.bg_path_label = QLabel("Kein Hintergrund")
        self.bg_path_label.setWordWrap(True)
        bg_layout.addWidget(self.bg_path_label)

        bg_layout.addWidget(QLabel("Blur"))
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(0, 200)
        self.slider_blur.setValue(0)
        self.slider_blur.valueChanged.connect(self._on_blur_changed)
        bg_layout.addWidget(self.slider_blur)

        bg_layout.addWidget(QLabel("Vignette"))
        self.slider_vignette = QSlider(Qt.Orientation.Horizontal)
        self.slider_vignette.setRange(0, 100)
        self.slider_vignette.setValue(0)
        self.slider_vignette.valueChanged.connect(self._on_vignette_changed)
        bg_layout.addWidget(self.slider_vignette)

        bg_layout.addWidget(QLabel("Opacity"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(30)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        bg_layout.addWidget(self.slider_opacity)

        layout.addWidget(bg_box)
        layout.addStretch()

    def _load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Audio laden",
            "",
            "Audio (*.mp3 *.wav *.flac *.aac *.ogg *.m4a)",
        )
        if path:
            self.state.audio_path = path
            self.state.set("audio_path", path)
            self.audio_info.setText(Path(path).name)
            self.analyze_requested.emit()

    def _load_background(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Hintergrund laden",
            "",
            "Bilder/Videos (*.png *.jpg *.jpeg *.mp4 *.mov *.gif)",
        )
        if path:
            self.state.background_path = path
            self.state.set("background_path", path)
            self.bg_path_label.setText(Path(path).name)

    def _on_blur_changed(self, value: int):
        self.state.bg_blur = value / 10.0
        self.state.set("bg_blur", self.state.bg_blur)

    def _on_vignette_changed(self, value: int):
        self.state.bg_vignette = value / 100.0
        self.state.set("bg_vignette", self.state.bg_vignette)

    def _on_opacity_changed(self, value: int):
        self.state.bg_opacity = value / 100.0
        self.state.set("bg_opacity", self.state.bg_opacity)
