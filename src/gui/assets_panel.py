"""Panel zum Laden von Audio, Hintergrund und Quotes."""

from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QGroupBox,
    QSlider, QHBoxLayout,
)

from src.gui.icons import get_icon
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
        self.btn_load_audio = QPushButton(" Audio laden")
        self.btn_load_audio.setIcon(get_icon("music"))
        self.btn_load_audio.setToolTip(
            "Audiodatei laden (MP3, WAV, FLAC, …) — die Analyse startet automatisch."
        )
        self.btn_load_audio.clicked.connect(self._load_audio)
        audio_layout.addWidget(self.btn_load_audio)
        self.audio_info = QLabel("Kein Audio geladen")
        self.audio_info.setWordWrap(True)
        audio_layout.addWidget(self.audio_info)
        layout.addWidget(audio_box)

        # Hintergrund
        bg_box = QGroupBox("Hintergrund")
        bg_layout = QVBoxLayout(bg_box)
        self.btn_load_bg = QPushButton(" Bild/Video laden")
        self.btn_load_bg.setIcon(get_icon("image"))
        self.btn_load_bg.setToolTip(
            "Bild oder Video als Hintergrund hinter dem Visualizer anzeigen."
        )
        self.btn_load_bg.clicked.connect(self._load_background)
        bg_layout.addWidget(self.btn_load_bg)

        self.bg_path_label = QLabel("Kein Hintergrund")
        self.bg_path_label.setWordWrap(True)
        bg_layout.addWidget(self.bg_path_label)

        self.slider_blur, self.lbl_blur = self._add_slider(
            bg_layout, "Weichzeichnen", 0, 200, 0,
            "Weichzeichnungs-Radius des Hintergrunds (0 = scharf).",
            lambda v: f"{v / 10.0:.1f}",
        )
        self.slider_blur.valueChanged.connect(self._on_blur_changed)

        self.slider_vignette, self.lbl_vignette = self._add_slider(
            bg_layout, "Vignette", 0, 100, 0,
            "Abdunklung der Hintergrund-Raender (0 % = aus).",
            lambda v: f"{v} %",
        )
        self.slider_vignette.valueChanged.connect(self._on_vignette_changed)

        self.slider_opacity, self.lbl_opacity = self._add_slider(
            bg_layout, "Deckkraft", 0, 100, 30,
            "Sichtbarkeit des Hintergrunds (100 % = volle Deckkraft).",
            lambda v: f"{v} %",
        )
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)

        layout.addWidget(bg_box)
        layout.addStretch()

    @staticmethod
    def _add_slider(parent_layout, title, minimum, maximum, value, tooltip, fmt):
        """Erzeugt einen Slider mit Titel-Zeile und Live-Wertanzeige."""
        row = QHBoxLayout()
        row.addWidget(QLabel(title))
        row.addStretch()
        value_label = QLabel(fmt(value))
        row.addWidget(value_label)
        parent_layout.addLayout(row)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setToolTip(tooltip)
        slider.valueChanged.connect(lambda v: value_label.setText(fmt(v)))
        parent_layout.addWidget(slider)
        return slider, value_label

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
