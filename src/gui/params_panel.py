"""Panel fuer Visualizer-Auswahl, Parameter und Post-Process."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QGroupBox,
    QGridLayout, QCheckBox, QLineEdit, QPushButton, QFileDialog, QDoubleSpinBox,
)

from src.gui.state import AppState
from src.gpu_visualizers import list_visualizers


class ParamsPanel(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Visualizer
        viz_box = QGroupBox("Visualizer")
        viz_layout = QVBoxLayout(viz_box)
        self.combo_viz = QComboBox()
        self.combo_viz.addItems(list_visualizers())
        self.combo_viz.currentTextChanged.connect(self._on_visualizer_changed)
        viz_layout.addWidget(self.combo_viz)
        layout.addWidget(viz_box)

        # Offset / Scale
        transform_box = QGroupBox("Transform")
        transform_layout = QGridLayout(transform_box)
        self.slider_offset_x = self._make_slider(-100, 100, 0)
        self.slider_offset_y = self._make_slider(-100, 100, 0)
        self.slider_scale = self._make_slider(50, 200, 100)

        transform_layout.addWidget(QLabel("Offset X"), 0, 0)
        transform_layout.addWidget(self.slider_offset_x, 0, 1)
        transform_layout.addWidget(QLabel("Offset Y"), 1, 0)
        transform_layout.addWidget(self.slider_offset_y, 1, 1)
        transform_layout.addWidget(QLabel("Scale"), 2, 0)
        transform_layout.addWidget(self.slider_scale, 2, 1)
        layout.addWidget(transform_box)

        # Post-Process
        pp_box = QGroupBox("Post-Process")
        pp_layout = QGridLayout(pp_box)
        self.slider_contrast = self._make_slider(0, 300, 100)
        self.slider_saturation = self._make_slider(0, 300, 100)
        self.slider_brightness = self._make_slider(-100, 100, 0)
        self.slider_warmth = self._make_slider(-100, 100, 0)
        self.slider_grain = self._make_slider(0, 100, 0)

        pp_layout.addWidget(QLabel("Contrast"), 0, 0)
        pp_layout.addWidget(self.slider_contrast, 0, 1)
        pp_layout.addWidget(QLabel("Saturation"), 1, 0)
        pp_layout.addWidget(self.slider_saturation, 1, 1)
        pp_layout.addWidget(QLabel("Brightness"), 2, 0)
        pp_layout.addWidget(self.slider_brightness, 2, 1)
        pp_layout.addWidget(QLabel("Warmth"), 3, 0)
        pp_layout.addWidget(self.slider_warmth, 3, 1)
        pp_layout.addWidget(QLabel("Grain"), 4, 0)
        pp_layout.addWidget(self.slider_grain, 4, 1)
        layout.addWidget(pp_box)

        # Intro-Einstellungen
        intro_box = QGroupBox("Intro")
        intro_layout = QGridLayout(intro_box)

        self.chk_intro_enabled = QCheckBox("Intro an Hauptvideo anhängen")
        self.chk_intro_enabled.setChecked(self.state.intro_enabled)
        self.chk_intro_enabled.setToolTip(
            "Ein kurzes Intro-Video vor das gerenderte Visualizer-Video setzen."
        )
        self.chk_intro_enabled.stateChanged.connect(self._on_intro_enabled_changed)
        intro_layout.addWidget(self.chk_intro_enabled, 0, 0, 1, 2)

        intro_layout.addWidget(QLabel("Intro-Datei"), 1, 0)
        intro_path_layout = QHBoxLayout()
        self.edit_intro_path = QLineEdit(self.state.intro_path or "")
        self.edit_intro_path.setPlaceholderText("Pfad zum Intro-Video (.mp4, .mov, ...)")
        self.edit_intro_path.textChanged.connect(self._on_intro_path_changed)
        self.btn_intro_browse = QPushButton("Durchsuchen...")
        self.btn_intro_browse.clicked.connect(self._browse_intro)
        intro_path_layout.addWidget(self.edit_intro_path)
        intro_path_layout.addWidget(self.btn_intro_browse)
        intro_layout.addLayout(intro_path_layout, 1, 1)

        intro_layout.addWidget(QLabel("Fade-Dauer"), 2, 0)
        self.spin_intro_fade = QDoubleSpinBox()
        self.spin_intro_fade.setRange(0.1, 2.0)
        self.spin_intro_fade.setSingleStep(0.1)
        self.spin_intro_fade.setDecimals(1)
        self.spin_intro_fade.setValue(self.state.intro_fade_duration)
        self.spin_intro_fade.setToolTip("Dauer des Crossfades zwischen Intro und Hauptvideo in Sekunden.")
        self.spin_intro_fade.valueChanged.connect(self._on_intro_fade_changed)
        intro_layout.addWidget(self.spin_intro_fade, 2, 1)

        layout.addWidget(intro_box)

        # Export-Einstellungen
        export_box = QGroupBox("Export")
        export_layout = QGridLayout(export_box)

        export_layout.addWidget(QLabel("Resolution"), 0, 0)
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(["1920x1080", "1280x720", "854x480", "3840x2160"])
        self.combo_resolution.setCurrentText(f"{self.state.resolution[0]}x{self.state.resolution[1]}")
        self.combo_resolution.setToolTip("Zielauflösung des gerenderten Videos.")
        self.combo_resolution.currentTextChanged.connect(self._on_resolution_changed)
        export_layout.addWidget(self.combo_resolution, 0, 1)

        export_layout.addWidget(QLabel("Render FPS"), 1, 0)
        fps_layout = QHBoxLayout()
        self.slider_render_fps = self._make_slider(24, 60, self.state.render_fps)
        self.slider_render_fps.setToolTip("Framerate des finalen Videos.")
        self.lbl_render_fps = QLabel(str(self.state.render_fps))
        self.lbl_render_fps.setMinimumWidth(24)
        self.slider_render_fps.valueChanged.connect(self._on_render_fps_changed)
        fps_layout.addWidget(self.slider_render_fps)
        fps_layout.addWidget(self.lbl_render_fps)
        export_layout.addLayout(fps_layout, 1, 1)

        export_layout.addWidget(QLabel("Codec"), 2, 0)
        self.combo_codec = QComboBox()
        self.combo_codec.addItems(["h264 (kompatibel)", "h265 / HEVC", "ProRes"])
        self.combo_codec.setCurrentText(self._codec_display(self.state.codec))
        self.combo_codec.setToolTip(
            "Video-Codec: h264 = kompatibel, h265 = kleinere Dateien, ProRes = hochwertig aber groß."
        )
        self.combo_codec.currentTextChanged.connect(self._on_codec_changed)
        export_layout.addWidget(self.combo_codec, 2, 1)

        export_layout.addWidget(QLabel("Quality"), 3, 0)
        self.combo_quality = QComboBox()
        self.combo_quality.addItems(["Low", "Medium", "High", "Lossless"])
        self.combo_quality.setCurrentText(self._quality_display(self.state.quality))
        self.combo_quality.setToolTip(
            "Qualität: Low = schnell/klein, High = scharf, Lossless = verlustfrei aber sehr groß."
        )
        self.combo_quality.currentTextChanged.connect(self._on_quality_changed)
        export_layout.addWidget(self.combo_quality, 3, 1)

        export_layout.addWidget(QLabel("GPU Encode"), 4, 0)
        self.chk_gpu_encode = QCheckBox()
        self.chk_gpu_encode.setChecked(self.state.gpu_encode)
        self.chk_gpu_encode.setToolTip(
            "Hardware-Encoding nutzen (NVENC/AMD/Intel) – deutlich schneller, falls verfügbar."
        )
        self.chk_gpu_encode.stateChanged.connect(self._on_gpu_encode_changed)
        export_layout.addWidget(self.chk_gpu_encode, 4, 1)

        layout.addWidget(export_box)
        layout.addStretch()

        # Signals verbinden
        self.slider_offset_x.valueChanged.connect(lambda v: self._set("viz_offset_x", v / 100.0))
        self.slider_offset_y.valueChanged.connect(lambda v: self._set("viz_offset_y", v / 100.0))
        self.slider_scale.valueChanged.connect(lambda v: self._set("viz_scale", v / 100.0))
        self.slider_contrast.valueChanged.connect(lambda v: self._set("pp_contrast", v / 100.0))
        self.slider_saturation.valueChanged.connect(lambda v: self._set("pp_saturation", v / 100.0))
        self.slider_brightness.valueChanged.connect(lambda v: self._set("pp_brightness", v / 100.0))
        self.slider_warmth.valueChanged.connect(lambda v: self._set("pp_warmth", v / 100.0))
        self.slider_grain.valueChanged.connect(lambda v: self._set("pp_grain", v / 100.0))

    def _make_slider(self, min_val: int, max_val: int, default: int):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        return slider

    def _on_visualizer_changed(self, text: str):
        self.state.visualizer_type = text
        self.state.viz_params = {}
        self.state.set("visualizer_type", text)

    def _on_resolution_changed(self, text: str):
        try:
            w, h = text.split("x")
            self.state.resolution = (int(w), int(h))
        except Exception:
            self.state.resolution = (1920, 1080)

    def _on_render_fps_changed(self, value: int):
        self.state.render_fps = value
        if hasattr(self, "lbl_render_fps"):
            self.lbl_render_fps.setText(str(value))

    def _on_codec_changed(self, text: str):
        codec_map = {
            "h264 (kompatibel)": "h264",
            "h265 / HEVC": "hevc",
            "ProRes": "prores",
        }
        self.state.codec = codec_map.get(text, "h264")

    def _on_quality_changed(self, text: str):
        quality_map = {
            "Low": "low",
            "Medium": "medium",
            "High": "high",
            "Lossless": "lossless",
        }
        self.state.quality = quality_map.get(text, "high")

    def _on_gpu_encode_changed(self, state):
        self.state.gpu_encode = bool(state)

    @staticmethod
    def _codec_display(codec: str) -> str:
        return {
            "h264": "h264 (kompatibel)",
            "hevc": "h265 / HEVC",
            "prores": "ProRes",
        }.get(codec, "h264 (kompatibel)")

    @staticmethod
    def _quality_display(quality: str) -> str:
        return {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
            "lossless": "Lossless",
        }.get(quality, "High")

    def _set(self, key: str, value):
        setattr(self.state, key, value)
        self.state.set(key, value)

    def _on_intro_enabled_changed(self, state):
        self.state.intro_enabled = bool(state)

    def _on_intro_path_changed(self, text: str):
        self.state.intro_path = text.strip() or None

    def _on_intro_fade_changed(self, value: float):
        self.state.intro_fade_duration = value

    def _browse_intro(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Intro-Video auswählen",
            "",
            "Video-Dateien (*.mp4 *.mov *.avi *.mkv)",
        )
        if path:
            self.edit_intro_path.setText(path)
