"""Panel fuer Visualizer-Auswahl, Parameter und Post-Process."""

from PyQt6.QtCore import Qt
import colorsys

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QGroupBox,
    QGridLayout, QCheckBox, QLineEdit, QPushButton, QFileDialog, QDoubleSpinBox,
    QColorDialog,
)

from src.gui.state import AppState
from src.gpu_visualizers import list_visualizers, get_visualizer


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
        self.combo_viz.setCurrentText(self.state.visualizer_type)
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

        # Farb-Einstellungen
        color_box = QGroupBox("Color")
        color_layout = QGridLayout(color_box)

        color_layout.addWidget(QLabel("Color Mode"), 0, 0)
        self.combo_color_mode = QComboBox()
        self.combo_color_mode.addItems(["chroma", "fixed", "monochrome", "warm", "cool"])
        self.combo_color_mode.setCurrentText(self.state.color_mode)
        self.combo_color_mode.currentTextChanged.connect(self._on_color_mode_changed)
        color_layout.addWidget(self.combo_color_mode, 0, 1)

        self.btn_primary_color, self.lbl_primary_color = self._create_color_picker(
            "Primary", self.state.primary_color
        )
        color_layout.addWidget(QLabel("Primary"), 1, 0)
        color_layout.addWidget(self.btn_primary_color, 1, 1)
        color_layout.addWidget(self.lbl_primary_color, 1, 2)

        self.btn_secondary_color, self.lbl_secondary_color = self._create_color_picker(
            "Secondary", self.state.secondary_color
        )
        color_layout.addWidget(QLabel("Secondary"), 2, 0)
        color_layout.addWidget(self.btn_secondary_color, 2, 1)
        color_layout.addWidget(self.lbl_secondary_color, 2, 2)

        self.btn_background_color, self.lbl_background_color = self._create_color_picker(
            "Background", self.state.background_color
        )
        color_layout.addWidget(QLabel("Background"), 3, 0)
        color_layout.addWidget(self.btn_background_color, 3, 1)
        color_layout.addWidget(self.lbl_background_color, 3, 2)

        color_layout.addWidget(QLabel("Saturation"), 4, 0)
        self.slider_color_saturation = self._make_slider(0, 100, int(self.state.color_saturation * 100))
        self.slider_color_saturation.valueChanged.connect(self._on_color_saturation_changed)
        color_layout.addWidget(self.slider_color_saturation, 4, 1)

        color_layout.addWidget(QLabel("Brightness"), 5, 0)
        self.slider_viz_brightness = self._make_slider(50, 200, int(self.state.viz_brightness * 100))
        self.slider_viz_brightness.valueChanged.connect(self._on_viz_brightness_changed)
        color_layout.addWidget(self.slider_viz_brightness, 5, 1)

        layout.addWidget(color_box)

        # Visualizer-spezifische Parameter
        self.viz_params_box = QGroupBox("Visualizer Params")
        self.viz_params_layout = QGridLayout(self.viz_params_box)
        layout.addWidget(self.viz_params_box)

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

        self.chk_intro_enabled = QCheckBox("Intro vor Hauptvideo setzen")
        self.chk_intro_enabled.setChecked(self.state.intro_enabled)
        self.chk_intro_enabled.setToolTip(
            "Ein kurzes Intro-Video wird vor das gerenderte Visualizer-Video gesetzt."
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
        self.spin_intro_fade.setToolTip("Dauer des Crossfades von Intro zu Hauptvideo in Sekunden.")
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

        self._rebuild_viz_params(self.state.visualizer_type)

    def _make_slider(self, min_val: int, max_val: int, default: int):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        return slider

    def _on_visualizer_changed(self, text: str):
        self.state.visualizer_type = text
        self.state.viz_params = {}
        self.state.set("visualizer_type", text)
        self._rebuild_viz_params(text)

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

    def _create_color_picker(self, label: str, initial_hex: str):
        btn = QPushButton()
        btn.setFixedSize(32, 20)
        btn.setToolTip(f"{label}-Farbe auswählen")
        lbl = QLabel(initial_hex)
        lbl.setMinimumWidth(60)
        self._update_color_button(btn, initial_hex)
        btn.clicked.connect(lambda: self._on_color_picked(label.lower(), btn, lbl))
        return btn, lbl

    def _update_color_button(self, btn: QPushButton, hex_color: str):
        btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555;")

    def _on_color_picked(self, key: str, btn: QPushButton, lbl: QLabel):
        current = QColor(getattr(self.state, f"{key}_color", "#FFFFFF"))
        color = QColorDialog.getColor(current, self, f"{key.capitalize()}-Farbe wählen")
        if not color.isValid():
            return
        hex_color = color.name().upper()
        setattr(self.state, f"{key}_color", hex_color)
        self._update_color_button(btn, hex_color)
        lbl.setText(hex_color)

        if key == "primary":
            r, g, b = color.redF(), color.greenF(), color.blueF()
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            self.state.base_hue = h
            self.state.color_saturation = s
            self.slider_color_saturation.setValue(int(s * 100))

    def _on_color_mode_changed(self, text: str):
        self.state.color_mode = text

    def _on_color_saturation_changed(self, value: int):
        self.state.color_saturation = value / 100.0

    def _on_viz_brightness_changed(self, value: int):
        self.state.viz_brightness = value / 100.0

    def _rebuild_viz_params(self, viz_name: str):
        """Baut die pro-Visualizer Parameter-Regler dynamisch auf."""
        # Alte Widgets entfernen
        while self.viz_params_layout.count():
            item = self.viz_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            viz_class = get_visualizer(viz_name)
        except Exception:
            self.viz_params_layout.addWidget(QLabel("Keine Parameter verfuegbar"), 0, 0)
            return

        specs = {}
        for source in (
            getattr(viz_class, "EFFECTS", {}),
            getattr(viz_class, "PARAMS", {}),
            getattr(viz_class, "COLOR_PARAMS", {}),
        ):
            specs.update(source)

        # Diese Werte werden bereits oben im Panel gesteuert
        blacklist = {
            "color_mode", "base_hue", "color_saturation",
            "primary_color", "secondary_color", "background_color",
        }

        row = 0
        for name, spec in sorted(specs.items()):
            if name in blacklist:
                continue
            if not isinstance(spec, (list, tuple)) or len(spec) != 4:
                continue
            default, min_val, max_val, step = spec
            # String-Parameter (min/max None) ueberspringen
            if min_val is None or max_val is None:
                continue
            try:
                min_val = float(min_val)
                max_val = float(max_val)
                step = float(step)
                default = float(default)
            except (TypeError, ValueError):
                continue

            current = self.state.viz_params.get(name, default)
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            # Dezimalstellen an step anpassen
            if step >= 1.0 and step == int(step):
                spin.setDecimals(0)
            elif step >= 0.1 and step == round(step, 1):
                spin.setDecimals(1)
            else:
                spin.setDecimals(2)
            spin.setValue(float(current))
            spin.valueChanged.connect(lambda v, n=name: self._on_viz_param_changed(n, v))

            self.viz_params_layout.addWidget(QLabel(name.replace("_", " ").title()), row, 0)
            self.viz_params_layout.addWidget(spin, row, 1)
            row += 1

        if row == 0:
            self.viz_params_layout.addWidget(QLabel("Keine weiteren Parameter"), 0, 0)

    def _on_viz_param_changed(self, name: str, value: float):
        self.state.viz_params[name] = value
        self.state.set("viz_params", self.state.viz_params)

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
