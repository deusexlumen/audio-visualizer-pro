"""Panel fuer Visualizer-Auswahl, Parameter und Post-Process."""

from PyQt6.QtCore import Qt
import colorsys

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QGroupBox,
    QGridLayout, QCheckBox, QLineEdit, QPushButton, QFileDialog, QDoubleSpinBox,
    QColorDialog, QFrame,
)

from src.gui.state import AppState
from src.gpu_visualizers import list_visualizers, get_visualizer
from src.visualizer_wizard import add_create_visualizer_button


class ParamsPanel(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        # Pro-Visualizer Parameter-Memory: {viz_name: {param_name: value}}
        self._viz_param_memory: dict[str, dict] = {}
        self._current_viz: str = state.visualizer_type
        self._updating: bool = False
        # Widget-Lookup fuer Zwei-Wege-Bindung: {param_name: widget}
        self._viz_param_widgets: dict[str, QWidget] = {}

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

        # Wizard-Button fuer benutzerdefinierte Visualizer
        add_create_visualizer_button(viz_layout, self.state, parent_window=self)

        layout.addWidget(viz_box)

        # Offset / Scale
        transform_box = QGroupBox("Transform")
        transform_layout = QGridLayout(transform_box)
        self.slider_offset_x, self.lbl_offset_x = self._make_labeled_slider(-100, 100, 0)
        self.slider_offset_y, self.lbl_offset_y = self._make_labeled_slider(-100, 100, 0)
        self.slider_scale, self.lbl_scale = self._make_labeled_slider(50, 200, 100)

        transform_layout.addWidget(QLabel("Offset X"), 0, 0)
        transform_layout.addWidget(self.slider_offset_x, 0, 1)
        transform_layout.addWidget(self.lbl_offset_x, 0, 2)
        transform_layout.addWidget(QLabel("Offset Y"), 1, 0)
        transform_layout.addWidget(self.slider_offset_y, 1, 1)
        transform_layout.addWidget(self.lbl_offset_y, 1, 2)
        transform_layout.addWidget(QLabel("Scale"), 2, 0)
        transform_layout.addWidget(self.slider_scale, 2, 1)
        transform_layout.addWidget(self.lbl_scale, 2, 2)
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
        self.slider_color_saturation, self.lbl_color_saturation = self._make_labeled_slider(
            0, 100, int(self.state.color_saturation * 100)
        )
        self.slider_color_saturation.valueChanged.connect(self._on_color_saturation_changed)
        color_layout.addWidget(self.slider_color_saturation, 4, 1)
        color_layout.addWidget(self.lbl_color_saturation, 4, 2)

        color_layout.addWidget(QLabel("Brightness"), 5, 0)
        self.slider_viz_brightness, self.lbl_viz_brightness = self._make_labeled_slider(
            50, 200, int(self.state.viz_brightness * 100)
        )
        self.slider_viz_brightness.valueChanged.connect(self._on_viz_brightness_changed)
        color_layout.addWidget(self.slider_viz_brightness, 5, 1)
        color_layout.addWidget(self.lbl_viz_brightness, 5, 2)

        layout.addWidget(color_box)

        # Visualizer-spezifische Parameter
        self.viz_params_box = QGroupBox("Visualizer Params")
        self.viz_params_layout = QGridLayout(self.viz_params_box)
        layout.addWidget(self.viz_params_box)

        # Post-Process
        pp_box = QGroupBox("Post-Process")
        pp_layout = QGridLayout(pp_box)
        self.slider_contrast, self.lbl_contrast = self._make_labeled_slider(0, 300, 100)
        self.slider_saturation, self.lbl_saturation = self._make_labeled_slider(0, 300, 100)
        self.slider_brightness, self.lbl_brightness = self._make_labeled_slider(-100, 100, 0)
        self.slider_warmth, self.lbl_warmth = self._make_labeled_slider(-100, 100, 0)
        self.slider_grain, self.lbl_grain = self._make_labeled_slider(0, 100, 0)

        pp_layout.addWidget(QLabel("Contrast"), 0, 0)
        pp_layout.addWidget(self.slider_contrast, 0, 1)
        pp_layout.addWidget(self.lbl_contrast, 0, 2)
        pp_layout.addWidget(QLabel("Saturation"), 1, 0)
        pp_layout.addWidget(self.slider_saturation, 1, 1)
        pp_layout.addWidget(self.lbl_saturation, 1, 2)
        pp_layout.addWidget(QLabel("Brightness"), 2, 0)
        pp_layout.addWidget(self.slider_brightness, 2, 1)
        pp_layout.addWidget(self.lbl_brightness, 2, 2)
        pp_layout.addWidget(QLabel("Warmth"), 3, 0)
        pp_layout.addWidget(self.slider_warmth, 3, 1)
        pp_layout.addWidget(self.lbl_warmth, 3, 2)
        pp_layout.addWidget(QLabel("Grain"), 4, 0)
        pp_layout.addWidget(self.slider_grain, 4, 1)
        pp_layout.addWidget(self.lbl_grain, 4, 2)
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
        self.combo_resolution.setToolTip("Zielaufloesung des gerenderten Videos.")
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
            "Qualitaet: Low = schnell/klein, High = scharf, Lossless = verlustfrei aber sehr groß."
        )
        self.combo_quality.currentTextChanged.connect(self._on_quality_changed)
        export_layout.addWidget(self.combo_quality, 3, 1)

        export_layout.addWidget(QLabel("GPU Encode"), 4, 0)
        self.chk_gpu_encode = QCheckBox()
        self.chk_gpu_encode.setChecked(self.state.gpu_encode)
        self.chk_gpu_encode.setToolTip(
            "Hardware-Encoding nutzen (NVENC/AMD/Intel) – deutlich schneller, falls verfuegbar."
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

        # Label-Updates fuer Transform/Post-Process Slider
        self.slider_offset_x.valueChanged.connect(lambda v: self.lbl_offset_x.setText(str(v)))
        self.slider_offset_y.valueChanged.connect(lambda v: self.lbl_offset_y.setText(str(v)))
        self.slider_scale.valueChanged.connect(lambda v: self.lbl_scale.setText(f"{v / 100.0:.2f}x"))
        self.slider_contrast.valueChanged.connect(lambda v: self.lbl_contrast.setText(f"{v / 100.0:.2f}x"))
        self.slider_saturation.valueChanged.connect(lambda v: self.lbl_saturation.setText(f"{v / 100.0:.2f}x"))
        self.slider_brightness.valueChanged.connect(lambda v: self.lbl_brightness.setText(str(v)))
        self.slider_warmth.valueChanged.connect(lambda v: self.lbl_warmth.setText(str(v)))
        self.slider_grain.valueChanged.connect(lambda v: self.lbl_grain.setText(str(v)))
        self.slider_color_saturation.valueChanged.connect(
            lambda v: self.lbl_color_saturation.setText(f"{v}%")
        )
        self.slider_viz_brightness.valueChanged.connect(
            lambda v: self.lbl_viz_brightness.setText(f"{v}%")
        )

        # Initialwerte fuer Label setzen
        self._refresh_slider_labels()

        # Zwei-Wege-Bindung mit AppState
        self.state.changed.connect(self._on_state_changed)

        # Memory initialisieren und erste Parameter-Regler aufbauen
        self._viz_param_memory[self._current_viz] = dict(self.state.viz_params)
        self._rebuild_viz_params(self.state.visualizer_type)

    def _make_slider(self, min_val: int, max_val: int, default: int):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        return slider

    def _make_labeled_slider(self, min_val: int, max_val: int, default: int):
        """Erzeugt einen Slider mit zugehoerigem Wert-Label."""
        slider = self._make_slider(min_val, max_val, default)
        label = QLabel(str(default))
        label.setMinimumWidth(40)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return slider, label

    def _refresh_slider_labels(self):
        """Aktualisiert alle Slider-Labels mit den aktuellen Werten."""
        self.lbl_offset_x.setText(str(self.slider_offset_x.value()))
        self.lbl_offset_y.setText(str(self.slider_offset_y.value()))
        self.lbl_scale.setText(f"{self.slider_scale.value() / 100.0:.2f}x")
        self.lbl_color_saturation.setText(f"{self.slider_color_saturation.value()}%")
        self.lbl_viz_brightness.setText(f"{self.slider_viz_brightness.value()}%")
        self.lbl_contrast.setText(f"{self.slider_contrast.value() / 100.0:.2f}x")
        self.lbl_saturation.setText(f"{self.slider_saturation.value() / 100.0:.2f}x")
        self.lbl_brightness.setText(str(self.slider_brightness.value()))
        self.lbl_warmth.setText(str(self.slider_warmth.value()))
        self.lbl_grain.setText(str(self.slider_grain.value()))

    def _on_state_changed(self, key: str):
        """Reagiert auf Aenderungen im AppState (Zwei-Wege-Bindung)."""
        if self._updating:
            return

        self._updating = True
        try:
            if key == "visualizer_type":
                self.combo_viz.setCurrentText(self.state.visualizer_type)
                self._apply_visualizer_change(self.state.visualizer_type)
            elif key == "viz_params":
                self._sync_viz_param_widgets()
            elif key == "color_mode":
                self.combo_color_mode.setCurrentText(self.state.color_mode)
            elif key == "primary_color":
                self._update_color_button(self.btn_primary_color, self.state.primary_color)
                self.lbl_primary_color.setText(self.state.primary_color)
            elif key == "secondary_color":
                self._update_color_button(self.btn_secondary_color, self.state.secondary_color)
                self.lbl_secondary_color.setText(self.state.secondary_color)
            elif key == "background_color":
                self._update_color_button(self.btn_background_color, self.state.background_color)
                self.lbl_background_color.setText(self.state.background_color)
            elif key == "color_saturation":
                self.slider_color_saturation.setValue(int(self.state.color_saturation * 100))
            elif key == "viz_brightness":
                self.slider_viz_brightness.setValue(int(self.state.viz_brightness * 100))
            elif key == "viz_offset_x":
                self.slider_offset_x.setValue(int(self.state.viz_offset_x * 100))
            elif key == "viz_offset_y":
                self.slider_offset_y.setValue(int(self.state.viz_offset_y * 100))
            elif key == "viz_scale":
                self.slider_scale.setValue(int(self.state.viz_scale * 100))
            elif key == "pp_contrast":
                self.slider_contrast.setValue(int(self.state.pp_contrast * 100))
            elif key == "pp_saturation":
                self.slider_saturation.setValue(int(self.state.pp_saturation * 100))
            elif key == "pp_brightness":
                self.slider_brightness.setValue(int(self.state.pp_brightness * 100))
            elif key == "pp_warmth":
                self.slider_warmth.setValue(int(self.state.pp_warmth * 100))
            elif key == "pp_grain":
                self.slider_grain.setValue(int(self.state.pp_grain * 100))
            elif key == "resolution":
                self.combo_resolution.setCurrentText(
                    f"{self.state.resolution[0]}x{self.state.resolution[1]}"
                )
            elif key == "render_fps":
                self.slider_render_fps.setValue(self.state.render_fps)
                self.lbl_render_fps.setText(str(self.state.render_fps))
            elif key == "codec":
                self.combo_codec.setCurrentText(self._codec_display(self.state.codec))
            elif key == "quality":
                self.combo_quality.setCurrentText(self._quality_display(self.state.quality))
            elif key == "gpu_encode":
                self.chk_gpu_encode.setChecked(self.state.gpu_encode)
            elif key == "intro_enabled":
                self.chk_intro_enabled.setChecked(self.state.intro_enabled)
            elif key == "intro_path":
                self.edit_intro_path.setText(self.state.intro_path or "")
            elif key == "intro_fade_duration":
                self.spin_intro_fade.setValue(self.state.intro_fade_duration)
        finally:
            self._updating = False

    def _sync_viz_param_widgets(self):
        """Synchronisiert die Visualizer-Parameter-Widgets mit state.viz_params."""
        for name, widget in self._viz_param_widgets.items():
            if name not in self.state.viz_params:
                continue
            value = self.state.viz_params[name]
            widget.blockSignals(True)
            try:
                if isinstance(widget, QCheckBox):
                    widget.setChecked(value > 0.5)
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
            finally:
                widget.blockSignals(False)

    def _on_visualizer_changed(self, text: str):
        """Wird von der ComboBox aufgerufen; leitet den Wechsel ueber den State."""
        self.state.visualizer_type = text

    def _apply_visualizer_change(self, new_viz: str):
        """Wechselt den Visualizer und merkt/stellt Parameter wieder her."""
        old_viz = self._current_viz
        if old_viz == new_viz:
            return

        if old_viz:
            self._viz_param_memory[old_viz] = dict(self.state.viz_params)

        self._current_viz = new_viz

        # Parameter fuer neuen Visualizer wiederherstellen oder Defaults laden
        if new_viz in self._viz_param_memory:
            self.state.viz_params = dict(self._viz_param_memory[new_viz])
        else:
            self.state.viz_params = self._default_viz_params(new_viz)

        self._rebuild_viz_params(new_viz)

    def _default_viz_params(self, viz_name: str) -> dict:
        """Liefert die Default-Parameter eines Visualizers (ohne globalen Blacklist)."""
        try:
            viz_class = get_visualizer(viz_name)
        except Exception:
            return {}

        # Diese Werte werden global im Panel gesteuert
        blacklist = {
            "color_mode", "base_hue", "color_saturation",
            "primary_color", "secondary_color", "background_color",
            "brightness",
        }

        defaults = {}
        for source in (
            getattr(viz_class, "EFFECTS", {}),
            getattr(viz_class, "PARAMS", {}),
            getattr(viz_class, "COLOR_PARAMS", {}),
        ):
            for name, spec in source.items():
                if name in blacklist:
                    continue
                if isinstance(spec, (list, tuple)) and len(spec) == 4:
                    defaults[name] = spec[0]
        return defaults

    def _on_resolution_changed(self, text: str):
        if self._updating:
            return
        try:
            w, h = text.split("x")
            self.state.resolution = (int(w), int(h))
        except Exception:
            self.state.resolution = (1920, 1080)

    def _on_render_fps_changed(self, value: int):
        if self._updating:
            return
        self.state.render_fps = value
        if hasattr(self, "lbl_render_fps"):
            self.lbl_render_fps.setText(str(value))

    def _on_codec_changed(self, text: str):
        if self._updating:
            return
        codec_map = {
            "h264 (kompatibel)": "h264",
            "h265 / HEVC": "hevc",
            "ProRes": "prores",
        }
        self.state.codec = codec_map.get(text, "h264")

    def _on_quality_changed(self, text: str):
        if self._updating:
            return
        quality_map = {
            "Low": "low",
            "Medium": "medium",
            "High": "high",
            "Lossless": "lossless",
        }
        self.state.quality = quality_map.get(text, "high")

    def _on_gpu_encode_changed(self, state):
        if self._updating:
            return
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
        if self._updating:
            return
        setattr(self.state, key, value)
        self.state.set(key, value)

    def _create_color_picker(self, label: str, initial_hex: str):
        btn = QPushButton()
        btn.setFixedSize(32, 20)
        btn.setToolTip(f"{label}-Farbe auswaehlen")
        lbl = QLabel(initial_hex)
        lbl.setMinimumWidth(60)
        self._update_color_button(btn, initial_hex)
        btn.clicked.connect(lambda: self._on_color_picked(label.lower(), btn, lbl))
        return btn, lbl

    def _update_color_button(self, btn: QPushButton, hex_color: str):
        btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555;")

    def _on_color_picked(self, key: str, btn: QPushButton, lbl: QLabel):
        if self._updating:
            return
        current = QColor(getattr(self.state, f"{key}_color", "#FFFFFF"))
        color = QColorDialog.getColor(current, self, f"{key.capitalize()}-Farbe waehlen")
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
        if self._updating:
            return
        self.state.color_mode = text

    def _on_color_saturation_changed(self, value: int):
        if self._updating:
            return
        self.state.color_saturation = value / 100.0

    def _on_viz_brightness_changed(self, value: int):
        if self._updating:
            return
        self.state.viz_brightness = value / 100.0

    def _group_params(self, viz_class, specs: dict) -> list[tuple[str, list[tuple[str, tuple]]]]:
        """Gruppiert Parameter nach PARAMS_GROUPS oder per Praefix."""
        groups_def = getattr(viz_class, "PARAMS_GROUPS", None)

        if groups_def:
            grouped: dict[str, list[tuple[str, tuple]]] = {name: [] for name in groups_def}
            grouped["Sonstige"] = []
            assigned = set()
            for name, spec in specs.items():
                placed = False
                for group_name, members in groups_def.items():
                    if name in members:
                        grouped[group_name].append((name, spec))
                        assigned.add(name)
                        placed = True
                        break
                if not placed:
                    grouped["Sonstige"].append((name, spec))
            # Leere Gruppen entfernen und Reihenfolge beibehalten
            result = []
            for name in list(groups_def.keys()) + ["Sonstige"]:
                if grouped[name]:
                    result.append((name, grouped[name]))
            return result

        # Fallback: Praefix-Gruppierung
        prefix_map: dict[str, list[tuple[str, tuple]]] = {}
        no_prefix = []
        for name, spec in specs.items():
            if "_" in name:
                prefix = name.split("_", 1)[0]
                prefix_map.setdefault(prefix, []).append((name, spec))
            else:
                no_prefix.append((name, spec))

        result = []
        for prefix in sorted(prefix_map.keys()):
            result.append((prefix.capitalize(), prefix_map[prefix]))
        if no_prefix:
            result.append(("Allgemein", no_prefix))
        return result

    def _rebuild_viz_params(self, viz_name: str):
        """Baut die pro-Visualizer Parameter-Regler dynamisch auf."""
        self._viz_param_widgets.clear()

        # Alte Widgets entfernen
        while self.viz_params_layout.count():
            item = self.viz_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        try:
            viz_class = get_visualizer(viz_name)
        except Exception:
            self.viz_params_layout.addWidget(QLabel("Keine Parameter verfuegbar"), 0, 0)
            self._add_reset_button(1)
            return

        specs = {}
        for source in (
            getattr(viz_class, "EFFECTS", {}),
            getattr(viz_class, "PARAMS", {}),
            getattr(viz_class, "COLOR_PARAMS", {}),
        ):
            specs.update(source)

        # Diese Werte werden bereits global im Panel gesteuert
        blacklist = {
            "color_mode", "base_hue", "color_saturation",
            "primary_color", "secondary_color", "background_color",
            "brightness",
        }

        # Parameter filtern und validieren
        valid_specs: dict[str, tuple] = {}
        for name, spec in specs.items():
            if name in blacklist:
                continue
            if not isinstance(spec, (list, tuple)) or len(spec) != 4:
                continue
            default, min_val, max_val, step = spec
            if min_val is None or max_val is None:
                continue
            try:
                min_val = float(min_val)
                max_val = float(max_val)
                step = float(step)
                default = float(default)
            except (TypeError, ValueError):
                continue
            valid_specs[name] = (default, min_val, max_val, step)

        if not valid_specs:
            self.viz_params_layout.addWidget(QLabel("Keine weiteren Parameter"), 0, 0)
            self._add_reset_button(1)
            return

        row = 0
        grouped = self._group_params(viz_class, valid_specs)

        for group_name, members in grouped:
            # Gruppen-Header
            header = QLabel(group_name)
            header.setStyleSheet("font-weight: bold; color: #cccccc; padding-top: 4px;")
            self.viz_params_layout.addWidget(header, row, 0, 1, 2)
            row += 1

            for name, spec in sorted(members):
                default, min_val, max_val, step = spec
                current = self.state.viz_params.get(name, default)
                is_boolean = (min_val == 0.0 and max_val == 1.0 and step == 1.0)

                self.viz_params_layout.addWidget(QLabel(name.replace("_", " ").title()), row, 0)

                if is_boolean:
                    chk = QCheckBox()
                    chk.setChecked(current > 0.5)
                    chk.stateChanged.connect(lambda s, n=name: self._on_viz_param_changed(n, float(s > 0)))
                    self.viz_params_layout.addWidget(chk, row, 1)
                    self._viz_param_widgets[name] = chk
                else:
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
                    self.viz_params_layout.addWidget(spin, row, 1)
                    self._viz_param_widgets[name] = spin
                row += 1

            # Visueller Trenner zwischen Gruppen
            if row > 0:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet("color: #444444; margin-top: 2px; margin-bottom: 2px;")
                self.viz_params_layout.addWidget(line, row, 0, 1, 2)
                row += 1

        self._add_reset_button(row)

    def _add_reset_button(self, row: int):
        """Fuegt den Reset-Button fuer Visualizer-Parameter hinzu."""
        self.btn_reset_viz_params = QPushButton("Auf Standardwerte zuruecksetzen")
        self.btn_reset_viz_params.setToolTip(
            "Setzt die Visualizer-spezifischen Parameter auf ihre Defaults zurueck."
        )
        self.btn_reset_viz_params.clicked.connect(self._on_reset_viz_params)
        self.viz_params_layout.addWidget(self.btn_reset_viz_params, row, 0, 1, 2)

    def _on_reset_viz_params(self):
        """Setzt die Parameter des aktuellen Visualizers auf Defaults zurueck."""
        defaults = self._default_viz_params(self._current_viz)
        self.state.viz_params = defaults
        self._viz_param_memory[self._current_viz] = dict(defaults)
        self._rebuild_viz_params(self._current_viz)

    def _on_viz_param_changed(self, name: str, value: float):
        if self._updating:
            return
        self.state.viz_params[name] = value
        self.state.set("viz_params", self.state.viz_params)

    def _on_intro_enabled_changed(self, state):
        if self._updating:
            return
        self.state.intro_enabled = bool(state)

    def _on_intro_path_changed(self, text: str):
        if self._updating:
            return
        self.state.intro_path = text.strip() or None

    def _on_intro_fade_changed(self, value: float):
        if self._updating:
            return
        self.state.intro_fade_duration = value

    def _browse_intro(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Intro-Video auswaehlen",
            "",
            "Video-Dateien (*.mp4 *.mov *.avi *.mkv)",
        )
        if path:
            self.edit_intro_path.setText(path)

    @staticmethod
    def _clear_layout(layout: QGridLayout):
        """Entfernt rekursiv alle Items aus einem Layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                ParamsPanel._clear_layout(item.layout())
