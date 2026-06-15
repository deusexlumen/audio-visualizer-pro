"""Panel fuer Visualizer-Auswahl, Parameter und Post-Process."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QSlider, QGroupBox,
    QGridLayout,
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

    def _set(self, key: str, value):
        setattr(self.state, key, value)
        self.state.set(key, value)
