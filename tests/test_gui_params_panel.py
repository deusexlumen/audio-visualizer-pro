"""Tests fuer das ParamsPanel (Farben, Visualizer, Export-Einstellungen)."""

from unittest.mock import patch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QColorDialog

from src.gui.params_panel import ParamsPanel
from src.gui.state import AppState


class _FakeColor:
    """Minimaler Ersatz fuer QColor im ColorDialog-Test."""

    def __init__(self, hex_color: str):
        self._hex = hex_color
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        self._r = r / 255.0
        self._g = g / 255.0
        self._b = b / 255.0

    def isValid(self):
        return True

    def name(self):
        return self._hex

    def redF(self):
        return self._r

    def greenF(self):
        return self._g

    def blueF(self):
        return self._b


def test_params_panel_initial_state(qtbot):
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    assert panel.combo_viz.count() > 0
    assert panel.combo_color_mode.currentText() == state.color_mode
    assert panel.lbl_primary_color.text() == state.primary_color
    assert panel.lbl_background_color.text() == state.background_color


def test_color_mode_changed_updates_state(qtbot):
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    idx = panel.combo_color_mode.findText("fixed")
    panel.combo_color_mode.setCurrentIndex(idx)

    assert state.color_mode == "fixed"


def test_primary_color_picker_updates_state(qtbot):
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    with patch.object(QColorDialog, "getColor", return_value=_FakeColor("#123456")):
        qtbot.mouseClick(panel.btn_primary_color, Qt.MouseButton.LeftButton)

    assert state.primary_color == "#123456"
    assert panel.lbl_primary_color.text() == "#123456"


def test_background_color_picker_updates_state(qtbot):
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    with patch.object(QColorDialog, "getColor", return_value=_FakeColor("#ABCDEF")):
        qtbot.mouseClick(panel.btn_background_color, Qt.MouseButton.LeftButton)

    assert state.background_color == "#ABCDEF"


def test_viz_brightness_slider_updates_state(qtbot):
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    panel.slider_viz_brightness.setValue(150)

    assert abs(state.viz_brightness - 1.5) < 0.01


def test_visualizer_params_rebuilt_on_change(qtbot):
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    # Lumina Core hat PARAMS -> es sollten Regler angelegt werden
    idx = panel.combo_viz.findText("lumina_core")
    panel.combo_viz.setCurrentIndex(idx)

    assert panel.viz_params_layout.count() > 0
    assert "core_intensity" in state.viz_params or any(
        panel.viz_params_layout.itemAt(i).widget() is not None
        for i in range(panel.viz_params_layout.count())
    )


def test_visualizer_param_spin_updates_state(qtbot):
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    idx = panel.combo_viz.findText("lumina_core")
    panel.combo_viz.setCurrentIndex(idx)

    # Suche den ersten QDoubleSpinBox und aendere seinen Wert
    spin = None
    for i in range(panel.viz_params_layout.count()):
        widget = panel.viz_params_layout.itemAt(i).widget()
        if widget is not None and widget.__class__.__name__ == "QDoubleSpinBox":
            spin = widget
            break

    assert spin is not None
    new_value = spin.value() + spin.singleStep()
    spin.setValue(new_value)

    param_label = panel.viz_params_layout.itemAt(
        panel.viz_params_layout.indexOf(spin) - 1
    ).widget().text().lower().replace(" ", "_")
    assert state.viz_params.get(param_label) == new_value
