"""Tests fuer das ParamsPanel (Farben, Visualizer, Export-Einstellungen)."""

from unittest.mock import patch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QColorDialog, QCheckBox, QDoubleSpinBox, QLabel

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


def _find_viz_param_widget(panel, param_name: str):
    """Sucht das Widget fuer einen Visualizer-Parameter anhand seines Labels."""
    target = param_name.replace("_", " ").title()
    for i in range(panel.viz_params_layout.count()):
        widget = panel.viz_params_layout.itemAt(i).widget()
        if isinstance(widget, QLabel) and widget.text() == target:
            # Naechstes Widget im Layout ist der Editor
            next_item = panel.viz_params_layout.itemAt(i + 1)
            if next_item is not None:
                return next_item.widget()
    return None


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

    spin = _find_viz_param_widget(panel, "core_intensity")
    assert isinstance(spin, QDoubleSpinBox)

    new_value = spin.value() + spin.singleStep()
    spin.setValue(new_value)

    assert state.viz_params.get("core_intensity") == new_value


def test_two_way_binding_updates_panel(qtbot):
    """Aenderungen am AppState muessen das Panel aktualisieren."""
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    state.color_mode = "fixed"
    assert panel.combo_color_mode.currentText() == "fixed"

    state.viz_brightness = 1.75
    assert panel.slider_viz_brightness.value() == 175
    assert panel.lbl_viz_brightness.text() == "175%"

    state.visualizer_type = "particle_swarm"
    assert panel.combo_viz.currentText() == "particle_swarm"


def test_boolean_param_renders_as_checkbox(qtbot):
    """Parameter mit min=0, max=1, step=1 sollen als QCheckBox erscheinen."""
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    idx = panel.combo_viz.findText("particle_swarm")
    panel.combo_viz.setCurrentIndex(idx)

    widget = _find_viz_param_widget(panel, "depth_enabled")
    assert isinstance(widget, QCheckBox)
    assert widget.isChecked()  # Default ist 1


def test_slider_labels_show_current_value(qtbot):
    """Transform-/Post-Process-/Color-Slider zeigen ihren Wert als Label."""
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    panel.slider_scale.setValue(150)
    assert panel.lbl_scale.text() == "1.50x"

    panel.slider_contrast.setValue(125)
    assert panel.lbl_contrast.text() == "1.25x"

    panel.slider_brightness.setValue(42)
    assert panel.lbl_brightness.text() == "42"


def test_visualizer_param_memory(qtbot):
    """Parameter eines Visualizers sollen beim Wechsel gemerkt werden."""
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    # Lumina Core waehlen und Parameter aendern
    panel.combo_viz.setCurrentIndex(panel.combo_viz.findText("lumina_core"))
    spin = _find_viz_param_widget(panel, "core_intensity")
    spin.setValue(2.5)
    assert state.viz_params["core_intensity"] == 2.5

    # Zu Particle Swarm wechseln
    panel.combo_viz.setCurrentIndex(panel.combo_viz.findText("particle_swarm"))
    assert state.visualizer_type == "particle_swarm"

    # Zurueck zu Lumina Core -> alter Wert muss erhalten sein
    panel.combo_viz.setCurrentIndex(panel.combo_viz.findText("lumina_core"))
    assert state.viz_params.get("core_intensity") == 2.5
    spin = _find_viz_param_widget(panel, "core_intensity")
    assert spin.value() == 2.5


def test_reset_viz_params_restores_defaults(qtbot):
    """Der Reset-Button setzt die Parameter auf die Defaults zurueck."""
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    panel.combo_viz.setCurrentIndex(panel.combo_viz.findText("lumina_core"))
    spin = _find_viz_param_widget(panel, "core_intensity")
    default = spin.value()

    spin.setValue(default + 1.0)
    assert state.viz_params["core_intensity"] == default + 1.0

    qtbot.mouseClick(panel.btn_reset_viz_params, Qt.MouseButton.LeftButton)

    assert state.viz_params["core_intensity"] == default
    spin = _find_viz_param_widget(panel, "core_intensity")
    assert spin.value() == default


def test_brightness_is_blacklisted(qtbot):
    """Der globale 'brightness'-Parameter darf nicht als Visualizer-Param erscheinen."""
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    # Lumina Core erbt 'brightness' aus BaseGPUVisualizer.EFFECTS
    panel.combo_viz.setCurrentIndex(panel.combo_viz.findText("lumina_core"))

    for i in range(panel.viz_params_layout.count()):
        widget = panel.viz_params_layout.itemAt(i).widget()
        if isinstance(widget, QLabel) and widget.text().lower().replace(" ", "_") == "brightness":
            raise AssertionError("brightness sollte nicht als Visualizer-Parameter angezeigt werden")

    assert "brightness" not in state.viz_params


def test_param_groups_rendered(qtbot):
    """Visualizer mit PARAMS_GROUPS sollen Gruppen-Header anzeigen."""
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    panel.combo_viz.setCurrentIndex(panel.combo_viz.findText("lumina_core"))

    headers = []
    for i in range(panel.viz_params_layout.count()):
        widget = panel.viz_params_layout.itemAt(i).widget()
        if isinstance(widget, QLabel) and widget.styleSheet() and "font-weight: bold" in widget.styleSheet():
            headers.append(widget.text())

    assert "Core" in headers
    assert "Ringe" in headers
