import numpy as np
import pytest
from PyQt6.QtCore import Qt
from src.gui.state import AppState
from src.gui.ki_panel import KIPanel


class DummyFeatures:
    """Minimale Audio-Features fuer KI-Panel-Tests ohne librosa/analyzer."""

    def __init__(self):
        self.duration = 10.0
        self.sample_rate = 44100
        self.fps = 30
        self.rms = np.array([0.1, 0.2, 0.3])
        self.onset = np.array([0.0, 0.5, 0.0])
        self.spectral_centroid = np.array([0.2, 0.3])
        self.zero_crossing_rate = np.array([0.01, 0.02])
        self.transient = np.array([0.0, 0.1])
        self.voice_clarity = np.array([0.4, 0.5])
        self.mode = "speech"
        self.tempo = 120.0
        self.key = None


@pytest.fixture
def state():
    return AppState()


def test_ki_panel_creates_widgets(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)
    assert panel.btn_auto_viz is not None
    assert panel.btn_optimize is not None
    assert panel.prompt_input is not None


def test_auto_viz_disabled_without_features(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)
    assert not panel.btn_auto_viz.isEnabled()


def test_auto_viz_enabled_when_features_available(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)
    state.features = {"duration": 10.0}
    assert panel.btn_auto_viz.isEnabled()


def test_auto_viz_recommendation(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)

    state.features = DummyFeatures()
    qtbot.mouseClick(panel.btn_auto_viz, Qt.MouseButton.LeftButton)

    text = panel.lbl_recommendation.text()
    assert any(name in text for name in panel._matcher.VISUAL_DESCRIPTIONS)
    assert panel.btn_apply_recommendation.isEnabled()


def test_apply_recommendation_updates_state(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)

    rec = panel._matcher.match(DummyFeatures())
    panel._last_recommendation = rec
    panel.lbl_recommendation.setText("test")
    panel.btn_apply_recommendation.setEnabled(True)

    original_viz = state.visualizer_type
    original_colors = dict(state.ki_suggested_colors)

    qtbot.mouseClick(panel.btn_apply_recommendation, Qt.MouseButton.LeftButton)

    assert state.visualizer_type == rec.visualizer
    assert state.visualizer_type != original_viz
    assert state.ki_suggested_colors == rec.colors
    assert state.ki_suggested_colors != original_colors
    assert any(rec.params[key] == state.viz_params.get(key) for key in rec.params)


def test_apply_optimize_result_updates_state(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)

    panel._apply_optimize_result({
        "params": {"foo": 1.0},
        "postprocess": {
            "contrast": 1.2,
            "saturation": 0.9,
            "brightness": 0.1,
            "warmth": 0.05,
            "film_grain": 0.2,
        },
        "background": {
            "blur": 0.5,
            "vignette": 0.3,
            "opacity": 0.7,
        },
        "colors": {
            "primary": "#FF0000",
            "secondary": "#00FF00",
            "background": "#0000FF",
        },
    })

    assert state.viz_params.get("foo") == 1.0
    assert state.pp_contrast == 1.2
    assert state.pp_saturation == 0.9
    assert state.pp_brightness == 0.1
    assert state.pp_warmth == 0.05
    assert state.pp_grain == 0.2
    assert state.bg_blur == 0.5
    assert state.bg_vignette == 0.3
    assert state.bg_opacity == 0.7
    assert state.ki_suggested_colors == {
        "primary": "#FF0000",
        "secondary": "#00FF00",
        "background": "#0000FF",
    }


def test_safe_float_ignores_invalid_values(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)

    state.pp_contrast = 1.0
    panel._apply_optimize_result({
        "postprocess": {
            "contrast": None,
            "saturation": "invalid",
        },
    })

    assert state.pp_contrast == 1.0
    assert state.pp_saturation == 1.0


def test_get_optimize_request_with_invalid_visualizer(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)

    state.features = DummyFeatures()
    state.visualizer_type = "nonexistent_visualizer"

    req = panel.get_optimize_request()

    assert req["param_specs"] == {}
    assert req["visualizer_type"] == "nonexistent_visualizer"
    assert "gemini" in req
    assert "current_params" in req
    assert "audio_features" in req


def test_features_change_resets_recommendation(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)

    rec = panel._matcher.match(DummyFeatures())
    panel._last_recommendation = rec
    panel.lbl_recommendation.setText("recommendation")
    panel.btn_apply_recommendation.setEnabled(True)

    state.features = DummyFeatures()

    assert panel._last_recommendation is None
    assert panel.lbl_recommendation.text() == "Noch keine Empfehlung"
    assert not panel.btn_apply_recommendation.isEnabled()


def test_apply_invalid_visualizer_shows_error(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)

    class FakeRec:
        visualizer = "nonexistent_visualizer"
        params = {}
        colors = {}

    panel._last_recommendation = FakeRec()
    panel.btn_apply_recommendation.setEnabled(True)

    qtbot.mouseClick(panel.btn_apply_recommendation, Qt.MouseButton.LeftButton)

    assert "nicht verf\u00fcgbar" in panel.lbl_recommendation.text()
