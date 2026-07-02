import numpy as np
import pytest
from PyQt6.QtCore import Qt
from unittest.mock import MagicMock

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
        self.spectral_rolloff = np.array([0.3, 0.4])
        self.zero_crossing_rate = np.array([0.01, 0.02])
        self.transient = np.array([0.0, 0.1])
        self.voice_clarity = np.array([0.4, 0.5])
        self.voice_band = np.array([0.3, 0.4])
        self.chroma = np.zeros((12, 3))
        self.mfcc = np.zeros((13, 3))
        self.tempogram = np.zeros((384, 3))
        self.beat_frames = np.array([0, 2])
        self.mode = "speech"
        self.tempo = 120.0
        self.key = None


@pytest.fixture
def state():
    return AppState()


def test_ki_panel_creates_widgets(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)
    assert panel.btn_optimize is not None
    assert panel.prompt_input is not None


def test_optimize_disabled_without_features_or_gemini(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)
    assert not panel.btn_optimize.isEnabled()

    state.features = {"duration": 10.0}
    # Ohne Gemini immer noch disabled
    assert not panel.btn_optimize.isEnabled()


def test_optimize_enabled_with_features_and_gemini(qtbot, state):
    panel = KIPanel(state, gemini=MagicMock())
    qtbot.addWidget(panel)
    state.features = DummyFeatures()
    assert panel.btn_optimize.isEnabled()


def test_optimize_runs_smartmatcher_and_requests_gemini(qtbot, state):
    gemini = MagicMock()
    panel = KIPanel(state, gemini=gemini)
    qtbot.addWidget(panel)

    state.features = DummyFeatures()
    received = []
    panel.optimize_requested.connect(lambda: received.append(True))

    qtbot.mouseClick(panel.btn_optimize, Qt.MouseButton.LeftButton)

    text = panel.lbl_recommendation.text()
    assert panel._last_recommendation is not None
    assert panel._last_recommendation.visualizer in text
    assert len(received) == 1
    assert state.ki_optimizing
    assert not panel.btn_optimize.isEnabled()


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
    assert state.primary_color == "#FF0000"
    assert state.secondary_color == "#00FF00"
    assert state.background_color == "#0000FF"
    assert state.color_mode == "fixed"


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
    panel = KIPanel(state, gemini=MagicMock())
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

    state.features = DummyFeatures()

    assert panel._last_recommendation is None
    assert "Noch keine Optimierung" in panel.lbl_recommendation.text()


def test_smartmatcher_colors_applied_to_state(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)

    rec = panel._matcher.match(DummyFeatures())
    panel._apply_recommendation(rec)

    assert state.color_mode == "fixed"
    assert state.primary_color == rec.colors["primary"]
    assert state.background_color == rec.colors["background"]


def test_get_optimize_request_includes_recommendation(qtbot, state):
    panel = KIPanel(state, gemini=MagicMock())
    qtbot.addWidget(panel)

    state.features = DummyFeatures()
    qtbot.mouseClick(panel.btn_optimize, Qt.MouseButton.LeftButton)

    req = panel.get_optimize_request()
    assert req["recommendation"] is not None
    assert req["recommendation"]["visualizer"] == panel._last_recommendation.visualizer


def test_get_optimize_request_contains_brightness_and_noisiness(qtbot, state):
    panel = KIPanel(state, gemini=MagicMock())
    qtbot.addWidget(panel)

    state.features = DummyFeatures()
    req = panel.get_optimize_request()

    assert "brightness" in req["audio_features"]
    assert "noisiness" in req["audio_features"]
    assert req["audio_features"]["brightness"] > 0


def test_apply_optimize_result_maps_quotes_to_state(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)

    panel._apply_optimize_result({
        "quotes": {
            "font_size": 60,
            "box_color": "#222222",
            "font_color": "#EEEEEE",
            "position": "center",
            "display_duration": 10.0,
            "auto_scale_font": True,
            "text_shadow_enabled": True,
            "box_gradient": True,
            "accent_line": True,
            "accent_line_color": "#FFAA00",
            "box_padding": 40,
            "box_radius": 20,
            "box_margin_bottom": 120,
            "max_width_ratio": 0.7,
            "fade_duration": 0.8,
            "line_spacing": 12,
            "max_font_size": 80,
            "max_chars_per_line": 35,
        }
    })

    qc = state.quote_config
    assert qc.font_size == 60
    assert qc.position == "center"
    assert qc.display_duration == 10.0
    assert qc.box_color == (34, 34, 34, 255)
    assert qc.font_color == (238, 238, 238, 255)
    assert qc.accent_line_color == (255, 170, 0, 255)


def test_apply_optimize_result_ignores_invalid_quotes(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)

    original_font_size = state.quote_config.font_size
    panel._apply_optimize_result({
        "quotes": {
            "font_size": -999,  # ungueltig (unter min)
            "position": "invalid_position",
        }
    })

    # Pydantic-Validierung setzt ungueltige Werte auf Defaults zurueck
    assert state.quote_config.font_size == original_font_size
    assert state.quote_config.position == "bottom"
