import pytest
from PyQt6.QtCore import QObject
from src.gui.state import AppState, SPIRAL_STATE_KEYS
from src.spiral_zoom import SpiralSettings


def test_state_initial_defaults():
    s = AppState()
    assert s.audio_path is None
    assert s.visualizer_type == "lumina_core"
    assert s.preview_width == 854


def test_state_set_emits_changed(qtbot):
    s = AppState()
    with qtbot.waitSignal(s.changed, timeout=100):
        s.visualizer_type = "voice_flow"


def test_state_to_dict_roundtrip():
    s = AppState()
    s.audio_path = "/tmp/test.mp3"
    s.bg_blur = 2.5
    data = s.to_dict()
    restored = AppState.from_dict(data)
    assert restored.audio_path == "/tmp/test.mp3"
    assert restored.bg_blur == 2.5


def test_apply_dict_updates_existing_instance(qtbot):
    """apply_dict() muss die bestehende Instanz aktualisieren und Signale feuern."""
    source = AppState()
    source.visualizer_type = "bass_temple"
    source.pp_bloom = 1.2
    source.resolution = (1280, 720)
    data = source.to_dict()

    target = AppState()
    received = []
    target.changed.connect(received.append)
    target.apply_dict(data)

    assert target.visualizer_type == "bass_temple"
    assert target.pp_bloom == 1.2
    assert target.resolution == (1280, 720)
    assert "visualizer_type" in received


def test_apply_dict_ignores_unknown_keys():
    """Unbekannte Schluessel (z.B. aus neueren Versionen) duerfen nicht crashen."""
    s = AppState()
    s.apply_dict({"version": 99, "zukunfts_feature": True, "bg_blur": 3.0})
    assert s.bg_blur == 3.0
    assert not hasattr(s, "zukunfts_feature")


def test_spiral_settings_roundtrip_and_reach_postprocess():
    s = AppState()
    s.pp_spiral_enabled = True
    s.pp_spiral_arms = -2
    s.pp_spiral_speed = -0.75
    restored = AppState.from_dict(s.to_dict())
    assert restored.pp_spiral_enabled is True
    assert restored.pp_spiral_arms == -2
    assert restored.pp_spiral_speed == -0.75
    pp = restored.get_postprocess()
    assert pp["spiral_enabled"] is True
    assert pp["spiral_arms"] == -2
    assert SpiralSettings.from_postprocess(pp).is_active


def test_old_project_without_spiral_keys_keeps_effect_off():
    data = AppState().to_dict()
    for key in SPIRAL_STATE_KEYS:
        data.pop(key)
    restored = AppState.from_dict(data)
    assert restored.pp_spiral_enabled is False
    assert not SpiralSettings.from_postprocess(restored.get_postprocess()).is_active


def test_spiral_state_defaults_match_renderer_defaults():
    s = AppState()
    d = SpiralSettings()
    for key in SPIRAL_STATE_KEYS:
        assert getattr(s, key) == getattr(d, key[len("pp_spiral_"):]), key


def test_spiral_keys_emit_changed(qtbot):
    s = AppState()
    received = []
    s.changed.connect(received.append)
    s.pp_spiral_ratio = 4.0
    assert "pp_spiral_ratio" in received


def test_apply_old_project_switches_running_spiral_off():
    """GUI-Ladepfad (apply_dict): ein altes Projekt ohne Spiral-Schluessel
    setzt einen in der Sitzung eingeschalteten Spiral-Zoom zurueck."""
    old = AppState().to_dict()
    for key in SPIRAL_STATE_KEYS:
        old.pop(key)
    s = AppState()
    s.pp_spiral_enabled = True
    s.pp_spiral_ratio = 4.0
    s.apply_dict(old)
    assert s.pp_spiral_enabled is False
    assert s.pp_spiral_ratio == SpiralSettings().ratio
    assert not SpiralSettings.from_postprocess(s.get_postprocess()).is_active
