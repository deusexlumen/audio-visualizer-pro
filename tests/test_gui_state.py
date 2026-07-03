import pytest
from PyQt6.QtCore import QObject
from src.gui.state import AppState


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
