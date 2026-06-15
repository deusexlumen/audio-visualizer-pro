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
