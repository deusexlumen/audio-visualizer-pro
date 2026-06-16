import pytest
from src.gui.state import AppState


KI_FIELD_DEFAULTS = {
    "ki_prompt": "",
    "ki_suggested_colors": {},
    "ki_status": "",
    "ki_error": False,
    "ki_optimizing": False,
    "quotes_extracting": False,
}


def test_app_state_has_ki_fields():
    s = AppState()
    for key, expected_default in KI_FIELD_DEFAULTS.items():
        assert hasattr(s, key)
        assert getattr(s, key) == expected_default


def test_ki_fields_emit_changed_signal(qtbot):
    s = AppState()
    received = []
    s.changed.connect(lambda name, container=received: container.append(name))

    s.ki_prompt = "Test prompt"
    s.ki_suggested_colors = {"primary": "#ff0000"}
    s.ki_status = "running"
    s.ki_error = True
    s.ki_optimizing = True
    s.quotes_extracting = True

    expected_keys = list(KI_FIELD_DEFAULTS.keys())
    assert received == expected_keys


def test_ki_serialization():
    s = AppState()
    s.ki_prompt = "Test prompt"
    s.ki_suggested_colors = {"primary": "#ff0000", "secondary": "#00ff00"}

    data = s.to_dict()
    assert data["ki_prompt"] == "Test prompt"
    assert data["ki_suggested_colors"] == {"primary": "#ff0000", "secondary": "#00ff00"}

    restored = AppState.from_dict(data)
    assert restored.ki_prompt == "Test prompt"
    assert restored.ki_suggested_colors == {"primary": "#ff0000", "secondary": "#00ff00"}


def test_ki_serialization_backwards_compatible():
    old_data = {"version": 1, "visualizer_type": "lumina_core"}
    restored = AppState.from_dict(old_data)
    assert restored.ki_prompt == ""
    assert restored.ki_suggested_colors == {}
