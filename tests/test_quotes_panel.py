import pytest
from PyQt6.QtCore import Qt
from src.gui.state import AppState
from src.gui.quotes_panel import QuotesPanel


@pytest.fixture
def state():
    return AppState()


def test_quotes_panel_creates_widgets(qtbot, state):
    panel = QuotesPanel(state, gemini=None)
    qtbot.addWidget(panel)
    assert panel.chk_enabled is not None
    assert panel.btn_extract is not None
    assert panel.list_quotes is not None


def test_quotes_panel_adds_demo_quote(qtbot, state):
    state.features = type("F", (), {"duration": 10.0})()
    panel = QuotesPanel(state, gemini=None)
    qtbot.addWidget(panel)
    qtbot.mouseClick(panel.btn_demo, Qt.MouseButton.LeftButton)
    assert len(state.quotes) > 0
