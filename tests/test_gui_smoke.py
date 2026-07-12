import pytest
from src.gui.main_window import MainWindow


def test_main_window_opens(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.preview_widget is not None
    assert window.right_tabs is not None
    assert window.right_tabs.count() == 4
