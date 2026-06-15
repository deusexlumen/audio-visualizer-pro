# tests/test_gui_styles.py
from src.gui.styles import Theme, build_app_stylesheet


def test_theme_colors_are_rgb_tuples():
    assert Theme.BACKGROUND == (10, 10, 15)
    assert Theme.ACCENT == (96, 176, 255)


def test_stylesheet_contains_background_color():
    qss = build_app_stylesheet()
    assert "#0a0a0f" in qss
    assert "QGroupBox" in qss
    assert "QPushButton" in qss
