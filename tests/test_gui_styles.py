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


def test_stylesheet_covers_all_major_widgets():
    """Der QSS-Vollausbau muss alle sichtbaren Widget-Typen abdecken."""
    qss = build_app_stylesheet()
    for widget in [
        "QTabWidget::pane", "QTabBar::tab", "QScrollBar:vertical",
        "QCheckBox::indicator", "QComboBox QAbstractItemView",
        "QListWidget::item", "QMenuBar", "QMenu::item",
        "QProgressBar::chunk", "QSplitter::handle", "QToolTip",
        "QMessageBox",
    ]:
        assert widget in qss, f"{widget} fehlt im Stylesheet"


def test_stylesheet_has_interaction_states():
    """Hover-, Fokus- und Disabled-Zustaende muessen definiert sein."""
    qss = build_app_stylesheet()
    for state in [":hover", ":focus", ":disabled", ":selected", ":pressed"]:
        assert state in qss, f"Zustand {state} fehlt im Stylesheet"
