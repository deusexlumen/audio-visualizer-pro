"""Entry-Point fuer die PyQt6-GUI."""

import sys
from pathlib import Path

from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QApplication

from src.app_logging import setup_logging, get_logger
from src.gui.main_window import MainWindow
from src.gui.styles import build_app_stylesheet

logger = get_logger(__name__)

_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


def _load_fonts():
    """Laedt gebuendelte Schriften (Inter), Fallback ist Segoe UI aus dem QSS."""
    if not _FONT_DIR.exists():
        return
    for font_file in _FONT_DIR.glob("*.ttf"):
        font_id = QFontDatabase.addApplicationFont(str(font_file))
        if font_id < 0:
            logger.warning(f"[GUI] Schrift konnte nicht geladen werden: {font_file.name}")


def run_app(argv=None):
    setup_logging()
    app = QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    _load_fonts()
    app.setStyleSheet(build_app_stylesheet())

    from src.gui.icons import get_app_icon
    app.setWindowIcon(get_app_icon())

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app(sys.argv))
