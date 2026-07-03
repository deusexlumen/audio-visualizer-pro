"""Entry-Point fuer die PyQt6-GUI."""

import sys
from PyQt6.QtWidgets import QApplication

from src.app_logging import setup_logging
from src.gui.main_window import MainWindow
from src.gui.styles import build_app_stylesheet


def run_app(argv=None):
    setup_logging()
    app = QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(build_app_stylesheet())

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app(sys.argv))
