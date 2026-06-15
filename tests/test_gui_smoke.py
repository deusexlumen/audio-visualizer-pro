# tests/test_gui_smoke.py
import sys
from PyQt6.QtWidgets import QApplication


def test_app_starts_and_exits():
    app = QApplication.instance() or QApplication(sys.argv)
    from src.gui.main_window import MainWindow
    window = MainWindow()
    window.show()
    window.close()
    assert window is not None
