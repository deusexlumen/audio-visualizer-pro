"""Entry-Point fuer die PyQt6-GUI."""

import sys
import threading
import traceback
from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QFontDatabase
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.app_logging import setup_logging, get_logger, LOG_DIR
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


def _show_crash_dialog(message: str, details: str) -> None:
    """Zeigt einen Fehlerdialog mit Traceback-Details und Log-Ordner-Button."""
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("Unerwarteter Fehler")
    box.setText(
        "Ein unerwarteter Fehler ist aufgetreten.\n\n"
        f"{message}\n\n"
        "Details stehen im Log (logs/app.log)."
    )
    box.setDetailedText(details)
    log_button = box.addButton("Log-Ordner öffnen", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Close)
    box.exec()
    if box.clickedButton() is log_button:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_DIR.resolve())))


def _install_exception_handlers() -> None:
    """Faengt unbehandelte Exceptions global ab (Qt schluckt Slot-Fehler sonst still)."""

    def handle_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical(f"Unbehandelte Exception:\n{details}")
        # Dialog nur zeigen, wenn eine QApplication laeuft (nicht in Tests ohne GUI)
        if QApplication.instance() is not None:
            try:
                _show_crash_dialog(str(exc_value), details)
            except Exception:
                # Dialog-Anzeige darf nie einen Folgecrash ausloesen
                logger.exception("Fehlerdialog konnte nicht angezeigt werden")

    def handle_thread_exception(args):
        handle_exception(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception


def run_app(argv=None):
    setup_logging()
    _install_exception_handlers()
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
