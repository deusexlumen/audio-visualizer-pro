"""Entry-Point fuer die PyQt6-GUI."""

import sys
import threading
import traceback

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QFontDatabase
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from src.app_logging import setup_logging, get_logger, LOG_DIR
from src.ffmpeg_locator import find_ffmpeg
from src.gui.main_window import MainWindow
from src.gui.styles import build_app_stylesheet
from src.paths import resource_path

logger = get_logger(__name__)

_FONT_DIR = resource_path("assets", "fonts")


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


def _check_ffmpeg(parent) -> None:
    """First-Run-Check: FFmpeg fehlt oft bei frischer Installation ohne Systemweite
    FFmpeg-Installation. Fragt einmalig nach Download-Erlaubnis statt beim ersten
    Render mit einer kryptischen FileNotFoundError abzubrechen."""
    if find_ffmpeg() is not None:
        return

    answer = QMessageBox.question(
        parent,
        "FFmpeg wird benötigt",
        "FFmpeg wurde nicht gefunden (weder im PATH noch lokal installiert).\n\n"
        "FFmpeg wird für Video-Encoding benötigt (~90 MB Download von gyan.dev).\n"
        "Jetzt herunterladen?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if answer != QMessageBox.StandardButton.Yes:
        logger.warning("[FFmpeg] Download abgelehnt — Rendern wird fehlschlagen, bis FFmpeg verfügbar ist.")
        return

    from src.gui.workers import FFmpegDownloadWorker

    progress = QProgressDialog("FFmpeg wird heruntergeladen…", "Abbrechen", 0, 0, parent)
    progress.setWindowTitle("FFmpeg-Download")
    progress.setMinimumDuration(0)
    progress.setAutoClose(True)
    progress.setCancelButton(None)  # kein sauberer Netzwerk-Abbruch moeglich

    worker = FFmpegDownloadWorker(parent)

    def on_progress(done: int, total: int):
        if total > 0:
            progress.setMaximum(total)
            progress.setValue(done)
        else:
            progress.setMaximum(0)  # unbestimmter Fortschritt, falls Content-Length fehlt

    def on_ready(path: str):
        progress.close()
        logger.info(f"[FFmpeg] Download abgeschlossen: {path}")

    def on_error(message: str, details: str):
        progress.close()
        logger.error(f"[FFmpeg] Download fehlgeschlagen: {details}")
        QMessageBox.critical(
            parent, "FFmpeg-Download fehlgeschlagen",
            f"FFmpeg konnte nicht heruntergeladen werden:\n{message}\n\n"
            "Bitte FFmpeg manuell installieren (https://ffmpeg.org/download.html).",
        )

    worker.download_progress.connect(on_progress)
    worker.download_ready.connect(on_ready)
    worker.download_error.connect(on_error)
    worker.start()
    progress.exec()


def run_app(argv=None):
    setup_logging()
    # Eine Startzeile ins Log: bei Support-Fragen ist als Erstes wichtig,
    # welche Version lief und ob es der installierte Build war.
    from src.gui import __version__
    logger.info(
        f"[App] Audio Visualizer Pro {__version__} gestartet "
        f"({'installiert' if getattr(sys, 'frozen', False) else 'aus dem Quellcode'})"
    )
    _install_exception_handlers()
    app = QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    _load_fonts()
    app.setStyleSheet(build_app_stylesheet())

    from src.gui.icons import get_app_icon
    app.setWindowIcon(get_app_icon())

    window = MainWindow()
    window.show()
    _check_ffmpeg(window)

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app(sys.argv))
