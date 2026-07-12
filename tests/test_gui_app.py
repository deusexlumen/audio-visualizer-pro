"""Tests fuer den globalen Exception-Handler der GUI."""

import sys
import threading
from unittest.mock import patch

from src.gui import app as gui_app


def _install_and_capture():
    """Installiert die Handler und gibt den neuen sys.excepthook zurueck."""
    old_sys_hook = sys.excepthook
    old_thread_hook = threading.excepthook
    gui_app._install_exception_handlers()
    new_hook = sys.excepthook
    # Original-Hooks wiederherstellen, damit andere Tests unbeeinflusst bleiben
    sys.excepthook = old_sys_hook
    threading.excepthook = old_thread_hook
    return new_hook


def test_excepthook_wird_installiert():
    old_hook = sys.excepthook
    new_hook = _install_and_capture()
    assert new_hook is not old_hook


def test_excepthook_loggt_traceback():
    hook = _install_and_capture()
    with patch.object(gui_app.logger, "critical") as mock_log, patch.object(
        gui_app, "_show_crash_dialog"
    ) as mock_dialog, patch(
        "src.gui.app.QApplication.instance", return_value=object()
    ):
        try:
            raise ValueError("Testfehler")
        except ValueError:
            hook(*sys.exc_info())

    assert mock_log.called
    logged = mock_log.call_args[0][0]
    assert "ValueError" in logged
    assert "Testfehler" in logged
    mock_dialog.assert_called_once()
    assert "Testfehler" in mock_dialog.call_args[0][0]


def test_excepthook_ohne_qapplication_kein_dialog():
    hook = _install_and_capture()
    with patch.object(gui_app.logger, "critical"), patch.object(
        gui_app, "_show_crash_dialog"
    ) as mock_dialog, patch(
        "src.gui.app.QApplication.instance", return_value=None
    ):
        try:
            raise RuntimeError("ohne GUI")
        except RuntimeError:
            hook(*sys.exc_info())

    mock_dialog.assert_not_called()


def test_excepthook_keyboardinterrupt_durchgereicht():
    hook = _install_and_capture()
    with patch.object(gui_app.logger, "critical") as mock_log, patch(
        "sys.__excepthook__"
    ) as mock_orig:
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            hook(*sys.exc_info())

    mock_orig.assert_called_once()
    mock_log.assert_not_called()


def test_dialogfehler_verursacht_keinen_folgecrash():
    hook = _install_and_capture()
    with patch.object(gui_app.logger, "critical"), patch.object(
        gui_app, "_show_crash_dialog", side_effect=RuntimeError("Dialog kaputt")
    ), patch("src.gui.app.QApplication.instance", return_value=object()):
        try:
            raise ValueError("Testfehler")
        except ValueError:
            # Darf keine Exception nach aussen werfen
            hook(*sys.exc_info())
