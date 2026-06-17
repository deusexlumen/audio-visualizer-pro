import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from src.gui.main_window import MainWindow


def test_main_window_has_three_tabs(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.right_tabs.count() == 3
    assert window.right_tabs.tabText(0) == "Params"
    assert window.right_tabs.tabText(1) == "KI"
    assert window.right_tabs.tabText(2) == "Quotes"


def test_render_button_shows_error_without_audio(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    with patch.object(QMessageBox, "critical", return_value=None) as mock_crit:
        qtbot.mouseClick(window.btn_render, Qt.MouseButton.LeftButton)

    assert window._render_worker is None
    mock_crit.assert_called_once()
    assert "Audio" in mock_crit.call_args[0][2]


def test_render_button_shows_error_without_features(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    audio = tmp_path / "test.wav"
    audio.touch()
    window.state.audio_path = str(audio)

    with patch.object(QMessageBox, "critical", return_value=None) as mock_crit:
        qtbot.mouseClick(window.btn_render, Qt.MouseButton.LeftButton)

    assert window._render_worker is None
    mock_crit.assert_called_once()
    assert "analysiert" in mock_crit.call_args[0][2]


def test_render_button_starts_render_worker(qtbot, tmp_path, dummy_audio_features):
    window = MainWindow()
    qtbot.addWidget(window)

    audio = tmp_path / "test.wav"
    audio.touch()
    window.state.audio_path = str(audio)
    window.state.features = dummy_audio_features

    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = False

    with patch("src.gui.workers.RenderWorker", return_value=mock_worker) as mock_cls, \
         patch.object(QMessageBox, "information", return_value=None):
        qtbot.mouseClick(window.btn_render, Qt.MouseButton.LeftButton)

    assert mock_cls.called
    config = mock_cls.call_args.args[0]
    assert config["audio_path"] == str(audio)
    assert config["visualizer_type"] == window.state.visualizer_type
    assert config["width"] == window.state.resolution[0]
    assert config["height"] == window.state.resolution[1]
    assert config["fps"] == window.state.render_fps
    assert config["codec"] == window.state.codec
    assert config["quality"] == window.state.quality
    assert config["gpu_encode"] == window.state.gpu_encode
    assert config["background_color"] == window.state.background_color

    mock_worker.start.assert_called_once()
    assert window.btn_render.text() == "⏳ Render..."
    assert window._render_worker is mock_worker


def test_preview_button_starts_preview_worker(qtbot, tmp_path, dummy_audio_features):
    window = MainWindow()
    qtbot.addWidget(window)

    audio = tmp_path / "test.wav"
    audio.touch()
    window.state.audio_path = str(audio)
    window.state.features = dummy_audio_features

    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = False

    with patch("src.gui.workers.PreviewWorker", return_value=mock_worker) as mock_cls:
        qtbot.mouseClick(window.btn_preview, Qt.MouseButton.LeftButton)

    assert mock_cls.called
    config = mock_cls.call_args.kwargs
    assert config["audio_path"] == str(audio)
    assert config["visualizer_type"] == window.state.visualizer_type
    assert config["background_color"] == window.state.background_color
    assert config["width"] == window.state.preview_width
    mock_worker.start.assert_called_once()
    assert window._preview_worker is mock_worker


def test_preview_timer_triggers_after_parameter_change(qtbot, tmp_path, dummy_audio_features):
    window = MainWindow()
    qtbot.addWidget(window)

    audio = tmp_path / "test.wav"
    audio.touch()
    window.state.audio_path = str(audio)
    window.state.features = dummy_audio_features

    with patch.object(window._preview_timer, "start") as mock_timer_start:
        window.state.color_mode = "fixed"
        qtbot.wait(100)
        mock_timer_start.assert_called_once_with(150)
