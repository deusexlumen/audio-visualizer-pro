from unittest.mock import patch, MagicMock
from PyQt6.QtCore import QCoreApplication
from src.gui.workers import PreviewWorker, RenderWorker, AIOptimizeWorker, QuoteExtractWorker


def test_preview_worker_emits_ready(qtbot):
    app = QCoreApplication.instance() or QCoreApplication([])
    worker = PreviewWorker(
        audio_path="/tmp/test.wav",
        visualizer_type="lumina_core",
        width=320,
        height=180,
        fps=30,
        preview_time_percent=0.3,
    )

    with patch("src.gui.workers.render_gpu_preview") as mock_render:
        from PIL import Image
        mock_render.return_value = Image.new("RGB", (320, 180), (0, 0, 0))

        with qtbot.waitSignal(worker.preview_ready, timeout=1000):
            worker.start()
            qtbot.waitUntil(lambda: not worker.isRunning(), timeout=1000)

    mock_render.assert_called_once()


def _mock_future(result):
    future = MagicMock()
    future.result.return_value = result
    return future


def test_ai_optimize_worker_emits_optimize_ready(qtbot):
    app = QCoreApplication.instance() or QCoreApplication([])
    gemini = MagicMock()
    expected_result = {"params": {"intensity": 1.5}, "colors": {"primary": "#ff0000"}}
    gemini.optimize_all_settings_async.return_value = _mock_future(expected_result)

    worker = AIOptimizeWorker(
        gemini=gemini,
        visualizer_type="lumina_core",
        current_params={"intensity": 1.0},
        audio_features={"rms": [0.5]},
        colors={"primary": "#000000"},
        param_specs={"intensity": {"min": 0.0, "max": 3.0}},
        user_prompt="make it brighter",
    )

    with qtbot.waitSignal(worker.optimize_ready, timeout=1000):
        worker.start()
        qtbot.waitUntil(lambda: not worker.isRunning(), timeout=1000)

    gemini.optimize_all_settings_async.assert_called_once_with(
        visualizer_type="lumina_core",
        current_params={"intensity": 1.0},
        audio_features={"rms": [0.5]},
        colors={"primary": "#000000"},
        param_specs={"intensity": {"min": 0.0, "max": 3.0}},
        user_prompt="make it brighter",
    )


def test_quote_extract_worker_emits_quotes_ready(qtbot):
    app = QCoreApplication.instance() or QCoreApplication([])
    gemini = MagicMock()
    expected_quotes = [
        {"text": "Quote one", "start_time": 1.0, "end_time": 3.0},
        {"text": "Quote two", "start_time": 5.0, "end_time": 7.0},
    ]
    gemini.extract_quotes_async.return_value = _mock_future(expected_quotes)

    worker = QuoteExtractWorker(
        gemini=gemini,
        audio_path="/tmp/test.wav",
        audio_duration=120.0,
        max_quotes=5,
    )

    with qtbot.waitSignal(worker.quotes_ready, timeout=1000):
        worker.start()
        qtbot.waitUntil(lambda: not worker.isRunning(), timeout=1000)

    gemini.extract_quotes_async.assert_called_once_with(
        audio_path="/tmp/test.wav",
        audio_duration=120.0,
        max_quotes=5,
    )


def _mock_error_future(error):
    future = MagicMock()
    future.result.side_effect = error
    return future


def test_ai_optimize_worker_emits_optimize_error(qtbot):
    app = QCoreApplication.instance() or QCoreApplication([])
    gemini = MagicMock()
    gemini.optimize_all_settings_async.return_value = _mock_error_future(
        RuntimeError("API error")
    )

    worker = AIOptimizeWorker(
        gemini=gemini,
        visualizer_type="lumina_core",
        current_params={"intensity": 1.0},
        audio_features={"rms": [0.5]},
        colors={"primary": "#000000"},
    )

    captured = []
    worker.optimize_error.connect(captured.append)
    worker.run()

    assert len(captured) == 1
    assert "API error" in captured[0]


def test_quote_extract_worker_emits_quotes_error(qtbot):
    app = QCoreApplication.instance() or QCoreApplication([])
    gemini = MagicMock()
    gemini.extract_quotes_async.return_value = _mock_error_future(
        RuntimeError("API error")
    )

    worker = QuoteExtractWorker(
        gemini=gemini,
        audio_path="/tmp/test.wav",
        audio_duration=120.0,
        max_quotes=5,
    )

    captured = []
    worker.quotes_error.connect(captured.append)
    worker.run()

    assert len(captured) == 1
    assert "API error" in captured[0]


def test_render_worker_emits_finished(qtbot):
    app = QCoreApplication.instance() or QCoreApplication([])
    config = {
        "audio_path": "/tmp/test.wav",
        "visualizer_type": "lumina_core",
        "output_path": "/tmp/output.mp4",
        "width": 320,
        "height": 180,
        "fps": 30,
    }

    mock_renderer = MagicMock()
    mock_renderer.render = MagicMock()

    with patch("src.gpu_renderer.GPUBatchRenderer", return_value=mock_renderer):
        worker = RenderWorker(config)
        with qtbot.waitSignal(worker.render_finished, timeout=1000):
            worker.start()
            qtbot.waitUntil(lambda: not worker.isRunning(), timeout=1000)

    mock_renderer.render.assert_called_once()
    mock_renderer.release.assert_called_once()
