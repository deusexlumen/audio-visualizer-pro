from unittest.mock import patch
from PyQt6.QtCore import QCoreApplication
from src.gui.workers import PreviewWorker


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
