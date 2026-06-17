"""
Tests fuer GPU-Live-Preview Modul.

Mocked AudioAnalyzer und GPUPreviewRenderer fuer schnelle, hardware-unabhaengige Tests.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import src.gpu_preview as gpu_preview
from src.types import AudioFeatures


@pytest.fixture
def dummy_features():
    """Minimal AudioFeatures fuer Preview-Tests."""
    return AudioFeatures(
        duration=10.0,
        sample_rate=44100,
        fps=30,
        frame_count=300,
        rms=np.random.rand(300).astype(np.float32),
        onset=np.random.rand(300).astype(np.float32),
        spectral_centroid=np.random.rand(300).astype(np.float32),
        spectral_rolloff=np.random.rand(300).astype(np.float32),
        zero_crossing_rate=np.random.rand(300).astype(np.float32),
        transient=np.random.rand(300).astype(np.float32),
        voice_clarity=np.random.rand(300).astype(np.float32),
        voice_band=np.random.rand(300).astype(np.float32),
        chroma=np.random.rand(12, 300).astype(np.float32),
        mfcc=np.random.rand(13, 300).astype(np.float32),
        tempogram=np.random.rand(384, 300).astype(np.float32),
        tempo=120.0,
        key="C",
        mode="music",
        beat_frames=np.array([0, 30, 60]),
    )


def _make_mock_renderer():
    """Erzeugt einen mockten Renderer, der valide RGB-Pixel liefert."""
    mock_renderer = MagicMock()
    mock_renderer.ctx = MagicMock()
    mock_renderer.fbo = MagicMock()
    mock_renderer.viz_fbo = MagicMock()
    mock_renderer.post_fbo = MagicMock()
    mock_renderer.fbo.read.return_value = b'\x00' * (480 * 270 * 3)
    mock_renderer.post_fbo.read.return_value = b'\x00' * (480 * 270 * 3)
    return mock_renderer


def _make_mock_viz():
    """Erzeugt einen mockten Visualizer."""
    return MagicMock()


class TestRenderGpuPreview:
    """Tests fuer render_gpu_preview."""

    @patch('src.gpu_preview.AudioAnalyzer')
    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_render_gpu_preview_success(self, mock_renderer_cls, mock_analyzer_cls, dummy_features):
        """Erfolgreiches Preview-Rendering sollte ein PIL Image zurueckgeben."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = dummy_features
        mock_analyzer_cls.return_value = mock_analyzer

        mock_renderer = _make_mock_renderer()
        mock_renderer_cls.return_value = mock_renderer

        img = gpu_preview.render_gpu_preview(
            audio_path="dummy.mp3",
            visualizer_type="lumina_core",
            width=480,
            height=270,
            fps=30,
        )

        assert img is not None
        assert img.size == (480, 270)
        mock_analyzer.analyze.assert_called_once()
        mock_renderer_cls.assert_called_once_with(width=480, height=270, fps=30)
        mock_renderer.release.assert_called_once()

    @patch('src.gpu_preview.AudioAnalyzer')
    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_render_gpu_preview_with_postprocess(self, mock_renderer_cls, mock_analyzer_cls, dummy_features):
        """Preview mit Post-Process sollte post_fbo verwenden."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = dummy_features
        mock_analyzer_cls.return_value = mock_analyzer

        mock_renderer = _make_mock_renderer()
        mock_renderer_cls.return_value = mock_renderer

        img = gpu_preview.render_gpu_preview(
            audio_path="dummy.mp3",
            visualizer_type="lumina_core",
            width=480,
            height=270,
            fps=30,
            postprocess={"contrast": 1.2, "saturation": 1.1},
        )

        assert img is not None
        mock_renderer._apply_postprocess.assert_called_once()
        mock_renderer.post_fbo.read.assert_called_once()

    @patch('src.gpu_preview.QuoteOverlayRenderer')
    @patch('src.gpu_preview.AudioAnalyzer')
    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_render_gpu_preview_with_quotes(self, mock_renderer_cls, mock_analyzer_cls, mock_quote_renderer_cls, dummy_features):
        """Preview mit Quotes sollte QuoteOverlayRenderer aufrufen."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = dummy_features
        mock_analyzer_cls.return_value = mock_analyzer

        mock_renderer = _make_mock_renderer()
        mock_renderer.fbo.read.return_value = b'\x00' * (480 * 270 * 3)
        mock_renderer_cls.return_value = mock_renderer

        mock_quote_renderer = MagicMock()
        mock_quote_renderer.apply.return_value = np.zeros((270, 480, 3), dtype=np.uint8)
        mock_quote_renderer_cls.return_value = mock_quote_renderer

        from src.types import Quote
        from src.quote_overlay import QuoteOverlayConfig
        quote_cfg = QuoteOverlayConfig(enabled=True)
        quotes = [Quote(text="Hello", start_time=1.0, end_time=3.0)]

        img = gpu_preview.render_gpu_preview(
            audio_path="dummy.mp3",
            visualizer_type="lumina_core",
            width=480,
            height=270,
            fps=30,
            quotes=quotes,
            quote_config=quote_cfg,
        )

        assert img is not None
        mock_quote_renderer_cls.assert_called_once()
        mock_quote_renderer.apply.assert_called_once()

    @patch('src.gpu_preview.get_visualizer')
    @patch('src.gpu_preview.AudioAnalyzer')
    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_render_gpu_preview_with_params(self, mock_renderer_cls, mock_analyzer_cls, mock_get_visualizer, dummy_features):
        """Preview mit params sollte viz.set_params aufrufen."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = dummy_features
        mock_analyzer_cls.return_value = mock_analyzer

        mock_viz = _make_mock_viz()
        mock_get_visualizer.return_value = lambda ctx, w, h: mock_viz

        mock_renderer = _make_mock_renderer()
        mock_renderer_cls.return_value = mock_renderer

        img = gpu_preview.render_gpu_preview(
            audio_path="dummy.mp3",
            visualizer_type="lumina_core",
            width=480,
            height=270,
            fps=30,
            params={"intensity": 1.5},
        )

        assert img is not None
        mock_viz.set_params.assert_called_once_with({"intensity": 1.5})

    @patch('src.gpu_preview.AudioAnalyzer')
    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_render_gpu_preview_failure_returns_none(self, mock_renderer_cls, mock_analyzer_cls):
        """Bei Exception sollte None zurueckgegeben werden."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = RuntimeError("Audio error")
        mock_analyzer_cls.return_value = mock_analyzer

        img = gpu_preview.render_gpu_preview(
            audio_path="dummy.mp3",
            visualizer_type="lumina_core",
        )

        assert img is None
