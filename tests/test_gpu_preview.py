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


class TestPreviewCache:
    """Tests fuer den Preview-Cache."""

    def teardown_method(self):
        """Cache nach jedem Test leeren."""
        gpu_preview._release_preview_cache()

    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_get_cached_renderer_creates_new(self, mock_renderer_cls):
        """Bei leerem Cache sollte ein neuer Renderer erstellt werden."""
        mock_renderer = MagicMock()
        mock_renderer.ctx = MagicMock()
        mock_renderer_cls.return_value = mock_renderer

        renderer, viz = gpu_preview._get_cached_renderer("lumina_core", 480, 270, 30)

        assert renderer is mock_renderer
        mock_renderer_cls.assert_called_once_with(width=480, height=270, fps=30)

    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_get_cached_renderer_reuses(self, mock_renderer_cls):
        """Bei gleichem Cache-Key sollte der Renderer wiederverwendet werden."""
        mock_renderer = MagicMock()
        mock_renderer.ctx = MagicMock()
        mock_renderer_cls.return_value = mock_renderer

        renderer1, viz1 = gpu_preview._get_cached_renderer("lumina_core", 480, 270, 30)
        renderer2, viz2 = gpu_preview._get_cached_renderer("lumina_core", 480, 270, 30)

        assert renderer1 is renderer2
        # Renderer sollte nur EINMAL erstellt worden sein
        mock_renderer_cls.assert_called_once()

    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_get_cached_renderer_invalidates(self, mock_renderer_cls):
        """Bei geaendertem Key sollte der alte Cache freigegeben werden."""
        # Side-Effect um verschiedene Mock-Instanzen zu erhalten
        mock_renderer_cls.side_effect = lambda **kwargs: MagicMock(ctx=MagicMock())

        # Sicherstellen dass Cache initialisiert ist
        gpu_preview._release_preview_cache()

        renderer1, viz1 = gpu_preview._get_cached_renderer("lumina_core", 480, 270, 30)
        renderer2, viz2 = gpu_preview._get_cached_renderer("spectrum_bars", 480, 270, 30)

        # Renderer sollte ZWEIMAL erstellt worden sein (Cache-Invalidierung)
        assert mock_renderer_cls.call_count == 2
        # Und es sollten unterschiedliche Instanzen sein
        assert renderer1 is not renderer2

    def test_release_preview_cache(self):
        """_release_preview_cache sollte den Cache leeren."""
        gpu_preview._PREVIEW_CACHE["key"] = ("test", 1, 2, 3)
        gpu_preview._PREVIEW_CACHE["renderer"] = MagicMock()
        gpu_preview._PREVIEW_CACHE["viz"] = MagicMock()

        gpu_preview._release_preview_cache()

        assert gpu_preview._PREVIEW_CACHE["key"] is None
        assert gpu_preview._PREVIEW_CACHE["renderer"] is None
        assert gpu_preview._PREVIEW_CACHE["viz"] is None


class TestRenderGpuPreview:
    """Tests fuer render_gpu_preview."""

    def teardown_method(self):
        """Cache nach jedem Test leeren."""
        gpu_preview._release_preview_cache()

    @patch('src.gpu_preview.AudioAnalyzer')
    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_render_gpu_preview_success(self, mock_renderer_cls, mock_analyzer_cls, dummy_features):
        """Erfolgreiches Preview-Rendering sollte ein PIL Image zurueckgeben."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = dummy_features
        mock_analyzer_cls.return_value = mock_analyzer

        mock_renderer = MagicMock()
        mock_renderer.ctx = MagicMock()
        mock_renderer.fbo = MagicMock()
        mock_renderer.viz_fbo = MagicMock()
        mock_renderer.post_fbo = MagicMock()
        mock_renderer.fbo.read.return_value = b'\x00' * (480 * 270 * 3)
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

    @patch('src.gpu_preview.AudioAnalyzer')
    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_render_gpu_preview_with_postprocess(self, mock_renderer_cls, mock_analyzer_cls, dummy_features):
        """Preview mit Post-Process sollte post_fbo verwenden."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = dummy_features
        mock_analyzer_cls.return_value = mock_analyzer

        mock_renderer = MagicMock()
        mock_renderer.ctx = MagicMock()
        mock_renderer.fbo = MagicMock()
        mock_renderer.viz_fbo = MagicMock()
        mock_renderer.post_fbo = MagicMock()
        mock_renderer.post_fbo.read.return_value = b'\x00' * (480 * 270 * 3)
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

    @patch('src.gpu_preview.AudioAnalyzer')
    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_render_gpu_preview_with_quotes(self, mock_renderer_cls, mock_analyzer_cls, dummy_features):
        """Preview mit Quotes sollte _render_quotes_gpu aufrufen."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = dummy_features
        mock_analyzer_cls.return_value = mock_analyzer

        mock_renderer = MagicMock()
        mock_renderer.ctx = MagicMock()
        mock_renderer.fbo = MagicMock()
        mock_renderer.viz_fbo = MagicMock()
        mock_renderer.fbo.read.return_value = b'\x00' * (480 * 270 * 3)
        mock_renderer_cls.return_value = mock_renderer

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
        mock_renderer._init_text_renderer.assert_called_once()
        mock_renderer._render_quotes_gpu.assert_called_once()

    @patch('src.gpu_preview.AudioAnalyzer')
    @patch('src.gpu_preview.GPUPreviewRenderer')
    def test_render_gpu_preview_with_params(self, mock_renderer_cls, mock_analyzer_cls, dummy_features):
        """Preview mit params sollte viz.set_params aufrufen."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = dummy_features
        mock_analyzer_cls.return_value = mock_analyzer

        mock_viz = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.ctx = MagicMock()
        mock_renderer.fbo = MagicMock()
        mock_renderer.viz_fbo = MagicMock()
        mock_renderer.fbo.read.return_value = b'\x00' * (480 * 270 * 3)
        mock_renderer_cls.return_value = mock_renderer

        # Cache initialisieren und viz manuell setzen
        gpu_preview._release_preview_cache()
        gpu_preview._PREVIEW_CACHE["key"] = ("lumina_core", 480, 270, 30)
        gpu_preview._PREVIEW_CACHE["renderer"] = mock_renderer
        gpu_preview._PREVIEW_CACHE["viz"] = mock_viz
        gpu_preview._PREVIEW_CACHE["viz_type"] = "lumina_core"

        img = gpu_preview.render_gpu_preview(
            audio_path="dummy.mp3",
            visualizer_type="lumina_core",
            width=480,
            height=270,
            fps=30,
            params={"intensity": 1.5},
        )

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
