"""
Erweiterte Tests fuer GPU-Renderer interne Methoden.

Deckt Debug-Speicherung, Audio-Muxing und Hintergrund-Textur-Loading ab.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.gpu_renderer import GPUBatchRenderer


class TestSaveDebug:
    """Tests fuer _save_debug."""

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_save_debug_success(self, mock_create_ctx):
        """_save_debug sollte eine PNG-Datei erstellen."""
        ctx = MagicMock()
        mock_create_ctx.return_value = ctx

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)

        mock_fbo = MagicMock()
        mock_fbo.read.return_value = b'\x00' * (64 * 64 * 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "debug.png"
            renderer._save_debug(mock_fbo, str(path))
            assert path.exists()

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_save_debug_failure_graceful(self, mock_create_ctx):
        """Fehler beim Speichern sollten abgefangen werden."""
        ctx = MagicMock()
        mock_create_ctx.return_value = ctx

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)

        mock_fbo = MagicMock()
        mock_fbo.read.side_effect = RuntimeError("FBO error")

        # Sollte nicht crashen
        renderer._save_debug(mock_fbo, "/invalid/path/debug.png")


class TestMuxAudio:
    """Tests fuer _mux_audio."""

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    @patch('src.gpu_renderer.subprocess.run')
    def test_mux_audio_success(self, mock_run, mock_create_ctx):
        """Erfolgreiches Muxing sollte ohne Fehler durchlaufen."""
        ctx = MagicMock()
        mock_create_ctx.return_value = ctx
        mock_run.return_value = MagicMock(returncode=0, stderr='')

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video:
            video.close()
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as audio:
                audio.close()
                try:
                    renderer._mux_audio(video.name, audio.name, "output.mp4")
                finally:
                    Path(video.name).unlink(missing_ok=True)
                    Path(audio.name).unlink(missing_ok=True)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == 'ffmpeg'
        assert '-c:v' in cmd
        assert 'copy' in cmd
        assert '-c:a' in cmd
        assert 'aac' in cmd

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    @patch('src.gpu_renderer.subprocess.run')
    def test_mux_audio_failure_raises(self, mock_run, mock_create_ctx):
        """FFmpeg-Fehler beim Muxing sollte RuntimeError werfen."""
        ctx = MagicMock()
        mock_create_ctx.return_value = ctx
        mock_run.return_value = MagicMock(returncode=1, stderr='mux error')

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)

        with pytest.raises(RuntimeError, match='Audio-Muxing'):
            renderer._mux_audio("video.mp4", Path("audio.mp3"), "output.mp4")


class TestLoadBackgroundTexture:
    """Tests fuer _load_background_texture."""

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_load_background_texture(self, mock_create_ctx):
        """Sollte eine ModernGL Textur aus einem Bild erstellen."""
        ctx = MagicMock()
        mock_tex = MagicMock()
        ctx.texture.return_value = mock_tex
        mock_create_ctx.return_value = ctx

        renderer = GPUBatchRenderer(width=100, height=100, fps=30)

        # Temporaeres RGB-Bild erstellen
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (200, 200), color=(255, 0, 0))
            img.save(f.name)
            img_path = f.name

        try:
            tex = renderer._load_background_texture(img_path, blur=0.0)
            assert tex is mock_tex
            assert ctx.texture.call_count >= 1
        finally:
            Path(img_path).unlink()

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_load_background_texture_with_blur(self, mock_create_ctx):
        """Sollte Blur anwenden wenn blur > 0.01."""
        ctx = MagicMock()
        mock_tex = MagicMock()
        ctx.texture.return_value = mock_tex
        mock_create_ctx.return_value = ctx

        renderer = GPUBatchRenderer(width=100, height=100, fps=30)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (200, 200), color=(0, 255, 0))
            img.save(f.name)
            img_path = f.name

        try:
            tex = renderer._load_background_texture(img_path, blur=5.0)
            assert tex is mock_tex
            assert ctx.texture.call_count >= 1
        finally:
            Path(img_path).unlink()


class TestRendererRelease:
    """Tests fuer release() Methode."""

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_release_frees_resources(self, mock_create_ctx):
        """release() sollte alle FBOs, Texturen und den Context freigeben."""
        ctx = MagicMock()
        mock_create_ctx.return_value = ctx

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)
        renderer.release()

        ctx.release.assert_called_once()
