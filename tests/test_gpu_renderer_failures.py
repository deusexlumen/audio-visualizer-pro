"""
Tests fuer Fehlerpfade des GPU-Batch-Renderers.

Ergaenzt test_gpu_renderer.py um die Faelle: FFmpeg stirbt mitten im Render
und der Encode-Thread meldet einen Schreibfehler. Alles ohne echte Hardware.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.gpu_renderer import GPUBatchRenderer
from src.types import AudioFeatures


@pytest.fixture
def mock_gl_context():
    """Vollstaendig gemockter ModernGL Context (wie in test_gpu_renderer.py)."""
    ctx = MagicMock()
    mock_texture = MagicMock()
    mock_texture.read.return_value = b'\x00' * (64 * 64 * 3)
    ctx.texture.return_value = mock_texture
    mock_fbo = MagicMock()
    mock_fbo.color_attachments = [mock_texture]
    mock_fbo.read.return_value = b'\x00' * (64 * 64 * 3)
    ctx.framebuffer.return_value = mock_fbo
    ctx.program.return_value = MagicMock()
    ctx.buffer.return_value = MagicMock()
    ctx.vertex_array.return_value = MagicMock()
    ctx.scope.return_value = MagicMock()
    return ctx


@pytest.fixture
def mock_features():
    frame_count = 30
    return AudioFeatures(
        duration=1.0,
        sample_rate=44100,
        fps=30,
        frame_count=frame_count,
        rms=np.random.rand(frame_count).astype(np.float32),
        onset=np.random.rand(frame_count).astype(np.float32),
        spectral_centroid=np.random.rand(frame_count).astype(np.float32),
        spectral_rolloff=np.random.rand(frame_count).astype(np.float32),
        zero_crossing_rate=np.random.rand(frame_count).astype(np.float32),
        transient=np.random.rand(frame_count).astype(np.float32),
        voice_clarity=np.random.rand(frame_count).astype(np.float32),
        voice_band=np.random.rand(frame_count).astype(np.float32),
        chroma=np.random.rand(12, frame_count).astype(np.float32),
        mfcc=np.random.rand(13, frame_count).astype(np.float32),
        tempogram=np.random.rand(384, frame_count).astype(np.float32),
        tempo=120.0,
        key="C",
        mode="music",
        beat_frames=np.array([0, 15, 29]),
    )


def _render(renderer, mock_features, **kwargs):
    """Fuehrt einen kurzen Preview-Render mit Temp-Dateien aus."""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as audio_file:
        audio_file.close()
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as output_file:
            output_file.close()
            try:
                renderer.render(
                    audio_path=audio_file.name,
                    visualizer_type='lumina_core',
                    output_path=output_file.name,
                    features=mock_features,
                    preview_mode=True,
                    preview_duration=0.3,
                    **kwargs,
                )
            finally:
                Path(audio_file.name).unlink(missing_ok=True)
                Path(output_file.name).unlink(missing_ok=True)


class TestFFmpegStirbtMidRender:
    """FFmpeg beendet sich unerwartet, nachdem der Render schon laeuft."""

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    @patch('src.gpu_renderer.subprocess.Popen')
    @patch('src.gpu_renderer.subprocess.run')
    def test_mid_render_tod_wirft_runtimeerror_mit_encoder(
        self, mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features
    ):
        mock_create_ctx.return_value = mock_gl_context
        mock_run.return_value = MagicMock(returncode=0, stderr='')

        mock_process = MagicMock()
        # Erste Frames laufen, dann stirbt FFmpeg (poll liefert Exit-Code)
        mock_process.poll.side_effect = [None, None, None, 1, 1, 1, 1, 1, 1, 1]
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)

        with pytest.raises(RuntimeError) as exc_info:
            _render(renderer, mock_features)

        msg = str(exc_info.value)
        assert "FFmpeg" in msg
        # Fehlermeldung nennt den Encoder, damit der Nutzer ihn pruefen kann
        assert "Encoder" in msg or "libx264" in msg


class TestEncodeThreadFehler:
    """Der Encode-Thread meldet einen Schreibfehler (z.B. Broken Pipe)."""

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    @patch('src.gpu_renderer.subprocess.Popen')
    @patch('src.gpu_renderer.subprocess.run')
    def test_stdin_schreibfehler_wird_propagiert(
        self, mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features
    ):
        mock_create_ctx.return_value = mock_gl_context
        mock_run.return_value = MagicMock(returncode=0, stderr='')

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Prozess "laeuft" weiter
        mock_process.returncode = 0
        mock_process.stdin.write.side_effect = BrokenPipeError("Pipe zu")
        mock_popen.return_value = mock_process

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)

        with pytest.raises(RuntimeError, match="Encode-Thread-Fehler"):
            _render(renderer, mock_features)
