"""
Tests für den GPU-Batch-Renderer.

Mocked ModernGL Context fuer Hardware-unabhaengige Tests.
Testet Initialisierung, FFmpeg-Cmd-Builder und den Render-Flow.
"""

import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.gpu_renderer import GPUBatchRenderer
from src.types import AudioFeatures


@pytest.fixture
def mock_gl_context():
    """Erzeugt einen vollstaendig gemockten ModernGL Context."""
    ctx = MagicMock()

    # Mock Texture
    mock_texture = MagicMock()
    mock_texture.read.return_value = b'\x00' * (640 * 480 * 3)
    ctx.texture.return_value = mock_texture

    # Mock Framebuffer
    mock_fbo = MagicMock()
    mock_fbo.color_attachments = [mock_texture]
    mock_fbo.read.return_value = b'\x00' * (640 * 480 * 3)
    ctx.framebuffer.return_value = mock_fbo

    # Mock Program
    mock_program = MagicMock()
    ctx.program.return_value = mock_program

    # Mock Buffer / VAO
    mock_buffer = MagicMock()
    ctx.buffer.return_value = mock_buffer
    mock_vao = MagicMock()
    ctx.vertex_array.return_value = mock_vao

    # Mock Scope
    mock_scope = MagicMock()
    ctx.scope.return_value = mock_scope

    return ctx


@pytest.fixture
def mock_features():
    """Minimal AudioFeatures fuer GPU-Renderer-Tests."""
    return AudioFeatures(
        duration=1.0,
        sample_rate=44100,
        fps=30,
        frame_count=30,
        rms=np.random.rand(30).astype(np.float32),
        onset=np.random.rand(30).astype(np.float32),
        spectral_centroid=np.random.rand(30).astype(np.float32),
        spectral_rolloff=np.random.rand(30).astype(np.float32),
        zero_crossing_rate=np.random.rand(30).astype(np.float32),
        transient=np.random.rand(30).astype(np.float32),
        voice_clarity=np.random.rand(30).astype(np.float32),
        voice_band=np.random.rand(30).astype(np.float32),
        chroma=np.random.rand(12, 30).astype(np.float32),
        mfcc=np.random.rand(13, 30).astype(np.float32),
        tempogram=np.random.rand(384, 30).astype(np.float32),
        tempo=120.0,
        key="C",
        mode="music",
        beat_frames=np.array([0, 15, 30]),
    )


class TestGPUBatchRendererInit:
    """Tests fuer die Initialisierung des GPU-Renderers."""

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_init_creates_context(self, mock_create_ctx, mock_gl_context):
        """__init__ sollte einen ModernGL Context und alle FBOs erstellen."""
        mock_create_ctx.return_value = mock_gl_context

        renderer = GPUBatchRenderer(width=640, height=480, fps=30)

        mock_create_ctx.assert_called_once()
        assert renderer.width == 640
        assert renderer.height == 480
        assert renderer.fps == 30
        assert renderer.ctx is mock_gl_context

        # 4 FBOs + dummy texture sollten erstellt worden sein
        assert mock_gl_context.framebuffer.call_count >= 4
        assert mock_gl_context.texture.call_count >= 4

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_release(self, mock_create_ctx, mock_gl_context):
        """release() sollte alle Ressourcen freigeben."""
        mock_create_ctx.return_value = mock_gl_context

        renderer = GPUBatchRenderer(width=640, height=480, fps=30)
        renderer.release()

        # ctx.release sollte aufgerufen werden
        mock_gl_context.release.assert_called_once()


class TestBuildFFmpegCmd:
    """Tests fuer den FFmpeg-Befehl-Builder."""

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_build_ffmpeg_cmd_h264_high(self, mock_create_ctx, mock_gl_context):
        """h264 + high sollte yuv444p verwenden."""
        mock_create_ctx.return_value = mock_gl_context
        renderer = GPUBatchRenderer(width=1920, height=1080, fps=60)

        cmd = renderer._build_ffmpeg_cmd("out.mp4", "h264", "high")

        assert "ffmpeg" in cmd
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "yuv444p" in cmd
        assert "-crf" in cmd
        assert "20" in cmd

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_build_ffmpeg_cmd_h264_low(self, mock_create_ctx, mock_gl_context):
        """h264 + low sollte yuv420p verwenden."""
        mock_create_ctx.return_value = mock_gl_context
        renderer = GPUBatchRenderer(width=1280, height=720, fps=30)

        cmd = renderer._build_ffmpeg_cmd("out.mp4", "h264", "low")

        assert "yuv420p" in cmd
        assert "ultrafast" in cmd
        assert "28" in cmd

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_build_ffmpeg_cmd_hevc(self, mock_create_ctx, mock_gl_context):
        """hevc sollte libx265 verwenden."""
        mock_create_ctx.return_value = mock_gl_context
        renderer = GPUBatchRenderer(width=1920, height=1080, fps=60)

        cmd = renderer._build_ffmpeg_cmd("out.mp4", "hevc", "high")

        assert "libx265" in cmd
        assert "hvc1" in cmd

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_build_ffmpeg_cmd_prores(self, mock_create_ctx, mock_gl_context):
        """prores sollte prores_ks verwenden."""
        mock_create_ctx.return_value = mock_gl_context
        renderer = GPUBatchRenderer(width=1920, height=1080, fps=60)

        cmd = renderer._build_ffmpeg_cmd("out.mp4", "prores", "high")

        assert "prores_ks" in cmd
        assert "yuv422p10le" in cmd
        assert "-b:v" in cmd
        assert "-crf" not in cmd

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    def test_build_ffmpeg_cmd_lossless(self, mock_create_ctx, mock_gl_context):
        """lossless sollte crf=0 und yuv444p verwenden."""
        mock_create_ctx.return_value = mock_gl_context
        renderer = GPUBatchRenderer(width=1920, height=1080, fps=60)

        cmd = renderer._build_ffmpeg_cmd("out.mp4", "h264", "lossless")

        assert "0" in cmd
        assert "slow" in cmd
        assert "yuv444p" in cmd


class TestRenderFlow:
    """Tests fuer den Haupt-Render-Flow mit gemocktem FFmpeg."""

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    @patch('src.gpu_renderer.subprocess.Popen')
    @patch('src.gpu_renderer.subprocess.run')
    def test_render_basic_flow(self, mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features):
        """Der Render-Flow sollte Frames generieren und an FFmpeg senden."""
        mock_create_ctx.return_value = mock_gl_context

        # Mock FFmpeg Prozess
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Noch laufend waehrend des Loops
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Mock subprocess.run fuer _mux_audio
        mock_run.return_value = MagicMock(returncode=0, stderr='')

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)

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
                        preview_duration=0.1,  # Nur 3 Frames
                    )
                finally:
                    Path(audio_file.name).unlink(missing_ok=True)
                    Path(output_file.name).unlink(missing_ok=True)

        # FFmpeg sollte gestartet worden sein
        mock_popen.assert_called_once()
        # Frames sollten geschrieben worden sein
        assert mock_process.stdin.write.call_count > 0
        # stdin sollte geschlossen werden
        mock_process.stdin.close.assert_called_once()
        assert mock_process.wait.call_count >= 1

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    @patch('src.gpu_renderer.subprocess.Popen')
    @patch('src.gpu_renderer.subprocess.run')
    def test_render_with_cancel(self, mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features):
        """Cancel-Event sollte den Render-Loop sofort beenden."""
        mock_create_ctx.return_value = mock_gl_context

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Noch laufend
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Mock subprocess.run fuer _mux_audio
        mock_run.return_value = MagicMock(returncode=0, stderr='')

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)
        cancel_event = threading.Event()
        cancel_event.set()  # Sofort abbrechen

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
                        cancel_event=cancel_event,
                    )
                finally:
                    Path(audio_file.name).unlink(missing_ok=True)
                    Path(output_file.name).unlink(missing_ok=True)

        # stdin sollte trotz Cancel geschlossen werden (finally-Block)
        mock_process.stdin.close.assert_called_once()

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    @patch('src.gpu_renderer.subprocess.Popen')
    @patch('src.gpu_renderer.subprocess.run')
    def test_render_with_progress_callback(self, mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features):
        """Progress-Callback sollte waehrend des Renderings aufgerufen werden."""
        mock_create_ctx.return_value = mock_gl_context

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Noch laufend
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Mock subprocess.run fuer _mux_audio
        mock_run.return_value = MagicMock(returncode=0, stderr='')

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)
        progress_calls = []

        def callback(frame, total):
            progress_calls.append((frame, total))

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
                        preview_duration=0.2,
                        progress_callback=callback,
                    )
                finally:
                    Path(audio_file.name).unlink(missing_ok=True)
                    Path(output_file.name).unlink(missing_ok=True)

        assert len(progress_calls) > 0

    @patch('src.gpu_renderer.moderngl.create_standalone_context')
    @patch('src.gpu_renderer.subprocess.Popen')
    @patch('src.gpu_renderer.subprocess.run')
    def test_render_ffmpeg_failure(self, mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features):
        """Wenn FFmpeg fehlschlaegt, sollte ein RuntimeError geworfen werden."""
        mock_create_ctx.return_value = mock_gl_context

        mock_process = MagicMock()
        mock_process.poll.return_value = 1
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        # Mock subprocess.run fuer _mux_audio (wird nicht erreicht, aber sicherheitshalber)
        mock_run.return_value = MagicMock(returncode=0, stderr='')

        renderer = GPUBatchRenderer(width=64, height=64, fps=30)

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as audio_file:
            audio_file.close()
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as output_file:
                output_file.close()
                try:
                    with pytest.raises(RuntimeError, match='FFmpeg'):
                        renderer.render(
                            audio_path=audio_file.name,
                            visualizer_type='lumina_core',
                            output_path=output_file.name,
                            features=mock_features,
                            preview_mode=True,
                            preview_duration=0.1,
                        )
                finally:
                    Path(audio_file.name).unlink(missing_ok=True)
                    Path(output_file.name).unlink(missing_ok=True)
