"""Renderer und Vorschau im Folgen-Modus (Bild + Lichtstrich + Text)."""

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import src.gpu_renderer as gpu_renderer
from src.episode.compositor import cover_crop
from src.episode.config import EpisodeLookConfig, EpisodeRenderSpec
from src.ffmpeg_locator import get_ffmpeg_path, get_ffprobe_path
from src.gpu_preview import render_gpu_preview
from src.gpu_renderer import GPUBatchRenderer
from src.types import Chapter, Quote
from tests.test_gpu_renderer import mock_features, mock_gl_context  # noqa: F401
from tests.test_gpu_renderer_timeline import _dummy_features, _write_wav

W, H = 320, 180


def _image(path):
    # Waagrechter Verlauf, senkrecht konstant: robust gegen kleinen Kamera-Zoom
    x = np.linspace(0, 1, 1536)[None, :, None]
    arr = np.broadcast_to(np.concatenate([40 + 160 * x, 90 + 60 * x, 180 - 120 * x], -1),
                          (1024, 1536, 3)).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _spec(tmp_path, cfg=None):
    return EpisodeRenderSpec(
        image_path=str(_image(tmp_path / "bild.png")),
        chapters=[Chapter(start=0.0, title="Einstieg")],
        quotes=[Quote(text="Ein kurzer Satz hier", start_time=0.5, end_time=1.5)],
        title="Testfolge",
        config=cfg or EpisodeLookConfig(),
    )


def _render(tmp_path, spec):
    wav = tmp_path / "a.wav"
    _write_wav(wav, seconds=2)
    out = tmp_path / "folge.mp4"
    r = GPUBatchRenderer(width=W, height=H, fps=30)
    try:
        r.render(audio_path=str(wav), visualizer_type="voice_line", output_path=str(out),
                 features=_dummy_features(seconds=2),
                 postprocess={"chromatic_aberration": 0.0, "film_grain": 0.0, "vignette": 0.0},
                 episode=spec)
    finally:
        r.release()
    return out


def _frame_count(path):
    res = subprocess.run([get_ffprobe_path(), "-v", "error", "-select_streams", "v:0",
                          "-count_frames", "-show_entries", "stream=nb_read_frames",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return int(res.stdout.strip())


def _frame_at(path, t, tmp_path):
    png = tmp_path / "frame.png"
    subprocess.run([get_ffmpeg_path(), "-y", "-v", "error", "-ss", str(t), "-i", str(path),
                    "-frames:v", "1", str(png)], check=True)
    return np.asarray(Image.open(png).convert("RGB")).astype(float)


@pytest.mark.gpu
def test_episode_render_writes_video(tmp_path, require_gpu):
    out = _render(tmp_path, _spec(tmp_path))
    assert _frame_count(out) == 60


@pytest.mark.gpu
def test_episode_frame_keeps_image_colours(tmp_path, require_gpu):
    spec = _spec(tmp_path, EpisodeLookConfig(card_duration=0.5))
    out = _render(tmp_path, spec)
    frame = _frame_at(out, 1.8, tmp_path)
    ref = np.asarray(cover_crop(Image.open(spec.image_path).convert("RGB"), W, H)).astype(float)
    # Obere Haelfte ohne Markenzeile vergleichen (Karte und Zitat sind vorbei)
    for c in range(3):
        assert abs(frame[30:90, :, c].mean() - ref[30:90, :, c].mean()) <= 6


@pytest.mark.gpu
def test_episode_preview_returns_image(tmp_path, require_gpu):
    wav = tmp_path / "a.wav"
    _write_wav(wav, seconds=2)
    img = render_gpu_preview(str(wav), "voice_line", width=480, height=270,
                             features=_dummy_features(seconds=2), preview_time_percent=0.5,
                             episode=_spec(tmp_path))
    assert img is not None and img.size == (480, 270)


@patch("src.gpu_renderer.moderngl.create_standalone_context")
@patch("src.gpu_renderer.subprocess.Popen")
@patch("src.gpu_renderer.subprocess.run")
def test_classic_render_ignores_episode_code(mock_run, mock_popen, mock_create_ctx,
                                             mock_gl_context, mock_features, tmp_path,
                                             monkeypatch):
    mock_create_ctx.return_value = mock_gl_context
    process = MagicMock()
    process.poll.return_value = None
    process.returncode = 0
    mock_popen.return_value = process
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    built = []
    monkeypatch.setattr(gpu_renderer, "EpisodeCompositor",
                        lambda *a, **k: built.append(1) or MagicMock())
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"")
    r = GPUBatchRenderer(width=64, height=64, fps=30)
    r.render(audio_path=str(audio), visualizer_type="lumina_core",
             output_path=str(tmp_path / "o.mp4"), features=mock_features,
             preview_mode=True, preview_duration=0.1)
    assert built == []
