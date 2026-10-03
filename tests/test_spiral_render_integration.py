"""Spiral-Zoom im Render-Pfad: Export-Schleife und Live-Vorschau."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.gpu_preview as gpu_preview
from src.gpu_renderer import GPUBatchRenderer
from src.gpu_visualizers import VISUALIZER_MAP
from src.gpu_visualizers.base import BaseGPUVisualizer
from src.render_common import compute_beat_intensity
from src.spiral_zoom import SpiralSettings, compute_spiral_phase
from tests.test_gpu_preview import _make_mock_renderer, dummy_features  # noqa: F401
from tests.test_gpu_renderer import mock_features, mock_gl_context  # noqa: F401

SPIRAL_ON = {"spiral_enabled": True, "spiral_arms": 1, "spiral_speed": 0.5, "spiral_beat": 0.3}


def _render_mocked(mock_run, mock_popen, mock_create_ctx, ctx, features, postprocess, tmp_path):
    mock_create_ctx.return_value = ctx
    process = MagicMock()
    process.poll.return_value = None
    process.returncode = 0
    mock_popen.return_value = process
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"")
    renderer = GPUBatchRenderer(width=64, height=64, fps=30)
    with patch.object(GPUBatchRenderer, "_apply_spiral") as spy:
        renderer.render(
            audio_path=str(audio),
            visualizer_type="lumina_core",
            output_path=str(tmp_path / "o.mp4"),
            features=features,
            preview_mode=True,
            preview_duration=0.1,  # 3 Frames
            postprocess=postprocess,
        )
    return spy, renderer


@patch("src.gpu_renderer.moderngl.create_standalone_context")
@patch("src.gpu_renderer.subprocess.Popen")
@patch("src.gpu_renderer.subprocess.run")
def test_batch_render_applies_spiral_per_frame(
    mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features, tmp_path
):
    spy, renderer = _render_mocked(
        mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features, SPIRAL_ON, tmp_path
    )
    s = SpiralSettings.from_postprocess(SPIRAL_ON)
    expected = compute_spiral_phase(
        mock_features.rms[:3], compute_beat_intensity(mock_features.beat_frames, 3, 30), 30, s
    )
    assert spy.call_count == 3
    for call, u in zip(spy.call_args_list, expected):
        settings, got_u, target = call.args
        assert settings == s
        assert got_u == pytest.approx(u)
        assert target is renderer.viz_fbo  # Visualizer-Ebene, nicht die Szene


@patch("src.gpu_renderer.moderngl.create_standalone_context")
@patch("src.gpu_renderer.subprocess.Popen")
@patch("src.gpu_renderer.subprocess.run")
def test_batch_render_without_spiral_never_calls_pass(
    mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features, tmp_path
):
    spy, _ = _render_mocked(
        mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features,
        {"bloom_intensity": 0.6}, tmp_path,
    )
    spy.assert_not_called()


@patch("src.gpu_preview.AudioAnalyzer")
@patch("src.gpu_preview.GPUPreviewRenderer")
def test_preview_uses_same_phase_as_export_frame(mock_renderer_cls, mock_analyzer_cls, dummy_features):
    mock_renderer = _make_mock_renderer()
    mock_renderer_cls.return_value = mock_renderer
    gpu_preview.render_gpu_preview(
        audio_path="dummy.mp3", visualizer_type="lumina_core", width=480, height=270,
        fps=30, features=dummy_features, preview_time_percent=0.5, postprocess=SPIRAL_ON,
    )
    s = SpiralSettings.from_postprocess(SPIRAL_ON)
    full = compute_spiral_phase(
        dummy_features.rms, compute_beat_intensity(dummy_features.beat_frames, 300, 30), 30, s
    )
    mock_renderer._apply_spiral.assert_called_once()
    settings, u, target = mock_renderer._apply_spiral.call_args.args
    assert settings == s
    assert u == pytest.approx(full[150])  # 10 s * 0.5 = 5 s = Frame 150
    assert target is mock_renderer.viz_fbo


@patch("src.gpu_preview.AudioAnalyzer")
@patch("src.gpu_preview.GPUPreviewRenderer")
def test_preview_without_spiral_never_calls_pass(mock_renderer_cls, mock_analyzer_cls, dummy_features):
    mock_renderer = _make_mock_renderer()
    mock_renderer_cls.return_value = mock_renderer
    gpu_preview.render_gpu_preview(
        audio_path="dummy.mp3", visualizer_type="lumina_core", width=480, height=270,
        fps=30, features=dummy_features, postprocess={"contrast": 1.1},
    )
    mock_renderer._apply_spiral.assert_not_called()


def _gradient_photo(tmp_path) -> str:
    from PIL import Image

    grad = np.linspace(0, 255, 160).astype(np.uint8)
    img = np.stack([np.tile(grad, (90, 1))] * 3, axis=-1)
    path = str(tmp_path / "bg.png")
    Image.fromarray(img).save(path)
    return path


def _preview(**kw) -> np.ndarray:
    img = gpu_preview.render_gpu_preview(
        audio_path="dummy.mp3", width=160, height=90, fps=30,
        preview_time_percent=0.5, background_opacity=1.0, **kw,
    )
    return np.asarray(img, dtype=np.int16)


@pytest.mark.gpu
@pytest.mark.parametrize("with_bg", [False, True])
def test_real_preview_off_is_bitidentical_on_differs(tmp_path, dummy_features, with_bg):
    """Echte GPU: aus = bitgleich zu ohne Schluessel; an = sichtbar anders (auch mit Foto)."""
    common = dict(
        visualizer_type="spectrum_bars", features=dummy_features,
        background_image=_gradient_photo(tmp_path) if with_bg else None,
    )
    base = _preview(**common)
    off = _preview(**common, postprocess={"spiral_enabled": False, "spiral_arms": 2})
    on = _preview(
        **common, postprocess={"spiral_enabled": True, "spiral_arms": 1, "spiral_rotation": 30}
    )
    assert np.array_equal(base, off)
    assert np.abs(on - base).mean() > 0.5


class _LeererViz(BaseGPUVisualizer):
    """Zeichnet nichts: die Visualizer-Ebene bleibt transparent."""

    def _setup(self):
        pass

    def render(self, features, time):
        pass


@pytest.mark.gpu
def test_real_preview_photo_stays_still(tmp_path, dummy_features, monkeypatch):
    """Nur der Visualizer wird verbogen: ohne Visualizer-Inhalt bleibt das Foto bitgleich."""
    monkeypatch.setitem(VISUALIZER_MAP, "_leer_test", _LeererViz)
    common = dict(
        visualizer_type="_leer_test", features=dummy_features,
        background_image=_gradient_photo(tmp_path),
    )
    off = _preview(**common)
    on = _preview(
        **common, postprocess={"spiral_enabled": True, "spiral_arms": 1, "spiral_rotation": 30}
    )
    assert np.array_equal(off, on)


def test_apply_spiral_creates_pass_lazily_without_init():
    """Auch ein Renderer ohne __init__ (z.B. in Tests) legt den Pass an."""
    renderer = GPUBatchRenderer.__new__(GPUBatchRenderer)
    renderer.ctx, renderer.width, renderer.height = MagicMock(), 64, 64
    target = MagicMock()
    renderer._apply_spiral(SpiralSettings(enabled=True), 0.25, target)
    assert renderer._spiral  # Pass angelegt
    target.use.assert_called()  # und auf das Ziel gezeichnet


def test_timeline_crossfade_spirals_the_blend_layer():
    """Bei Timeline-Ueberblendung liegt der Visualizer in viz_fbo_blend."""
    renderer = GPUBatchRenderer.__new__(GPUBatchRenderer)  # ohne GL-Kontext
    renderer.viz_fbo = MagicMock()
    renderer.viz_fbo_blend = MagicMock()
    blend_tex = renderer.viz_fbo_blend.color_attachments[0]
    assert renderer._viz_fbo_holding(blend_tex) is renderer.viz_fbo_blend
    assert renderer._viz_fbo_holding(renderer.viz_fbo.color_attachments[0]) is renderer.viz_fbo
