"""Tests fuer den Visualizer voice_line (Stimm-Lichtstrich im Folgen-Look)."""

import moderngl
import numpy as np
import pytest

from src.gpu_visualizers import AUTO_EXCLUDED_VISUALIZERS, get_visualizer, list_visualizers

W, H = 256, 144


@pytest.fixture
def gl():
    ctx = moderngl.create_standalone_context()
    yield ctx
    ctx.release()


def _features(voice):
    n = 30
    return {"rms": np.full(n, voice, np.float32), "onset": np.zeros(n, np.float32),
            "spectral_centroid": np.zeros(n, np.float32),
            "chroma": np.zeros((12, n), np.float32),
            "voice_band": np.full(n, voice, np.float32),
            "fps": 30, "frame_count": n, "mode": "speech"}


def _render(gl, voice):
    tex = gl.texture((W, H), 3, dtype="f2")
    fbo = gl.framebuffer(color_attachments=[tex])
    viz = get_visualizer("voice_line")(gl, W, H)
    fbo.use()
    gl.clear(0.0, 0.0, 0.0)
    viz.render(_features(voice), 0.5)
    # Zeile 0 der Auslese = Bildoberkante im Video (wie beim Hintergrund)
    return np.frombuffer(fbo.read(components=3, dtype="f2"), np.float16).astype(
        np.float32).reshape(H, W, 3)


def test_voice_line_registered():
    assert "voice_line" in list_visualizers()


@pytest.mark.gpu
def test_voice_line_brighter_with_voice(gl):
    assert _render(gl, 0.9).mean() > 1.5 * _render(gl, 0.0).mean()


@pytest.mark.gpu
def test_voice_line_only_near_bottom(gl):
    img = _render(gl, 0.9)
    assert img[: int(0.7 * H)].max() < 1e-3
    assert img[int(0.85 * H):].max() > 0.1


def test_voice_line_not_offered_to_auto(dummy_audio_features):
    from src.ai_matcher import SmartMatcher
    m = SmartMatcher()
    for mode in ("speech", "music"):
        feats = dummy_audio_features.model_copy(update={"mode": mode})
        scores = m._compute_suitability_scores(m._extract_features(feats))
        assert "voice_line" not in scores
        assert m.match(feats).visualizer != "voice_line"
    assert "voice_line" in AUTO_EXCLUDED_VISUALIZERS


def test_scene_timeline_candidates_exclude_voice_line(qtbot, dummy_audio_features):
    from src.gui.workers import FullAIWorker
    captured = {}

    class FakeGemini:
        def generate_scene_timeline(self, segments_stats, available_visualizers, mode):
            captured["list"] = list(available_visualizers)
            return []

    worker = FullAIWorker(dummy_audio_features, gemini=FakeGemini(), use_gemini=True)
    worker.run()  # synchron, ohne Thread
    assert captured["list"] and "voice_line" not in captured["list"]
