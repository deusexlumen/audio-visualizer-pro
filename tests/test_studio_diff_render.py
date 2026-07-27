"""Tests für den Differenz-Render (Spec §3.2, §3.4)."""

import numpy as np
import pytest

from src.studio.probe import ProbeRenderer, probe_resolution
from src.studio.types import MeasureConstraints

pytestmark = pytest.mark.gpu


def test_probe_resolution_policy():
    assert probe_resolution(3840, 2160) == (960, 540)   # Ziel/4 > 480p
    assert probe_resolution(1920, 1080) == (854, 480)   # 480p-Minimum greift
    assert probe_resolution(1280, 720) == (854, 480)
    # Aspect-identisch bei Nicht-16:9
    w, h = probe_resolution(2000, 1000)
    assert w / h == pytest.approx(2.0, abs=0.01)


@pytest.fixture
def probe():
    p = ProbeRenderer(width=160, height=90, fps=30)
    yield p
    p.release()


def test_contribution_zero_when_capped(probe, dummy_audio_features):
    from src.render_common import build_features_dict
    from src.gpu_visualizers import get_visualizer

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    viz_cls = get_visualizer("spectrum_bars")
    viz = viz_cls(probe.ctx, 160, 90)
    constraints = MeasureConstraints(alpha_cap=0.0, alpha_from_luma=True)
    a, b = probe.render_pair(viz, features_dict, 0.5, None, {}, constraints)
    contrib = probe.contribution_map(a, b)
    assert float(contrib.max()) < 1e-3


def test_contribution_scales_with_cap(probe, dummy_audio_features):
    from src.render_common import build_features_dict
    from src.gpu_visualizers import get_visualizer

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    viz_cls = get_visualizer("spectrum_bars")
    bg = np.full((90, 160, 3), 40, dtype=np.uint8)
    bg_tex = probe.ctx.texture((160, 90), 3, bg.tobytes())

    def energy(cap):
        viz = viz_cls(probe.ctx, 160, 90)
        constraints = MeasureConstraints(alpha_cap=cap, alpha_from_luma=True)
        a = probe.render_frame(viz, features_dict, 0.5, bg_tex, {}, constraints)
        b = probe.render_frame(
            viz, features_dict, 0.5, bg_tex, {},
            MeasureConstraints(alpha_cap=0.0, alpha_from_luma=True),
        )
        return float(probe.contribution_map(a, b).mean())

    assert energy(1.0) > energy(0.3) > 0.0
