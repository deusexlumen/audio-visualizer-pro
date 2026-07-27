"""Rausch-Aufhebung im Differenz-Render (C15, Spec §3.2.1).

Der positive Fall muss PASSEN; die Negativkontrolle zeigt, dass der Test
überhaupt etwas prüft (unterschiedlicher Seed => Rauschboden sichtbar).
"""

import numpy as np
import pytest

from src.studio.probe import ProbeRenderer
from src.studio.types import MeasureConstraints

pytestmark = pytest.mark.gpu

GRAINY_PP = {"film_grain": 0.5, "bloom_intensity": 0.0}


@pytest.fixture
def probe():
    p = ProbeRenderer(width=160, height=90, fps=30)
    yield p
    p.release()


def _black_viz(probe):
    """Visualizer, der nichts zeichnet (Alpha 0, Farbe 0)."""
    from src.gpu_visualizers.pulsing_core import PulsingCoreGPU
    viz = PulsingCoreGPU(probe.ctx, 160, 90)
    viz.set_params({"bg_brightness": 0.0})
    return viz


def test_noise_cancels_with_identical_seed(probe, dummy_audio_features):
    from src.render_common import build_features_dict
    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    viz = _black_viz(probe)
    # Beide Renders mit cap=0 => kein Visualizer-Beitrag; Grain aktiv.
    # Identisches u_time => contrib muss trotz Grain ~0 sein.
    c = MeasureConstraints(alpha_cap=0.0)
    a = probe.render_frame(viz, features_dict, 0.5, None, GRAINY_PP, c)
    b = probe.render_frame(viz, features_dict, 0.5, None, GRAINY_PP, c)
    contrib = probe.contribution_map(a, b)
    assert float(contrib.max()) < 1e-4


def test_noise_visible_with_different_seed(probe, dummy_audio_features):
    from src.render_common import build_features_dict
    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    viz = _black_viz(probe)
    c = MeasureConstraints(alpha_cap=0.0)
    a = probe.render_frame(viz, features_dict, 0.5, None, GRAINY_PP, c)
    b = probe.render_frame(viz, features_dict, 0.516, None, GRAINY_PP, c)
    contrib = probe.contribution_map(a, b)
    # Negativkontrolle: anderer Seed => Rauschboden muss sichtbar sein.
    # Abweichung zum Plan (Schwelle 0.005): gemessen ~7.2e-4. Grund: Das
    # Messraster konvertiert sRGB -> Linear-Light; auf schwarzem Grund wird
    # der Grain-Delta dabei um Faktor ~12.9 komprimiert (sRGB-Kennlinie).
    # 5e-4 liegt knapp unter dem deterministisch reproduzierbaren Messwert
    # und trennt trotzdem klar vom grain-freien Boden (max 1 LSB, Test 3).
    assert float(contrib.mean()) > 5e-4


def test_grain_free_forces_zero_grain(probe, dummy_audio_features):
    from src.render_common import build_features_dict
    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    viz = _black_viz(probe)
    c = MeasureConstraints(alpha_cap=0.0, grain_free=True)
    a = probe.render_frame(viz, features_dict, 0.5, None, GRAINY_PP, c)
    b = probe.render_frame(viz, features_dict, 0.516, None, GRAINY_PP, c)
    contrib = probe.contribution_map(a, b)
    # grain_free unterdrückt Grain (C15 Regel 3). Verbleibender Boden bei
    # unterschiedlichem u_time: exakt 1 sRGB-LSB (0.0003035 linear) durch das
    # zeit-geseedete Triangular-Dither (gpu_renderer.py:1142), das unabhängig
    # von film_grain läuft. Praktisch irrelevant: Faktor ~66 unter der
    # M5-Schwelle (0.02) — und im produktiven Pfad (render_pair, identisches
    # u_time) hebt sich das Dither ohnehin auf (Test 1). Plan-Schwelle 1e-4
    # ist daher auf 1e-3 angehoben (gleiche Größenordnung wie
    # test_studio_diff_render.py).
    assert float(contrib.max()) < 1e-3
