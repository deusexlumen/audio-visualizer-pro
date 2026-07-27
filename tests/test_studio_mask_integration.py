"""End-to-End: Subjekt-Maske + Cap wirken messbar (Spec §16 P1 DoD)."""

import numpy as np
import pytest

from src.studio.constraints import ConstraintSet
from src.studio.probe import ProbeRenderer

pytestmark = pytest.mark.gpu


@pytest.fixture
def probe():
    p = ProbeRenderer(width=160, height=90, fps=30)
    yield p
    p.release()


def _features_dict(dummy_audio_features):
    from src.render_common import build_features_dict
    return build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )


def _composite_viz(probe):
    """Composite-Stack (alpha=1.0 überall) — der C14-Hauptfall."""
    from src.gpu_visualizers.composite import make_recipe_visualizer_class
    recipe = {
        "name": "test_glow",
        "layers": [
            {"block": "core_glow", "blend": "add",
             "params": {"size": 0.5, "intensity": 1.6}, "mappings": []}
        ],
    }
    cls = make_recipe_visualizer_class(recipe)
    return cls(probe.ctx, 160, 90)


def test_maske_stellt_subjektregion_frei(probe, dummy_audio_features):
    fd = _features_dict(dummy_audio_features)
    viz = _composite_viz(probe)
    mask = np.zeros((90, 160), dtype=np.float32)
    mask[:, :80] = 1.0  # Subjekt links
    cs = ConstraintSet(max_overlay_alpha=1.0, subject_strength=1.0)
    a, b = probe.render_pair(
        viz, fd, 0.5, None, {}, cs.to_measure_constraints(),
        subject_mask=mask,
    )
    contrib = probe.contribution_map(a, b)
    h, w = contrib.shape
    assert contrib[:, : w // 2].mean() < 0.02   # Subjektregion frei
    assert contrib[:, w // 2 :].mean() > 0.05   # Rest aktiv


def test_cap_wirkt_post_fx_bei_composite_stack(probe, dummy_audio_features):
    fd = _features_dict(dummy_audio_features)
    pp = {"bloom_intensity": 1.5, "bloom_threshold": 0.5}
    viz = _composite_viz(probe)

    def energy(cap):
        cs = ConstraintSet(max_overlay_alpha=cap)
        a, b = probe.render_pair(
            viz, fd, 0.5, None, pp, cs.to_measure_constraints()
        )
        return float(probe.contribution_map(a, b).mean())

    # Cap regelt auch bei vollem Bloom und alpha=1.0-Stack (Spec §6.4)
    assert energy(1.0) > energy(0.3) > 0.0


def test_keine_maske_kein_unterschied_zu_p0(probe, dummy_audio_features):
    fd = _features_dict(dummy_audio_features)
    viz = _composite_viz(probe)
    cs = ConstraintSet(max_overlay_alpha=0.0)
    a, b = probe.render_pair(viz, fd, 0.5, None, {}, cs.to_measure_constraints())
    contrib = probe.contribution_map(a, b)
    assert float(contrib.max()) < 1e-3
