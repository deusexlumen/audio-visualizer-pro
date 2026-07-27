"""Tests für die Luma-Alpha-Ableitung im Blit-Shader (C14, Spec §6.1).

Dokumentiert den Defekt: Visualizer mit alpha=1.0 (u.a. Composite-Stacks)
decken heute die gesamte Fläche ab — ein Cap darauf wäre ein
Vollbild-Schleier, keine Deckungsregel.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def renderer(shared_gl_context):
    from src.gpu_renderer import GPUPreviewRenderer
    r = GPUPreviewRenderer(width=64, height=64, fps=30)
    yield r
    r.release()


def _viz_texture(renderer):
    """64x64 RGBA: alpha=1.0 überall, aber nur ein helles Quadrat —
    exakt das Ausgabeverhalten von Composite-Stacks (composite.py:137)."""
    data = np.zeros((64, 64, 4), dtype=np.uint8)
    data[16:48, 16:48, :3] = 255
    data[..., 3] = 255
    return renderer.ctx.texture((64, 64), 4, data.tobytes())


def _blit_to_array(renderer, viz_tex, **kwargs):
    renderer.fbo.use()
    renderer.ctx.clear(0.5, 0.5, 0.5, 1.0)  # grauer „Hintergrund"
    renderer._blit_viz_to_fbo(viz_tex, **kwargs)
    # ACHTUNG: renderer.fbo ist HDR (dtype f16, gpu_renderer.py:82) —
    # read() braucht dtype="f2", ein uint8-Read wäre Datensalat.
    raw = renderer.fbo.read(components=3, dtype="f2")
    return np.frombuffer(raw, dtype=np.float16).reshape(64, 64, 3).astype(np.float32)


def test_default_behavior_documents_defect(renderer):
    out = _blit_to_array(renderer, _viz_texture(renderer))
    # Bestand: schwarze Regionen ersetzen den Hintergrund vollständig.
    assert out[4, 4].mean() < 0.05
    assert out[32, 32].mean() > 0.9


def test_luma_alpha_frees_black_regions(renderer):
    out = _blit_to_array(
        renderer, _viz_texture(renderer), alpha_from_luma=True
    )
    # Studio: schwarze Regionen bleiben Hintergrund, Quadrat bleibt sichtbar.
    assert out[4, 4].mean() == pytest.approx(0.5, abs=0.05)
    assert out[32, 32].mean() > 0.9


def test_alpha_cap_zero_yields_background(renderer):
    out = _blit_to_array(
        renderer, _viz_texture(renderer), alpha_cap=0.0, alpha_from_luma=True
    )
    # Fundament der Messtechnik: cap=0 => kein Visualizer-Beitrag.
    assert out.mean() == pytest.approx(0.5, abs=0.02)
