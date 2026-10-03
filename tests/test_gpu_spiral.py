"""GPU-Tests fuer den Spiral-Zoom-Pass: Shader gegen CPU-Spiegel."""

import moderngl
import numpy as np
import pytest

from src.gpu_spiral import SpiralZoomPass
from src.spiral_zoom import SpiralSettings, spiral_map, spiral_uniforms

pytestmark = pytest.mark.gpu

# Zweierpotenzen: Mip-Stufen teilen sauber, der lineare Verlauf bleibt exakt linear
W, H = 256, 128


@pytest.fixture
def gl():
    ctx = moderngl.create_standalone_context()
    yield ctx
    ctx.release()


def _scene(ctx, img):
    tex = ctx.texture((W, H), 4, img.astype("f2").tobytes(), dtype="f2")
    return ctx.framebuffer(color_attachments=[tex])


def _gradient():
    """Farbe = (x/B, y/H, 0.5, 1): bleibt in jeder Mip-Stufe linear."""
    img = np.zeros((H, W, 4), dtype=np.float32)
    img[..., 0] = ((np.arange(W) + 0.5) / W)[None, :]
    img[..., 1] = ((np.arange(H) + 0.5) / H)[:, None]
    img[..., 2] = 0.5
    img[..., 3] = 1.0
    return img


def _read(fbo):
    data = np.frombuffer(fbo.read(components=4, dtype="f2"), dtype=np.float16)
    return data.reshape(H, W, 4).astype(np.float32)


def _pixel_grid():
    py, px = np.mgrid[0:H, 0:W]
    return px.ravel() + 0.5, py.ravel() + 0.5


def test_disabled_leaves_scene_untouched(gl):
    img = _gradient()
    fbo = _scene(gl, img)
    SpiralZoomPass(gl, W, H).apply(fbo, SpiralSettings(enabled=False, arms=2), 0.3)
    np.testing.assert_array_equal(_read(fbo), img.astype("f2").astype(np.float32))


def test_hdr_values_survive(gl):
    """Kein clamp im Pass: HDR-Werte > 1 bleiben erhalten."""
    img = np.full((H, W, 4), 4.0, dtype=np.float32)
    fbo = _scene(gl, img)
    SpiralZoomPass(gl, W, H).apply(fbo, SpiralSettings(enabled=True, arms=1), 0.4)
    np.testing.assert_allclose(_read(fbo)[..., :3], 4.0, atol=1e-2)


def test_ring_is_identity_even_with_blending_left_on(gl):
    img = _gradient()
    fbo = _scene(gl, img)
    s = SpiralSettings(enabled=True, arms=0, rotation=0.0, ratio=2.5, feather=0.15)
    gl.enable(moderngl.BLEND)
    gl.blend_func = moderngl.ONE, moderngl.ONE  # wie nach dem Luma-Alpha-Blit
    SpiralZoomPass(gl, W, H).apply(fbo, s, 0.0)
    out = _read(fbo)
    px, py = _pixel_grid()
    zr = np.hypot(px - W / 2, py - H / 2)
    r_in = 0.5 * H / 2.5
    ring = (zr > r_in * 1.05) & (zr < r_in * 0.85 * 2.5 * 0.95)
    assert ring.sum() > 500
    np.testing.assert_allclose(
        out.reshape(-1, 4)[ring, :3], img.reshape(-1, 4)[ring, :3], atol=2e-3
    )


def test_alpha_is_warped_with_color(gl):
    """Deckung (Alpha) wandert mit der Farbe — wichtig fuer den Luma-/
    Occlusion-Alpha-Blit ueber ein Hintergrundbild."""
    img = _gradient()
    img[..., 3] = 0.2 + 0.6 * img[..., 0]  # Alpha-Verlauf von links nach rechts
    fbo = _scene(gl, img)
    s = SpiralSettings(enabled=True, arms=1, rotation=30.0, ratio=2.5)
    SpiralZoomPass(gl, W, H).apply(fbo, s, 0.37)
    out = _read(fbo).reshape(-1, 4)

    px, py = _pixel_grid()
    m = spiral_map(spiral_uniforms(s, W, H, 0.37), px, py)
    w_in = m["w_in"]
    expected = (0.2 + 0.6 * m["q"][:, 0] / W) * (1 - w_in) + (0.2 + 0.6 * m["q_in"][:, 0] / W) * w_in
    c = np.array([W / 2, H / 2])
    r_lim = 0.85 * 0.5 * min(W, H)
    uses_in = w_in > 0
    ok = (
        (m["jac"] <= 4.0)
        & (np.linalg.norm(m["q"] - c, axis=1) <= r_lim)
        & (~uses_in | ((m["jac_in"] <= 4.0) & (np.linalg.norm(m["q_in"] - c, axis=1) <= r_lim)))
        & (np.hypot(px - W / 2, py - H / 2) >= 4.0)
    )
    assert ok.sum() > 200
    np.testing.assert_allclose(out[ok, 3], expected[ok], atol=0.01)


@pytest.mark.parametrize("arms", [0, 1, -2])
@pytest.mark.parametrize("rotation", [0.0, 30.0])
@pytest.mark.parametrize("u", [0.0, 0.37])
def test_matches_cpu_mirror(gl, arms, rotation, u):
    img = _gradient()
    fbo = _scene(gl, img)
    s = SpiralSettings(enabled=True, arms=arms, rotation=rotation, ratio=2.5)
    SpiralZoomPass(gl, W, H).apply(fbo, s, u)
    out = _read(fbo).reshape(-1, 4)

    px, py = _pixel_grid()
    m = spiral_map(spiral_uniforms(s, W, H, u), px, py)
    color = lambda q: np.stack([q[:, 0] / W, q[:, 1] / H], axis=-1)  # noqa: E731
    w_in = m["w_in"][:, None]
    expected = color(m["q"]) * (1 - w_in) + color(m["q_in"]) * w_in

    # Nur Pixel vergleichen, deren Abtastung weit genug vom Bildrand liegt
    # (Randklemmung) und nicht aus sehr groben Mip-Stufen kommt.
    c = np.array([W / 2, H / 2])
    r_lim = 0.85 * 0.5 * min(W, H)
    uses_in = m["w_in"] > 0
    ok = (
        (m["jac"] <= 4.0)
        & (np.linalg.norm(m["q"] - c, axis=1) <= r_lim)
        & (~uses_in | ((m["jac_in"] <= 4.0) & (np.linalg.norm(m["q_in"] - c, axis=1) <= r_lim)))
        & (np.hypot(px - W / 2, py - H / 2) >= 4.0)
    )
    assert ok.sum() > 200, "Test prueft zu wenige Pixel"
    np.testing.assert_allclose(out[ok, :2], expected[ok], atol=0.01)
