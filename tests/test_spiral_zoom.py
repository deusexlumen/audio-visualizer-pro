"""Tests fuer src/spiral_zoom.py: Einstellungen, Abbildung, Zoom-Phase."""

import numpy as np
import pytest

from src.spiral_zoom import SpiralSettings, spiral_map, spiral_uniforms

W, H = 1920, 1080


def _on(**kw):
    return SpiralSettings(enabled=True, **kw)


def _pixels(w, h, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, w, n), rng.uniform(0, h, n)


class TestSettings:
    def test_defaults_are_off(self):
        s = SpiralSettings.from_postprocess(None)
        assert s.enabled is False
        assert s.is_active is False

    def test_reads_and_clamps_postprocess(self):
        s = SpiralSettings.from_postprocess({
            "spiral_enabled": True,
            "spiral_arms": 7.4,
            "spiral_ratio": 99,
            "spiral_rotation": -500,
            "spiral_speed": "kaputt",
            "spiral_mix": float("nan"),
        })
        assert s.enabled is True
        assert s.arms == 3
        assert isinstance(s.arms, int)
        assert s.ratio == 6.0
        assert s.rotation == -90.0
        assert s.speed == SpiralSettings().speed  # Unsinn -> Default
        assert s.mix == SpiralSettings().mix

    def test_mix_zero_is_inactive(self):
        assert _on(mix=0.0).is_active is False
        assert _on(mix=0.5).is_active is True


class TestSpiralMap:
    def test_ring_is_identity_without_motion(self):
        """arms=0, keine Drehung, u=0: der Quell-Ring bildet sich auf sich selbst ab."""
        s = _on(arms=0, rotation=0.0, ratio=2.5, feather=0.15)
        uni = spiral_uniforms(s, W, H, 0.0)
        r_in = 0.5 * min(W, H) / 2.5
        rr = np.linspace(r_in * 1.02, r_in * 0.85 * 2.5 * 0.98, 50)
        ang = np.linspace(-3.0, 3.0, 50)
        px = W / 2 + rr * np.cos(ang)
        py = H / 2 + rr * np.sin(ang)
        m = spiral_map(uni, px, py)
        np.testing.assert_allclose(m["q"][:, 0], px, atol=1e-6)
        np.testing.assert_allclose(m["q"][:, 1], py, atol=1e-6)
        assert np.all(m["w_in"] == 0.0)

    def test_core_shows_scaled_copy(self):
        """Innerhalb des Innenradius erscheint das Bild um K vergroessert abgetastet."""
        s = _on(arms=0, rotation=0.0, ratio=3.0, feather=0.15)
        uni = spiral_uniforms(s, W, H, 0.0)
        r_in = 0.5 * H / 3.0
        zx, zy = 0.3 * r_in, -0.2 * r_in
        m = spiral_map(uni, [W / 2 + zx], [H / 2 + zy])
        np.testing.assert_allclose(m["q"][0], [W / 2 + 3 * zx, H / 2 + 3 * zy], atol=1e-6)
        assert m["jac"][0] == pytest.approx(3.0)

    @pytest.mark.parametrize("arms,rot", [(0, 0.0), (1, 0.0), (-2, 30.0), (3, -45.0)])
    def test_period_is_one_level(self, arms, rot):
        """map(u) == map(u + 1): deshalb reicht dem Shader u mod 1."""
        s = _on(arms=arms, rotation=rot, ratio=2.5)
        px, py = _pixels(W, H)
        uni_a = spiral_uniforms(s, W, H, 0.3)
        uni_b = dict(uni_a)
        uni_b["u_u"] = 1.3
        a = spiral_map(uni_a, px, py)
        b = spiral_map(uni_b, px, py)
        np.testing.assert_allclose(a["q"], b["q"], atol=1e-6)
        np.testing.assert_allclose(a["w_in"], b["w_in"], atol=1e-9)

    @pytest.mark.parametrize("arms", [-3, -1, 1, 2])
    def test_no_seam_on_the_left(self, arms):
        """Ganzzahlige Arme: am atan-Sprung links der Mitte bleibt die Abbildung stetig."""
        s = _on(arms=arms, rotation=20.0, ratio=2.5)
        uni = spiral_uniforms(s, W, H, 0.4)
        xs = W / 2 - np.linspace(50, 500, 40)
        above = spiral_map(uni, xs, np.full_like(xs, H / 2 + 1e-4))
        below = spiral_map(uni, xs, np.full_like(xs, H / 2 - 1e-4))
        np.testing.assert_allclose(above["q"], below["q"], atol=0.01)
        np.testing.assert_allclose(above["w_in"], below["w_in"], atol=1e-3)

    def test_samples_stay_inside_the_frame(self):
        """Alle benutzten Abtastpunkte liegen im Inkreis (nie ausserhalb des Bildes)."""
        rng = np.random.default_rng(1)
        c = np.array([W / 2, H / 2])
        r_out = 0.5 * min(W, H)
        for _ in range(20):
            s = _on(
                arms=int(rng.integers(-3, 4)),
                rotation=float(rng.uniform(-90, 90)),
                ratio=float(rng.uniform(1.5, 6.0)),
                feather=float(rng.uniform(0.02, 0.5)),
            )
            uni = spiral_uniforms(s, W, H, float(rng.uniform(0, 1)))
            px, py = _pixels(W, H, seed=int(rng.integers(1_000_000)))
            m = spiral_map(uni, px, py)
            assert np.all(np.linalg.norm(m["q"] - c, axis=1) <= r_out + 1e-6)
            used = m["w_in"] > 0
            assert np.all(np.linalg.norm(m["q_in"][used] - c, axis=1) <= r_out + 1e-6)

    @pytest.mark.parametrize("size", [(480, 270), (3840, 2160), (1080, 1920)])
    def test_same_picture_at_every_resolution(self, size):
        """Normierte Punkte (relativ zu min(B, H)) bilden in jeder Aufloesung gleich ab."""
        s = _on(arms=1, rotation=15.0, ratio=3.0)

        def normalized(w, h):
            m0 = min(w, h)
            zn = np.random.default_rng(7).uniform(-0.5, 0.5, (500, 2))
            m = spiral_map(
                spiral_uniforms(s, w, h, 0.25),
                w / 2 + zn[:, 0] * m0,
                h / 2 + zn[:, 1] * m0,
            )
            return (m["q"] - [w / 2, h / 2]) / m0, m["w_in"]

        q_ref, w_ref = normalized(1920, 1080)
        q, w = normalized(*size)
        np.testing.assert_allclose(q, q_ref, atol=1e-9)
        np.testing.assert_allclose(w, w_ref, atol=1e-9)
