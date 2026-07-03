"""
Tests fuer src/gpu_bloom.py: .cube-LUT-Parser und Bloom-Parameter.
"""

import numpy as np
import pytest

from src.gpu_bloom import load_cube_lut
from config.schemas import PostProcessConfig


class TestLoadCubeLut:
    """Tests fuer den .cube-LUT-Parser."""

    def test_parses_valid_lut(self, mock_lut_file):
        """Gueltige 2x2x2-LUT sollte korrekt geparst werden."""
        arr = load_cube_lut(mock_lut_file)
        assert arr.shape == (2, 2, 2, 3)
        assert arr.dtype == np.float32
        # Erster Eintrag (0,0,0), letzter Eintrag (1,1,1) laut Fixture
        assert np.allclose(arr[0, 0, 0], [0.0, 0.0, 0.0])
        assert np.allclose(arr[1, 1, 1], [1.0, 1.0, 1.0])
        # .cube: rot variiert am schnellsten -> arr[b][g][r]
        assert np.allclose(arr[0, 0, 1], [1.0, 0.0, 0.0])
        assert np.allclose(arr[0, 1, 0], [0.0, 1.0, 0.0])
        assert np.allclose(arr[1, 0, 0], [0.0, 0.0, 1.0])

    def test_values_clamped(self, tmp_path):
        """Werte ausserhalb 0-1 sollten geclampt werden."""
        lut = tmp_path / "over.cube"
        lut.write_text(
            "LUT_3D_SIZE 2\n" + "\n".join(["-0.5 0.5 1.5"] * 8), encoding="utf-8"
        )
        arr = load_cube_lut(str(lut))
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_missing_size_raises(self, tmp_path):
        """Fehlende LUT_3D_SIZE sollte ValueError werfen."""
        lut = tmp_path / "kaputt.cube"
        lut.write_text("0.0 0.0 0.0\n1.0 1.0 1.0\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_cube_lut(str(lut))

    def test_wrong_entry_count_raises(self, tmp_path):
        """Falsche Anzahl an Eintraegen sollte ValueError werfen."""
        lut = tmp_path / "unvollstaendig.cube"
        lut.write_text("LUT_3D_SIZE 2\n0.0 0.0 0.0\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_cube_lut(str(lut))

    def test_ignores_comments_and_title(self, tmp_path):
        """Kommentare und TITLE-Zeilen sollten ignoriert werden."""
        lut = tmp_path / "kommentare.cube"
        lut.write_text(
            "# Kommentar\nTITLE \"Test\"\nLUT_3D_SIZE 2\n"
            + "\n".join(["0.5 0.5 0.5"] * 8),
            encoding="utf-8",
        )
        arr = load_cube_lut(str(lut))
        assert arr.shape == (2, 2, 2, 3)
        assert np.allclose(arr, 0.5)


class TestBloomSchema:
    """Tests fuer die neuen Post-Process-Schema-Felder."""

    def test_defaults(self):
        cfg = PostProcessConfig()
        assert cfg.bloom_intensity == 0.6
        assert cfg.bloom_threshold == 1.0
        assert cfg.bloom_radius == 1.0
        assert cfg.exposure == 1.0
        assert cfg.lut_strength == 1.0

    def test_bounds_validated(self):
        with pytest.raises(Exception):
            PostProcessConfig(bloom_intensity=5.0)
        with pytest.raises(Exception):
            PostProcessConfig(exposure=0.0)
        with pytest.raises(Exception):
            PostProcessConfig(lut_strength=2.0)


@pytest.mark.gpu
class TestBloomPassGPU:
    """GPU-Smoke-Test fuer die Bloom-Kette (wird ohne GPU uebersprungen)."""

    def test_bloom_brightens_hdr_content(self, require_gpu):
        """Bloom sollte helle HDR-Bereiche in die Umgebung ausbluten lassen."""
        import moderngl
        from src.gpu_bloom import BloomPass

        ctx = moderngl.create_standalone_context()
        try:
            w, h = 256, 256
            scene_tex = ctx.texture((w, h), 4, dtype='f2')
            scene_fbo = ctx.framebuffer(color_attachments=[scene_tex])

            # Szene: schwarzer Hintergrund mit hellem HDR-Quadrat in der Mitte
            data = np.zeros((h, w, 4), dtype=np.float16)
            data[112:144, 112:144] = [4.0, 4.0, 4.0, 1.0]  # HDR-Wert > threshold
            scene_tex.write(data.tobytes())

            bloom = BloomPass(ctx, w, h)
            bloom.apply(scene_fbo, scene_tex, intensity=1.0, threshold=1.0)

            result = np.frombuffer(
                scene_fbo.read(components=3, dtype='f2'), dtype=np.float16
            ).reshape((h, w, 3))

            # Pixel NEBEN dem Quadrat muessen jetzt Licht abbekommen haben
            halo = float(result[128, 160, :].max())
            corner = float(result[8, 8, :].max())
            assert halo > 0.01, "Bloom-Halo neben dem hellen Quadrat fehlt"
            assert halo > corner, "Halo sollte heller sein als die Bildecke"

            bloom.release()
        finally:
            ctx.release()
