"""
Tests fuer GPU-Text-Rendering mit SDF-Font-Atlas.

Mocked PIL Font und ModernGL Context fuer hardware-unabhaengige Tests.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.gpu_text_renderer import GlyphInfo, SDFFontAtlas, GPUTextRenderer


class TestGlyphInfo:
    """Tests fuer das GlyphInfo Datenmodell."""

    def test_glyph_info_creation(self):
        """GlyphInfo sollte alle Attribute korrekt speichern."""
        g = GlyphInfo(char="A", x=10, y=20, w=30, h=40,
                      bearing_x=5, bearing_y=6, advance=35)
        assert g.char == "A"
        assert g.x == 10
        assert g.y == 20
        assert g.w == 30
        assert g.h == 40
        assert g.bearing_x == 5
        assert g.bearing_y == 6
        assert g.advance == 35


class TestComputeSDF:
    """Tests fuer die SDF-Berechnung (reine NumPy/SciPy Logik)."""

    def test_compute_sdf_shape(self):
        """SDF sollte die gleiche Shape wie die Eingabe haben."""
        bitmap = np.zeros((32, 32), dtype=np.float32)
        bitmap[10:22, 10:22] = 1.0  # Rechteck in der Mitte
        sdf = SDFFontAtlas._compute_sdf(bitmap, spread=8.0)
        assert sdf.shape == bitmap.shape
        # scipy.distance_transform_edt gibt float64 zurueck, clamped auf float32
        assert sdf.dtype in (np.float32, np.float64)

    def test_compute_sdf_range(self):
        """SDF-Werte sollten im Bereich 0.0..1.0 liegen."""
        bitmap = np.random.rand(16, 16).astype(np.float32)
        sdf = SDFFontAtlas._compute_sdf(bitmap, spread=4.0)
        assert sdf.min() >= 0.0
        assert sdf.max() <= 1.0

    def test_compute_sdf_inside_outside(self):
        """Innen sollte < 0.5, aussen > 0.5 sein (ungefaehr)."""
        bitmap = np.zeros((64, 64), dtype=np.float32)
        bitmap[20:44, 20:44] = 1.0
        sdf = SDFFontAtlas._compute_sdf(bitmap, spread=8.0)
        # Mitte des Rechtecks (innen)
        assert sdf[32, 32] < 0.5
        # Weit draussen
        assert sdf[5, 5] > 0.5


class TestSDFFontAtlasAccessors:
    """Tests fuer Atlas-Metadaten ohne Font-Rendering."""

    def test_get_glyph_missing(self):
        """Nicht-existierende Glyphe sollte None zurueckgeben."""
        atlas = SDFFontAtlas("dummy.ttf")
        assert atlas.get_glyph("X") is None

    def test_get_glyph_existing(self):
        """Existierende Glyphe sollte korrekte GlyphInfo zurueckgeben."""
        atlas = SDFFontAtlas("dummy.ttf")
        atlas.glyphs["A"] = GlyphInfo(char="A", x=0, y=0, w=10, h=10,
                                       bearing_x=1, bearing_y=2, advance=8)
        g = atlas.get_glyph("A")
        assert g is not None
        assert g.char == "A"
        assert g.advance == 8

    def test_get_uv(self):
        """UV-Koordinaten sollten korrekt berechnet werden."""
        atlas = SDFFontAtlas("dummy.ttf")
        atlas.atlas_width = 100
        atlas.atlas_height = 100
        glyph = GlyphInfo(char="B", x=10, y=20, w=30, h=40,
                          bearing_x=0, bearing_y=0, advance=30)
        u1, v1, u2, v2 = atlas.get_uv(glyph)
        assert u1 == 0.1
        assert v1 == 0.2
        assert u2 == 0.4
        assert v2 == 0.6


class TestSDFFontAtlasBuild:
    """Tests fuer den Atlas-Build mit gemocktem PIL Font."""

    @patch.object(SDFFontAtlas, '_generate_atlas')
    def test_build_creates_texture(self, mock_generate):
        """build() sollte eine ModernGL Textur zurueckgeben."""
        ctx = MagicMock()
        mock_tex = MagicMock()
        ctx.texture.return_value = mock_tex

        atlas = SDFFontAtlas("dummy.ttf", font_size=16, sdf_size=16, padding=4)
        atlas.texture_data = np.random.rand(100, 100).astype(np.float32)
        atlas.atlas_width = 100
        atlas.atlas_height = 100
        atlas.glyphs["A"] = GlyphInfo("A", 0, 0, 10, 10, 1, 1, 8)

        tex = atlas.build(ctx)

        assert tex is mock_tex
        assert ctx.texture.call_count >= 1
        assert atlas.texture_data is not None


class TestGPUTextRenderer:
    """Tests fuer den GPU-Text-Renderer."""

    @pytest.fixture
    def mock_atlas(self):
        """Erzeugt einen gemockten Atlas mit bekannten Glyphen."""
        atlas = MagicMock(spec=SDFFontAtlas)
        atlas.sdf_size = 64
        atlas.atlas_width = 512
        atlas.atlas_height = 512
        atlas.glyphs = {
            "A": GlyphInfo("A", 0, 0, 40, 50, 2, 10, 36),
            "B": GlyphInfo("B", 50, 0, 40, 50, 2, 10, 36),
        }
        atlas.get_glyph.side_effect = lambda c: atlas.glyphs.get(c)
        atlas.get_uv.return_value = (0.0, 0.0, 0.1, 0.1)
        return atlas

    @pytest.fixture
    def mock_ctx(self):
        """Erzeugt einen gemockten ModernGL Context."""
        ctx = MagicMock()
        ctx.program.return_value = MagicMock()
        ctx.buffer.return_value = MagicMock()
        ctx.vertex_array.return_value = MagicMock()
        return ctx

    def test_init(self, mock_ctx, mock_atlas):
        """GPUTextRenderer sollte Shader, VBOs und VAO erstellen."""
        mock_tex = MagicMock()
        renderer = GPUTextRenderer(mock_ctx, mock_atlas, mock_tex, width=800, height=600)

        assert renderer.width == 800
        assert renderer.height == 600
        assert renderer.atlas is mock_atlas
        assert renderer.texture is mock_tex
        mock_ctx.program.assert_called_once()
        mock_ctx.buffer.assert_called()
        mock_ctx.vertex_array.assert_called_once()

    def test_render_text_sets_uniforms(self, mock_ctx, mock_atlas):
        """render_text sollte Shader-Uniforms korrekt setzen."""
        mock_tex = MagicMock()
        renderer = GPUTextRenderer(mock_ctx, mock_atlas, mock_tex)

        renderer.render_text("AB", x=100, y=200, size=32.0,
                             color=(1.0, 0.5, 0.0), alpha=0.9,
                             glow=0.3, glow_color=(1.0, 1.0, 1.0),
                             outline_width=0.05, outline_color=(0.0, 0.0, 0.0),
                             shadow_offset=(2.0, 2.0), shadow_color=(0.0, 0.0, 0.0),
                             shadow_alpha=0.5)

        # VAO sollte mit den korrekten Instanzen gerendert werden
        vao = mock_ctx.vertex_array.return_value
        vao.render.assert_called_once()
        call_kwargs = vao.render.call_args[1]
        assert call_kwargs["instances"] == 2  # "AB" = 2 Zeichen

    def test_render_text_align_center(self, mock_ctx, mock_atlas):
        """Text-Alignment 'center' sollte Cursor verschieben."""
        mock_tex = MagicMock()
        renderer = GPUTextRenderer(mock_ctx, mock_atlas, mock_tex)

        # Render mit center alignment — sollte nicht crashen
        renderer.render_text("A", x=100, y=100, size=32.0, align="center")

        # Render mit right alignment
        renderer.render_text("B", x=100, y=100, size=32.0, align="right")

    def test_render_text_empty(self, mock_ctx, mock_atlas):
        """Leerer Text sollte keinen Draw Call ausloesen."""
        mock_tex = MagicMock()
        renderer = GPUTextRenderer(mock_ctx, mock_atlas, mock_tex)
        vao = mock_ctx.vertex_array.return_value

        renderer.render_text("", x=0, y=0)

        # VAO render sollte NICHT aufgerufen werden (keine Instanzen)
        vao.render.assert_not_called()

    def test_render_text_unknown_chars(self, mock_ctx, mock_atlas):
        """Unbekannte Zeichen sollten uebersprungen werden."""
        mock_tex = MagicMock()
        renderer = GPUTextRenderer(mock_ctx, mock_atlas, mock_tex)
        vao = mock_ctx.vertex_array.return_value

        renderer.render_text("???", x=0, y=0, size=32.0)

        # VAO render sollte nicht aufgerufen werden, da keine Glyphen gefunden
        vao.render.assert_not_called()

    def test_render_text_max_chars(self, mock_ctx, mock_atlas):
        """Zu langer Text sollte bei _max_chars abgeschnitten werden."""
        mock_tex = MagicMock()
        renderer = GPUTextRenderer(mock_ctx, mock_atlas, mock_tex)
        renderer._max_chars = 2
        vao = mock_ctx.vertex_array.return_value

        renderer.render_text("AAA", x=0, y=0, size=32.0)

        # Nur 2 Instanzen sollten gerendert werden
        vao.render.assert_called_once()
        call_kwargs = vao.render.call_args[1]
        assert call_kwargs["instances"] <= 2

    def test_release(self, mock_ctx, mock_atlas):
        """release() sollte alle GPU-Ressourcen freigeben."""
        mock_tex = MagicMock()
        renderer = GPUTextRenderer(mock_ctx, mock_atlas, mock_tex)

        renderer.release()

        mock_tex.release.assert_called_once()
        mock_ctx.vertex_array.return_value.release.assert_called_once()
