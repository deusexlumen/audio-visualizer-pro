"""
GPU-beschleunigte Text-Rendering mit Signed-Distance-Field (SDF) Fonts.

Erzeugt einen Font-Atlas mit SDF-Glyphen und rendert Text als instanzierte
Quads auf der GPU. Unterstuetzt weiche Kanten, Glow und Outlines.

Vorteile gegenueber PIL-Overlay:
- Kein CPU-seitiges Framebuffer-Read fuer Text
- Beliebige Skalierung ohne Pixelation
- Glow/Outline im Shader berechnet
- Sehr schnell (ein Draw Call pro Text-Zeile)
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import moderngl


class GlyphInfo:
    """Metadaten fuer eine einzelne Glyphe im Atlas."""
    __slots__ = ["char", "x", "y", "w", "h", "bearing_x", "bearing_y", "advance"]

    def __init__(self, char: str, x: int, y: int, w: int, h: int,
                 bearing_x: int, bearing_y: int, advance: int):
        self.char = char
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.bearing_x = bearing_x
        self.bearing_y = bearing_y
        self.advance = advance


class SDFFontAtlas:
    """
    Erzeugt einen SDF-Font-Atlas aus einer TTF-Datei.

    Usage:
        atlas = SDFFontAtlas("arial.ttf", font_size=64, sdf_size=64)
        texture = atlas.build(ctx)  # ModernGL Context
    """

    # Druckbare ASCII-Zeichen
    CHARS = "".join(chr(c) for c in range(32, 127))

    def __init__(self, font_path: str, font_size: int = 64, sdf_size: int = 64,
                 padding: int = 4, spread: float = 8.0):
        """
        Args:
            font_path: Pfad zur TTF-Datei.
            font_size: Groesse des Fonts fuer die Rasterisierung.
            sdf_size: Zielgroesse jeder Glyphe im Atlas.
            padding: Pixel-Padding um jede Glyphe.
            spread: Maximale Distanz fuer das SDF (in Pixeln).
        """
        self.font_path = font_path
        self.font_size = font_size
        self.sdf_size = sdf_size
        self.padding = padding
        self.spread = spread
        self.glyphs: dict[str, GlyphInfo] = {}
        self.atlas_width = 0
        self.atlas_height = 0
        self.texture_data: np.ndarray | None = None

    def build(self, ctx: moderngl.Context) -> moderngl.Texture:
        """Baut den Atlas und gibt eine ModernGL-Textur zurueck."""
        self._generate_atlas()
        # Konvertiere float32 -> uint8 fuer die Textur (normalized)
        data_u8 = (np.clip(self.texture_data, 0.0, 1.0) * 255).astype(np.uint8)
        # PIL ist top-down, OpenGL erwartet bottom-up
        data_u8 = np.flipud(data_u8)
        tex = ctx.texture((self.atlas_width, self.atlas_height), 1,
                          data_u8.tobytes(), dtype="f1")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def _generate_atlas(self):
        """Generiert die Atlas-Textur und Glyphen-Metadaten."""
        font = ImageFont.truetype(self.font_path, self.font_size)

        # Raster-Groesse pro Glyphe (mit Padding)
        cell_size = self.sdf_size + self.padding * 2
        chars_per_row = 16
        num_rows = math.ceil(len(self.CHARS) / chars_per_row)

        self.atlas_width = chars_per_row * cell_size
        self.atlas_height = num_rows * cell_size
        atlas = np.zeros((self.atlas_height, self.atlas_width), dtype=np.float32)

        # Hilfs-Groesse fuer die initiale Bitmap (hoeher aufgeloest fuer bessere Qualitaet)
        render_scale = 4
        render_size = self.sdf_size * render_scale
        render_pad = self.padding * render_scale
        render_spread = self.spread * render_scale

        for idx, char in enumerate(self.CHARS):
            row = idx // chars_per_row
            col = idx % chars_per_row
            atlas_x = col * cell_size
            atlas_y = row * cell_size

            # Glyph-Metadaten von PIL holen
            bbox = font.getbbox(char)
            if bbox is None:
                bbox = (0, 0, render_size, render_size)
            l, t, r, b = bbox
            gw = max(1, r - l)
            gh = max(1, b - t)
            advance = font.getlength(char)

            # Hochaufloesende Bitmap rendern
            img = Image.new("L", (gw + render_pad * 2, gh + render_pad * 2), 0)
            draw = ImageDraw.Draw(img)
            draw.text((render_pad - l, render_pad - t), char, font=font, fill=255)
            bitmap = np.array(img, dtype=np.float32) / 255.0

            # SDF berechnen
            sdf = self._compute_sdf(bitmap, render_spread)

            # Downsamplen auf Zielgroesse
            from PIL import Image as PILImage
            sdf_img = PILImage.fromarray((sdf * 255).astype(np.uint8))
            target_w = self.sdf_size + self.padding * 2
            target_h = self.sdf_size + self.padding * 2
            sdf_img = sdf_img.resize((target_w, target_h), PILImage.LANCZOS)
            sdf_small = np.array(sdf_img, dtype=np.float32) / 255.0

            # In Atlas einfuegen
            atlas[atlas_y:atlas_y + target_h, atlas_x:atlas_x + target_w] = sdf_small

            # Metadaten speichern (in font_size-Koordinaten, NICHT skaliert)
            # font.getbbox() und font.getlength() geben bereits Werte fuer
            # self.font_size (z.B. 64px) zurueck, nicht fuer render_size.
            # Die Skalierung auf Zielgroesse erfolgt spaeter in render_text().
            self.glyphs[char] = GlyphInfo(
                char=char,
                x=atlas_x,
                y=atlas_y,
                w=target_w,
                h=target_h,
                bearing_x=int(l),
                bearing_y=int(-t),
                advance=int(advance),
            )

        self.texture_data = atlas

    @staticmethod
    def _compute_sdf(bitmap: np.ndarray, spread: float) -> np.ndarray:
        """
        Berechnet ein Signed Distance Field aus einer Bitmap.

        Args:
            bitmap: 2D Array mit Werten 0.0 (Hintergrund) bis 1.0 (Vordergrund).
            spread: Maximale Distanz in Pixeln.

        Returns:
            2D Array mit SDF-Werten im Bereich 0.0..1.0
            (0.0 = innen, 0.5 = Kante, 1.0 = aussen).
        """
        # Maske fuer innen (Pixel > 0.5 gelten als "inside")
        inside = bitmap > 0.5

        # Distanz von innen nach aussen
        dist_out = ndimage.distance_transform_edt(~inside)
        # Distanz von aussen nach innen
        dist_in = ndimage.distance_transform_edt(inside)

        # Kombiniere: negativ innen, positiv aussen
        signed = dist_out - dist_in

        # Normalisiere auf 0..1 mit spread als Grenze
        # signed = -spread -> 0.0 (tief innen)
        # signed = 0      -> 0.5 (Kante)
        # signed = +spread -> 1.0 (weit aussen)
        sdf = (signed / (spread * 2.0)) + 0.5
        sdf = np.clip(sdf, 0.0, 1.0)

        return sdf

    def get_glyph(self, char: str) -> GlyphInfo | None:
        """Gibt die Glyphen-Info fuer ein Zeichen zurueck."""
        return self.glyphs.get(char)

    def get_uv(self, glyph: GlyphInfo) -> tuple[float, float, float, float]:
        """Berechnet UV-Koordinaten fuer eine Glyphe im Atlas."""
        u1 = glyph.x / self.atlas_width
        v1 = glyph.y / self.atlas_height
        u2 = (glyph.x + glyph.w) / self.atlas_width
        v2 = (glyph.y + glyph.h) / self.atlas_height
        return u1, v1, u2, v2


class GPUTextRenderer:
    """
    Rendert Text auf der GPU mit einem SDF-Font-Atlas.

    Usage:
        atlas = SDFFontAtlas("arial.ttf")
        tex = atlas.build(ctx)
        renderer = GPUTextRenderer(ctx, atlas, tex)
        renderer.render_text("Hello GPU", x=100, y=100, size=48, color=(1,1,1))
    """

    def __init__(self, ctx: moderngl.Context, atlas: SDFFontAtlas, texture: moderngl.Texture,
                 width: int = 1920, height: int = 1080):
        self.ctx = ctx
        self.atlas = atlas
        self.texture = texture
        self.width = width
        self.height = height
        
        # Neuer Text-Shader mit Outline + Drop-Shadow
        self._prog = ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;

            in vec2 in_vertex_pos;   // (-1,-1) .. (1,1)
            in vec2 in_char_pos;     // Pixel-Position des Quads
            in vec2 in_char_size;    // halbe Breite/Hoehe
            in vec4 in_uv;           // u1, v1, u2, v2
            in vec3 in_color;
            in float in_alpha;

            out vec2 v_uv;
            out vec3 v_color;
            out float v_alpha;

            void main() {
                vec2 pixel_pos = in_char_pos + in_vertex_pos * in_char_size;
                vec2 ndc = (pixel_pos / u_resolution) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);

                // UV-Mapping basierend auf Quad-Position
                v_uv = mix(in_uv.xy, in_uv.zw, in_vertex_pos * 0.5 + 0.5);
                v_color = in_color;
                v_alpha = in_alpha;
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D u_atlas;
            uniform float u_smoothing;
            uniform float u_glow;
            uniform vec3 u_glow_color;
            uniform float u_outline_width;
            uniform vec3 u_outline_color;
            uniform vec2 u_shadow_offset;
            uniform vec3 u_shadow_color;
            uniform float u_shadow_alpha;

            in vec2 v_uv;
            in vec3 v_color;
            in float v_alpha;
            out vec4 f_color;

            void main() {
                float sdf = texture(u_atlas, v_uv).r;

                // Weiche Kanten: smoothstep um die Mitte (0.5)
                // SDF: 0.0 = innen (sichtbar), 1.0 = aussen (unsichtbar)
                float alpha = 1.0 - smoothstep(0.5 - u_smoothing, 0.5 + u_smoothing, sdf);

                // Outline (Kontur um die Glyphe)
                float outline = 0.0;
                if (u_outline_width > 0.0) {
                    float outline_edge = 0.5 - u_outline_width;
                    outline = smoothstep(outline_edge - u_smoothing, outline_edge + u_smoothing, sdf);
                    outline = max(0.0, outline - (1.0 - alpha));
                }

                // Drop-Shadow (versetztes SDF-Sampling im Atlas)
                float shadow = 0.0;
                if (length(u_shadow_offset) > 0.0) {
                    float shadow_sdf = texture(u_atlas, v_uv + u_shadow_offset).r;
                    shadow = 1.0 - smoothstep(0.5 - u_smoothing, 0.5 + u_smoothing, shadow_sdf);
                    // Shadow nur dort, wo weder Fill noch Outline ist
                    shadow = max(0.0, shadow - alpha - outline) * u_shadow_alpha;
                }

                // Glow: Bereich ausserhalb der Kante mit exponentiellem Abfall
                float glow = 0.0;
                if (u_glow > 0.0) {
                    float outer_sdf = 1.0 - sdf;
                    glow = exp(-outer_sdf * outer_sdf * 8.0) * u_glow;
                }

                vec3 final_color = v_color * alpha
                                 + u_outline_color * outline
                                 + u_shadow_color * shadow
                                 + u_glow_color * glow;
                float final_alpha = max(max(alpha, outline), max(shadow, glow)) * v_alpha;

                f_color = vec4(final_color, final_alpha);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self._quad_vbo = ctx.buffer(quad.tobytes())

        # Instanz-VBO: pos, size, uv, color, alpha
        self._max_chars = 500
        self._instance_data = np.zeros((self._max_chars, 12), dtype=np.float32)
        self._instance_vbo = ctx.buffer(reserve=self._max_chars * 12 * 4, dynamic=True)
        self._vao = ctx.vertex_array(
            self._prog,
            [
                (self._quad_vbo, "2f", "in_vertex_pos"),
                (self._instance_vbo, "2f 2f 4f 3f 1f /i",
                 "in_char_pos", "in_char_size", "in_uv", "in_color", "in_alpha"),
            ],
        )

    def render_text(self, text: str, x: float, y: float, size: float = 32.0,
                    color: tuple = (1.0, 1.0, 1.0), alpha: float = 1.0,
                    align: str = "left", glow: float = 0.0,
                    glow_color: tuple = (1.0, 1.0, 1.0),
                    smoothing: float = 0.25,
                    outline_width: float = 0.0,
                    outline_color: tuple = (0.0, 0.0, 0.0),
                    shadow_offset: tuple = (0.0, 0.0),
                    shadow_color: tuple = (0.0, 0.0, 0.0),
                    shadow_alpha: float = 0.5):
        """
        Rendert einen Text-String auf der GPU.

        Args:
            text: Der anzuzeigende Text.
            x, y: Position in Pixeln (Baseline).
            size: Schriftgroesse in Pixeln.
            color: RGB-Farbe (0.0-1.0).
            alpha: Gesamt-Alpha (0.0-1.0).
            align: 'left', 'center', 'right'.
            glow: Glow-Intensitaet (0.0-1.0).
            glow_color: RGB-Farbe des Glows.
            smoothing: Kantenglaettung (0.0 = hart, 0.5 = sehr weich).
        """
        # Text-Breite berechnen fuer Alignment
        total_width = 0.0
        for char in text:
            g = self.atlas.get_glyph(char)
            if g:
                scale = size / self.atlas.sdf_size
                total_width += g.advance * scale

        if align == "center":
            cursor_x = x - total_width / 2.0
        elif align == "right":
            cursor_x = x - total_width
        else:
            cursor_x = x

        cursor_y = y
        instance_idx = 0
        scale = size / self.atlas.sdf_size

        for char in text:
            g = self.atlas.get_glyph(char)
            if not g:
                continue
            if instance_idx >= self._max_chars:
                break

            # Quad-Position und Groesse
            px = cursor_x + g.bearing_x * scale
            py = cursor_y - g.bearing_y * scale
            pw = g.w * scale / 2.0
            ph = g.h * scale / 2.0

            u1, v1, u2, v2 = self.atlas.get_uv(g)

            self._instance_data[instance_idx] = [
                px + pw, py + ph,  # Mitte des Quads
                pw, ph,
                u1, v1, u2, v2,
                color[0], color[1], color[2],
                alpha,
            ]
            instance_idx += 1
            cursor_x += g.advance * scale

        if instance_idx == 0:
            return

        # Shader-Uniforms
        self._prog["u_resolution"].value = (self.width, self.height)
        self._prog["u_smoothing"].value = smoothing
        self._prog["u_glow"].value = glow
        self._prog["u_glow_color"].value = glow_color
        self._prog["u_outline_width"].value = outline_width
        self._prog["u_outline_color"].value = outline_color
        shadow_uv_offset = (
            shadow_offset[0] / max(self.atlas.atlas_width, 1),
            shadow_offset[1] / max(self.atlas.atlas_height, 1)
        )
        self._prog["u_shadow_offset"].value = shadow_uv_offset
        self._prog["u_shadow_color"].value = shadow_color
        self._prog["u_shadow_alpha"].value = shadow_alpha

        self.texture.use(location=0)
        self._prog["u_atlas"].value = 0

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._instance_vbo.write(self._instance_data[:instance_idx].tobytes())
        self._vao.render(mode=moderngl.TRIANGLE_STRIP, instances=instance_idx)
        # Blending wird vom Aufrufer verwaltet (State-Machine-Sicherheit)

    def release(self):
        """Gibt alle GPU-Ressourcen (Shader, VBOs, VAO, Atlas-Textur) frei.

        Muss explizit aufgerufen werden, bevor ein neuer Atlas/Renderer
        erzeugt wird, um Memory-Leaks im VRAM zu vermeiden.
        """
        if hasattr(self, '_vao') and self._vao is not None:
            self._vao.release()
            self._vao = None
        if hasattr(self, '_instance_vbo') and self._instance_vbo is not None:
            self._instance_vbo.release()
            self._instance_vbo = None
        if hasattr(self, '_quad_vbo') and self._quad_vbo is not None:
            self._quad_vbo.release()
            self._quad_vbo = None
        if hasattr(self, '_prog') and self._prog is not None:
            self._prog.release()
            self._prog = None
        if hasattr(self, 'texture') and self.texture is not None:
            self.texture.release()
            self.texture = None
