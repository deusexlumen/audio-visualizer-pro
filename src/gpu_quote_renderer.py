"""
GPU-beschleunigter Quote-Renderer.

Rendert Key-Zitate als abgerundete Box + Text direkt auf der GPU.
Ersetzt den langsamen PIL-basierten QuoteOverlayRenderer im Render-Loop.

Features:
- Abgerundete Hintergrund-Box mit SDF-basierten Ecken
- Text via GPUTextRenderer (SDF-Font-Atlas)
- Fade-In/Out via Alpha-Uniform
- Glow-Pulse Effekt
- Schatten
- Position: bottom, center, top
"""

import numpy as np
import moderngl
from typing import Optional, List

from .gpu_text_renderer import SDFFontAtlas, GPUTextRenderer
from .gemini_integration import Quote


class GPUQuoteRenderer:
    """
    Rendert Quote-Overlays komplett auf der GPU.
    """

    def __init__(self, ctx: moderngl.Context, font_path: str,
                 width: int = 1920, height: int = 1080):
        self.ctx = ctx
        self.width = width
        self.height = height

        # Font-Atlas + Text-Renderer
        self._atlas = SDFFontAtlas(font_path, font_size=64, sdf_size=64)
        self._font_tex = self._atlas.build(ctx)
        self._text_renderer = GPUTextRenderer(ctx, self._atlas, self._font_tex,
                                               width=width, height=height)

        # --- Box-Shader (abgerundete Ecken via SDF) ---
        self._box_prog = ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;
            in vec2 in_pos;
            void main() {
                vec2 ndc = (in_pos / u_resolution) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            uniform vec2 u_resolution;
            uniform vec4 u_box;      // x, y, w, h (in Pixeln)
            uniform float u_radius;  // Ecken-Radius
            uniform vec4 u_color;    // RGBA
            uniform float u_alpha;   // Global-Alpha (Fade)
            uniform vec2 u_shadow;   // Shadow-Offset
            uniform vec4 u_shadow_color;
            uniform float u_shadow_alpha;
            uniform float u_blur;    // Hintergrund-Blur (simuliert)

            out vec4 f_color;

            float sdRoundedBox(vec2 p, vec2 b, float r) {
                vec2 q = abs(p) - b + r;
                return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
            }

            void main() {
                vec2 pixel = gl_FragCoord.xy;
                vec2 center = u_box.xy + u_box.zw * 0.5;
                vec2 half_size = u_box.zw * 0.5;
                float radius = min(u_radius, min(half_size.x, half_size.y));

                // Shadow
                float shadow_dist = sdRoundedBox(pixel - center - u_shadow, half_size, radius);
                float shadow_alpha = u_shadow_alpha * (1.0 - smoothstep(-2.0, 2.0, shadow_dist));

                // Box
                float dist = sdRoundedBox(pixel - center, half_size, radius);
                float box_alpha = 1.0 - smoothstep(-1.0, 1.0, dist);

                // Blur-Simulation (einfach: weichere Kante)
                float blur_edge = smoothstep(-u_blur, u_blur, -dist);
                box_alpha *= blur_edge;

                vec3 final_color = u_shadow_color.rgb * shadow_alpha
                                 + u_color.rgb * box_alpha;
                float final_alpha = max(shadow_alpha, box_alpha) * u_alpha;

                f_color = vec4(final_color, final_alpha);
            }
            """,
        )

        # Fullscreen-Quad für Box
        box_quad = np.array([
            [0.0, 0.0], [width, 0.0], [0.0, height],
            [width, 0.0], [width, height], [0.0, height],
        ], dtype=np.float32)
        self._box_vbo = ctx.buffer(box_quad.tobytes())
        self._box_vao = ctx.vertex_array(self._box_prog, [(self._box_vbo, "2f", "in_pos")])

        # Cache für aktives Zitat (vermeidet Neuberechnung)
        self._cached_quote_text: Optional[str] = None
        self._cached_lines: List[str] = []
        self._cached_box_w: float = 0.0
        self._cached_box_h: float = 0.0

    def render(self, quote: Quote, config, time_seconds: float,
               frame_idx: int = None) -> None:
        """
        Rendert ein Quote-Overlay auf den aktuellen FBO.
        """
        if quote is None:
            return

        # --- Fade-Alpha berechnen ---
        fade = getattr(config, 'fade_duration', 0.6)
        display_dur = getattr(config, 'display_duration', 8.0)
        effective_end = min(quote.end_time, quote.start_time + display_dur)
        latency = getattr(config, 'latency_offset', 0.0)
        t = time_seconds - latency

        if t < quote.start_time or t > effective_end:
            return

        if t < quote.start_time + fade:
            alpha = (t - quote.start_time) / fade
        elif t > effective_end - fade:
            alpha = (effective_end - t) / fade
        else:
            alpha = 1.0
        alpha = max(0.0, min(1.0, alpha))
        if alpha <= 0.01:
            return

        # --- Text vorbereiten (caching) ---
        if self._cached_quote_text != quote.text:
            self._cached_quote_text = quote.text
            self._cached_lines = self._wrap_text(quote.text, config)
            # Box-Größe schätzen
            scale = getattr(config, 'scale', 1.0)
            font_size = int(config.font_size * scale)
            line_h = font_size * 1.4
            padding = int(config.box_padding * scale)
            max_w = 0
            for line in self._cached_lines:
                w = self._text_width(line, font_size)
                max_w = max(max_w, w)
            self._cached_box_w = min(max_w + padding * 2,
                                     self.width * config.max_width_ratio)
            self._cached_box_h = len(self._cached_lines) * line_h + padding * 2

        lines = self._cached_lines
        scale = getattr(config, 'scale', 1.0)
        font_size = int(config.font_size * scale)
        line_height_px = font_size * 1.4
        padding = int(config.box_padding * scale)
        box_w = self._cached_box_w
        box_h = self._cached_box_h

        # --- Position berechnen ---
        offset_x = getattr(config, 'offset_x', 0)
        offset_y = getattr(config, 'offset_y', 0)
        margin = getattr(config, 'box_margin_bottom', 100)

        box_x = (self.width - box_w) / 2.0 + offset_x
        if config.position == "bottom":
            box_y = self.height - box_h - margin + offset_y
        elif config.position == "top":
            box_y = margin + offset_y
        else:  # center
            box_y = (self.height - box_h) / 2.0 + offset_y

        # Clamp
        box_x = max(0.0, min(box_x, self.width - box_w))
        box_y = max(0.0, min(box_y, self.height - box_h))

        # --- Box rendern ---
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        box_color = config.box_color
        if len(box_color) == 3:
            box_color = (*box_color, 200)
        shadow = getattr(config, 'shadow_offset', (3, 3))
        shadow_color = getattr(config, 'shadow_color', (0, 0, 0, 120))
        if len(shadow_color) == 3:
            shadow_color = (*shadow_color, 120)

        self._box_prog["u_resolution"].value = (self.width, self.height)
        self._box_prog["u_box"].value = (box_x, box_y, box_w, box_h)
        self._box_prog["u_radius"].value = float(config.box_radius * scale)
        self._box_prog["u_color"].value = (
            box_color[0] / 255.0,
            box_color[1] / 255.0,
            box_color[2] / 255.0,
            box_color[3] / 255.0,
        )
        self._box_prog["u_alpha"].value = alpha
        self._box_prog["u_shadow"].value = shadow
        self._box_prog["u_shadow_color"].value = (
            shadow_color[0] / 255.0,
            shadow_color[1] / 255.0,
            shadow_color[2] / 255.0,
            shadow_color[3] / 255.0,
        )
        self._box_prog["u_shadow_alpha"].value = shadow_color[3] / 255.0
        blur = getattr(config, 'compensation_blur', 12.0) * scale
        self._box_prog["u_blur"].value = blur

        self._box_vao.render(mode=moderngl.TRIANGLES)

        # --- Text rendern ---
        text_color = getattr(config, 'font_color', (255, 255, 255))
        text_rgb = (text_color[0] / 255.0, text_color[1] / 255.0, text_color[2] / 255.0)

        text_x = box_x + box_w / 2.0
        text_y = box_y + padding + font_size * 0.85

        glow = getattr(config, 'glow_pulse', False) and alpha < 1.0
        glow_val = 0.3 if glow else 0.0

        align = getattr(config, 'text_align', 'center')

        self._text_renderer.render_multiline_text(
            lines, text_x, text_y,
            line_height=1.4, size=font_size,
            color=text_rgb, alpha=alpha,
            align=align,
            glow=glow_val, glow_color=text_rgb,
            smoothing=0.25,
            outline_width=0.0,
            shadow_offset=(1.0, 1.0),
            shadow_color=(0.0, 0.0, 0.0),
            shadow_alpha=0.4,
        )

        # Blending bleibt aktiv (Aufrufer verwaltet State)

    def _wrap_text(self, text: str, config) -> List[str]:
        """Bricht Text in Zeilen um."""
        import textwrap
        max_chars = config.max_chars_per_line
        return textwrap.wrap(text, width=max_chars, break_long_words=False,
                             replace_whitespace=False)

    def _text_width(self, text: str, size: float) -> float:
        """Berechnet Text-Breite in Pixeln."""
        total = 0.0
        scale = size / self._atlas.sdf_size
        for char in text:
            g = self._atlas.get_glyph(char)
            if g:
                total += g.advance * scale
        return total

    def release(self):
        """Gibt alle GPU-Ressourcen frei."""
        if hasattr(self, '_box_vao') and self._box_vao:
            self._box_vao.release()
            self._box_vao = None
        if hasattr(self, '_box_vbo') and self._box_vbo:
            self._box_vbo.release()
            self._box_vbo = None
        if hasattr(self, '_box_prog') and self._box_prog:
            self._box_prog.release()
            self._box_prog = None
        if hasattr(self, '_text_renderer') and self._text_renderer:
            self._text_renderer.release()
            self._text_renderer = None
