"""
Spiral-Zoom-Pass (Droste/Escher) fuer den GPU-Renderer.

Verbiegt eine HDR-Ebene (im Renderer: die Visualizer-Ebene vor dem Blit
ueber den Hintergrund) in
logarithmisch-polaren Koordinaten. Mathematik und CPU-Gegenstueck stehen in
src/spiral_zoom.py — dieser Shader muss spiral_map() exakt folgen
(tests/test_gpu_spiral.py vergleicht beide).

Ablauf pro Frame:
1. Ebene in eine eigene Textur kopieren (aus der Ziel-Textur kann man nicht
   gleichzeitig lesen) und Mipmaps bauen — tiefe Ebenen werden stark
   verkleinert und wuerden ohne Mipmaps flimmern.
2. Vollbild-Pass zurueck in die Ebene, Blending aus (ueberschreiben).
   Alpha wird mit verbogen (Luma-/Occlusion-Alpha im Blit bleibt gueltig).

HDR bleibt erhalten: kein clamp, kein sRGB — Tonemapping macht zentral der
Renderer.
"""

import moderngl

from .app_logging import get_logger
from .gpu_visualizers.base import TEXTURED_VERTEX_SHADER, create_textured_quad
from .spiral_zoom import SpiralSettings, spiral_uniforms

logger = get_logger(__name__)


_SPIRAL_FRAGMENT = """
#version 330
uniform sampler2D u_src;
uniform vec2 u_size;
uniform float u_r_out;
uniform vec2 u_m;
uniform vec2 u_per;
uniform float u_ln_r;
uniform float u_lf;
uniform float u_u;
uniform float u_mix;
in vec2 v_uv;
out vec4 f_color;

// Abtastung auf Radius R_out * exp(ln_rad) und Winkel ang um die Bildmitte.
// Die Mip-Stufe folgt dem Massstab der Abbildung (Quell-Texel pro Pixel).
vec4 sample_at(float ln_rad, float ang, float jac) {
    vec2 q = 0.5 * u_size + u_r_out * exp(ln_rad) * vec2(cos(ang), sin(ang));
    return textureLod(u_src, q / u_size, log2(max(jac, 1e-6)));
}

void main() {
    vec4 orig = textureLod(u_src, v_uv, 0.0);
    // Radien relativ zu R_out (aufloesungsunabhaengig, auch mit Spiralarmen)
    vec2 z = (v_uv * u_size - 0.5 * u_size) / u_r_out;
    float zr = max(length(z), 1e-6);
    vec2 w = vec2(log(zr), atan(z.y, z.x));
    // Bildschirm -> Quelle (komplexe Multiplikation mit M), dann Phase abziehen
    vec2 ws = vec2(w.x * u_m.x - w.y * u_m.y, w.x * u_m.y + w.y * u_m.x) - u_u * u_per;
    // Um ganze Ebenen verschieben, bis der Punkt im Quell-Ring liegt
    float k = -floor((ws.x - u_ln_r - u_lf) / u_per.x);
    vec2 p = ws + k * u_per;
    float m_abs = length(u_m);
    vec4 col = sample_at(p.x, p.y, exp(p.x) * m_abs / zr);
    // An der Innenkante des Rings weich in die naechste Ebene ueberblenden
    float w_in = 1.0 - smoothstep(u_lf, 0.0, p.x - u_ln_r);
    if (w_in > 0.0) {
        float ln_in = p.x + u_per.x;
        col = mix(col, sample_at(ln_in, p.y + u_per.y, exp(ln_in) * m_abs / zr), w_in);
    }
    f_color = mix(orig, col, u_mix);
}
"""


class SpiralZoomPass:
    """Droste/Escher-Verbiegung einer HDR-Ebene (in place).

    Der Aufrufer rendert die Ebene in ein f16-FBO und ruft danach
    apply(fbo, settings, u) auf — im Renderer vor dem Visualizer-Blit.
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        self._src = ctx.texture((width, height), 4, dtype="f2")
        self._src.repeat_x = False
        self._src.repeat_y = False
        # Kopie FBO -> FBO: eine Kopie direkt in die Textur klemmt HDR auf 1.0
        self._src_fbo = ctx.framebuffer(color_attachments=[self._src])
        self._prog = ctx.program(
            vertex_shader=TEXTURED_VERTEX_SHADER, fragment_shader=_SPIRAL_FRAGMENT
        )
        self._vao, self._vbo = create_textured_quad(ctx, self._prog)

    def apply(self, scene_fbo, settings: SpiralSettings, u: float):
        """Verbiegt scene_fbo in place; tut nichts, wenn der Effekt aus ist."""
        if not settings.is_active:
            return
        self.ctx.copy_framebuffer(self._src_fbo, scene_fbo)
        self._src.build_mipmaps()
        self._src.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        scene_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        for name, value in spiral_uniforms(settings, self.width, self.height, u).items():
            self._set(name, value)
        self._set("u_src", 0)
        self._src.use(location=0)
        self._vao.render(mode=moderngl.TRIANGLE_STRIP)

    def _set(self, name: str, value):
        # Vom Treiber wegoptimierte Uniforms fehlen im Programm (KeyError)
        try:
            self._prog[name].value = value
        except KeyError:
            pass

    def release(self):
        """Gibt alle GPU-Ressourcen des Passes frei."""
        for obj in (self._src_fbo, self._src, self._prog, self._vao, self._vbo):
            try:
                obj.release()
            except Exception:
                pass
