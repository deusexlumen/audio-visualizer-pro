"""
Echter HDR-Bloom-Pass fuer den GPU-Renderer.

Klassische Bloom-Kette:
1. Threshold-Pass mit Soft-Knee: nur helle (HDR-)Bereiche extrahieren
2. Progressive Downsample-Kette (bilinear, halbe Aufloesung pro Stufe)
3. Tent-Upsample mit additiver Akkumulation (weicher, grosser Glow)
4. Additives Compositing auf die HDR-Szene (vor dem Tonemapping)

Zusaetzlich: .cube-LUT-Parser fuer GPU-basiertes Color-Grading.
"""

import numpy as np
import moderngl

from .app_logging import get_logger
from .gpu_visualizers.base import TEXTURED_VERTEX_SHADER, create_textured_quad

logger = get_logger(__name__)


_THRESHOLD_FRAGMENT = """
#version 330
uniform sampler2D u_texture;
uniform float u_threshold;
uniform float u_knee;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec3 c = max(texture(u_texture, v_uv).rgb, 0.0);
    float brightness = max(c.r, max(c.g, c.b));
    // Soft-Knee: weicher Uebergang um den Threshold statt harter Kante
    float knee = max(u_knee, 1e-5);
    float soft = clamp(brightness - u_threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float contribution = max(soft, brightness - u_threshold) / max(brightness, 1e-5);
    f_color = vec4(c * max(contribution, 0.0), 1.0);
}
"""

_DOWNSAMPLE_FRAGMENT = """
#version 330
uniform sampler2D u_texture;
uniform vec2 u_texel;
in vec2 v_uv;
out vec4 f_color;

void main() {
    // 4-Tap-Box-Filter mit Halbpixel-Offsets (nutzt bilineare Filterung)
    vec3 c = vec3(0.0);
    c += texture(u_texture, v_uv + u_texel * vec2(-1.0, -1.0)).rgb;
    c += texture(u_texture, v_uv + u_texel * vec2( 1.0, -1.0)).rgb;
    c += texture(u_texture, v_uv + u_texel * vec2(-1.0,  1.0)).rgb;
    c += texture(u_texture, v_uv + u_texel * vec2( 1.0,  1.0)).rgb;
    f_color = vec4(c * 0.25, 1.0);
}
"""

_UPSAMPLE_FRAGMENT = """
#version 330
uniform sampler2D u_texture;
uniform vec2 u_texel;
uniform float u_intensity;
in vec2 v_uv;
out vec4 f_color;

void main() {
    // 3x3-Tent-Filter fuer weiches Upsampling
    vec3 c = vec3(0.0);
    c += texture(u_texture, v_uv + u_texel * vec2(-1.0,  1.0)).rgb * 1.0;
    c += texture(u_texture, v_uv + u_texel * vec2( 0.0,  1.0)).rgb * 2.0;
    c += texture(u_texture, v_uv + u_texel * vec2( 1.0,  1.0)).rgb * 1.0;
    c += texture(u_texture, v_uv + u_texel * vec2(-1.0,  0.0)).rgb * 2.0;
    c += texture(u_texture, v_uv + u_texel * vec2( 0.0,  0.0)).rgb * 4.0;
    c += texture(u_texture, v_uv + u_texel * vec2( 1.0,  0.0)).rgb * 2.0;
    c += texture(u_texture, v_uv + u_texel * vec2(-1.0, -1.0)).rgb * 1.0;
    c += texture(u_texture, v_uv + u_texel * vec2( 0.0, -1.0)).rgb * 2.0;
    c += texture(u_texture, v_uv + u_texel * vec2( 1.0, -1.0)).rgb * 1.0;
    f_color = vec4(c / 16.0 * u_intensity, 1.0);
}
"""


class BloomPass:
    """Progressive Downsample/Upsample-Bloom-Kette auf HDR-Texturen.

    Der Aufrufer rendert die Szene in ein f16-FBO und ruft danach
    apply(scene_fbo, scene_texture, ...) auf — das Ergebnis wird additiv
    auf die Szene gemischt (vor dem Tonemapping).
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int, levels: int = 5):
        self.ctx = ctx
        self.width = width
        self.height = height

        # Mip-Kette: halbe Aufloesung pro Stufe, mindestens 8px
        self._sizes = []
        w, h = max(width // 2, 1), max(height // 2, 1)
        for _ in range(levels):
            if w < 8 or h < 8:
                break
            self._sizes.append((w, h))
            w, h = max(w // 2, 1), max(h // 2, 1)

        self._textures = []
        self._fbos = []
        for size in self._sizes:
            tex = ctx.texture(size, 4, dtype='f2')
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False
            self._textures.append(tex)
            self._fbos.append(ctx.framebuffer(color_attachments=[tex]))

        self._threshold_prog = ctx.program(
            vertex_shader=TEXTURED_VERTEX_SHADER, fragment_shader=_THRESHOLD_FRAGMENT
        )
        self._down_prog = ctx.program(
            vertex_shader=TEXTURED_VERTEX_SHADER, fragment_shader=_DOWNSAMPLE_FRAGMENT
        )
        self._up_prog = ctx.program(
            vertex_shader=TEXTURED_VERTEX_SHADER, fragment_shader=_UPSAMPLE_FRAGMENT
        )
        self._threshold_vao, self._threshold_vbo = create_textured_quad(ctx, self._threshold_prog)
        self._down_vao, self._down_vbo = create_textured_quad(ctx, self._down_prog)
        self._up_vao, self._up_vbo = create_textured_quad(ctx, self._up_prog)

    def apply(self, scene_fbo, scene_texture, intensity: float = 0.6,
              threshold: float = 1.0, radius: float = 1.0):
        """Berechnet Bloom aus scene_texture und addiert ihn auf scene_fbo.

        Args:
            scene_fbo: Ziel-FBO (HDR-Szene), auf das der Bloom addiert wird.
            scene_texture: Farbtextur der Szene (Input).
            intensity: Staerke des addierten Blooms (0 = aus).
            threshold: Helligkeits-Schwelle (1.0 = nur HDR-Werte >1 leuchten).
            radius: Skaliert die Upsample-Streuung (0.5-2.0).
        """
        if intensity <= 0.0 or not self._fbos:
            return

        # 1. Threshold-Pass in die groesste Mip-Stufe
        self._fbos[0].use()
        self.ctx.disable(moderngl.BLEND)
        self._threshold_prog["u_texture"].value = 0
        self._threshold_prog["u_threshold"].value = float(threshold)
        self._threshold_prog["u_knee"].value = float(threshold) * 0.5
        scene_texture.use(location=0)
        self._threshold_vao.render(mode=moderngl.TRIANGLE_STRIP)

        # 2. Downsample-Kette
        for i in range(1, len(self._fbos)):
            src_size = self._sizes[i - 1]
            self._fbos[i].use()
            self._down_prog["u_texture"].value = 0
            self._down_prog["u_texel"].value = (0.5 / src_size[0], 0.5 / src_size[1])
            self._textures[i - 1].use(location=0)
            self._down_vao.render(mode=moderngl.TRIANGLE_STRIP)

        # 3. Upsample mit additiver Akkumulation (klein -> gross)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self._fbos) - 1, 0, -1):
            src_size = self._sizes[i]
            self._fbos[i - 1].use()
            self._up_prog["u_texture"].value = 0
            self._up_prog["u_texel"].value = (
                float(radius) / src_size[0], float(radius) / src_size[1]
            )
            self._up_prog["u_intensity"].value = 1.0
            self._textures[i].use(location=0)
            self._up_vao.render(mode=moderngl.TRIANGLE_STRIP)

        # 4. Additiv auf die Szene mischen.
        # Normalisierung ueber die Stufenzahl: die Upsample-Kette akkumuliert
        # pro Stufe Energie, ohne Division wuerde Bloom die Szene ueberstrahlen.
        scene_fbo.use()
        self._up_prog["u_texture"].value = 0
        self._up_prog["u_texel"].value = (
            float(radius) / self._sizes[0][0], float(radius) / self._sizes[0][1]
        )
        self._up_prog["u_intensity"].value = float(intensity) / max(len(self._fbos), 1)
        self._textures[0].use(location=0)
        self._up_vao.render(mode=moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def release(self):
        """Gibt alle GPU-Ressourcen der Bloom-Kette frei."""
        for obj in (self._fbos + self._textures +
                    [self._threshold_prog, self._down_prog, self._up_prog,
                     self._threshold_vao, self._threshold_vbo,
                     self._down_vao, self._down_vbo,
                     self._up_vao, self._up_vbo]):
            try:
                obj.release()
            except Exception:
                pass
        self._fbos = []
        self._textures = []


def load_cube_lut(path: str) -> np.ndarray:
    """Laedt eine .cube-LUT-Datei als 3D-Array.

    Args:
        path: Pfad zur .cube-Datei.

    Returns:
        Float32-Array der Shape (N, N, N, 3) mit Werten 0.0-1.0.
        .cube-Reihenfolge: rot variiert am schnellsten -> Array[b][g][r].

    Raises:
        ValueError: Wenn die Datei keine gueltige 3D-LUT enthaelt.
    """
    size = None
    data = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            if upper.startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE")):
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    data.append([float(v) for v in parts])
                except ValueError:
                    continue

    if size is None:
        raise ValueError(f"Keine LUT_3D_SIZE in .cube-Datei gefunden: {path}")
    if len(data) != size ** 3:
        raise ValueError(
            f"Ungueltige .cube-Datei: erwartet {size ** 3} Eintraege, "
            f"gefunden {len(data)}: {path}"
        )

    arr = np.array(data, dtype=np.float32).reshape((size, size, size, 3))
    return np.clip(arr, 0.0, 1.0)
