"""
Voice Line - duenner Stimm-Lichtstrich fuer den Folgen-Look (Podcast).

Eine waagrechte Linie nahe der Bildunterkante, deren Helligkeit der Stimme
folgt (voice_band, Rueckfall rms). Zur Mitte hin am hellsten, an den Enden
ausgeblendet; der Rest bleibt schwarz, damit der Compositor das Folgenbild
unveraendert durchlaesst. Nicht fuer Auto-Auswahl gedacht (siehe
AUTO_EXCLUDED_VISUALIZERS).

HDR-Ausgabe ohne lokales Clamp (Tonemapping macht zentral der Renderer).
"""

import moderngl

from .base import (
    BaseGPUVisualizer,
    FULLSCREEN_VERTEX_SHADER,
    compose_fragment,
    create_fullscreen_quad,
)


class VoiceLineGPU(BaseGPUVisualizer):
    """Duenner Lichtstrich, der mit der Stimme atmet."""

    PARAMS = {
        'y_position': (0.07, 0.0, 0.5, 0.01),
        'thickness_px': (2.0, 1.0, 8.0, 0.5),
        'width': (0.88, 0.2, 1.0, 0.01),
        'intensity': (1.0, 0.0, 3.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Linie": ["y_position", "thickness_px", "width", "intensity"],
    }

    COLOR_PARAMS = {
        **BaseGPUVisualizer.COLOR_PARAMS,
        'line_color': '#F2B866',
    }

    def __init__(self, ctx, width, height):
        self._level = 0.0  # geglaettete Stimm-Huelle
        self._last_time = 0.0
        super().__init__(ctx, width, height)

    def _setup(self):
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_y;          // Abstand von der Video-Unterkante (0..1)
            uniform float u_thickness;  // Pixel
            uniform float u_width;      // Anteil der Bildbreite
            uniform float u_level;      // Stimm-Huelle 0..1
            uniform float u_intensity;
            uniform vec3 u_color;
            out vec4 f_color;

            void main() {
                // gl_FragCoord.y = 0 ist die Video-Oberkante (siehe Readback)
                float line_y = (1.0 - u_y) * u_resolution.y;
                float d = abs(gl_FragCoord.y - line_y) / max(u_thickness, 0.5);
                float core = exp(-d * d);
                float glow = 0.3 * exp(-d * d / 16.0);
                float x = abs(gl_FragCoord.x / u_resolution.x - 0.5);
                float half_w = 0.5 * u_width;
                float ends = 1.0 - smoothstep(half_w * 0.55, half_w, x);
                float gain = (0.25 + 1.6 * u_level) * u_intensity;
                f_color = vec4(u_color * (core + glow) * ends * gain, 1.0);
            }
            """
        )
        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        voice = float(f.get("voice_band", f.get("rms", 0.0)))

        # Kurze Glaettung, damit die Linie atmet statt zu flackern;
        # bei Spruengen (Vorschau, Neustart) direkt auf den Wert setzen
        dt = time - self._last_time
        self._last_time = time
        if 0.0 < dt <= 0.1:
            self._level += (voice - self._level) * min(1.0, dt * 12.0)
        else:
            self._level = voice

        p = self.params
        color = p.get("line_color", "#F2B866")
        try:
            rgb = self._hex_to_rgb(color)
        except Exception:
            rgb = self._hex_to_rgb("#F2B866")

        prog = self.prog
        prog["u_resolution"].value = (self.width, self.height)
        prog["u_y"].value = float(p["y_position"])
        prog["u_thickness"].value = float(p["thickness_px"])
        prog["u_width"].value = float(p["width"])
        prog["u_level"].value = float(self._level)
        prog["u_intensity"].value = float(p["intensity"])
        prog["u_color"].value = rgb
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
