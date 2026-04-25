"""
GPU-beschleunigter Pulsing-Core-Visualizer mit ModernGL.

Nutzt einen Fullscreen-Quad und Distance-Field-Rendering im Fragment-Shader.
Der zentrale Kreis pulsiert mit RMS, Ringe reagieren auf Onsets,
und die Farbe aendert sich basierend auf dem dominanten Chroma-Ton.
"""

import numpy as np
import moderngl

from .base import BaseGPUVisualizer


_VERTEX_SHADER = """
#version 330
in vec2 in_position;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330
uniform vec2 u_resolution;
uniform float u_rms;
uniform float u_onset;
uniform vec3 u_color;
uniform float u_pulse_intensity;
uniform int u_ring_count;
uniform float u_glow_radius;
out vec4 f_color;

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);

    float radius = 0.1 + u_rms * 0.15 * u_pulse_intensity;
    float glow = exp(-dist * dist / (radius * radius * 2.0 / u_glow_radius));

    // Konzentrische Ringe
    float ring = 0.0;
    for (int i = 1; i <= 8; i++) {
        if (i > u_ring_count) break;
        float ringRadius = radius + float(i) * 0.06;
        float ringWidth = 0.015;
        float ringGlow = smoothstep(ringRadius + ringWidth, ringRadius, dist)
                       * smoothstep(ringRadius - ringWidth, ringRadius, dist);
        ring += ringGlow * (0.2 + u_onset * 0.4);
    }

    vec3 color = u_color * glow + u_color * ring * u_onset * 0.7;

    // Subtiler Hintergrund-Glow
    float bgGlow = exp(-dist * dist / ((radius + 0.2) * (radius + 0.2) * 3.0)) * u_rms * 0.15;
    color += u_color * bgGlow;

    f_color = vec4(color, 1.0);
}
"""


class PulsingCoreGPU(BaseGPUVisualizer):
    """Pulsing-Core-Visualizer mit Distance-Field-Rendering auf der GPU.

    Ein einzelner Fullscreen-Quad deckt den gesamten Bildschirm ab.
    Alle Formen werden im Fragment-Shader ueber Distanzberechnungen gerendert.
    """

    PARAMS = {
        'pulse_intensity': (1.0, 0.0, 3.0, 0.1),
        'ring_count': (3, 1, 8, 1),
        'glow_radius': (1.0, 0.2, 3.0, 0.1),
        'bg_brightness': (0.05, 0.0, 0.3, 0.01),
    }

    def _setup(self):
        """Initialisiert Shader, VBO und VAO fuer den Fullscreen-Quad."""
        self.prog = self.ctx.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )
        self.prog["u_resolution"].value = (self.width, self.height)

        # Fullscreen-Quad: 4 Vertices, 2 Dreiecke (Clip-Space: -1 bis +1)
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f", "in_position")],
        )

    @staticmethod
    def _chroma_to_color(chroma: np.ndarray) -> tuple:
        """Wandelt ein Chroma-Vektor in eine elegante, gedaempfte RGB-Farbe um."""
        chroma = np.asarray(chroma).flatten()
        if chroma.size < 12:
            chroma = np.pad(chroma, (0, 12 - chroma.size))

        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        x = np.sum(chroma * np.cos(angles))
        y = np.sum(chroma * np.sin(angles))

        hue = np.arctan2(y, x) / (2.0 * np.pi)
        if hue < 0:
            hue += 1.0

        strength = float(np.max(chroma))
        saturation = min(0.35, 0.2 + 0.15 * strength)
        value = min(0.7, 0.5 + 0.2 * strength)

        return PulsingCoreGPU._hsv_to_rgb(hue, saturation, value)

    def render(self, features: dict, time: float):
        """Rendert einen Frame mit aktuellem RMS, Onset und Chroma-Farbe.

        Args:
            features: Dictionary mit Audio-Features fuer alle Frames.
            time: Aktuelle Zeit in Sekunden.
        """
        frame_idx = int(time * features.get("fps", 30))
        frame_idx = max(0, min(frame_idx, features.get("frame_count", 0) - 1))

        f = self._get_feature_at_frame(features, frame_idx)

        rms = f["rms"]
        onset = f["onset"]
        chroma = f["chroma"]

        # Farbe aus dominantem Chroma-Ton ableiten
        color = self._chroma_to_color(chroma)

        # Uniforms aktualisieren
        self.prog["u_rms"].value = float(rms)
        self.prog["u_onset"].value = float(onset)
        self.prog["u_color"].value = color
        self.prog["u_pulse_intensity"].value = float(self.params['pulse_intensity'])
        self.prog["u_ring_count"].value = int(self.params['ring_count'])
        self.prog["u_glow_radius"].value = float(self.params['glow_radius'])

        # Zeichnen
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
