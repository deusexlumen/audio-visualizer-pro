"""
GPU-beschleunigter Pulsing-Core-Visualizer mit ModernGL.

Nutzt einen Fullscreen-Quad und Distance-Field-Rendering im Fragment-Shader.
Der zentrale Kreis pulsiert mit RMS, Ringe reagieren auf Onsets,
und die Farbe aendert sich basierend auf dem aktuellen color_mode.
"""

import numpy as np
import moderngl

from .base import (
    BaseGPUVisualizer,
    FULLSCREEN_VERTEX_SHADER,
    create_fullscreen_quad,
)


_FRAGMENT_SHADER = """
#version 330
uniform vec2 u_resolution;
uniform float u_rms;
uniform float u_onset;
uniform float u_beat_intensity;
uniform vec3 u_color;
uniform float u_pulse_intensity;
uniform float u_base_radius;
uniform int u_ring_count;
uniform float u_ring_spacing;
uniform float u_ring_width;
uniform float u_glow_radius;
uniform float u_trail_length;
uniform float u_trail_decay;
uniform float u_bg_brightness;
uniform float u_brightness;
out vec4 f_color;

void main() {
    // Aspektkorrektur: Kreise bleiben rund, unabhaengig von der Aufloesung
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv * aspect, center * aspect);

    float radius = u_base_radius + u_rms * 0.15 * u_pulse_intensity;
    float glow = exp(-dist * dist / (radius * radius * 2.0 / u_glow_radius));

    // Konzentrische Ringe
    float ring = 0.0;
    for (int i = 1; i <= 8; i++) {
        if (i > u_ring_count) break;
        float ringRadius = radius + float(i) * u_ring_spacing;
        float ringWidth = u_ring_width;
        float ringGlow = smoothstep(ringRadius + ringWidth, ringRadius, dist)
                       * smoothstep(ringRadius - ringWidth, ringRadius, dist);
        ring += ringGlow * (0.2 + max(u_onset, u_beat_intensity) * 0.4);
    }

    vec3 color = u_color * glow + u_color * ring * u_onset * 0.7;

    // Trail-Echo-Ringe
    int trails = int(u_trail_length);
    for (int t = 1; t <= 8; t++) {
        if (t > trails) break;
        float trailFade = pow(u_trail_decay, float(t));
        float trailRadius = max(0.02, radius - float(t) * 0.03);
        float trailGlow = exp(-dist * dist / (trailRadius * trailRadius * 2.0 / u_glow_radius));
        color += u_color * trailGlow * 0.12 * trailFade;
    }

    // Subtiler Hintergrund-Glow
    float bgGlow = exp(-dist * dist / ((radius + 0.2) * (radius + 0.2) * 3.0)) * u_rms * u_bg_brightness;
    color += u_color * bgGlow;

    f_color = vec4(color * u_brightness, 1.0);
}
"""


class PulsingCoreGPU(BaseGPUVisualizer):
    """Pulsing-Core-Visualizer mit Distance-Field-Rendering auf der GPU.

    Ein einzelner Fullscreen-Quad deckt den gesamten Bildschirm ab.
    Alle Formen werden im Fragment-Shader ueber Distanzberechnungen gerendert.
    """

    PARAMS = {
        'pulse_intensity': (1.0, 0.0, 3.0, 0.1),
        'base_radius': (0.1, 0.02, 0.3, 0.01),
        'ring_count': (3, 1, 8, 1),
        'ring_spacing': (0.06, 0.02, 0.15, 0.01),
        'ring_width': (0.015, 0.005, 0.05, 0.005),
        'glow_radius': (1.0, 0.2, 3.0, 0.1),
        'bg_brightness': (0.15, 0.0, 0.5, 0.01),
    }

    PARAMS_GROUPS = {
        "Puls": ["pulse_intensity", "base_radius"],
        "Ringe": ["ring_count", "ring_spacing", "ring_width"],
        "Erscheinungsbild": ["glow_radius", "bg_brightness"],
    }

    def _setup(self):
        """Initialisiert Shader, VBO und VAO fuer den Fullscreen-Quad."""
        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

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
        beat_intensity = f.get("beat_intensity", onset)
        chroma = f["chroma"]

        # Farbe aus dem konfigurierten color_mode ableiten
        color = self._chroma_to_color(chroma)

        # Uniforms aktualisieren
        self.prog["u_rms"].value = float(rms)
        self.prog["u_onset"].value = float(onset)
        self.prog["u_beat_intensity"].value = float(beat_intensity)
        self.prog["u_color"].value = color
        self.prog["u_pulse_intensity"].value = float(self.params['pulse_intensity'])
        self.prog["u_base_radius"].value = float(self.params['base_radius'])
        self.prog["u_ring_count"].value = int(self.params['ring_count'])
        self.prog["u_ring_spacing"].value = float(self.params['ring_spacing'])
        self.prog["u_ring_width"].value = float(self.params['ring_width'])
        self.prog["u_glow_radius"].value = float(self.params['glow_radius'])
        self.prog["u_trail_length"].value = float(self.params.get('trail_length', 0))
        self.prog["u_trail_decay"].value = float(self.params.get('trail_decay', 0.7))
        self.prog["u_bg_brightness"].value = float(self.params['bg_brightness'])
        self.prog["u_brightness"].value = float(self.params.get('brightness', 1.0))

        # Zeichnen
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
