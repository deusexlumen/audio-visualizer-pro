"""
Speech Focus - Minimalistischer Podcast-Visualizer.

Sehr reduziert, professionell und nie ablenkend.
- Fast schwarzer Hintergrund
- Eine schlanke, horizontale Wellenform-Linie in der Mitte
- Segmentierter VU-Meter am rechten Rand
- Sanfte Reaktion auf Sprache (RMS-gesteuert)
- Dezente Akzentfarbe (soft cyan oder warm amber) nur bei Sprache
- Keine harten Beats, keine Explosionen

Psychologische Vorgabe: Die Visualisierung darf NIEMALS
vom gesprochenen Wort ablenken.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


_VERTEX_SHADER = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_FRAGMENT_SHADER = """
#version 330
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_rms;
uniform float u_line_thickness;
uniform int u_vu_segments;
uniform float u_response_speed;
uniform float u_accent_hue;

out vec4 f_color;

// === Basic utilities inline ===
float remap(float v, float i_min, float i_max, float o_min, float o_max) {
    return o_min + (v - i_min) * (o_max - o_min) / (i_max - i_min + 1e-8);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float aspect = u_resolution.x / u_resolution.y;

    // Sehr dunkler, fast schwarzer Hintergrund
    vec3 bg = vec3(0.012, 0.012, 0.014);
    vec3 col = bg;

    // Akzentfarbe als Hue (0.52 = soft cyan, 0.08 = warm amber)
    vec3 accent = hsv2rgb(vec3(u_accent_hue, 0.55, 0.32));
    vec3 dimAccent = accent * 0.35;

    // Sprach-Gate: Akzent nur bei vorhandener Sprache
    float speech = smoothstep(0.03, 0.10, u_rms);

    // Reaktiver RMS-Wert (durch response_speed skalierbar)
    float reactiveRms = u_rms * u_response_speed;

    // --- Minimale Wellenform-Linie in der Mitte ---
    float centerY = 0.5;
    float wave = sin(uv.x * 8.0 + u_time * 1.2) * reactiveRms * 0.025;
    wave += sin(uv.x * 16.0 - u_time * 0.8) * reactiveRms * 0.012;

    float lineDist = abs(uv.y - (centerY + wave));
    float lineThick = u_line_thickness / u_resolution.y;
    float lineMask = 1.0 - smoothstep(0.0, lineThick, lineDist);

    // Farbe: dezentes Grau bei Stille, sanfter Akzent bei Sprache
    vec3 lineCol = mix(vec3(0.08, 0.08, 0.09), accent, speech * 0.55);
    float lineBright = lineMask * (0.05 + reactiveRms * 0.22);
    lineBright = min(lineBright, 0.35); // Helligkeit gecappt
    col += lineCol * lineBright;

    // --- Sehr feine Hilfslinien (25%, 50%, 75%) ---
    for (int i = 1; i < 4; i++) {
        float gy = float(i) * 0.25;
        float gDist = abs(uv.y - gy);
        float gLine = 1.0 - smoothstep(0.0, 1.0 / u_resolution.y, gDist);
        col += vec3(0.022) * gLine * 0.45;
    }

    // --- VU-Meter am rechten Rand ---
    float vuRight = 0.96;
    float vuW = 0.008;
    float vuH = 0.55;
    float vuBottom = 0.225;
    float vuLeft = vuRight - vuW;

    // Hintergrund-Schiene des VU-Meters
    float inVuX = smoothstep(vuLeft - 0.002, vuLeft, uv.x)
                * smoothstep(vuRight + 0.002, vuRight, uv.x);
    float inVuY = smoothstep(vuBottom - 0.005, vuBottom, uv.y)
                * smoothstep(vuBottom + vuH + 0.005, vuBottom + vuH, uv.y);
    float vuTrack = inVuX * inVuY;
    col += vec3(0.025) * vuTrack;

    // Segmente des VU-Meters
    float segH = vuH / float(u_vu_segments);
    float fillH = reactiveRms * vuH * 0.98;
    float relY = uv.y - vuBottom;

    if (relY > 0.0 && relY < fillH && uv.x > vuLeft && uv.x < vuRight) {
        float segIdx = floor(relY / segH);
        float segFrac = fract(relY / segH);
        float gap = 0.16;
        float segActive = smoothstep(0.0, gap, segFrac)
                        * smoothstep(1.0, 1.0 - gap, segFrac);

        float segBright = 0.08 + reactiveRms * 0.18;
        segBright = min(segBright, 0.32); // Helligkeit gecappt

        // Obere Segmente leuchten etwas staerker
        float segNorm = segIdx / float(u_vu_segments);
        vec3 segCol = mix(dimAccent, accent, segNorm * 0.5 + speech * 0.35);

        col += segCol * segActive * segBright;
    }

    // --- Globales Helligkeits-Cap bei 0.4 ---
    col = clamp(col, 0.0, 0.4);

    // --- Film Grain (extrem subtil) ---
    float grain = (hash(gl_FragCoord.xy + fract(u_time * 73.0) * 100.0) - 0.5) * 0.01;
    col += grain;

    f_color = vec4(col, 1.0);
}
"""


class SpeechFocusGPU(BaseGPUVisualizer):
    """
    Speech Focus - Minimalistischer, podcast-optimierter GPU-Visualizer.

    Sehr dunkler Hintergrund, eine schlanke Wellenform-Linie,
    segmentierter VU-Meter und dezente Akzentfarben nur bei Sprache.
    """

    PARAMS = {
        'line_thickness': (2.0, 0.5, 6.0, 0.5),
        'vu_segments': (12, 4, 24, 1),
        'response_speed': (0.8, 0.2, 1.5, 0.1),
        # Hue-Wert fuer Akzentfarbe: 0.52 = soft cyan, 0.08 = warm amber
        'accent_color': (0.52, 0.0, 1.0, 0.01),
    }

    def _setup(self):
        """Initialisiert Shader, VBO und VAO fuer den Fullscreen-Quad."""
        self.prog = self.ctx.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )
        self.prog["u_resolution"].value = (self.width, self.height)

        # Fullscreen-Quad als Triangle-Strip
        quad = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)

        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f", "in_pos")],
        )

    def render(self, features: dict, time: float):
        """Rendert einen Frame mit aktuellen Audio-Features.

        Args:
            features: Dictionary mit Audio-Feature-Arrays.
            time: Aktuelle Zeit in Sekunden.
        """
        frame_idx = int(time * features.get("fps", 30))
        frame_idx = max(0, min(frame_idx, features.get("frame_count", 0) - 1))

        f = self._get_feature_at_frame(features, frame_idx)
        rms = f["rms"]
        onset = f["onset"]

        self.prog["u_resolution"].value = (self.width, self.height)
        self.prog["u_time"].value = time
        self.prog["u_rms"].value = float(rms)
        self.prog["u_line_thickness"].value = float(self.params["line_thickness"])
        self.prog["u_vu_segments"].value = int(self.params["vu_segments"])
        self.prog["u_response_speed"].value = float(self.params["response_speed"])
        self.prog["u_accent_hue"].value = float(self.params["accent_color"])

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
