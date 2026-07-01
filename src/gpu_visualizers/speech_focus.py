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
uniform vec3 u_accent_color;
uniform vec3 u_background_color;
uniform float u_brightness;
uniform float u_wave_amp;
uniform float u_line_brightness;
uniform float u_accent_intensity;
uniform float u_grain_amount;
uniform float u_brightness_cap;

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

    vec3 bg = u_background_color;
    vec3 col = bg;

    vec3 accent = u_accent_color * u_brightness;
    vec3 dimAccent = accent * 0.35;

    // Sprach-Gate: Akzent nur bei vorhandener Sprache
    float speech = smoothstep(0.03, 0.10, u_rms);

    // Reaktiver RMS-Wert (durch response_speed skalierbar)
    float reactiveRms = u_rms * u_response_speed;

    // --- Minimale Wellenform-Linie in der Mitte ---
    float centerY = 0.5;
    float wave = sin(uv.x * 8.0 + u_time * 1.2) * reactiveRms * u_wave_amp;
    wave += sin(uv.x * 16.0 - u_time * 0.8) * reactiveRms * (u_wave_amp * 0.5);

    float lineDist = abs(uv.y - (centerY + wave));
    float lineThick = u_line_thickness / u_resolution.y;
    float lineMask = 1.0 - smoothstep(0.0, lineThick, lineDist);

    // Farbe: dezentes Grau bei Stille, sanfter Akzent bei Sprache
    vec3 lineCol = mix(vec3(0.08, 0.08, 0.09), accent, speech * u_accent_intensity);
    float lineBright = lineMask * u_line_brightness * (0.2 + reactiveRms * 1.1);
    lineBright = min(lineBright, u_brightness_cap * 0.9);
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
        segBright = min(segBright, u_brightness_cap * 0.8);

        // Obere Segmente leuchten etwas staerker
        float segNorm = segIdx / float(u_vu_segments);
        vec3 segCol = mix(dimAccent, accent, segNorm * 0.5 + speech * 0.35);

        col += segCol * segActive * segBright;
    }

    // --- Globales Helligkeits-Cap ---
    col = clamp(col, 0.0, u_brightness_cap);

    // --- Film Grain (subtil) ---
    float grain = (hash(gl_FragCoord.xy + fract(u_time * 73.0) * 100.0) - 0.5) * u_grain_amount;
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
        # Farb-Modus wird ueber color_mode / primary_color / secondary_color gesteuert
        'wave_amp': (0.025, 0.0, 0.08, 0.005),
        'line_brightness': (0.18, 0.05, 0.5, 0.01),
        'accent_intensity': (0.55, 0.0, 1.0, 0.05),
        'grain_amount': (0.01, 0.0, 0.05, 0.005),
        'brightness_cap': (0.4, 0.1, 0.8, 0.05),
        # Hintergrundfarbe als Hex-String (Tupel-Form noetig wegen PARAMS-Merge)
        'background_color': ('#060607',),
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
        chroma = f["chroma"]

        # Farben ueber den gemeinsamen color_mode erzeugen
        accent_rgb = self._chroma_to_color(chroma)
        bg_hex = self.params.get("background_color", "#060607")
        if isinstance(bg_hex, str) and bg_hex.startswith("#"):
            bg_rgb = self._hex_to_rgb(bg_hex)
        else:
            bg_rgb = (0.024, 0.024, 0.027)

        self.prog["u_resolution"].value = (self.width, self.height)
        self.prog["u_time"].value = time
        self.prog["u_rms"].value = float(rms)
        self.prog["u_line_thickness"].value = float(self.params["line_thickness"])
        self.prog["u_vu_segments"].value = int(self.params["vu_segments"])
        self.prog["u_response_speed"].value = float(self.params["response_speed"])
        self.prog["u_accent_color"].value = accent_rgb
        self.prog["u_background_color"].value = bg_rgb
        self.prog["u_brightness"].value = float(self.params.get("brightness", 1.0))
        self.prog["u_wave_amp"].value = float(self.params["wave_amp"])
        self.prog["u_line_brightness"].value = float(self.params["line_brightness"])
        self.prog["u_accent_intensity"].value = float(self.params["accent_intensity"])
        self.prog["u_grain_amount"].value = float(self.params["grain_amount"])
        self.prog["u_brightness_cap"].value = float(self.params["brightness_cap"])

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
