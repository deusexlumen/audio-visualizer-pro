"""
GPU-beschleunigter Bass Temple Visualizer mit ModernGL.

Dunkler, aggressiver Fullscreen-Fragment-Shader fuer Dubstep/Deep Bass.
Zentrale Tempel-SDF-Form pulst mit Bass, vertikale Bass-Balken reagieren
auf RMS, Stroboskop-Flashes bei Onsets, Shockwave-Ringe expandieren
vom Zentrum, und chromatische Aberration bei starken Beats.

Farbpalette: Deep Purple, Crimson Red, Near-Black.
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
uniform float u_time;
uniform float u_rms;
uniform float u_onset;
uniform float u_beat_intensity;
uniform float u_spectral_centroid;
uniform vec3 u_chroma_color;
uniform float u_bass_intensity;
uniform float u_strobe_threshold;
uniform float u_color_shift;
uniform float u_shockwave_speed;

out vec4 f_color;

// === Grundlegende Utilities (inline) ===
float remap(float v, float i_min, float i_max, float o_min, float o_max) {
    return o_min + (v - i_min) * (o_max - o_min) / (i_max - i_min + 1e-8);
}

mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float hash(float n) { return fract(sin(n) * 43758.5453123); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float noise(float x) {
    float i = floor(x);
    float f = fract(x);
    return mix(hash(i), hash(i + 1.0), smoothstep(0.0, 1.0, f));
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p, int octaves) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < octaves; i++) {
        v += a * noise(p);
        p = rot * p * 2.0 + vec2(100.0);
        a *= 0.5;
    }
    return v;
}

// === SDF-Hilfsfunktionen ===
float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdHexagon(vec2 p, float r) {
    vec2 q = abs(p);
    return max(q.x * 0.866025 + q.y * 0.5, q.y) - r;
}

// === Tempel-SDF ===
float templeSDF(vec2 p, float pulse) {
    // Zentraler hexagonaler Monolith
    float d = sdHexagon(p * vec2(1.0, 0.58), 0.18 * pulse);

    // Dachkappe
    float cap = sdBox(p - vec2(0.0, 0.13 * pulse), vec2(0.19 * pulse, 0.012));
    d = min(d, cap);

    // Sockel
    float base = sdBox(p + vec2(0.0, 0.15 * pulse), vec2(0.21 * pulse, 0.012));
    d = min(d, base);

    // Seitenpfeiler
    float pillarL = sdBox(p - vec2(-0.24 * pulse, 0.0), vec2(0.025, 0.18) * pulse);
    float pillarR = sdBox(p - vec2( 0.24 * pulse, 0.0), vec2(0.025, 0.18) * pulse);
    d = min(d, pillarL);
    d = min(d, pillarR);

    // Innere Tuer (subtrahiert)
    float door = sdBox(p - vec2(0.0, -0.04 * pulse), vec2(0.04 * pulse, 0.08 * pulse));
    d = max(d, -door);

    // Horizontale Fugen fuer architektonische Details
    float groove1 = sdBox(p - vec2(0.0,  0.05 * pulse), vec2(0.14 * pulse, 0.003));
    float groove2 = sdBox(p - vec2(0.0, -0.06 * pulse), vec2(0.14 * pulse, 0.003));
    d = max(d, -groove1);
    d = max(d, -groove2);

    return d;
}

// === Vertikale Bass-Balken ===
float bassBars(vec2 p, float rms, float intensity) {
    float bar = 1e6;
    float barCount = 14.0;
    float w = 1.0 / barCount;

    for (float i = 0.0; i < barCount; i += 1.0) {
        float x = (i + 0.5) * w;
        // Hoehe skaliert mit RMS und leichter Wellen-Variation
        float h = rms * intensity * (0.4 + 0.6 * abs(sin(i * 1.3 + u_time * 3.0)));
        h = max(h, 0.005);
        float bx = abs(p.x - x);
        float by = p.y - h * 0.5;
        float b = sdBox(vec2(bx, by), vec2(w * 0.35, h * 0.5));
        bar = min(bar, b);
    }
    return bar;
}

// === Shockwave-Ringe ===
float shockwave(vec2 p, float time, float speed, float intensity) {
    float d = length(p);
    float sw = 0.0;
    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        float t = time * speed + fi * 1.57;
        float phase = fract(t);
        float radius = phase * 0.75;
        float strength = exp(-phase * 5.0) * intensity;
        float ring = smoothstep(radius + 0.018, radius, d)
                   * smoothstep(radius - 0.018, radius, d);
        sw += ring * strength;
    }
    return sw;
}

// === Szene ===
vec3 scene(vec2 uv, vec2 uv_full, float time) {
    // Dunkler Hintergrund mit subtilem Purpur-Stich
    vec3 col = vec3(0.008, 0.003, 0.012);

    // Hintergrund-Rauschen via FBM
    float bgNoise = fbm(uv * 2.5 + time * 0.12, 4);
    col += vec3(0.02, 0.0, 0.035) * bgNoise;

    // Vignette
    float vig = 1.0 - smoothstep(0.25, 1.1, length(uv));
    col *= 0.4 + 0.6 * vig;

    // Farbpalette: Deep Purple, Crimson, Near-Black
    vec3 purple = vec3(0.12, 0.0, 0.22);
    vec3 crimson = vec3(0.55, 0.0, 0.08);
    vec3 black = vec3(0.0, 0.0, 0.0);

    // Highlight-Farbe mischt zwischen Palette und Chroma-Farbe
    vec3 highlight = mix(purple, crimson, u_rms);
    highlight = mix(highlight, u_chroma_color, u_color_shift * 0.5);

    // Puls-Faktor aus RMS und Beat-Intensitaet
    float pulse = 1.0 + u_rms * u_bass_intensity * 0.35 + u_beat_intensity * 0.15;

    // --- Tempel ---
    float temple = templeSDF(uv, pulse);
    float templeMask = smoothstep(0.006, -0.006, temple);
    vec3 templeCol = mix(black, highlight, 0.5 + u_rms * 0.5);

    // Tempel-Kanten-Glow
    float templeEdge = exp(-abs(temple) * 90.0) * (u_rms * 2.5 + u_beat_intensity * 1.5);
    col += mix(crimson, highlight, 0.5) * templeEdge;

    col = mix(col, templeCol, templeMask);

    // Inneres Tempel-Glow (durch die Tuer)
    float innerGlow = exp(-length(uv) * length(uv) * 8.0) * u_rms * 0.6;
    col += crimson * innerGlow;

    // --- Bass-Balken (links und rechts) ---
    float barIntensity = u_bass_intensity * 0.4;
    float leftBars = bassBars(vec2(fract(uv_full.x * 2.857), uv_full.y * 0.5), u_rms, barIntensity);
    float rightBars = bassBars(vec2(fract((uv_full.x - 0.65) / 0.35), uv_full.y * 0.5), u_rms, barIntensity);

    float barMaskL = smoothstep(0.006, -0.006, leftBars) * step(uv_full.x, 0.35);
    float barMaskR = smoothstep(0.006, -0.006, rightBars) * step(0.65, uv_full.x);

    // Balken-Glow
    float barGlowL = exp(-abs(leftBars) * 50.0) * step(uv_full.x, 0.35);
    float barGlowR = exp(-abs(rightBars) * 50.0) * step(0.65, uv_full.x);
    col += highlight * (barGlowL + barGlowR) * 0.4;

    col = mix(col, highlight * 0.85, barMaskL);
    col = mix(col, highlight * 0.85, barMaskR);

    // --- Shockwave-Ringe ---
    float sw = shockwave(uv, time, u_shockwave_speed, u_beat_intensity + u_onset * 0.5);
    col += vec3(0.45, 0.04, 0.15) * sw;

    // --- Stroboskop-Flash ---
    float strobe = step(u_strobe_threshold, u_onset);
    col = mix(col, vec3(0.92, 0.92, 1.0), strobe * 0.55);

    // --- Spektrale Funken ---
    float sparkles = hash(uv * 120.0 + time * 12.0);
    sparkles = pow(sparkles, 25.0) * u_spectral_centroid * 3.0;
    col += vec3(0.35, 0.08, 0.45) * sparkles;

    return col;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
    vec2 uv_full = gl_FragCoord.xy / u_resolution.xy;

    // Chromatische Aberration auf starken Beats
    float caStrength = smoothstep(0.35, 0.65, u_onset) * u_beat_intensity;
    vec2 caOffset = vec2(caStrength * 0.004, 0.0);

    vec3 col;
    col.r = scene(uv + caOffset, uv_full, u_time).r;
    col.g = scene(uv,           uv_full, u_time).g;
    col.b = scene(uv - caOffset, uv_full, u_time).b;

    // Kontrast-Boost bei Beat
    col = pow(col, vec3(0.88));
    col *= 1.0 + u_rms * 0.35 + u_beat_intensity * 0.2;

    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


class BassTempleGPU(BaseGPUVisualizer):
    """Dunkler, aggressiver GPU-Visualizer fuer Dubstep/Deep Bass.

    Rendert eine zentrale, bass-reaktive Tempel-SDF-Form mit vertikalen
    Bass-Balken, Stroboskop-Flashes, expandierenden Shockwave-Ringen
    und chromatischer Aberration bei starken Beats.
    """

    PARAMS = {
        'bass_intensity': (1.2, 0.0, 3.0, 0.1),
        'strobe_threshold': (0.55, 0.0, 1.0, 0.05),
        'color_shift': (0.0, 0.0, 1.0, 0.05),
        'shockwave_speed': (2.5, 0.5, 6.0, 0.1),
    }

    def _setup(self):
        """Initialisiert Shader, VBO und VAO fuer den Fullscreen-Quad."""
        self.prog = self.ctx.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )
        self.prog["u_resolution"].value = (self.width, self.height)

        # Fullscreen-Quad: 4 Vertices als Triangle-Strip (Clip-Space -1 bis +1)
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

    def render(self, features: dict, time: float):
        """Rendert einen Frame mit aktuellen Audio-Features.

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
        spectral_centroid = f["spectral_centroid"]

        # beat_intensity aus Features oder als Fallback aus Onset berechnen
        beat_intensity = features.get("beat_intensity", None)
        if beat_intensity is not None and len(np.atleast_1d(beat_intensity)) > frame_idx:
            beat_intensity = float(np.atleast_1d(beat_intensity)[frame_idx])
        else:
            beat_intensity = min(onset * 1.5, 1.0)

        # Farbe aus dominantem Chroma-Ton ableiten
        chroma_color = self._chroma_to_color(chroma)

        # Uniforms aktualisieren
        self.prog["u_time"].value = float(time)
        self.prog["u_rms"].value = float(rms)
        self.prog["u_onset"].value = float(onset)
        self.prog["u_beat_intensity"].value = float(beat_intensity)
        self.prog["u_spectral_centroid"].value = float(spectral_centroid)
        self.prog["u_chroma_color"].value = chroma_color
        self.prog["u_bass_intensity"].value = float(self.params['bass_intensity'])
        self.prog["u_strobe_threshold"].value = float(self.params['strobe_threshold'])
        self.prog["u_color_shift"].value = float(self.params['color_shift'])
        self.prog["u_shockwave_speed"].value = float(self.params['shockwave_speed'])

        # Zeichnen
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
