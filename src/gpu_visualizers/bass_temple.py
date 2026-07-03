"""
GPU-beschleunigter Bass Temple Visualizer mit ModernGL.

Dunkler, aggressiver Fullscreen-Fragment-Shader fuer Dubstep/Deep Bass.
Zentrale Tempel-SDF-Form pulst mit Bass, vertikale Bass-Balken reagieren
auf RMS, Stroboskop-Flashes bei Onsets, Shockwave-Ringe expandieren
vom Zentrum, und chromatische Aberration bei starken Beats.

Farbpalette: Respektiert color_mode ueber u_primary_color / u_secondary_color
/ u_bg_color statt hartkodierter Werte.
"""

import numpy as np
import moderngl

from .base import (
    BaseGPUVisualizer,
    FULLSCREEN_VERTEX_SHADER,
    LYGIA_MATH_GLSL,
    LYGIA_NOISE_GLSL,
    LYGIA_SDF_GLSL,
    SHADER_COMMON_GLSL,
    compose_fragment,
    create_fullscreen_quad,
)


_FRAGMENT_BODY = """
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_rms;
uniform float u_onset;
uniform float u_beat_intensity;
uniform float u_spectral_centroid;
uniform vec3 u_chroma_color;
uniform vec3 u_primary_color;
uniform vec3 u_secondary_color;
uniform vec3 u_bg_color;
uniform vec3 u_flash_color;
uniform vec3 u_sparkle_color;
uniform float u_bass_intensity;
uniform float u_strobe_threshold;
uniform float u_strobe_intensity;
uniform float u_color_shift;
uniform float u_shockwave_speed;
uniform float u_temple_scale;
uniform float u_bar_count;
uniform float u_contrast_gamma;
uniform float u_vignette_strength;
uniform float u_noise_intensity;
uniform float u_sparkle_intensity;

out vec4 f_color;

// === SDF-Hilfsfunktionen (nicht in der Lygia-Bibliothek) ===
float sdHexagon(vec2 p, float r) {
    vec2 q = abs(p);
    return max(q.x * 0.866025 + q.y * 0.5, q.y) - r;
}

// === Tempel-SDF ===
float templeSDF(vec2 p, float pulse) {
    float scale = u_temple_scale;
    // Zentraler hexagonaler Monolith
    float d = sdHexagon(p * vec2(1.0, 0.58), 0.18 * pulse * scale);

    // Dachkappe
    float cap = sdBox(p - vec2(0.0, 0.13 * pulse * scale), vec2(0.19 * pulse * scale, 0.012));
    d = min(d, cap);

    // Sockel
    float base = sdBox(p + vec2(0.0, 0.15 * pulse * scale), vec2(0.21 * pulse * scale, 0.012));
    d = min(d, base);

    // Seitenpfeiler
    float pillarL = sdBox(p - vec2(-0.24 * pulse * scale, 0.0), vec2(0.025, 0.18) * pulse * scale);
    float pillarR = sdBox(p - vec2( 0.24 * pulse * scale, 0.0), vec2(0.025, 0.18) * pulse * scale);
    d = min(d, pillarL);
    d = min(d, pillarR);

    // Innere Tuer (subtrahiert)
    float door = sdBox(p - vec2(0.0, -0.04 * pulse * scale), vec2(0.04 * pulse * scale, 0.08 * pulse * scale));
    d = max(d, -door);

    // Horizontale Fugen fuer architektonische Details
    float groove1 = sdBox(p - vec2(0.0,  0.05 * pulse * scale), vec2(0.14 * pulse * scale, 0.003));
    float groove2 = sdBox(p - vec2(0.0, -0.06 * pulse * scale), vec2(0.14 * pulse * scale, 0.003));
    d = max(d, -groove1);
    d = max(d, -groove2);

    return d;
}

// === Vertikale Bass-Balken ===
float bassBars(vec2 p, float rms, float intensity) {
    float bar = 1e6;
    float barCount = u_bar_count;
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
    // Hintergrund aus color_mode
    vec3 col = u_bg_color;

    // Hintergrund-Rauschen via FBM
    float bgNoise = fbm(uv * 2.5 + time * 0.12, 4);
    col += mix(u_bg_color, u_primary_color, 0.3) * bgNoise * u_noise_intensity;

    // Vignette
    float vig = 1.0 - smoothstep(0.25, 1.1, length(uv));
    col *= 1.0 - u_vignette_strength * (1.0 - vig);

    // Dunkle Basis- und Highlight-Farbe
    vec3 dark = u_bg_color * 0.2;

    // Highlight-Farbe mischt zwischen primaerer und sekundaerer Farbe
    vec3 highlight = mix(u_primary_color, u_secondary_color, u_rms);
    highlight = mix(highlight, u_chroma_color, u_color_shift * 0.5);

    // Puls-Faktor aus RMS und Beat-Intensitaet
    float pulse = 1.0 + u_rms * u_bass_intensity * 0.35 + u_beat_intensity * 0.15;

    // --- Tempel ---
    float temple = templeSDF(uv, pulse);
    // Pixelgenaue weiche Kante (fwidth-basiert, aufloesungsunabhaengig)
    float templeMask = aafill(temple);
    vec3 templeCol = mix(dark, highlight, 0.5 + u_rms * 0.5);

    // Tempel-Kanten-Glow
    float templeEdge = exp(-abs(temple) * 90.0) * (u_rms * 2.5 + u_beat_intensity * 1.5);
    col += mix(u_secondary_color, highlight, 0.5) * templeEdge;

    col = mix(col, templeCol, templeMask);

    // Inneres Tempel-Glow (durch die Tuer)
    float innerGlow = exp(-length(uv) * length(uv) * 8.0) * u_rms * 0.6;
    col += u_secondary_color * innerGlow;

    // --- Bass-Balken (links und rechts) ---
    float barIntensity = u_bass_intensity * 0.4;
    float leftBars = bassBars(vec2(fract(uv_full.x * 2.857), uv_full.y * 0.5), u_rms, barIntensity);
    float rightBars = bassBars(vec2(fract((uv_full.x - 0.65) / 0.35), uv_full.y * 0.5), u_rms, barIntensity);

    float barMaskL = aafill(leftBars) * step(uv_full.x, 0.35);
    float barMaskR = aafill(rightBars) * step(0.65, uv_full.x);

    // Balken-Glow
    float barGlowL = exp(-abs(leftBars) * 50.0) * step(uv_full.x, 0.35);
    float barGlowR = exp(-abs(rightBars) * 50.0) * step(0.65, uv_full.x);
    col += highlight * (barGlowL + barGlowR) * 0.4;

    col = mix(col, highlight * 0.85, barMaskL);
    col = mix(col, highlight * 0.85, barMaskR);

    // --- Shockwave-Ringe ---
    float sw = shockwave(uv, time, u_shockwave_speed, u_beat_intensity + u_onset * 0.5);
    col += u_secondary_color * sw;

    // --- Stroboskop-Flash ---
    float strobe = step(u_strobe_threshold, u_onset);
    col = mix(col, u_flash_color, strobe * u_strobe_intensity);

    // --- Spektrale Funken ---
    float sparkles = hash(uv * 120.0 + time * 12.0);
    sparkles = pow(sparkles, 25.0) * u_spectral_centroid * 3.0;
    col += u_sparkle_color * sparkles * u_sparkle_intensity;

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

    // Kontrast-Boost bei Beat (Gamma, per Parameter)
    col = pow(max(col, 0.0), vec3(u_contrast_gamma));
    col *= 1.0 + u_rms * 0.35 + u_beat_intensity * 0.2;

    // HDR-Ausgabe: kein Clamp mehr — Highlight-Kompression uebernimmt
    // zentral der ACES-Pass im Renderer.
    f_color = vec4(max(col, 0.0), 1.0);
}
"""

_FRAGMENT_SHADER = compose_fragment(
    _FRAGMENT_BODY,
    includes=(LYGIA_MATH_GLSL, LYGIA_NOISE_GLSL, LYGIA_SDF_GLSL, SHADER_COMMON_GLSL),
)


class BassTempleGPU(BaseGPUVisualizer):
    """Dunkler, aggressiver GPU-Visualizer fuer Dubstep/Deep Bass.

    Rendert eine zentrale, bass-reaktive Tempel-SDF-Form mit vertikalen
    Bass-Balken, Stroboskop-Flashes, expandierenden Shockwave-Ringen
    und chromatischer Aberration bei starken Beats.
    """

    PARAMS = {
        'bass_intensity': (1.2, 0.0, 3.0, 0.1),
        'strobe_threshold': (0.55, 0.0, 1.0, 0.05),
        'strobe_intensity': (0.55, 0.0, 1.0, 0.05),
        'color_shift': (0.0, 0.0, 1.0, 0.05),
        'shockwave_speed': (2.5, 0.5, 6.0, 0.1),
        'temple_scale': (1.0, 0.5, 2.0, 0.1),
        'bar_count': (14.0, 4.0, 32.0, 1.0),
        'contrast_gamma': (0.88, 0.5, 1.5, 0.01),
        'vignette_strength': (0.6, 0.0, 1.0, 0.05),
        'noise_intensity': (0.2, 0.0, 1.0, 0.05),
        'sparkle_intensity': (1.0, 0.0, 3.0, 0.1),
    }

    def _setup(self):
        """Initialisiert Shader, VBO und VAO fuer den Fullscreen-Quad."""
        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

    def _color_tuple(self, value):
        """Hilfsmethode: Konvertiert Hex-String oder Sequenz in RGB-Tupel."""
        if value is None:
            return None
        if isinstance(value, str) and value.startswith('#'):
            return self._hex_to_rgb(value)
        try:
            rgb = tuple(float(v) for v in value)
            if len(rgb) >= 3:
                return rgb[:3]
        except Exception:
            pass
        return None

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
        beat_intensity = f.get("beat_intensity", min(onset * 1.5, 1.0))

        # Farben aus color_mode ableiten
        primary_color = self._chroma_to_color(chroma)

        secondary = self._color_tuple(self.params.get('secondary_color'))
        if secondary is not None:
            secondary_color = secondary
        else:
            h, s, v = self._rgb_to_hsv(*primary_color)
            secondary_color = self._hsv_to_rgb((h + 0.5) % 1.0, min(1.0, s * 1.2), v)

        background = self._color_tuple(self.params.get('background_color'))
        if background is not None:
            bg_color = background
        else:
            h, s, v = self._rgb_to_hsv(*primary_color)
            bg_color = self._hsv_to_rgb(h, s * 0.3, v * 0.08)

        # Flash-Farbe: aufgehellte Sekundaerfarbe
        h, s, v = self._rgb_to_hsv(*secondary_color)
        flash_color = self._hsv_to_rgb(h, s * 0.5, min(1.0, v * 1.5))

        # Funken-Farbe: Primaerfarbe
        sparkle_color = primary_color

        # Uniforms aktualisieren
        self.prog["u_time"].value = float(time)
        self.prog["u_rms"].value = float(rms)
        self.prog["u_onset"].value = float(onset)
        self.prog["u_beat_intensity"].value = float(beat_intensity)
        self.prog["u_spectral_centroid"].value = float(spectral_centroid)
        self.prog["u_chroma_color"].value = primary_color
        self.prog["u_primary_color"].value = primary_color
        self.prog["u_secondary_color"].value = secondary_color
        self.prog["u_bg_color"].value = bg_color
        self.prog["u_flash_color"].value = flash_color
        self.prog["u_sparkle_color"].value = sparkle_color
        self.prog["u_bass_intensity"].value = float(self.params['bass_intensity'])
        self.prog["u_strobe_threshold"].value = float(self.params['strobe_threshold'])
        self.prog["u_strobe_intensity"].value = float(self.params['strobe_intensity'])
        self.prog["u_color_shift"].value = float(self.params['color_shift'])
        self.prog["u_shockwave_speed"].value = float(self.params['shockwave_speed'])
        self.prog["u_temple_scale"].value = float(self.params['temple_scale'])
        self.prog["u_bar_count"].value = float(self.params['bar_count'])
        self.prog["u_contrast_gamma"].value = float(self.params['contrast_gamma'])
        self.prog["u_vignette_strength"].value = float(self.params['vignette_strength'])
        self.prog["u_noise_intensity"].value = float(self.params['noise_intensity'])
        self.prog["u_sparkle_intensity"].value = float(self.params['sparkle_intensity'])

        # Zeichnen
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
