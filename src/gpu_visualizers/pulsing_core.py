"""
GPU-beschleunigter Pulsing-Core-Visualizer mit ModernGL.

Signature-Ueberarbeitung: mehrschichtiger HDR-Kern (heller Innenkern >1.0 fuer
kraeftigen Bloom), fbm-Korona, beat-getriggerte Schockwellen-Ringe ueber
u_impact und dezente Orbit-Partikel. Distance-Field-Rendering im Fragment-Shader
auf einem einzelnen Fullscreen-Quad.
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


class PulsingCoreGPU(BaseGPUVisualizer):
    """Pulsing-Core-Visualizer mit mehrschichtigem HDR-Distance-Field-Rendering."""

    PARAMS = {
        'pulse_intensity': (1.0, 0.0, 3.0, 0.1),
        'base_radius': (0.1, 0.02, 0.3, 0.01),
        'core_glow': (1.4, 0.3, 3.0, 0.05),
        'corona_amount': (0.5, 0.0, 1.5, 0.05),
        'ring_count': (3, 1, 8, 1),
        'ring_spacing': (0.06, 0.02, 0.15, 0.01),
        'ring_width': (0.015, 0.005, 0.05, 0.005),
        'shockwave_strength': (1.0, 0.0, 2.0, 0.05),
        'particle_count': (14, 0, 32, 1),
        'particle_strength': (0.6, 0.0, 1.5, 0.05),
        'glow_radius': (1.0, 0.2, 3.0, 0.1),
        'bg_brightness': (0.15, 0.0, 0.5, 0.01),
    }

    PARAMS_GROUPS = {
        "Puls": ["pulse_intensity", "base_radius", "core_glow", "corona_amount"],
        "Ringe": ["ring_count", "ring_spacing", "ring_width", "shockwave_strength"],
        "Partikel": ["particle_count", "particle_strength"],
        "Erscheinungsbild": ["glow_radius", "bg_brightness"],
    }

    def _setup(self):
        """Initialisiert Shader, VBO und VAO fuer den Fullscreen-Quad."""
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_rms;
            uniform float u_onset;
            uniform float u_impact;
            uniform float u_beat_intensity;
            uniform vec3 u_color;
            uniform vec3 u_secondary_color;
            uniform vec3 u_background_color;
            uniform float u_pulse_intensity;
            uniform float u_base_radius;
            uniform float u_core_glow;
            uniform float u_corona_amount;
            uniform int u_ring_count;
            uniform float u_ring_spacing;
            uniform float u_ring_width;
            uniform float u_shockwave_strength;
            uniform int u_particle_count;
            uniform float u_particle_strength;
            uniform float u_glow_radius;
            uniform float u_bg_brightness;
            uniform float u_brightness;
            out vec4 f_color;

            void main() {
                // Zentriert, aspektkorrigiert -> Kreise bleiben rund
                vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
                uv.x *= u_resolution.x / u_resolution.y;
                float dist = length(uv);
                float ang = atan(uv.y, uv.x);

                vec3 col = u_background_color * u_bg_brightness;

                float radius = u_base_radius + u_rms * 0.15 * u_pulse_intensity;

                // === Mehrschichtiger Kern ===
                // Weiss-heisser Innenkern deutlich ueber 1.0 -> kraeftiger Bloom
                float core = exp(-dist * dist / (radius * radius * 0.6));
                col += mix(u_color, vec3(1.0), 0.6) * core * u_core_glow * 2.2;
                // Weicher Farb-Halo um den Kern
                float halo = exp(-dist * dist / (radius * radius * 2.0 / u_glow_radius));
                col += u_color * halo * (0.6 + u_rms * 0.6);

                // === fbm-Korona: flackernder Saum am Kernrand ===
                float corona = fbm(vec2(ang * 3.0, u_time * 0.6), 4);
                float coronaBand = exp(-pow((dist - radius * 1.35) * 6.0, 2.0));
                col += u_color * coronaBand * corona * u_corona_amount * (0.5 + u_onset);

                // === Konzentrische Ringe ===
                for (int i = 1; i <= 8; i++) {
                    if (i > u_ring_count) break;
                    float fi = float(i);
                    float ringRadius = radius + fi * u_ring_spacing;
                    float ringWidth = u_ring_width + u_rms * 0.004;
                    float ringDist = abs(dist - ringRadius) - ringWidth;
                    float ringGlow = exp(-ringDist * ringDist * 1800.0);
                    vec3 ringColor = mix(u_color, u_secondary_color, clamp(fi * 0.18, 0.0, 1.0));
                    col += ringColor * ringGlow * (0.25 + max(u_onset, u_beat_intensity) * 0.6);
                }

                // === Beat-Schockwelle: expandierender Ring auf Transienten ===
                float wavePhase = fract(u_time * 0.6);
                float waveRadius = wavePhase * 1.2;
                float wave = exp(-pow((dist - waveRadius) * 10.0, 2.0));
                col += mix(u_color, u_secondary_color, 0.5)
                     * wave * u_impact * u_shockwave_strength * (1.0 - wavePhase);

                // === Orbit-Partikel ===
                for (int p = 0; p < 32; p++) {
                    if (p >= u_particle_count) break;
                    float fp = float(p);
                    float orbit = radius + 0.12 + hash12(vec2(fp, 3.0)) * 0.5;
                    float speed = 0.3 + hash12(vec2(fp, 7.0)) * 0.7;
                    float pa = fp * 2.399963 + u_time * speed;
                    vec2 pp = vec2(cos(pa), sin(pa)) * orbit;
                    float pd = length(uv - pp);
                    float sparkle = exp(-pd * pd * 1400.0);
                    col += mix(u_color, vec3(1.0), 0.4)
                         * sparkle * u_particle_strength * (0.4 + u_beat_intensity * 0.8);
                }

                // Subtiler Hintergrund-Glow, atmet mit RMS
                float bgGlow = exp(-dist * dist / ((radius + 0.25) * (radius + 0.25) * 3.0));
                col += u_color * bgGlow * u_rms * u_bg_brightness;

                // HDR-Ausgabe: zentrales ACES-Tonemapping im Renderer
                col = max(col, 0.0) * u_brightness;
                f_color = vec4(col, 1.0);
            }
            """,
            includes=(LYGIA_MATH_GLSL, LYGIA_NOISE_GLSL, LYGIA_SDF_GLSL, SHADER_COMMON_GLSL),
        )

        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

    def render(self, features: dict, time: float):
        """Rendert einen Frame mit aktuellem RMS, Onset, Transient und Chroma-Farbe."""
        f = self._features_at_time(features, time)
        uniforms = self._map_features_to_uniforms(f, mode="music")

        color = self._chroma_to_color(uniforms["u_chroma"])

        def _rgb_from_hex(value, default):
            if isinstance(value, str) and value.startswith('#'):
                try:
                    return self._hex_to_rgb(value)
                except Exception:
                    pass
            return default

        secondary = _rgb_from_hex(self.params.get("secondary_color"), (0.0, 0.8, 1.0))
        background = _rgb_from_hex(self.params.get("background_color"), (0.02, 0.02, 0.04))

        self.prog["u_resolution"].value = (self.width, self.height)
        self.prog["u_time"].value = time
        self.prog["u_rms"].value = uniforms["u_energy"]
        self.prog["u_onset"].value = uniforms["u_beat"]
        self.prog["u_impact"].value = uniforms["u_impact"]
        self.prog["u_beat_intensity"].value = uniforms.get("u_beat_intensity", uniforms["u_beat"])
        self.prog["u_color"].value = color
        self.prog["u_secondary_color"].value = secondary
        self.prog["u_background_color"].value = background
        self.prog["u_pulse_intensity"].value = float(self.params['pulse_intensity'])
        self.prog["u_base_radius"].value = float(self.params['base_radius'])
        self.prog["u_core_glow"].value = float(self.params['core_glow'])
        self.prog["u_corona_amount"].value = float(self.params['corona_amount'])
        self.prog["u_ring_count"].value = int(self.params['ring_count'])
        self.prog["u_ring_spacing"].value = float(self.params['ring_spacing'])
        self.prog["u_ring_width"].value = float(self.params['ring_width'])
        self.prog["u_shockwave_strength"].value = float(self.params['shockwave_strength'])
        self.prog["u_particle_count"].value = int(self.params['particle_count'])
        self.prog["u_particle_strength"].value = float(self.params['particle_strength'])
        self.prog["u_glow_radius"].value = float(self.params['glow_radius'])
        self.prog["u_bg_brightness"].value = float(self.params['bg_brightness'])
        self.prog["u_brightness"].value = float(self.params.get('brightness', 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
