"""
Nebula Drift - Signature Musik-Visualizer.

Treibende fbm-Nebelwolken mit einem Feld aus glitzernden Partikeln.
Beat-Intensitaet laesst die Nebel pulsieren und Partikel aufblitzen; der
Farbton driftet langsam entlang des Chroma. Fuer Ambient bis Big-Room-EDM.
Ein einzelner Fullscreen-Fragment-Shader, HDR-Ausgabe.
"""

import moderngl
from .base import (
    BaseGPUVisualizer,
    FULLSCREEN_VERTEX_SHADER,
    LYGIA_MATH_GLSL,
    LYGIA_NOISE_GLSL,
    SHADER_COMMON_GLSL,
    compose_fragment,
    create_fullscreen_quad,
)


class NebulaDriftGPU(BaseGPUVisualizer):
    """fbm-Nebel + Partikelfeld, beat-getrieben, mit Chroma-Hue-Drift."""

    PARAMS = {
        'nebula_scale': (2.2, 0.5, 5.0, 0.1),
        'drift_speed': (0.12, 0.0, 0.6, 0.01),
        'nebula_density': (0.8, 0.2, 2.0, 0.05),
        'beat_pulse': (0.6, 0.0, 1.5, 0.05),
        'particle_count': (40, 0, 96, 2),
        'particle_strength': (0.7, 0.0, 1.5, 0.05),
        'hue_drift': (0.15, 0.0, 0.6, 0.01),
        'glow_strength': (1.0, 0.2, 2.5, 0.05),
        'bg_brightness': (0.14, 0.0, 0.5, 0.01),
    }

    PARAMS_GROUPS = {
        "Nebel": ["nebula_scale", "nebula_density", "drift_speed"],
        "Beat": ["beat_pulse", "hue_drift"],
        "Partikel": ["particle_count", "particle_strength"],
        "Erscheinungsbild": ["glow_strength", "bg_brightness"],
    }

    def _setup(self):
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_energy;
            uniform float u_impact;
            uniform float u_beat_intensity;
            uniform vec3 u_color;
            uniform vec3 u_secondary_color;
            uniform vec3 u_background_color;
            uniform float u_nebula_scale;
            uniform float u_drift_speed;
            uniform float u_nebula_density;
            uniform float u_beat_pulse;
            uniform int u_particle_count;
            uniform float u_particle_strength;
            uniform float u_hue_drift;
            uniform float u_glow_strength;
            uniform float u_bg_brightness;
            uniform float u_brightness;
            out vec4 f_color;

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                vec2 p = uv * 2.0 - 1.0;
                p.x *= u_resolution.x / u_resolution.y;

                vec3 col = u_background_color * u_bg_brightness;

                float t = u_time * u_drift_speed;
                float pulse = 1.0 + u_beat_intensity * u_beat_pulse;

                // === Geschichteter fbm-Nebel ===
                vec2 q = p * u_nebula_scale;
                float n1 = fbm(q + vec2(t, t * 0.6), 5);
                float n2 = fbm(q * 1.8 - vec2(t * 0.4, t), 4);
                float nebula = pow(clamp(n1 * 0.7 + n2 * 0.5, 0.0, 1.0), 1.6);
                nebula *= u_nebula_density * pulse;

                // Hue-Drift ueber die Nebelmasse
                vec3 hsvA = rgb2hsv(u_color);
                vec3 hsvB = rgb2hsv(u_secondary_color);
                float driftT = fract(hsvA.x + n2 * u_hue_drift + u_time * 0.01);
                vec3 nebCol = hsv2rgb(vec3(driftT, mix(hsvA.y, hsvB.y, n1), 1.0));
                col += nebCol * nebula * u_glow_strength * (0.5 + u_energy * 0.6);

                // Dichte Kern-Gluten dort, wo der Nebel am dicksten ist
                col += mix(nebCol, vec3(1.0), 0.5) * smoothstep(0.7, 1.1, nebula) * 0.8;

                // === Partikelfeld ===
                for (int i = 0; i < 96; i++) {
                    if (i >= u_particle_count) break;
                    float fi = float(i);
                    // Langsam driftende Sternposition
                    vec2 seed = vec2(fi * 0.137, fi * 0.317);
                    float px = fract(hash12(seed) + t * (0.1 + hash12(seed + 5.0) * 0.2)) * 2.0 - 1.0;
                    px *= u_resolution.x / u_resolution.y;
                    float py = (hash12(seed + 1.0) * 2.0 - 1.0);
                    py += sin(u_time * (0.3 + hash12(seed + 2.0)) + fi) * 0.04;
                    vec2 pp = vec2(px, py);

                    float d = length(p - pp);
                    float twinkle = 0.5 + 0.5 * sin(u_time * 3.0 + fi * 1.7);
                    float spark = exp(-d * d * 1600.0) * twinkle;
                    col += mix(u_color, vec3(1.0), 0.5)
                         * spark * u_particle_strength * (0.4 + u_beat_intensity * 0.9);
                }

                // Beat-Aufhellung des gesamten Nebels (weich, kein Hard-Flash)
                col += nebCol * u_impact * 0.15;

                col = max(col, 0.0) * u_brightness;
                f_color = vec4(col, 1.0);
            }
            """,
            includes=(LYGIA_MATH_GLSL, LYGIA_NOISE_GLSL, SHADER_COMMON_GLSL),
        )
        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        uniforms = self._map_features_to_uniforms(f, mode="music")

        color = self._chroma_to_color(uniforms["u_chroma"])
        h, s, v = self._rgb_to_hsv(*color)
        secondary = self._hsv_to_rgb((h + 0.4) % 1.0, s, v)

        bg = self.params.get("background_color")
        if isinstance(bg, str) and bg.startswith("#"):
            try:
                bg_rgb = self._hex_to_rgb(bg)
            except Exception:
                bg_rgb = (0.01, 0.01, 0.03)
        else:
            bg_rgb = (0.01, 0.01, 0.03)

        self.prog["u_resolution"].value = (self.width, self.height)
        self.prog["u_time"].value = time
        self.prog["u_energy"].value = uniforms["u_energy"]
        self.prog["u_impact"].value = uniforms["u_impact"]
        self.prog["u_beat_intensity"].value = uniforms.get("u_beat_intensity", uniforms["u_beat"])
        self.prog["u_color"].value = color
        self.prog["u_secondary_color"].value = secondary
        self.prog["u_background_color"].value = bg_rgb
        self.prog["u_nebula_scale"].value = float(self.params["nebula_scale"])
        self.prog["u_drift_speed"].value = float(self.params["drift_speed"])
        self.prog["u_nebula_density"].value = float(self.params["nebula_density"])
        self.prog["u_beat_pulse"].value = float(self.params["beat_pulse"])
        self.prog["u_particle_count"].value = int(self.params["particle_count"])
        self.prog["u_particle_strength"].value = float(self.params["particle_strength"])
        self.prog["u_hue_drift"].value = float(self.params["hue_drift"])
        self.prog["u_glow_strength"].value = float(self.params["glow_strength"])
        self.prog["u_bg_brightness"].value = float(self.params["bg_brightness"])
        self.prog["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
