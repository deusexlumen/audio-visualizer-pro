"""
Voice Flow - Signature Podcast-Visualizer.

Organischer, atmender Flow fuer Podcasts und Sprach-Inhalte.
- Langsame FBM-Noise Wellen (nie ablenkend)
- Voice-Clarity steuert sanfte Intensitaet
- Keine harten Beats oder Explosionen
- Atmender, meditativer Rhythmus
- Soft-Chroma Farbverlaeufe

Psychologische Vorgabe: Die Visualisierung darf NIEMALS
vom gesprochenen Wort ablenken.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class VoiceFlowGPU(BaseGPUVisualizer):
    """
    Voice Flow - Sanfter, organischer Podcast-Visualizer.
    Fliesst wie Wasser, atmet wie ein Organismus.
    """

    PARAMS = {
        'flow_speed': (0.15, 0.05, 0.5, 0.01),
        'wave_depth': (0.4, 0.1, 0.8, 0.05),
        'color_saturation': (0.5, 0.2, 0.8, 0.05),
        'breathe_intensity': (0.3, 0.0, 0.6, 0.05),
        'detail_level': (3, 1, 5, 1),
    }

    def _setup(self):
        self._prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
            #version 330
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_voice;
            uniform vec3 u_color;
            uniform float u_flow_speed;
            uniform float u_wave_depth;
            uniform float u_color_saturation;
            uniform float u_breathe_intensity;
            uniform float u_detail_level;

            out vec4 f_color;

            // === Lygia Math ===
            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }
            mat2 rot2(float a) {
                float c = cos(a), s = sin(a);
                return mat2(c, -s, s, c);
            }

            // === Lygia Noise ===
            float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
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
                for (int i = 0; i < 5; i++) {
                    if (i >= octaves) break;
                    v += a * noise(p);
                    p = rot * p * 2.0 + vec2(100.0);
                    a *= 0.5;
                }
                return v;
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                uv.x *= u_resolution.x / u_resolution.y;

                // Langsamer Atem-Takt (ca. 6-10 BPM)
                float breathe = sin(u_time * 0.8) * 0.5 + 0.5;
                float breatheAmt = u_breathe_intensity * (0.5 + breathe * 0.5);

                // Voice-Clarity steuert sanfte Wellen-Amplitude
                float voiceAmt = u_voice * u_wave_depth;

                // Mehrere uebereinanderliegende Noise-Ebenen
                vec2 p = uv * 2.0;
                float t = u_time * u_flow_speed;

                // Ebene 1: Grosse, langsame Wellen
                float n1 = fbm(p + vec2(t * 0.3, t * 0.2), int(u_detail_level));

                // Ebene 2: Mittlere Details (Voice-reaktiv)
                float n2 = fbm(p * 2.0 + vec2(-t * 0.2, t * 0.4) + n1 * 0.5, int(u_detail_level));

                // Ebene 3: Feine Details (sehr subtil)
                float n3 = fbm(p * 4.0 + vec2(t * 0.1, -t * 0.15) + n2 * 0.3, max(1, int(u_detail_level) - 1));

                // Kombiniere mit Gewichtung
                float finalNoise = n1 * 0.5 + n2 * 0.35 + n3 * 0.15;

                // Sanfte Wellen-Displacement
                float wave = sin(uv.x * 3.14159 * 2.0 + t + finalNoise * 2.0) * voiceAmt;
                wave += sin(uv.x * 1.5 + t * 0.7) * breatheAmt * 0.3;

                // Farbgebung: Chroma-basiert, aber sehr gedämpft
                vec3 baseColor = u_color;
                float hueShift = finalNoise * 0.05 + wave * 0.02;
                vec3 color1 = hsv2rgb(vec3(fract(baseColor.x + hueShift), u_color_saturation * 0.6, 0.4 + wave * 0.1));
                vec3 color2 = hsv2rgb(vec3(fract(baseColor.x + hueShift + 0.15), u_color_saturation * 0.5, 0.3 + wave * 0.08));

                // Verlauf von oben nach unten
                float grad = uv.y + wave * 0.1;
                vec3 col = mix(color2, color1, grad);

                // Sehr sanfter Glow in der Mitte
                float centerDist = length((uv - vec2(0.5 * u_resolution.x / u_resolution.y, 0.5)) * vec2(1.0, 0.6));
                float centerGlow = exp(-centerDist * centerDist * 3.0) * u_voice * 0.15;
                col += baseColor * centerGlow;

                // Dunkler Hintergrund (nie heller als 0.4)
                col = clamp(col, 0.0, 0.4 + u_voice * 0.1);

                // Sehr sanftes Film Grain (optional, subtil)
                float grain = hash(gl_FragCoord.xy + fract(u_time * 100.0) * 100.0) * 0.02 - 0.01;
                col += grain;

                f_color = vec4(col, 1.0);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self._vao = self.ctx.vertex_array(self._prog, [(vbo, "2f", "in_pos")])

    def render(self, features: dict, time: float):
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        uniforms = self._map_features_to_uniforms(f, mode="speech")

        color = self._chroma_to_color(uniforms["u_chroma"])

        self._prog["u_resolution"].value = (self.width, self.height)
        self._prog["u_time"].value = time
        self._prog["u_voice"].value = uniforms["u_flow"]
        self._prog["u_color"].value = color
        self._prog["u_flow_speed"].value = self.params["flow_speed"]
        self._prog["u_wave_depth"].value = self.params["wave_depth"]
        self._prog["u_color_saturation"].value = self.params["color_saturation"]
        self._prog["u_breathe_intensity"].value = self.params["breathe_intensity"]
        self._prog["u_detail_level"].value = self.params["detail_level"]

        self._vao.render(mode=moderngl.TRIANGLE_STRIP)
