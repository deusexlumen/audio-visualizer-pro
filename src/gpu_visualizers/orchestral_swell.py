"""
Orchestral Swell - GPU-Visualizer fuer klassische Musik.

Eleganter, warmer Visualizer fuer Orchester- und Kammermusik:
- Aufsteigende goldene Partikel wie Glut und Staub im Konzertsaallicht
- Dynamik-basiertes Schwellen: forte = mehr Partikel, heller, weiter
- piano = weniger Partikel, gedimmter, kontrollierter
- Langsame, sanfte Bewegung mit Sinus-Wellen
- Warme Farbpalette: Gold, Bernstein, sanftes Orange, tiefes Burgunder
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class OrchestralSwellGPU(BaseGPUVisualizer):
    """
    Orchestral Swell - Eleganter GPU-Visualizer fuer klassische Musik-Dynamik.
    """

    PARAMS = {
        'swell_intensity': (1.0, 0.2, 2.0, 0.05),
        'particle_count': (64, 8, 128, 8),
        'gold_tint': (0.5, 0.0, 1.0, 0.05),
        'dynamics_response': (1.2, 0.5, 2.5, 0.1),
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
            uniform float u_rms;
            uniform float u_beat_intensity;
            uniform float u_swell_intensity;
            uniform float u_particle_count;
            uniform float u_gold_tint;
            uniform float u_dynamics_response;

            out vec4 f_color;

            // === Utilities ===
            float remap(float v, float i_min, float i_max, float o_min, float o_max) {
                return o_min + (v - i_min) * (o_max - o_min) / (i_max - i_min + 1e-8);
            }

            float hash(float n) { return fract(sin(n) * 43758.5453123); }
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
                for (int i = 0; i < octaves; i++) {
                    v += a * noise(p);
                    p = rot * p * 2.0 + vec2(100.0);
                    a *= 0.5;
                }
                return v;
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                float aspect = u_resolution.x / u_resolution.y;
                uv.x *= aspect;

                float t = u_time;
                float rms = u_rms;
                float beat = u_beat_intensity;
                float dyn = rms * u_dynamics_response;

                // === Deep warm background ===
                vec3 col = vec3(0.03, 0.015, 0.008);

                // Subtle warm vignette
                vec2 center = vec2(aspect * 0.5, 0.5);
                float dist = length(uv - center);
                col *= smoothstep(1.2, 0.2, dist);

                // === Central warm glow (spotlight effect) ===
                float spotGlow = exp(-dist * dist * 2.0) * (0.1 + dyn * 0.4);
                vec3 spotColor = vec3(1.0, 0.85, 0.6);
                col += spotColor * spotGlow * u_swell_intensity;

                // === Particles (embers/dust) ===
                int activeParticles = int(u_particle_count * (0.4 + rms * 0.6));
                float spread = 1.0 + dyn * 2.5;
                float globalBright = 0.5 + dyn * 0.5;

                for (int i = 0; i < 128; i++) {
                    if (float(i) >= u_particle_count) break;
                    if (i >= activeParticles) break;

                    float fi = float(i);
                    float seed = fi * 1.618033;

                    // Base position
                    float px = hash(seed * 7.13) * aspect;
                    float py = fract(hash(seed * 13.37) + t * (0.02 + hash(seed * 3.71) * 0.03 + dyn * 0.015));

                    // Spread from center when loud
                    px = (px - aspect * 0.5) * spread + aspect * 0.5;

                    // Gentle drift
                    px += sin(t * 0.4 + fi * 0.73) * 0.04 * spread;
                    px += sin(t * 0.9 + fi * 1.19) * 0.015 * spread;

                    vec2 pPos = vec2(px, py);
                    float d = length(uv - pPos);

                    // Particle size
                    float pSize = 0.004 + hash(seed * 5.23) * 0.006 + dyn * 0.006;
                    float glow = exp(-d * d / (pSize * pSize));

                    // Warm palette
                    vec3 pColor;
                    float ci = hash(seed * 11.11);
                    if (ci < 0.25) {
                        pColor = vec3(1.0, 0.84, 0.0);      // gold
                    } else if (ci < 0.5) {
                        pColor = vec3(1.0, 0.65, 0.1);      // amber
                    } else if (ci < 0.75) {
                        pColor = vec3(1.0, 0.5, 0.2);       // soft orange
                    } else {
                        pColor = vec3(0.55, 0.08, 0.12);    // deep burgundy
                    }

                    // Gold tint bias
                    pColor = mix(pColor, vec3(1.0, 0.84, 0.0), u_gold_tint * 0.25);

                    // Brightness per particle + dynamics
                    float pBright = (0.3 + hash(seed * 9.99) * 0.5) * globalBright;
                    pBright *= (1.0 + beat * 0.6 * hash(seed * 2.71));

                    col += pColor * glow * pBright * u_swell_intensity;
                }

                // === Subtle light rays from top ===
                vec2 rayOrigin = vec2(aspect * 0.5, 1.05);
                vec2 rayDir = uv - rayOrigin;
                float rayAngle = atan(rayDir.x, -rayDir.y);
                float rayLen = length(rayDir);

                float rayNoise = fbm(vec2(rayAngle * 2.0 + t * 0.05, t * 0.08), 3);
                float rays = pow(max(0.0, sin(rayAngle * 5.0 + rayNoise * 1.5)), 6.0);
                rays *= exp(-rayLen * rayLen * 1.5);
                rays *= (0.2 + dyn * 0.8);
                rays *= smoothstep(0.0, 0.3, uv.y); // fade at bottom

                vec3 rayColor = mix(vec3(1.0, 0.9, 0.7), vec3(1.0, 0.7, 0.4), dyn);
                col += rayColor * rays * 0.06 * u_swell_intensity;

                // === Film grain ===
                float grain = hash(gl_FragCoord.xy + fract(t * 100.0) * 100.0) * 0.015 - 0.0075;
                col += grain;

                // Tone mapping
                col = col / (1.0 + col * 0.5);

                f_color = vec4(col, 1.0);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self._vao = self.ctx.vertex_array(self._prog, [(vbo, "2f", "in_pos")])

    def render(self, features: dict, time: float):
        frame_idx = int(time * features.get("fps", 30))
        frame_idx = max(0, min(frame_idx, features.get("frame_count", 0) - 1))

        rms = float(features["rms"][frame_idx])
        onset = float(features["onset"][frame_idx])
        beat_intensity = float(features.get("beat_intensity", [0.0] * 30)[frame_idx])
        self._prog["u_resolution"].value = (self.width, self.height)
        self._prog["u_time"].value = time
        self._prog["u_rms"].value = rms
        self._prog["u_beat_intensity"].value = beat_intensity
        self._prog["u_swell_intensity"].value = self.params["swell_intensity"]
        self._prog["u_particle_count"].value = self.params["particle_count"]
        self._prog["u_gold_tint"].value = self.params["gold_tint"]
        self._prog["u_dynamics_response"].value = self.params["dynamics_response"]

        self._vao.render(mode=moderngl.TRIANGLE_STRIP)
