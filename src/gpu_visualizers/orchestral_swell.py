"""
Orchestral Swell - GPU-Visualizer fuer klassische Musik.

Eleganter, warmer Visualizer fuer Orchester- und Kammermusik:
- Aufsteigende goldene Partikel wie Glut und Staub im Konzertsaallicht
- Dynamik-basiertes Schwellen: forte = mehr Partikel, heller, weiter
- piano = weniger Partikel, gedimmter, kontrollierter
- Langsame, sanfte Bewegung mit Sinus-Wellen
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class OrchestralSwellGPU(BaseGPUVisualizer):
    """
    Orchestral Swell - Eleganter GPU-Visualizer fuer klassische Musik-Dynamik.
    """

    COLOR_PARAMS = {
        'color_mode': 'warm',     # Orchestral-Look: warme Toene als Default
        'base_hue': 0.10,         # 0.0-1.0, nur fuer 'fixed'
        'color_saturation': 0.75, # 0.0-1.0
    }

    PARAMS = {
        'swell_intensity': (1.0, 0.2, 2.0, 0.05),
        'particle_count': (64, 8, 128, 8),
        'gold_tint': (0.5, 0.0, 1.0, 0.05),
        'dynamics_response': (1.2, 0.5, 2.5, 0.1),
        'bg_brightness': (0.08, 0.0, 0.5, 0.01),
        'vignette_strength': (0.6, 0.0, 1.5, 0.05),
        'spotlight_strength': (0.3, 0.0, 1.0, 0.05),
        'ray_strength': (0.06, 0.0, 0.3, 0.01),
        'grain_amount': (0.015, 0.0, 0.1, 0.005),
        'particle_spread': (1.0, 0.0, 4.0, 0.1),
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
            uniform float u_bg_brightness;
            uniform float u_vignette_strength;
            uniform float u_spotlight_strength;
            uniform float u_ray_strength;
            uniform float u_grain_amount;
            uniform float u_particle_spread;
            uniform float u_brightness;
            uniform vec3 u_primary_color;
            uniform vec3 u_secondary_color;
            uniform vec3 u_background_color;

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
                vec3 col = u_background_color * u_bg_brightness;

                // Subtle warm vignette
                vec2 center = vec2(aspect * 0.5, 0.5);
                float dist = length(uv - center);
                col *= smoothstep(1.2, 0.2, dist * u_vignette_strength);

                // === Central warm glow (spotlight effect) ===
                float spotGlow = exp(-dist * dist * 2.0) * (0.1 + dyn * 0.4);
                vec3 spotColor = mix(u_primary_color, vec3(1.0, 0.95, 0.85), 0.25);
                col += spotColor * spotGlow * u_spotlight_strength * u_swell_intensity;

                // === Particles (embers/dust) ===
                int activeParticles = int(u_particle_count * (0.4 + rms * 0.6));
                float spread = 1.0 + dyn * 2.5 * u_particle_spread;
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

                    // Palette aus primary/secondary abgeleitet
                    float ci = hash(seed * 11.11);
                    vec3 pColor;
                    if (ci < 0.33) {
                        pColor = u_primary_color;
                    } else if (ci < 0.66) {
                        pColor = mix(u_primary_color, u_secondary_color, 0.5);
                    } else {
                        pColor = u_secondary_color;
                    }

                    // Gold tint bias
                    pColor = mix(pColor, u_primary_color, u_gold_tint * 0.25);

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

                vec3 rayColor = mix(u_primary_color, u_secondary_color, dyn);
                col += rayColor * rays * u_ray_strength * u_swell_intensity;

                // === Film grain ===
                float grain = hash(gl_FragCoord.xy + fract(t * 100.0) * 100.0) * u_grain_amount - u_grain_amount * 0.5;
                col += grain;

                // Tone mapping
                col = col / (1.0 + col * 0.5);

                f_color = vec4(col * u_brightness, 1.0);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self._vao = self.ctx.vertex_array(self._prog, [(vbo, "2f", "in_pos")])

    def render(self, features: dict, time: float):
        frame_idx = int(time * features.get("fps", 30))
        frame_idx = max(0, min(frame_idx, features.get("frame_count", 0) - 1))

        def _safe_float(arr, idx, default=0.0):
            if arr is None:
                return default
            if hasattr(arr, "__len__") and len(arr) > idx >= 0:
                return float(arr[idx])
            return default

        rms = _safe_float(features.get("rms"), frame_idx, 0.0)
        onset = _safe_float(features.get("onset"), frame_idx, 0.0)
        chroma = features.get("chroma")
        if chroma is not None and hasattr(chroma, "shape") and len(chroma.shape) > 1:
            if chroma.shape[0] == 12 and chroma.shape[1] > frame_idx >= 0:
                chroma_frame = chroma[:, frame_idx]
            elif chroma.shape[1] == 12 and chroma.shape[0] > frame_idx >= 0:
                chroma_frame = chroma[frame_idx, :]
            else:
                chroma_frame = np.zeros(12, dtype=np.float32)
        elif chroma is not None and hasattr(chroma, "__len__") and len(chroma) > frame_idx >= 0:
            chroma_frame = chroma[frame_idx]
        else:
            chroma_frame = np.zeros(12, dtype=np.float32)

        beat_intensity_arr = features.get("beat_intensity")
        if beat_intensity_arr is not None and hasattr(beat_intensity_arr, "__len__") and len(beat_intensity_arr) > frame_idx >= 0:
            beat_intensity = float(beat_intensity_arr[frame_idx])
        else:
            beat_intensity = min(onset * 1.5, 1.0)

        # Farben aus dem konfigurierten color_mode ableiten
        primary_color = self._chroma_to_color(chroma_frame)
        # Sekundaere Farbe: etwas waermer/heller, falls nicht via Parameter gesetzt
        secondary_param = self.params.get('secondary_color')
        if secondary_param and isinstance(secondary_param, str) and secondary_param.startswith('#'):
            secondary_color = self._hex_to_rgb(secondary_param)
        else:
            secondary_color = (
                min(1.0, primary_color[0] * 1.1 + 0.1),
                min(1.0, primary_color[1] * 0.9 + 0.05),
                min(1.0, primary_color[2] * 0.7),
            )

        background_param = self.params.get('background_color')
        if background_param and isinstance(background_param, str) and background_param.startswith('#'):
            background_color = self._hex_to_rgb(background_param)
        else:
            background_color = (0.03, 0.015, 0.008)

        self._prog["u_resolution"].value = (self.width, self.height)
        self._prog["u_time"].value = time
        self._prog["u_rms"].value = rms
        self._prog["u_beat_intensity"].value = beat_intensity
        self._prog["u_swell_intensity"].value = self.params["swell_intensity"]
        self._prog["u_particle_count"].value = self.params["particle_count"]
        self._prog["u_gold_tint"].value = self.params["gold_tint"]
        self._prog["u_dynamics_response"].value = self.params["dynamics_response"]
        self._prog["u_bg_brightness"].value = self.params["bg_brightness"]
        self._prog["u_vignette_strength"].value = self.params["vignette_strength"]
        self._prog["u_spotlight_strength"].value = self.params["spotlight_strength"]
        self._prog["u_ray_strength"].value = self.params["ray_strength"]
        self._prog["u_grain_amount"].value = self.params["grain_amount"]
        self._prog["u_particle_spread"].value = self.params["particle_spread"]
        self._prog["u_brightness"].value = self.params.get("brightness", 1.0)
        self._prog["u_primary_color"].value = primary_color
        self._prog["u_secondary_color"].value = secondary_color
        self._prog["u_background_color"].value = background_color

        self._vao.render(mode=moderngl.TRIANGLE_STRIP)
