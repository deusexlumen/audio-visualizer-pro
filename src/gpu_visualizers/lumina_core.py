"""
Lumina Core - Signature Musik-Visualizer.

Ein zentraler, leuchtender Kern mit:
- FBM-Noise Displacement fuer organische Oberflaeche
- Transient-Reaktive Explosionen (Kick/Snare)
- Rotierende Ringe mit Chroma-Farben
- Chromatic Aberration & Glow im Shader
- Phong-Lighting fuer 3D-Look

Optimiert fuer Musik: Reagiert stark auf Beats und Transienten.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class LuminaCoreGPU(BaseGPUVisualizer):
    """
    Lumina Core - Der ultimative Musik-Visualizer.
    Ein leuchtender, pulsierender Kern mit organischem Noise und Beat-Reaktion.
    """

    PARAMS = {
        'core_intensity': (1.2, 0.5, 3.0, 0.1),
        'ring_count': (4, 1, 8, 1),
        'noise_scale': (2.0, 0.5, 5.0, 0.1),
        'glow_strength': (0.8, 0.0, 2.0, 0.1),
        'chromatic_aberration': (0.003, 0.0, 0.02, 0.001),
        'rotation_speed': (0.3, 0.0, 1.0, 0.05),
        'core_base_radius': (0.15, 0.05, 0.4, 0.01),
        'ring_base_radius': (0.25, 0.1, 0.5, 0.01),
        'ring_spacing': (0.08, 0.02, 0.15, 0.01),
        'ring_width': (0.008, 0.001, 0.05, 0.001),
        'noise_amount': (0.03, 0.0, 0.1, 0.005),
        'pulse_intensity': (0.3, 0.0, 1.0, 0.05),
        'specular_power': (32.0, 4.0, 128.0, 1.0),
        'bg_brightness': (0.5, 0.0, 2.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Core": ["core_intensity", "core_base_radius", "pulse_intensity"],
        "Ringe": ["ring_count", "ring_base_radius", "ring_spacing", "ring_width", "rotation_speed"],
        "Noise": ["noise_scale", "noise_amount"],
        "Glow & Licht": ["glow_strength", "specular_power"],
        "Effekte": ["chromatic_aberration"],
        "Hintergrund": ["bg_brightness"],
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
            uniform float u_onset;
            uniform float u_transient;
            uniform float u_centroid;
            uniform float u_beat_intensity;
            uniform vec3 u_color;
            uniform vec3 u_secondary_color;
            uniform vec3 u_background_color;
            uniform float u_core_intensity; // Kern-Intensitaet (wird verwendet)
            uniform float u_ring_count;
            uniform float u_noise_scale;
            uniform float u_glow_strength;
            uniform float u_chromatic_aberration;
            uniform float u_rotation_speed;
            uniform float u_core_base_radius;
            uniform float u_ring_base_radius;
            uniform float u_ring_spacing;
            uniform float u_ring_width;
            uniform float u_noise_amount;
            uniform float u_pulse_intensity;
            uniform float u_specular_power;
            uniform float u_bg_brightness;
            uniform float u_brightness;

            out vec4 f_color;

            // === Lygia Math ===
            float remap(float v, float i_min, float i_max, float o_min, float o_max) {
                return o_min + (v - i_min) * (o_max - o_min) / (i_max - i_min + 1e-8);
            }
            mat2 rot2(float a) {
                float c = cos(a), s = sin(a);
                return mat2(c, -s, s, c);
            }

            // === Lygia Noise ===
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

            // === SDF ===
            float sdCircle(vec2 p, float r) { return length(p) - r; }

            // === Lighting ===
            vec3 phong(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 color, float specPower) {
                float diff = max(dot(normal, lightDir), 0.0);
                vec3 reflectDir = reflect(-lightDir, normal);
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), specPower);
                return color * diff + vec3(1.0) * spec * 0.5;
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
                uv.x *= u_resolution.x / u_resolution.y;

                vec3 col = u_background_color * u_bg_brightness;
                vec2 center = vec2(0.0);

                // === Transient-Explosion ===
                float explosion = smoothstep(0.3, 1.0, max(u_transient, u_beat_intensity)) * 0.15;

                // === Zentrale Kugel mit FBM Displacement ===
                vec2 p = uv - center;
                float baseRadius = (u_core_base_radius + u_rms * 0.08 + explosion) * u_core_intensity;
                float n = fbm(p * u_noise_scale * 10.0 + u_time * 0.5, 4);
                float displacedRadius = baseRadius + n * u_noise_amount * u_centroid;
                float d = sdCircle(p, displacedRadius);

                // 3D Normal aus Noise-Gradient approximieren
                float n_x = fbm((p + vec2(0.001, 0.0)) * u_noise_scale * 10.0 + u_time * 0.5, 4);
                float n_y = fbm((p + vec2(0.0, 0.001)) * u_noise_scale * 10.0 + u_time * 0.5, 4);
                vec3 normal = normalize(vec3((n - n_x) * 30.0, (n - n_y) * 30.0, 1.0));

                vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
                vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0));

                // Kugel-Farbe mit Chroma
                vec3 sphereColor = u_color * (0.8 + u_rms * 0.4);
                vec3 lit = phong(normal, lightDir, viewDir, sphereColor, u_specular_power);

                // Kugel-Glow
                float glow = exp(-d * d * 80.0) * u_glow_strength;
                col += lit * glow * 2.0;
                col += sphereColor * exp(-max(d, 0.0) * 8.0) * 0.5;

                // === Ringe ===
                int rings = int(u_ring_count);
                for (int i = 0; i < 8; i++) {
                    if (i >= rings) break;
                    float fi = float(i);
                    float ringRadius = u_ring_base_radius + fi * u_ring_spacing + u_rms * 0.03;
                    float ringWidth = u_ring_width + u_rms * 0.004;

                    // Rotation
                    float angle = u_time * u_rotation_speed * (1.0 + fi * 0.3) * (mod(fi, 2.0) < 1.0 ? 1.0 : -1.0);
                    vec2 rp = rot2(angle) * p;

                    float ringDist = abs(sdCircle(rp, ringRadius)) - ringWidth;

                    // Ring-Farbe aus Primary/Secondary ableiten
                    vec3 ringColor = mix(u_color, u_secondary_color, clamp(fi * 0.15, 0.0, 1.0));

                    // Beat-Puls auf Ringen
                    float pulse = 1.0 + u_onset * u_pulse_intensity * sin(rp.y * 20.0 + u_time * 5.0);

                    float ringGlow = exp(-ringDist * ringDist * 2000.0) * u_glow_strength * pulse;
                    col += ringColor * ringGlow;
                }

                // === Transient Flash ===
                col += u_color * explosion * 0.5;

                // === Chromatic Aberration ===
                if (u_chromatic_aberration > 0.0) {
                    float ca = u_chromatic_aberration * (u_rms + u_onset);
                    vec2 caOffset = normalize(uv) * ca;
                    // Simuliere CA durch RGB-Shift
                    col.r += glow * 0.2 * ca * 100.0;
                    col.b += glow * 0.2 * ca * 100.0;
                }

                // Tonemapping
                col = col / (1.0 + col);
                col = pow(col, vec3(0.95));
                col *= u_brightness;

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

        self._prog["u_resolution"].value = (self.width, self.height)
        self._prog["u_time"].value = time
        self._prog["u_rms"].value = uniforms["u_energy"]
        self._prog["u_onset"].value = uniforms["u_beat"]
        self._prog["u_transient"].value = uniforms["u_impact"]
        self._prog["u_centroid"].value = uniforms["u_detail"]
        self._prog["u_beat_intensity"].value = uniforms.get("u_beat_intensity", uniforms["u_beat"])
        self._prog["u_color"].value = color
        self._prog["u_secondary_color"].value = secondary
        self._prog["u_background_color"].value = background
        self._prog["u_core_intensity"].value = self.params["core_intensity"]
        self._prog["u_ring_count"].value = self.params["ring_count"]
        self._prog["u_noise_scale"].value = self.params["noise_scale"]
        self._prog["u_glow_strength"].value = self.params["glow_strength"]
        self._prog["u_chromatic_aberration"].value = self.params["chromatic_aberration"]
        self._prog["u_rotation_speed"].value = self.params["rotation_speed"]
        self._prog["u_core_base_radius"].value = self.params["core_base_radius"]
        self._prog["u_ring_base_radius"].value = self.params["ring_base_radius"]
        self._prog["u_ring_spacing"].value = self.params["ring_spacing"]
        self._prog["u_ring_width"].value = self.params["ring_width"]
        self._prog["u_noise_amount"].value = self.params["noise_amount"]
        self._prog["u_pulse_intensity"].value = self.params["pulse_intensity"]
        self._prog["u_specular_power"].value = self.params["specular_power"]
        self._prog["u_bg_brightness"].value = self.params["bg_brightness"]
        self._prog["u_brightness"].value = self.params["brightness"]

        self._vao.render(mode=moderngl.TRIANGLE_STRIP)
