"""
Spectrum Genesis - Signature Hybrid-Visualizer.

Kombiniert die Staerke von Musik- und Podcast-Visualisierung:
- Spectrum Bars mit SDF-Glow
- Wellenform-Overlay
- Beat-Reaktion + Voice-Flow kombiniert
- Chroma-Farbverlaeufe
- Chromatic Aberration bei starken Beats

Optimiert fuer Hybrid-Audio: Sprache mit Musik-Begleitung.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class SpectrumGenesisGPU(BaseGPUVisualizer):
    """
    Spectrum Genesis - Der Allrounder.
    Bars, Wellenform und Glow in einem professionellen Package.
    """

    PARAMS = {
        'bar_count': (64, 16, 128, 8),
        'bar_height': (0.35, 0.1, 0.7, 0.05),
        'wave_intensity': (0.6, 0.0, 1.5, 0.1),
        'wave_frequency': (10.0, 1.0, 40.0, 1.0),
        'wave_complexity': (3, 1, 6, 1),
        'glow_radius': (12.0, 4.0, 30.0, 2.0),
        'color_shift': (0.15, 0.0, 1.0, 0.05),
        'beat_flash': (0.4, 0.0, 1.0, 0.05),
        'peak_hold': (1.0, 0.0, 1.0, 1.0),
        'peak_decay': (0.02, 0.005, 0.1, 0.005),
        'reflection': (0.35, 0.0, 1.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Balken": ["bar_count", "bar_height", "glow_radius", "color_shift"],
        "Wellenform": ["wave_intensity", "wave_frequency", "wave_complexity"],
        "Extras": ["beat_flash", "peak_hold", "peak_decay", "reflection"],
    }

    def _setup(self):
        self._prog = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;
            in vec2 in_pos;
            in vec2 in_center;
            in vec2 in_size;
            in vec3 in_color;
            in float in_alpha;
            out vec3 v_color;
            out float v_alpha;
            out vec2 v_local;
            void main() {
                vec2 pixel = in_center + in_pos * in_size;
                vec2 ndc = (pixel / u_resolution) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
                v_color = in_color;
                v_alpha = in_alpha;
                v_local = in_pos;
            }
            """,
            fragment_shader="""
            #version 330
            in vec3 v_color;
            in float v_alpha;
            in vec2 v_local;
            uniform float u_glow_radius;
            out vec4 f_color;
            void main() {
                // Rechteckige Balken mit abgerundeten Ecken statt Kreisen
                vec2 q = abs(v_local);
                float d = max(q.x, q.y);
                if (d > 1.0) discard;

                float core = 1.0 - smoothstep(0.85, 1.0, d);
                // glow_radius steuert die Weite des Außenleuchtens
                float glow = exp(-max(0.0, d - 0.5) * max(0.0, d - 0.5) * u_glow_radius);
                vec3 col = v_color * (core + glow * 0.8);
                float a = (core * 0.95 + glow * 0.5) * v_alpha;
                f_color = vec4(col, a);
            }
            """,
        )

        # Fullscreen shader fuer Wellenform
        self._wave_prog = self.ctx.program(
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
            uniform float u_voice;
            uniform float u_wave_intensity;
            uniform float u_wave_frequency;
            uniform int u_wave_complexity;
            uniform float u_beat_flash;
            uniform vec3 u_color;
            out vec4 f_color;

            void main() {
                vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
                uv.x *= u_resolution.x / u_resolution.y;

                // Wellenform: Anzahl und Frequenz der Komponenten per Parameter
                float wave = 0.0;
                for (int i = 0; i < u_wave_complexity; i++) {
                    float fi = float(i);
                    float freq = u_wave_frequency * (1.0 + fi * 0.5);
                    float phase = u_time * (2.0 + fi);
                    float amp = 1.0 / (1.0 + fi * 0.7);
                    float source = mix(u_rms, u_voice, float(i == 1));
                    if (i == 2) source = u_onset;
                    wave += sin(uv.x * freq + phase) * source * u_wave_intensity * amp;
                }

                float dist = abs(uv.y - wave);
                float line = exp(-dist * dist * 200.0);

                // Beat-Flash als Overlay, Intensitaet per Parameter
                float flash = u_onset * u_beat_flash;

                vec3 col = u_color * line + vec3(flash);
                f_color = vec4(col, line * 0.8 + flash);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        # Bar VAO: bis zu 4 Instanzen pro Balken (obere/untere Haelfte + 2 Peak-Caps)
        self._bar_max = 128
        # Peak-Hold-Zustand pro Balken (bleibt ueber Frames erhalten)
        self._peaks = np.zeros(self._bar_max, dtype=np.float32)
        self._bar_slots = self._bar_max * 4
        self._bar_data = np.zeros((self._bar_slots, 8), dtype=np.float32)
        self._bar_vbo = self.ctx.buffer(reserve=self._bar_slots * 8 * 4, dynamic=True)
        quad_vbo = self.ctx.buffer(quad.tobytes())
        self._bar_vao = self.ctx.vertex_array(
            self._prog,
            [
                (quad_vbo, "2f", "in_pos"),
                (self._bar_vbo, "2f 2f 3f 1f /i", "in_center", "in_size", "in_color", "in_alpha"),
            ],
        )

        # Wave VAO
        wave_vbo = self.ctx.buffer(quad.tobytes())
        self._wave_vao = self.ctx.vertex_array(self._wave_prog, [(wave_vbo, "2f", "in_pos")])

    def render(self, features: dict, time: float):
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        mode = f.get("mode", "hybrid")
        uniforms = self._map_features_to_uniforms(f, mode=mode)

        color = self._chroma_to_color(uniforms["u_chroma"])
        bar_count = min(int(self.params["bar_count"]), self._bar_max)
        bar_height = self.params["bar_height"]
        glow = self.params["glow_radius"]
        color_shift = self.params["color_shift"]
        beat_flash = self.params["beat_flash"]
        peak_hold = self.params.get("peak_hold", 1.0) > 0.5
        peak_decay = float(self.params.get("peak_decay", 0.02))
        reflection = float(self.params.get("reflection", 0.35))

        # === Bars generieren ===
        bar_w = self.width / bar_count
        max_h = self.height * bar_height
        instance_idx = 0

        base_h, base_s, base_v = self._rgb_to_hsv(*color)

        for i in range(bar_count):
            # Simulierte Bar-Hoehe aus Features
            phase = (i / bar_count) * np.pi * 4 + frame_idx * 0.1
            h = (np.sin(phase) * 0.3 + uniforms["u_energy"] * 0.5 + uniforms["u_impact"] * 0.3) * max_h
            h = max(2.0, h)

            # Peak-Hold: Spitzenwert langsam absinken lassen
            norm_h = h / max_h if max_h > 0 else 0.0
            if norm_h >= self._peaks[i]:
                self._peaks[i] = norm_h
            else:
                self._peaks[i] = max(norm_h, self._peaks[i] - peak_decay)

            # Farbverlauf: bei Monochrom/Farblos einfach Helligkeit modulieren,
            # sonst den Hue entlang der Balken verschieben (Chroma-Sweep).
            val = base_v * (0.4 + (h / max_h) * 0.6)
            if base_s < 0.05:
                bar_rgb = self._hsv_to_rgb(base_h, base_s, val)
            else:
                hue = (base_h + (i / bar_count) * color_shift) % 1.0
                sat = base_s * (0.7 + uniforms["u_energy"] * 0.3)
                bar_rgb = self._hsv_to_rgb(hue, sat, val)

            x = i * bar_w + bar_w / 2.0
            cy = self.height / 2.0

            # Obere Haelfte
            if instance_idx < self._bar_slots:
                self._bar_data[instance_idx] = [
                    x, cy - h / 2.0, bar_w / 2.0, h / 2.0,
                    bar_rgb[0], bar_rgb[1], bar_rgb[2], 1.0
                ]
                instance_idx += 1

            # Untere Haelfte (Reflexion: gedimmt fuer Tiefenwirkung)
            if instance_idx < self._bar_slots:
                refl_alpha = 0.35 + reflection * 0.65
                self._bar_data[instance_idx] = [
                    x, cy + h / 2.0, bar_w / 2.0, h / 2.0,
                    bar_rgb[0] * (0.5 + reflection * 0.5),
                    bar_rgb[1] * (0.5 + reflection * 0.5),
                    bar_rgb[2] * (0.5 + reflection * 0.5),
                    refl_alpha
                ]
                instance_idx += 1

            # Peak-Hold-Cap: heller, duenner Balken an der Spitze
            if peak_hold and instance_idx < self._bar_slots:
                peak_h = self._peaks[i] * max_h
                cap_half = max(1.5, bar_w * 0.08)
                cap_rgb = self._hsv_to_rgb(
                    (base_h + (i / bar_count) * color_shift) % 1.0,
                    base_s * 0.4, min(1.0, val + 0.4)
                ) if base_s >= 0.05 else self._hsv_to_rgb(base_h, base_s, min(1.0, val + 0.4))
                self._bar_data[instance_idx] = [
                    x, cy - peak_h - cap_half, bar_w / 2.0, cap_half,
                    cap_rgb[0], cap_rgb[1], cap_rgb[2], 1.0
                ]
                instance_idx += 1

        # === Rendern ===
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Bars
        if instance_idx > 0:
            self._prog["u_resolution"].value = (self.width, self.height)
            self._prog["u_glow_radius"].value = glow
            self._bar_vbo.write(self._bar_data[:instance_idx].tobytes())
            self._bar_vao.render(mode=moderngl.TRIANGLE_STRIP, instances=instance_idx)

        # Wellenform Overlay
        self._wave_prog["u_resolution"].value = (self.width, self.height)
        self._wave_prog["u_time"].value = time
        self._wave_prog["u_rms"].value = uniforms["u_energy"]
        self._wave_prog["u_onset"].value = uniforms["u_beat"]
        self._wave_prog["u_voice"].value = uniforms["u_flow"]
        self._wave_prog["u_wave_intensity"].value = self.params["wave_intensity"]
        self._wave_prog["u_wave_frequency"].value = self.params["wave_frequency"]
        self._wave_prog["u_wave_complexity"].value = int(self.params["wave_complexity"])
        self._wave_prog["u_beat_flash"].value = beat_flash
        self._wave_prog["u_color"].value = color
        self._wave_vao.render(mode=moderngl.TRIANGLE_STRIP)

        self.ctx.disable(moderngl.BLEND)
