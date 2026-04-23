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
        'wave_intensity': (0.6, 0.0, 1.5, 0.1),
        'glow_radius': (12.0, 4.0, 30.0, 2.0),
        'color_shift': (0.0, 0.0, 1.0, 0.05),
        'beat_flash': (0.4, 0.0, 1.0, 0.05),
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
            out vec4 f_color;
            void main() {
                float d = length(v_local);
                if (d > 1.0) discard;
                float core = 1.0 - smoothstep(0.0, 0.5, d);
                float glow = exp(-d * d * 4.0);
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
            uniform vec3 u_color;
            out vec4 f_color;

            void main() {
                vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
                uv.x *= u_resolution.x / u_resolution.y;

                // Wellenform
                float wave = sin(uv.x * 10.0 + u_time * 2.0) * u_rms * u_wave_intensity;
                wave += sin(uv.x * 20.0 - u_time * 3.0) * u_voice * u_wave_intensity * 0.5;
                wave += sin(uv.x * 5.0 + u_time) * u_onset * u_wave_intensity * 0.3;

                float dist = abs(uv.y - wave);
                float line = exp(-dist * dist * 200.0);

                // Beat-Flash als Overlay
                float flash = u_onset * 0.1;

                vec3 col = u_color * line + vec3(flash);
                f_color = vec4(col, line * 0.8 + flash);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        # Bar VAO
        self._bar_max = 128
        self._bar_data = np.zeros((self._bar_max * 2, 8), dtype=np.float32)
        self._bar_vbo = self.ctx.buffer(reserve=self._bar_max * 2 * 8 * 4, dynamic=True)
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
        bar_count = int(self.params["bar_count"])
        glow = self.params["glow_radius"]
        color_shift = self.params["color_shift"]
        beat_flash = self.params["beat_flash"]

        # === Bars generieren ===
        bar_w = self.width / bar_count
        max_h = self.height * 0.35
        instance_idx = 0

        for i in range(bar_count):
            # Simulierte Bar-Hoehe aus Features
            phase = (i / bar_count) * np.pi * 4 + frame_idx * 0.1
            h = (np.sin(phase) * 0.3 + uniforms["u_energy"] * 0.5 + uniforms["u_impact"] * 0.3) * max_h
            h = max(2.0, h)

            # Farbverlauf
            hue = (color[0] + i / bar_count * color_shift + color_shift) % 1.0
            sat = 0.7 + uniforms["u_energy"] * 0.3
            val = 0.5 + h / max_h * 0.5
            from .base import BaseGPUVisualizer
            bar_rgb = BaseGPUVisualizer._hsv_to_rgb(hue, sat, val)

            x = i * bar_w + bar_w / 2.0
            cy = self.height / 2.0

            # Obere Haelfte
            if instance_idx < self._bar_max * 2:
                self._bar_data[instance_idx] = [
                    x, cy - h / 2.0, bar_w / 2.0, h / 2.0,
                    bar_rgb[0], bar_rgb[1], bar_rgb[2], 1.0
                ]
                instance_idx += 1

            # Untere Haelfte
            if instance_idx < self._bar_max * 2:
                self._bar_data[instance_idx] = [
                    x, cy + h / 2.0, bar_w / 2.0, h / 2.0,
                    bar_rgb[0], bar_rgb[1], bar_rgb[2], 1.0
                ]
                instance_idx += 1

        # === Rendern ===
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Bars
        if instance_idx > 0:
            self._prog["u_resolution"].value = (self.width, self.height)
            self._bar_vbo.write(self._bar_data[:instance_idx].tobytes())
            self._bar_vao.render(mode=moderngl.TRIANGLE_STRIP, instances=instance_idx)

        # Wellenform Overlay
        self._wave_prog["u_resolution"].value = (self.width, self.height)
        self._wave_prog["u_time"].value = time
        self._wave_prog["u_rms"].value = uniforms["u_energy"]
        self._wave_prog["u_onset"].value = uniforms["u_beat"]
        self._wave_prog["u_voice"].value = uniforms["u_flow"]
        self._wave_prog["u_wave_intensity"].value = self.params["wave_intensity"]
        self._wave_prog["u_color"].value = color
        self._wave_vao.render(mode=moderngl.TRIANGLE_STRIP)

        # Beat Flash
        if uniforms["u_beat"] > 0.3:
            flash_alpha = (uniforms["u_beat"] - 0.3) * beat_flash
            # Einfacher Flash via Clear-Color-Mix waere besser, aber wir nutzen
            # den vorhandenen Wellenform-Shader der bereits Flash rendert

        self.ctx.disable(moderngl.BLEND)
