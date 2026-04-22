"""
GPU-beschleunigte typografische Visualisierung fuer Podcasts.

Symmetrische Wellenform-Balken, Mittellinie, Beat-Indikator und Fortschrittsbalken.
Text-Overlays werden vom Post-Processing-System uebernommen.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class TypographicGPU(BaseGPUVisualizer):
    """
    Minimalistische Podcast-Visualisierung mit symmetrischen Balken.
    """

    PARAMS = {
        'bar_width': (3, 1, 10, 1),
        'bar_spacing': (1, 0, 5, 1),
        'animation_speed': (0.2, 0.0, 1.0, 0.05),
    }

    def _setup(self):
        """Initialisiere Shader und VBOs."""
        # --- Rechteck-Shader (Balken, Progress, Indikator) ---
        self._rect_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;

            in vec2 in_vertex_pos;
            in vec2 in_rect_pos;
            in vec2 in_rect_size;
            in vec3 in_color;
            in float in_alpha;

            out vec3 v_color;
            out float v_alpha;

            void main() {
                vec2 pixel_pos = in_rect_pos + in_vertex_pos * in_rect_size;
                vec2 ndc = (pixel_pos / u_resolution) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
                v_color = in_color;
                v_alpha = in_alpha;
            }
            """,
            fragment_shader="""
            #version 330
            uniform float u_brightness;
            in vec3 v_color;
            in float v_alpha;
            out vec4 f_color;
            void main() {
                f_color = vec4(v_color * u_brightness, v_alpha);
            }
            """,
        )

        # --- Line-Shader (Mittellinie) ---
        self._line_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;
            in vec2 in_pos;
            in vec3 in_color;
            in float in_alpha;
            out vec3 v_color;
            out float v_alpha;
            void main() {
                vec2 ndc = (in_pos / u_resolution) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
                v_color = in_color;
                v_alpha = in_alpha;
            }
            """,
            fragment_shader="""
            #version 330
            uniform float u_brightness;
            in vec3 v_color;
            in float v_alpha;
            out vec4 f_color;
            void main() { f_color = vec4(v_color * u_brightness, v_alpha); }
            """,
        )

        # Quad-VBO
        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self._quad_vbo = self.ctx.buffer(quad.tobytes())

        max_rects = 500
        self._rect_data = np.zeros((max_rects, 8), dtype=np.float32)
        self._rect_vbo = self.ctx.buffer(reserve=max_rects * 8 * 4, dynamic=True)
        self._rect_vao = self.ctx.vertex_array(
            self._rect_prog,
            [
                (self._quad_vbo, "2f", "in_vertex_pos"),
                (self._rect_vbo, "2f 2f 3f 1f /i", "in_rect_pos", "in_rect_size", "in_color", "in_alpha"),
            ],
        )

        self._line_vbo = self.ctx.buffer(reserve=100 * 6 * 4, dynamic=True)
        self._line_vao = self.ctx.vertex_array(
            self._line_prog,
            [(self._line_vbo, "2f 3f 1f", "in_pos", "in_color", "in_alpha")],
        )

    def render(self, features: dict, time: float):
        """Rendert symmetrische Balken, Mittellinie und Indikatoren."""
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        rms = f["rms"]
        onset = f["onset"]
        spectral = f["spectral_centroid"]
        progress = f.get("progress", time / features.get("duration", 1.0))

        bar_w = int(self.params["bar_width"])
        spacing = int(self.params["bar_spacing"])
        anim_speed = self.params["animation_speed"]

        num_bars = self.width // (bar_w + spacing)
        wave_y = self.height / 2.0
        max_h = self.height * 0.3

        # Farben
        primary = (0.0, 0.78, 1.0)      # Cyan
        secondary = (1.0, 0.39, 0.39)   # Rot

        rect_idx = 0

        # --- Balken ---
        for i in range(num_bars):
            phase = (i / num_bars) * np.pi * 4
            t_off = frame_idx * anim_speed
            wave = np.sin(phase + t_off) * rms
            wave += np.sin(phase * 2.5 + t_off * 1.3) * 0.25
            wave = max(-1.0, min(1.0, wave))
            bar_h = abs(wave) * max_h

            # Farbverlauf
            if i < num_bars // 2:
                ratio = i / (num_bars // 2) if num_bars // 2 > 0 else 0
                color = (
                    primary[0] * ratio + secondary[0] * (1 - ratio),
                    primary[1] * ratio + secondary[1] * (1 - ratio),
                    primary[2] * ratio + secondary[2] * (1 - ratio),
                )
            else:
                ratio = (i - num_bars // 2) / (num_bars // 2) if num_bars // 2 > 0 else 0
                color = (
                    secondary[0] * ratio + primary[0] * (1 - ratio),
                    secondary[1] * ratio + primary[1] * (1 - ratio),
                    secondary[2] * ratio + primary[2] * (1 - ratio),
                )

            x = i * (bar_w + spacing) + bar_w / 2.0

            # Oberer Balken
            if rect_idx < len(self._rect_data):
                self._rect_data[rect_idx] = [
                    x, wave_y - bar_h / 2.0,
                    bar_w / 2.0, bar_h / 2.0,
                    color[0], color[1], color[2], 1.0
                ]
                rect_idx += 1

            # Unterer Balken
            if rect_idx < len(self._rect_data):
                self._rect_data[rect_idx] = [
                    x, wave_y + bar_h / 2.0,
                    bar_w / 2.0, bar_h / 2.0,
                    color[0], color[1], color[2], 1.0
                ]
                rect_idx += 1

        # --- Mittellinie ---
        line_color = (max(0.0, primary[0] - 0.4), max(0.0, primary[1] - 0.4), max(0.0, primary[2] - 0.4))
        line_verts = np.array([
            [0.0, wave_y, *line_color, 1.0],
            [self.width, wave_y, *line_color, 1.0],
        ], dtype=np.float32)

        # --- Beat-Indikator (Ring) ---
        if onset > 0.3:
            radius = 20.0 + onset * 30.0
            if rect_idx < len(self._rect_data):
                self._rect_data[rect_idx] = [
                    self.width / 2.0, wave_y,
                    radius, radius,
                    primary[0], primary[1], primary[2], 0.6
                ]
                rect_idx += 1

        # --- Fortschrittsbalken ---
        bar_y = self.height - 30.0
        bar_width = self.width * 0.6
        bar_x = (self.width - bar_width) / 2.0

        # Hintergrund
        if rect_idx < len(self._rect_data):
            self._rect_data[rect_idx] = [
                bar_x + bar_width / 2.0, bar_y + 3.0,
                bar_width / 2.0, 3.0,
                0.2, 0.2, 0.2, 1.0
            ]
            rect_idx += 1

        # Füllung
        progress_width = bar_width * progress
        if rect_idx < len(self._rect_data) and progress_width > 0:
            self._rect_data[rect_idx] = [
                bar_x + progress_width / 2.0, bar_y + 3.0,
                progress_width / 2.0, 3.0,
                primary[0], primary[1], primary[2], 1.0
            ]
            rect_idx += 1

        # --- Rendern ---
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        brightness = self.params.get("brightness", 1.0)
        self._rect_prog["u_brightness"].value = brightness
        self._line_prog["u_brightness"].value = brightness

        # Linien-Width aus Parameter
        line_width_val = self.params.get("line_width", 0.003)
        self.ctx.line_width = max(1.0, line_width_val * 400.0)

        # Mittellinie
        self._line_prog["u_resolution"].value = (self.width, self.height)
        self._line_vbo.write(line_verts.tobytes())
        self._line_vao.render(mode=moderngl.LINES)

        # Rechtecke
        if rect_idx > 0:
            self._rect_prog["u_resolution"].value = (self.width, self.height)
            self._rect_vbo.write(self._rect_data[:rect_idx].tobytes())
            self._rect_vao.render(mode=moderngl.TRIANGLE_STRIP, instances=rect_idx)

        self.ctx.line_width = 1.0
        self.ctx.disable(moderngl.BLEND)
