"""
GPU-beschleunigte Frequency Flower.

Bluetenblaetter als Dreiecke, Stempel und Pollen als instanced Quads,
wachsender Staengel als Line Strip.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class FrequencyFlowerGPU(BaseGPUVisualizer):
    """
    Organische Blumen-Form mit audio-reaktiven Bluetenblaettern.
    """

    PARAMS = {
        'num_petals': (8, 4, 16, 1),
        'layer_count': (3, 1, 6, 1),
    }

    def _setup(self):
        """Initialisiere Shader und VBOs."""
        # --- Polygon-Shader (Bluetenblaetter, Staengel) ---
        self._poly_prog = self.ctx.program(
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
            in vec3 v_color;
            in float v_alpha;
            out vec4 f_color;
            void main() {
                f_color = vec4(v_color, v_alpha);
            }
            """,
        )

        # --- Partikel-Shader (Stempel, Pollen) ---
        self._particle_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;
            in vec2 in_vertex_pos;
            in vec2 in_particle_pos;
            in vec3 in_color;
            in float in_size;
            in float in_alpha;
            out vec3 v_color;
            out float v_alpha;
            out vec2 v_local;
            void main() {
                vec2 pixel_pos = in_particle_pos + in_vertex_pos * in_size;
                vec2 ndc = (pixel_pos / u_resolution) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
                v_color = in_color;
                v_alpha = in_alpha;
                v_local = in_vertex_pos;
            }
            """,
            fragment_shader="""
            #version 330
            in vec3 v_color;
            in float v_alpha;
            in vec2 v_local;
            out vec4 f_color;
            void main() {
                float dist = length(v_local);
                if (dist > 1.0) discard;
                float core = 1.0 - smoothstep(0.0, 0.5, dist);
                float glow = exp(-dist * dist * 4.0);
                f_color = vec4(v_color * (core + glow * 0.7), (core * 0.9 + glow * 0.4) * v_alpha);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self._quad_vbo = self.ctx.buffer(quad.tobytes())

        # Polygon-VBO (Bluetenblaetter + Staengel)
        max_poly_verts = 5000
        self._poly_vbo = self.ctx.buffer(reserve=max_poly_verts * 6 * 4, dynamic=True)
        self._poly_vao = self.ctx.vertex_array(
            self._poly_prog,
            [(self._poly_vbo, "2f 3f 1f", "in_pos", "in_color", "in_alpha")],
        )

        # Partikel-VBO
        max_particles = 100
        self._particle_data = np.zeros((max_particles, 7), dtype=np.float32)
        self._particle_vbo = self.ctx.buffer(reserve=max_particles * 7 * 4, dynamic=True)
        self._particle_vao = self.ctx.vertex_array(
            self._particle_prog,
            [
                (self._quad_vbo, "2f", "in_vertex_pos"),
                (self._particle_vbo, "2f 3f 1f 1f /i", "in_particle_pos", "in_color", "in_size", "in_alpha"),
            ],
        )

        self.center = (self.width / 2.0, self.height / 2.0)
        self.base_petal_length = min(self.width, self.height) / 3.0
        self.rotation = 0.0

    def _append_petal(self, verts, center, angle, length, width, color, alpha, rms):
        """Fuegt ein Bluetenblatt als Dreiecke hinzu (Fan vom Zentrum)."""
        tip_x = center[0] + np.cos(angle) * length
        tip_y = center[1] + np.sin(angle) * length

        side_angle1 = angle - np.pi / 6.0
        side_angle2 = angle + np.pi / 6.0
        side_len = width * 0.8

        side1_x = center[0] + np.cos(angle) * (length * 0.5) + np.cos(side_angle1) * side_len
        side1_y = center[1] + np.sin(angle) * (length * 0.5) + np.sin(side_angle1) * side_len
        side2_x = center[0] + np.cos(angle) * (length * 0.5) + np.cos(side_angle2) * side_len
        side2_y = center[1] + np.sin(angle) * (length * 0.5) + np.sin(side_angle2) * side_len

        base_width = width * 0.4
        base1_x = center[0] + np.cos(angle - np.pi / 4.0) * base_width
        base1_y = center[1] + np.sin(angle - np.pi / 4.0) * base_width
        base2_x = center[0] + np.cos(angle + np.pi / 4.0) * base_width
        base2_y = center[1] + np.sin(angle + np.pi / 4.0) * base_width

        outline = tuple(max(0.0, c - 0.15) for c in color)

        # 5 Dreiecke fuer das Bluetenblatt (Fan)
        triangles = [
            (center, (base1_x, base1_y), (side1_x, side1_y)),
            (center, (side1_x, side1_y), (tip_x, tip_y)),
            (center, (tip_x, tip_y), (side2_x, side2_y)),
            (center, (side2_x, side2_y), (base2_x, base2_y)),
            (center, (base2_x, base2_y), (base1_x, base1_y)),
        ]
        for a, b, c in triangles:
            verts.append([*a, *color, alpha])
            verts.append([*b, *color, alpha])
            verts.append([*c, *color, alpha])

        # Highlight in der Mitte (kleiner Kreis als Partikel)
        return (center[0] + tip_x) / 2.0, (center[1] + tip_y) / 2.0

    def render(self, features: dict, time: float):
        """Rendert Bluetenblaetter, Stempel, Pollen und Staengel."""
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        rms = f["rms"]
        onset = f["onset"]
        chroma = f["chroma"]
        spectral = f["spectral_centroid"]
        progress = f.get("progress", time / features.get("duration", 1.0))

        if chroma is not None and chroma.size > 0:
            dominant = int(np.argmax(chroma))
            base_hue = dominant / 12.0
        else:
            dominant = 0
            base_hue = 0.5

        num_petals = int(self.params["num_petals"])
        num_layers = int(self.params["layer_count"])
        cx, cy = self.center

        # Farbpalette
        petal_colors = []
        for i in range(num_petals):
            hue = (base_hue + i / num_petals * 0.3) % 1.0
            ci = (dominant + i) % 12
            strength = chroma[ci] if chroma is not None else 0.5
            sat = min(0.35, 0.25 + strength * 0.1)
            val = min(0.7, 0.4 + rms * 0.3)
            petal_colors.append(self._hsv_to_rgb(hue, sat, val))

        # Rotation
        self.rotation += 0.003 + rms * 0.02
        if onset > 0.4:
            self.rotation += 0.08

        poly_verts = []
        particle_idx = 0

        # --- Bluetenblaetter ---
        for layer in range(num_layers - 1, -1, -1):
            layer_scale = 1.0 - (layer * 0.25)
            layer_rot = self.rotation + (layer * np.pi / num_petals)

            for i in range(num_petals):
                angle = (i / num_petals) * np.pi * 2.0 + layer_rot
                ci = (dominant + i + layer * 2) % 12
                strength = chroma[ci] if chroma is not None else 0.5
                petal_length = self.base_petal_length * layer_scale * (0.6 + strength * 0.5 + rms * 0.3)
                petal_width = 30.0 * layer_scale * (1.0 + rms * 0.5)

                base_color = petal_colors[(i + layer) % num_petals]
                if layer > 0:
                    layer_color = tuple(c * (1.0 - layer * 0.15) for c in base_color)
                else:
                    layer_color = base_color

                mid_x, mid_y = self._append_petal(
                    poly_verts, self.center, angle, petal_length, petal_width, layer_color, 1.0, rms
                )

                # Highlight in der Mitte
                if particle_idx < len(self._particle_data):
                    hi_color = tuple(min(1.0, c + 0.23) for c in layer_color)
                    self._particle_data[particle_idx] = [
                        mid_x, mid_y,
                        hi_color[0], hi_color[1], hi_color[2],
                        petal_width * 0.15, 0.8
                    ]
                    particle_idx += 1

        # --- Bluetenmitte (Stempel) ---
        center_radius = 25.0 + rms * 30.0
        center_hue = (base_hue + 0.5) % 1.0
        center_color = self._hsv_to_rgb(center_hue, 0.35, 0.7)

        # Stempel als Kreis (Line Loop)
        segments = 32
        for i in range(segments):
            a1 = (i / segments) * np.pi * 2.0
            a2 = ((i + 1) / segments) * np.pi * 2.0
            poly_verts.append([cx + np.cos(a1) * center_radius, cy + np.sin(a1) * center_radius, *center_color, 1.0])
            poly_verts.append([cx + np.cos(a2) * center_radius, cy + np.sin(a2) * center_radius, *center_color, 1.0])

        # Stempel-Details (kleine Punkte)
        num_stamens = 5
        for i in range(num_stamens):
            a = (i / num_stamens) * np.pi * 2.0
            dist = center_radius * 0.5
            sx = cx + np.cos(a) * dist
            sy = cy + np.sin(a) * dist
            stamen_size = 5.0 + rms * 8.0
            stamen_color = self._hsv_to_rgb((center_hue + i * 0.1) % 1.0, 0.35, 0.7)
            if particle_idx < len(self._particle_data):
                self._particle_data[particle_idx] = [
                    sx, sy, stamen_color[0], stamen_color[1], stamen_color[2], stamen_size, 1.0
                ]
                particle_idx += 1

        # --- Pollen bei Beats ---
        if onset > 0.3 or rms > 0.7:
            np.random.seed(frame_idx // 5)
            num_pollen = int(10 + rms * 20)
            for i in range(num_pollen):
                if particle_idx >= len(self._particle_data):
                    break
                angle = np.random.random() * np.pi * 2.0
                dist = np.random.uniform(self.base_petal_length * 0.3, self.base_petal_length * 1.2)
                px = cx + np.cos(angle) * dist
                py = cy + np.sin(angle) * dist
                size = 2.0 + rms * 4.0 + onset * 3.0
                ci = i % len(petal_colors)
                self._particle_data[particle_idx] = [
                    px, py, petal_colors[ci][0], petal_colors[ci][1], petal_colors[ci][2], size, 1.0
                ]
                particle_idx += 1

        # --- Staengel ---
        stem_length = self.height * 0.4 * progress
        if stem_length >= 10.0:
            stem_top = cy + 20.0
            stem_bottom = stem_top + stem_length
            stem_x = cx
            bend = np.sin(progress * 10.0) * 20.0
            stem_hue = (base_hue + 0.3) % 1.0
            stem_color = self._hsv_to_rgb(stem_hue, 0.35, 0.4)
            stem_width = 8.0

            points_left = []
            points_right = []
            for i in range(int(stem_length)):
                y = stem_top + i
                x_offset = np.sin(i * 0.02 + progress * 5.0) * bend
                x = stem_x + x_offset
                w = stem_width * (1.0 - i / stem_length * 0.3)
                points_left.append((x - w / 2.0, y))
                points_right.append((x + w / 2.0, y))

            if len(points_left) > 1:
                # Staengel als Triangle Strip
                for i in range(len(points_left) - 1):
                    poly_verts.append([*points_left[i], *stem_color, 1.0])
                    poly_verts.append([*points_right[i], *stem_color, 1.0])
                    poly_verts.append([*points_left[i + 1], *stem_color, 1.0])
                    poly_verts.append([*points_left[i + 1], *stem_color, 1.0])
                    poly_verts.append([*points_right[i], *stem_color, 1.0])
                    poly_verts.append([*points_right[i + 1], *stem_color, 1.0])

        # --- Rendern ---
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Polygone (Bluetenblaetter + Staengel)
        if poly_verts:
            arr = np.array(poly_verts, dtype=np.float32)
            self._poly_prog["u_resolution"].value = (self.width, self.height)
            self._poly_vbo.write(arr.tobytes())
            self._poly_vao.render(mode=moderngl.TRIANGLES)

        # Partikel (Stempel + Pollen + Highlights)
        if particle_idx > 0:
            self._particle_prog["u_resolution"].value = (self.width, self.height)
            self._particle_vbo.write(self._particle_data[:particle_idx].tobytes())
            self._particle_vao.render(mode=moderngl.TRIANGLE_STRIP, instances=particle_idx)

        self.ctx.disable(moderngl.BLEND)
