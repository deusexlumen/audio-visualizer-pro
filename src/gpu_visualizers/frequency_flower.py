"""
GPU-beschleunigte Frequency Flower.

Bluetenblaetter als Dreiecke, Stempel und Pollen als instanced Quads,
wachsender Staengel als Triangle Strip.
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
        'petal_width': (30.0, 10.0, 80.0, 1.0),
        'rotation_speed': (1.0, 0.0, 3.0, 0.1),
        'beat_rotation_boost': (0.08, 0.0, 0.5, 0.01),
        'center_size': (25.0, 10.0, 60.0, 1.0),
        'pollen_threshold': (0.3, 0.0, 1.0, 0.05),
        'stem_growth': (0.4, 0.0, 1.0, 0.05),
        'stem_bend': (20.0, 0.0, 60.0, 1.0),
    }

    def _setup(self):
        """Initialisiere Shader und VBOs."""
        # --- Polygon-Shader (Bluetenblaetter, Staengel, Stempel) ---
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
            uniform float u_brightness;
            in vec3 v_color;
            in float v_alpha;
            out vec4 f_color;
            void main() {
                f_color = vec4(v_color * u_brightness, v_alpha);
            }
            """,
        )

        # --- Partikel-Shader (Stempel, Pollen, Highlights) ---
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
            uniform float u_brightness;
            in vec3 v_color;
            in float v_alpha;
            in vec2 v_local;
            out vec4 f_color;
            void main() {
                float dist = length(v_local);
                if (dist > 1.0) discard;
                float core = 1.0 - smoothstep(0.0, 0.5, dist);
                float glow = exp(-dist * dist * 4.0);
                vec3 col = v_color * (core + glow * 0.7);
                f_color = vec4(col * u_brightness, (core * 0.9 + glow * 0.4) * v_alpha);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self._quad_vbo = self.ctx.buffer(quad.tobytes())

        # Polygon-VBO (Bluetenblaetter + Staengel + Stempel)
        max_poly_verts = 6000
        self._poly_vbo = self.ctx.buffer(reserve=max_poly_verts * 6 * 4, dynamic=True)
        self._poly_vao = self.ctx.vertex_array(
            self._poly_prog,
            [(self._poly_vbo, "2f 3f 1f", "in_pos", "in_color", "in_alpha")],
        )

        # Partikel-VBO
        max_particles = 200
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

    def _append_petal(self, verts, center, angle, length, width, color, alpha):
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

        # Highlight-Position in der Blattmitte
        return (center[0] + tip_x) / 2.0, (center[1] + tip_y) / 2.0

    def _shift_hue(self, rgb, shift):
        """Verschiebt den Hue eines RGB-Tupels um shift (0.0-1.0)."""
        h, s, v = self._rgb_to_hsv(*rgb)
        return self._hsv_to_rgb((h + shift) % 1.0, s, v)

    def render(self, features: dict, time: float):
        """Rendert Bluetenblaetter, Stempel, Pollen und Staengel."""
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        rms = f["rms"]
        onset = f["onset"]
        chroma = f["chroma"]
        progress = f.get("progress", time / features.get("duration", 1.0))

        base_color = self._chroma_to_color(chroma)

        num_petals = int(self.params["num_petals"])
        num_layers = int(self.params["layer_count"])
        cx, cy = self.center

        # Farbpalette: Basisfarbe mit leichter Hue-Verschiebung pro Blatt
        petal_colors = []
        for i in range(num_petals):
            shift = (i / max(num_petals, 1)) * 0.15
            petal_colors.append(self._shift_hue(base_color, shift))

        # Rotation
        rotation_speed = self.params["rotation_speed"]
        beat_boost = self.params["beat_rotation_boost"]
        self.rotation += 0.003 * rotation_speed + rms * 0.02 * rotation_speed
        if onset > 0.4:
            self.rotation += beat_boost

        poly_verts = []
        particle_idx = 0
        petal_width_base = self.params["petal_width"]

        # --- Bluetenblaetter ---
        for layer in range(num_layers - 1, -1, -1):
            layer_scale = 1.0 - (layer * 0.25)
            layer_rot = self.rotation + (layer * np.pi / num_petals)

            for i in range(num_petals):
                angle = (i / num_petals) * np.pi * 2.0 + layer_rot
                ci = (int(np.argmax(chroma)) + i + layer * 2) % 12 if chroma is not None and chroma.size > 0 else 0
                strength = chroma[ci] if chroma is not None and chroma.size > 0 else 0.5
                petal_length = self.base_petal_length * layer_scale * (0.6 + strength * 0.5 + rms * 0.3)
                petal_width = petal_width_base * layer_scale * (1.0 + rms * 0.5)

                base_color_local = petal_colors[(i + layer) % num_petals]
                if layer > 0:
                    layer_color = tuple(c * (1.0 - layer * 0.15) for c in base_color_local)
                else:
                    layer_color = base_color_local

                mid_x, mid_y = self._append_petal(
                    poly_verts, self.center, angle, petal_length, petal_width, layer_color, 1.0
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
        center_size = self.params["center_size"]
        center_radius = center_size + rms * 30.0
        center_color = self._shift_hue(base_color, 0.5)

        # Stempel als gefuellter Kreis (Triangle Fan)
        segments = 32
        for i in range(segments):
            a1 = (i / segments) * np.pi * 2.0
            a2 = ((i + 1) / segments) * np.pi * 2.0
            poly_verts.append([cx, cy, *center_color, 1.0])
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
            stamen_color = self._shift_hue(center_color, i * 0.1)
            if particle_idx < len(self._particle_data):
                self._particle_data[particle_idx] = [
                    sx, sy, stamen_color[0], stamen_color[1], stamen_color[2], stamen_size, 1.0
                ]
                particle_idx += 1

        # --- Pollen bei Beats ---
        pollen_threshold = self.params["pollen_threshold"]
        if onset > pollen_threshold or rms > 0.7:
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
        stem_growth = self.params["stem_growth"]
        stem_length = self.height * stem_growth * progress
        if stem_length >= 10.0:
            stem_top = cy + 20.0
            stem_bottom = stem_top + stem_length
            stem_x = cx
            stem_bend = self.params["stem_bend"]
            bend = np.sin(progress * 10.0) * stem_bend
            stem_color = self._shift_hue(base_color, 0.3)
            stem_width = 8.0

            points_left = []
            points_right = []
            max_stem_points = 200
            step = max(1.0, stem_length / max_stem_points)
            for i in range(int(stem_length / step)):
                t = i * step
                y = stem_top + t
                x_offset = np.sin(t * 0.02 + progress * 5.0) * bend
                x = stem_x + x_offset
                w = stem_width * (1.0 - t / stem_length * 0.3)
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

        brightness = self.params.get("brightness", 1.0)
        self._poly_prog["u_brightness"].value = brightness
        self._particle_prog["u_brightness"].value = brightness

        # Polygone (Bluetenblaetter + Staengel + Stempel)
        if poly_verts:
            arr = np.array(poly_verts, dtype=np.float32)
            data = arr.tobytes()
            # VBO bei Bedarf vergroessern
            if len(data) > self._poly_vbo.size:
                self._poly_vbo = self.ctx.buffer(data, dynamic=True)
                self._poly_vao = self.ctx.vertex_array(
                    self._poly_prog,
                    [(self._poly_vbo, "2f 3f 1f", "in_pos", "in_color", "in_alpha")],
                )
            else:
                self._poly_vbo.write(data)
            self._poly_prog["u_resolution"].value = (self.width, self.height)
            self._poly_vao.render(mode=moderngl.TRIANGLES)

        # Partikel (Stempel + Pollen + Highlights)
        if particle_idx > 0:
            self._particle_prog["u_resolution"].value = (self.width, self.height)
            self._particle_vbo.write(self._particle_data[:particle_idx].tobytes())
            self._particle_vao.render(mode=moderngl.TRIANGLE_STRIP, instances=particle_idx)

        self.ctx.disable(moderngl.BLEND)
