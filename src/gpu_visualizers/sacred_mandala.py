"""
GPU-beschleunigtes Sacred Geometry Mandala.

Heilige Geometrie als Linien-Muster (Flower of Life, Polygone, Sterne)
plus instanced Partikel bei hohem RMS.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class SacredMandalaGPU(BaseGPUVisualizer):
    """
    Sacred Geometry Mandala mit rotierenden, audio-reaktiven Mustern.
    """

    PARAMS = {
        'rotation_speed': (0.005, 0.001, 0.02, 0.001),
    }

    def _setup(self):
        """Initialisiere Shader und VBOs."""
        # --- Line-Shader (alle geometrischen Formen) ---
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
            in vec3 v_color;
            in float v_alpha;
            out vec4 f_color;
            void main() { f_color = vec4(v_color, v_alpha); }
            """,
        )

        # --- Partikel-Shader (Energie-Partikel) ---
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

        # Line-VBO (gross genug fuer alle Geometrie)
        self._max_line_verts = 5000
        self._line_vbo = self.ctx.buffer(reserve=self._max_line_verts * 6 * 4, dynamic=True)
        self._line_vao = self.ctx.vertex_array(
            self._line_prog,
            [(self._line_vbo, "2f 3f 1f", "in_pos", "in_color", "in_alpha")],
        )

        # Partikel-VBO
        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self._quad_vbo = self.ctx.buffer(quad.tobytes())
        self._max_particles = 50
        self._particle_data = np.zeros((self._max_particles, 7), dtype=np.float32)
        self._particle_vbo = self.ctx.buffer(reserve=self._max_particles * 7 * 4, dynamic=True)
        self._particle_vao = self.ctx.vertex_array(
            self._particle_prog,
            [
                (self._quad_vbo, "2f", "in_vertex_pos"),
                (self._particle_vbo, "2f 3f 1f 1f /i", "in_particle_pos", "in_color", "in_size", "in_alpha"),
            ],
        )

        self.rotation = 0.0
        self.base_radius = min(self.width, self.height) / 3.0

    def _append_circle(self, verts, cx, cy, radius, color, alpha, segments=32):
        """Fuegt einen Kreis als Liniensegmente hinzu."""
        for i in range(segments):
            a1 = (i / segments) * np.pi * 2.0
            a2 = ((i + 1) / segments) * np.pi * 2.0
            verts.append([cx + np.cos(a1) * radius, cy + np.sin(a1) * radius, *color, alpha])
            verts.append([cx + np.cos(a2) * radius, cy + np.sin(a2) * radius, *color, alpha])

    def _append_polygon(self, verts, cx, cy, radius, sides, rotation, color, alpha, rms):
        """Fuegt ein Polygon als Linien hinzu."""
        points = []
        for i in range(sides):
            angle = (i / sides) * np.pi * 2.0 + rotation
            points.append([cx + np.cos(angle) * radius, cy + np.sin(angle) * radius])
        for i in range(sides):
            p1 = points[i]
            p2 = points[(i + 1) % sides]
            verts.append([*p1, *color, alpha])
            verts.append([*p2, *color, alpha])
        # Linien zum Zentrum bei hohem RMS
        if rms > 0.5:
            for p in points:
                verts.append([*p, *color, alpha * 0.4])
                verts.append([cx, cy, *color, alpha * 0.4])

    def _append_star(self, verts, cx, cy, radius, points_count, rotation, color, alpha, rms):
        """Fuegt einen Stern als Linien hinzu."""
        outer = []
        inner = []
        for i in range(points_count):
            a = (i / points_count) * np.pi * 2.0 + rotation
            outer.append([cx + np.cos(a) * radius, cy + np.sin(a) * radius])
            ia = ((i + 0.5) / points_count) * np.pi * 2.0 + rotation
            ir = radius * 0.4 * (1.0 + rms * 0.3)
            inner.append([cx + np.cos(ia) * ir, cy + np.sin(ia) * ir])
        all_p = []
        for i in range(points_count):
            all_p.append(outer[i])
            all_p.append(inner[i])
        for i in range(len(all_p)):
            p1 = all_p[i]
            p2 = all_p[(i + 1) % len(all_p)]
            verts.append([*p1, *color, alpha])
            verts.append([*p2, *color, alpha])

    def render(self, features: dict, time: float):
        """Rendert Mandala-Layer, Verbindungen und Partikel."""
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        rms = f["rms"]
        onset = f["onset"]
        chroma = f["chroma"]
        spectral = f["spectral_centroid"]

        if chroma is not None and chroma.size > 0:
            dominant = int(np.argmax(chroma))
            base_hue = dominant / 12.0
        else:
            dominant = 0
            base_hue = 0.5

        # Farbpalette
        colors = []
        for i, strength in enumerate(chroma if chroma is not None else [0.5] * 12):
            rgb = self._hsv_to_rgb(i / 12.0, 0.6 + strength * 0.4, 0.5 + strength * 0.5)
            colors.append(rgb)

        # Rotation
        self.rotation += self.params["rotation_speed"] + rms * 0.02
        if onset > 0.4:
            self.rotation += self.params["rotation_speed"] * 20.0

        cx, cy = self.width / 2.0, self.height / 2.0
        line_verts = []

        # 1. Flower of Life (6 Kreise + Zentrum)
        num_circles = 6
        circle_radius = self.base_radius * 0.3 * (1.0 + rms * 0.3)
        for i in range(num_circles):
            angle = (i / num_circles) * np.pi * 2.0 + self.rotation
            x = cx + np.cos(angle) * (self.base_radius * 0.5)
            y = cy + np.sin(angle) * (self.base_radius * 0.5)
            ci = i % 12
            strength = float(chroma[ci]) if chroma is not None else 0.5
            c_rgb = self._hsv_to_rgb(ci / 12.0, 0.7 + strength * 0.3, 0.5 + strength * 0.5)
            self._append_circle(line_verts, x, y, circle_radius, c_rgb, 1.0)
        # Zentraler Kreis
        self._append_circle(line_verts, cx, cy, circle_radius * (1.0 + rms * 0.5), colors[dominant], 1.0, segments=48)

        # 2. Sechseck
        hex_color = colors[(dominant + 2) % 12]
        self._append_polygon(line_verts, cx, cy, self.base_radius * 0.75, 6, self.rotation, hex_color, 1.0, rms)

        # 3. Stern (12-spitzig)
        star_color = colors[(dominant + 4) % 12]
        self._append_star(line_verts, cx, cy, self.base_radius * 0.5, 12, self.rotation * -1.5, star_color, 1.0, rms)

        # 4. Inneres Dreieck
        tri_color = colors[(dominant + 6) % 12]
        self._append_polygon(line_verts, cx, cy, self.base_radius * 0.3, 3, self.rotation * 2.0, tri_color, 1.0, rms)

        # 5. Zentrumspuls (als Kreis)
        pulse_r = 20.0 + rms * 60.0
        self._append_circle(line_verts, cx, cy, pulse_r, colors[dominant], 1.0, segments=48)

        # 6. Verbindungslinien bei Beat
        if onset > 0.3:
            num_lines = 12
            for i in range(num_lines):
                angle = (i / num_lines) * np.pi * 2.0 + self.rotation
                x1 = cx + np.cos(angle) * self.base_radius
                y1 = cy + np.sin(angle) * self.base_radius
                x2 = cx + np.cos(angle) * (self.base_radius + self.base_radius * 0.2 * onset)
                y2 = cy + np.sin(angle) * (self.base_radius + self.base_radius * 0.2 * onset)
                line_verts.append([x1, y1, *colors[dominant], onset])
                line_verts.append([x2, y2, *colors[dominant], onset])

        # --- Energie-Partikel bei hohem RMS ---
        p_idx = 0
        if rms > 0.6:
            np.random.seed(frame_idx // 10)
            num_particles = int(rms * 20)
            for i in range(num_particles):
                if p_idx >= self._max_particles:
                    break
                angle = np.random.random() * np.pi * 2.0
                distance = self.base_radius * (0.8 + np.random.random() * 0.5)
                x = cx + np.cos(angle) * distance
                y = cy + np.sin(angle) * distance
                size = 2.0 + rms * 4.0
                ci = i % len(colors)
                self._particle_data[p_idx] = [x, y, colors[ci][0], colors[ci][1], colors[ci][2], size, 1.0]
                p_idx += 1

        # --- Rendern ---
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Linien
        if line_verts:
            arr = np.array(line_verts, dtype=np.float32)
            if len(arr) > self._max_line_verts:
                arr = arr[:self._max_line_verts]
            self._line_prog["u_resolution"].value = (self.width, self.height)
            self._line_vbo.write(arr.tobytes())
            self._line_vao.render(mode=moderngl.LINES)

        # Partikel
        if p_idx > 0:
            self._particle_prog["u_resolution"].value = (self.width, self.height)
            self._particle_vbo.write(self._particle_data[:p_idx].tobytes())
            self._particle_vao.render(mode=moderngl.TRIANGLE_STRIP, instances=p_idx)

        self.ctx.disable(moderngl.BLEND)
