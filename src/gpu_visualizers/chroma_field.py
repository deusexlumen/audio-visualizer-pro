"""
GPU-beschleunigtes Chroma-Feld mit Partikeln, Verbindungslinien und Chroma-Ring.

Partikel repraesentieren die 12 Halbtoene und pulsen mit der jeweiligen Chroma-Staerke.
Nahe Partikel werden mit leuchtenden Linien verbunden.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class ChromaFieldGPU(BaseGPUVisualizer):
    """
    Partikel-Feld, das auf die 12 Halbtoene reagiert.
    GPU-beschleunigt via instanced Quads fuer Partikel und GL_LINES fuer Verbindungen.
    """

    PARAMS = {
        'field_resolution': (100, 50, 200, 10),
        'connection_dist': (100, 50, 200, 10),
        'particle_size': (8, 3, 20, 1),
    }

    def _setup(self):
        """Initialisiere Shader, VBOs und Partikel."""
        # --- Partikel-Shader (instanced Quads mit Glow) ---
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

                vec3 final_color = v_color * (core + glow * 0.3);
                float alpha = (core * 0.9 + glow * 0.4) * v_alpha;
                f_color = vec4(final_color, alpha);
            }
            """,
        )

        # --- Line-Shader (Verbindungen + Chroma-Ring) ---
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
            void main() {
                f_color = vec4(v_color, v_alpha);
            }
            """,
        )

        # Quad-VBO fuer Partikel
        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self._quad_vbo = self.ctx.buffer(quad.tobytes())

        max_particles = 200
        self._max_particles = max_particles
        self._instance_data = np.zeros((max_particles, 7), dtype=np.float32)
        self._instance_vbo = self.ctx.buffer(reserve=max_particles * 7 * 4, dynamic=True)

        self._particle_vao = self.ctx.vertex_array(
            self._particle_prog,
            [
                (self._quad_vbo, "2f", "in_vertex_pos"),
                (self._instance_vbo, "2f 3f 1f 1f /i", "in_particle_pos", "in_color", "in_size", "in_alpha"),
            ],
        )

        # Line-VBO (Verbindungen + Ring)
        max_lines = 2000
        self._line_vbo = self.ctx.buffer(reserve=max_lines * 6 * 4, dynamic=True)
        self._line_vao = self.ctx.vertex_array(
            self._line_prog,
            [(self._line_vbo, "2f 3f 1f", "in_pos", "in_color", "in_alpha")],
        )

        self._init_particles()

    def _on_params_changed(self):
        self._init_particles()

    def _init_particles(self):
        """Initialisiere Partikel mit zufaelligen Positionen und Noten-Zuordnung."""
        count = int(self.params["field_resolution"])
        count = min(count, self._max_particles)
        self._particles = []
        np.random.seed(42)

        for i in range(count):
            self._particles.append({
                "base_x": np.random.randint(50, self.width - 50),
                "base_y": np.random.randint(50, self.height - 50),
                "x": 0.0,
                "y": 0.0,
                "size": np.random.randint(3, 12),
                "note": i % 12,
                "phase": np.random.random() * np.pi * 2,
                "speed": 0.5 + np.random.random() * 1.5,
            })

    def render(self, features: dict, time: float):
        """Rendert Partikel-Feld, Verbindungen und Chroma-Ring."""
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        rms = f["rms"]
        onset = f["onset"]
        chroma = f["chroma"]
        spectral = f["spectral_centroid"]

        conn_dist = self.params["connection_dist"]
        base_size = self.params["particle_size"]

        # --- Partikel aktualisieren und Instanz-Array fuellen ---
        instance_idx = 0
        t = frame_idx * 0.05

        for p in self._particles:
            note_strength = float(chroma[p["note"]]) if chroma is not None else 0.5

            p["x"] = p["base_x"] + np.sin(t * p["speed"] + p["phase"]) * 30.0 * note_strength
            p["y"] = p["base_y"] + np.cos(t * p["speed"] * 0.7 + p["phase"]) * 20.0 * note_strength

            if onset > 0.3:
                pulse = 1.0 + onset * 0.5
                p["x"] += np.sin(frame_idx * 0.5) * 10.0 * pulse

            # Farbe basierend auf Note
            hue = p["note"] / 12.0
            sat = 0.25 + note_strength * 0.1
            val = 0.4 + note_strength * 0.3
            rgb = self._hsv_to_rgb(hue, sat, val)

            size = float(p["size"] * (0.8 + rms * 0.4) * base_size / 8.0)
            alpha = 0.5 + note_strength * 0.5

            if instance_idx < self._max_particles:
                self._instance_data[instance_idx] = [p["x"], p["y"], rgb[0], rgb[1], rgb[2], size, alpha]
                instance_idx += 1

        # --- Verbindungslinien bauen ---
        line_verts = []
        n = len(self._particles)
        for i in range(n):
            p1 = self._particles[i]
            for j in range(i + 1, min(i + 5, n)):
                p2 = self._particles[j]
                dx = p1["x"] - p2["x"]
                dy = p1["y"] - p2["y"]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < conn_dist:
                    note_strength = float(chroma[p1["note"]]) if chroma is not None else 0.5
                    alpha = (1.0 - dist / conn_dist) * 0.35 * note_strength
                    rgb = self._hsv_to_rgb(p1["note"] / 12.0, 0.2 + note_strength * 0.15, 0.5)
                    line_verts.append([p1["x"], p1["y"], rgb[0], rgb[1], rgb[2], alpha])
                    line_verts.append([p2["x"], p2["y"], rgb[0], rgb[1], rgb[2], alpha])

        # --- Chroma-Ring in der Mitte ---
        cx, cy = self.width / 2.0, self.height / 2.0
        ring_radius = 80.0 + rms * 40.0
        for note in range(12):
            angle = (note / 12.0) * np.pi * 2.0 - np.pi / 2.0
            note_strength = float(chroma[note]) if chroma is not None else 0.0
            x = cx + np.cos(angle) * ring_radius
            y = cy + np.sin(angle) * ring_radius
            rgb = self._hsv_to_rgb(note / 12.0, 0.2 + note_strength * 0.15, 0.4 + note_strength * 0.3)
            dot_size = 8.0 + note_strength * 12.0
            # Ring-Partikel als Instanzen hinzufuegen
            if instance_idx < self._max_particles:
                self._instance_data[instance_idx] = [x, y, rgb[0], rgb[1], rgb[2], dot_size, 1.0]
                instance_idx += 1

        # --- Rendern ---
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Verbindungslinien
        if line_verts:
            line_arr = np.array(line_verts, dtype=np.float32)
            self._line_prog["u_resolution"].value = (self.width, self.height)
            self._line_vbo.write(line_arr.tobytes())
            self._line_vao.render(mode=moderngl.LINES)

        # Partikel + Ring
        if instance_idx > 0:
            self._particle_prog["u_resolution"].value = (self.width, self.height)
            self._instance_vbo.write(self._instance_data[:instance_idx].tobytes())
            self._particle_vao.render(mode=moderngl.TRIANGLE_STRIP, instances=instance_idx)

        self.ctx.disable(moderngl.BLEND)
