"""
GPU-beschleunigte Liquid Blobs / MetaBalls.

Blobs als instanced Quads mit Glow, Verbindungslinien zwischen nahen Blobs,
Beat-Ringe und Highlights.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class LiquidBlobsGPU(BaseGPUVisualizer):
    """
    Flüssige Blob-Visualisierung mit MetaBall-aehnlichem Rendering.
    """

    PARAMS = {
        'blob_count': (6, 3, 12, 1),
        'fluidity': (0.7, 0.1, 1.0, 0.1),
        'blob_base_radius': (60.0, 20.0, 150.0, 5.0),
        'radius_variation': (40.0, 0.0, 100.0, 5.0),
        'connection_distance': (1.2, 0.5, 3.0, 0.1),
        'ring_count': (3, 0, 8, 1),
        'ring_base_radius': (100.0, 50.0, 300.0, 10.0),
        'ring_spacing': (80.0, 20.0, 200.0, 5.0),
    }

    def _setup(self):
        """Initialisiere Shader und VBOs."""
        # --- Blob-Shader (instanced Quads mit weichem Rand) ---
        self._blob_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;
            in vec2 in_vertex_pos;
            in vec2 in_blob_pos;
            in vec3 in_color;
            in float in_size;
            in float in_alpha;
            out vec3 v_color;
            out float v_alpha;
            out vec2 v_local;
            void main() {
                vec2 pixel_pos = in_blob_pos + in_vertex_pos * in_size;
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
                float core = 1.0 - smoothstep(0.0, 0.6, dist);
                float glow = exp(-dist * dist * 3.0);
                vec3 col = v_color * (core + glow * 0.5) * u_brightness;
                float alpha = (core * 0.95 + glow * 0.4) * v_alpha;
                f_color = vec4(col, alpha);
            }
            """,
        )

        # --- Line-Shader (Verbindungen + Beat-Ringe) ---
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

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self._quad_vbo = self.ctx.buffer(quad.tobytes())

        max_blobs = 12
        self._max_instances = max_blobs * 3  # Blob + Highlight + Glow
        self._blob_data = np.zeros((self._max_instances, 7), dtype=np.float32)
        self._blob_vbo = self.ctx.buffer(reserve=self._max_instances * 7 * 4, dynamic=True)
        self._blob_vao = self.ctx.vertex_array(
            self._blob_prog,
            [
                (self._quad_vbo, "2f", "in_vertex_pos"),
                (self._blob_vbo, "2f 3f 1f 1f /i", "in_blob_pos", "in_color", "in_size", "in_alpha"),
            ],
        )

        self._max_line_verts = 2000
        self._line_vbo = self.ctx.buffer(reserve=self._max_line_verts * 6 * 4, dynamic=True)
        self._line_vao = self.ctx.vertex_array(
            self._line_prog,
            [(self._line_vbo, "2f 3f 1f", "in_pos", "in_color", "in_alpha")],
        )

        self._init_blobs()

    def _on_params_changed(self):
        self._init_blobs()

    def _init_blobs(self):
        """Initialisiere Blob-Positionen und Eigenschaften."""
        count = int(self.params["blob_count"])
        base_radius = self.params["blob_base_radius"]
        radius_variation = self.params["radius_variation"]
        self._blobs = []
        for i in range(count):
            self._blobs.append({
                "base_radius": base_radius + np.random.random() * radius_variation,
                "phase": np.random.random() * np.pi * 2.0,
                "x": 0.0,
                "y": 0.0,
                "current_radius": 0.0,
                "color": (0.6, 0.6, 0.6),
            })

    def _shift_hue(self, rgb, shift):
        """Verschiebt den Hue eines RGB-Tupels um shift (0.0-1.0)."""
        h, s, v = self._rgb_to_hsv(*rgb)
        return self._hsv_to_rgb((h + shift) % 1.0, s, v)

    def render(self, features: dict, time: float):
        """Rendert Blobs, Verbindungen, Beat-Ringe und Highlights."""
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        rms = f["rms"]
        onset = f["onset"]
        chroma = f["chroma"]

        base_color = self._chroma_to_color(chroma)

        fluidity = self.params["fluidity"]
        t = frame_idx * 0.02

        # --- Blob-Positionen aktualisieren ---
        for i, blob in enumerate(self._blobs):
            blob["phase"] += 0.01 + rms * 0.05 * fluidity
            base_x = self.width / 2.0 + np.sin(t + i) * (self.width * 0.3)
            base_y = self.height / 2.0 + np.cos(t * 0.7 + i * 1.3) * (self.height * 0.25)
            noise_x = np.sin(t * 3.0 + i * 2.0) * rms * 100.0
            noise_y = np.cos(t * 2.5 + i * 1.5) * rms * 100.0
            blob["x"] = base_x + noise_x
            blob["y"] = base_y + noise_y
            blob["current_radius"] = blob["base_radius"] * (0.8 + rms * 0.6)

            blob["color"] = self._shift_hue(base_color, i * 0.15)

        # --- Instanz-Array fuellen ---
        instance_idx = 0

        for blob in self._blobs:
            # Aeusserer Glow
            if instance_idx < self._max_instances:
                glow_color = tuple(max(0.0, c - 0.15) for c in blob["color"])
                self._blob_data[instance_idx] = [
                    blob["x"], blob["y"],
                    glow_color[0], glow_color[1], glow_color[2],
                    blob["current_radius"] + 25.0, 0.35
                ]
                instance_idx += 1

            # Haupt-Blob
            if instance_idx < self._max_instances:
                self._blob_data[instance_idx] = [
                    blob["x"], blob["y"],
                    blob["color"][0], blob["color"][1], blob["color"][2],
                    blob["current_radius"], 1.0
                ]
                instance_idx += 1

            # Innerer Highlight
            if instance_idx < self._max_instances:
                hi_color = tuple(min(1.0, c + 0.1) for c in blob["color"])
                offset = blob["current_radius"] * 0.3
                self._blob_data[instance_idx] = [
                    blob["x"] - offset, blob["y"] - offset,
                    hi_color[0], hi_color[1], hi_color[2],
                    blob["current_radius"] * 0.35, 0.8
                ]
                instance_idx += 1

        # --- Verbindungslinien ---
        line_verts = []
        connection_distance = self.params["connection_distance"]
        for i, b1 in enumerate(self._blobs):
            for j, b2 in enumerate(self._blobs[i + 1:], i + 1):
                dx = b1["x"] - b2["x"]
                dy = b1["y"] - b2["y"]
                dist = np.sqrt(dx * dx + dy * dy)
                threshold = b1["current_radius"] + b2["current_radius"]
                if dist < threshold * connection_distance:
                    strength = 1.0 - (dist / (threshold * connection_distance))
                    mixed = tuple((b1["color"][k] + b2["color"][k]) / 2.0 for k in range(3))
                    alpha = strength * 0.6
                    line_verts.append([b1["x"], b1["y"], *mixed, alpha])
                    line_verts.append([b2["x"], b2["y"], *mixed, alpha])

        # --- Beat-Ringe ---
        if onset > 0.4:
            cx, cy = self.width / 2.0, self.height / 2.0
            ring_count = int(self.params["ring_count"])
            ring_base = self.params["ring_base_radius"]
            ring_spacing = self.params["ring_spacing"]
            for i in range(ring_count):
                radius = ring_base + i * ring_spacing + onset * 50.0
                ring_color = self._shift_hue(base_color, i * 0.1)
                alpha = onset * 0.4 * (1.0 - i / max(ring_count, 1))
                segments = 48
                for j in range(segments):
                    a1 = (j / segments) * np.pi * 2.0
                    a2 = ((j + 1) / segments) * np.pi * 2.0
                    line_verts.append([cx + np.cos(a1) * radius, cy + np.sin(a1) * radius, *ring_color, alpha])
                    line_verts.append([cx + np.cos(a2) * radius, cy + np.sin(a2) * radius, *ring_color, alpha])

        # --- Rendern ---
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        brightness = self.params.get("brightness", 1.0)
        self._blob_prog["u_brightness"].value = brightness
        self._line_prog["u_brightness"].value = brightness

        # Verbindungen + Ringe (nur geschriebene Vertices rendern)
        if line_verts:
            arr = np.array(line_verts, dtype=np.float32)
            self._line_prog["u_resolution"].value = (self.width, self.height)
            self._line_vbo.write(arr.tobytes())
            self._line_vao.render(mode=moderngl.LINES, vertices=len(line_verts))

        # Blobs
        if instance_idx > 0:
            self._blob_prog["u_resolution"].value = (self.width, self.height)
            self._blob_vbo.write(self._blob_data[:instance_idx].tobytes())
            self._blob_vao.render(mode=moderngl.TRIANGLE_STRIP, instances=instance_idx)

        self.ctx.disable(moderngl.BLEND)
