"""
GPU-beschleunigter Neon Wave Circle.

Konzentrische, wellenfoermige Ringe als Line Strips,
Partikel-Strahlen bei Beats und Frequenz-Balken am aeusseren Ring.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class NeonWaveCircleGPU(BaseGPUVisualizer):
    """
    Neon Wave Circle mit audio-reaktiven Ringen und Strahlen.
    """

    PARAMS = {
        'circle_count': (5, 2, 10, 1),
        'wave_amplitude': (1.0, 0.2, 3.0, 0.1),
    }

    def _setup(self):
        """Initialisiere Shader und VBOs."""
        # --- Line-Shader (Ringe, Strahlen, Balken) ---
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

        # --- Quad-Shader (Kern) ---
        self._quad_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;
            in vec2 in_vertex_pos;
            in vec2 in_center;
            in vec3 in_color;
            in float in_size;
            in float in_alpha;
            out vec3 v_color;
            out float v_alpha;
            out vec2 v_local;
            void main() {
                vec2 pixel_pos = in_center + in_vertex_pos * in_size;
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
                float core = 1.0 - smoothstep(0.0, 0.6, dist);
                float glow = exp(-dist * dist * 3.0);
                f_color = vec4(v_color * (core + glow * 0.6), (core * 0.95 + glow * 0.4) * v_alpha);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self._quad_vbo = self.ctx.buffer(quad.tobytes())

        max_line_verts = 8000
        self._line_vbo = self.ctx.buffer(reserve=max_line_verts * 6 * 4, dynamic=True)
        self._line_vao = self.ctx.vertex_array(
            self._line_prog,
            [(self._line_vbo, "2f 3f 1f", "in_pos", "in_color", "in_alpha")],
        )

        self._core_vbo = self.ctx.buffer(reserve=4 * 7 * 4, dynamic=True)
        self._core_vao = self.ctx.vertex_array(
            self._quad_prog,
            [
                (self._quad_vbo, "2f", "in_vertex_pos"),
                (self._core_vbo, "2f 3f 1f 1f /i", "in_center", "in_color", "in_size", "in_alpha"),
            ],
        )

        self.center = (self.width / 2.0, self.height / 2.0)
        self.max_radius = min(self.width, self.height) / 2.0 - 50.0

    def _append_wavy_ring(self, verts, center, base_radius, amplitude, wave_count, phase, color, alpha, rms, segments=100):
        """Fuegt einen wellenfoermigen Ring als Line Strip hinzu."""
        points = []
        for i in range(segments + 1):
            angle = (i / segments) * np.pi * 2.0
            wave = np.sin(angle * wave_count + phase) * amplitude
            wave += np.sin(angle * wave_count * 2.0 + phase * 1.5) * amplitude * 0.3
            r = base_radius + wave
            x = center[0] + np.cos(angle) * r
            y = center[1] + np.sin(angle) * r
            points.append([x, y])
        for i in range(len(points) - 1):
            verts.append([*points[i], *color, alpha])
            verts.append([*points[i + 1], *color, alpha])

    def render(self, features: dict, time: float):
        """Rendert Ringe, Kern, Strahlen und Frequenz-Balken."""
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

        primary = self._hsv_to_rgb(base_hue, 0.9, 1.0)
        secondary = self._hsv_to_rgb((base_hue + 0.33) % 1.0, 0.8, 1.0)

        num_rings = int(self.params["circle_count"])
        wave_amp = self.params["wave_amplitude"]
        t = frame_idx * 0.03
        cx, cy = self.center
        line_verts = []

        # --- Ring-Daten initialisieren (falls noch nicht geschehen) ---
        if not hasattr(self, '_ring_data'):
            self._ring_data = []
            for i in range(num_rings):
                self._ring_data.append({
                    "base_radius": (i + 1) * (self.max_radius / num_rings),
                    "amplitude": 10.0 + i * 5.0,
                    "frequency": 0.1 + i * 0.02,
                    "phase": i * np.pi / 3.0,
                    "wave_count": 6 + i * 2,
                })

        # --- Ringe von aussen nach innen ---
        for i in range(num_rings - 1, -1, -1):
            ring = self._ring_data[i]
            audio_boost = 1.0 + rms * 0.5
            if onset > 0.3:
                audio_boost += onset * 0.5
            current_radius = ring["base_radius"] * audio_boost
            current_amp = ring["amplitude"] * (0.5 + rms * 1.5) * wave_amp

            ring_hue = (base_hue + i * 0.1) % 1.0
            ring_sat = 0.8 + spectral * 0.2
            ring_val = 0.5 + rms * 0.5

            if i % 2 == 0:
                ring_color = tuple(primary[j] * (0.5 + ring_val * 0.5) for j in range(3))
            else:
                ring_color = tuple(secondary[j] * (0.5 + ring_val * 0.5) for j in range(3))

            # Haupt-Ring
            self._append_wavy_ring(
                line_verts, self.center, current_radius, current_amp,
                ring["wave_count"], t + ring["phase"], ring_color, 1.0, rms
            )

            # Glow fuer aeussere Ringe
            if i < 2:
                glow_color = tuple(c * 0.3 for c in ring_color)
                self._append_wavy_ring(
                    line_verts, self.center, current_radius + 4.0, current_amp,
                    ring["wave_count"], t + ring["phase"], glow_color, 0.5, rms, segments=80
                )

        # --- Kern (gefuellter Kreis) ---
        core_radius = 30.0 + rms * 40.0
        core_data = np.array([[
            cx, cy,
            primary[0], primary[1], primary[2],
            core_radius, 1.0
        ]], dtype=np.float32)

        # --- Partikel-Strahlen bei Beats ---
        if onset > 0.4:
            num_rays = 12
            ray_length = self.max_radius * (0.3 + onset * 0.5)
            for i in range(num_rays):
                angle = (i / num_rays) * np.pi * 2.0 + t * 0.5
                start_r = 30.0 + rms * 40.0
                sx = cx + np.cos(angle) * start_r
                sy = cy + np.sin(angle) * start_r
                ex = cx + np.cos(angle) * (self.max_radius + ray_length)
                ey = cy + np.sin(angle) * (self.max_radius + ray_length)
                for j in range(3):
                    oa = j * 0.02
                    ox = cx + np.cos(angle + oa) * (self.max_radius + ray_length * 0.8)
                    oy = cy + np.sin(angle + oa) * (self.max_radius + ray_length * 0.8)
                    ray_color = tuple(primary[k] * (1.0 - j * 0.3) for k in range(3))
                    line_verts.append([sx, sy, *ray_color, 0.8])
                    line_verts.append([ox, oy, *ray_color, 0.8])

        # --- Frequenz-Balken am aeusseren Ring ---
        num_bars = 12
        bar_w = 15.0
        max_bar_h = 40.0
        ring_r = self.max_radius + 20.0
        for i in range(num_bars):
            angle = (i / num_bars) * np.pi * 2.0 - np.pi / 2.0
            bar_h = max_bar_h * (chroma[i] if chroma is not None else 0.5) * (0.5 + rms)
            bx = cx + np.cos(angle) * ring_r
            by = cy + np.sin(angle) * ring_r
            ex = cx + np.cos(angle) * (ring_r + bar_h)
            ey = cy + np.sin(angle) * (ring_r + bar_h)
            bar_color = self._hsv_to_rgb(i / 12.0, 0.9, 0.8 + (chroma[i] if chroma is not None else 0.0) * 0.2)
            # Balken als dicke Linie
            line_verts.append([bx, by, *bar_color, 1.0])
            line_verts.append([ex, ey, *bar_color, 1.0])

        # --- Rendern ---
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Linien
        if line_verts:
            arr = np.array(line_verts, dtype=np.float32)
            self._line_prog["u_resolution"].value = (self.width, self.height)
            self._line_vbo.write(arr.tobytes())
            self._line_vao.render(mode=moderngl.LINES)

        # Kern
        self._quad_prog["u_resolution"].value = (self.width, self.height)
        self._core_vbo.write(core_data.tobytes())
        self._core_vao.render(mode=moderngl.TRIANGLE_STRIP, instances=1)

        self.ctx.disable(moderngl.BLEND)
