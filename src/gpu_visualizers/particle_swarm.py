"""
GPU-beschleunigtes Partikel-System mit Trails, Glow und Tiefen-Simulation.

Physik-Update auf CPU (150-500 Partikel sind trivial),
Rendering auf GPU via instanced Quads mit weichem Kreis-Fragment-Shader.
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class ParticleSwarmGPU(BaseGPUVisualizer):
    """
    Professionelles GPU-Partikel-System mit Trails und Glow.

    Partikel werden als instanzierte Quads gerendert – jede Instanz
traegt Position, Farbe, Groesse und Alpha. Der Fragment-Shader
    zeichnet einen weichen, leuchtenden Kreis mit exponentiellem Glow.
    """

    PARAMS = {
        'particle_count': (150, 50, 500, 10),
        'explosion_threshold': (0.4, 0.1, 0.9, 0.05),
        'glow_size': (3, 0, 10, 1),
        'glow_strength': (0.7, 0.0, 2.0, 0.1),
        'trail_length': (5, 0, 10, 1),
        'depth_enabled': (1, 0, 1, 1),
        'speed_scale': (1.0, 0.2, 3.0, 0.1),
        'center_force': (0.04, 0.0, 0.2, 0.01),
        'friction': (0.985, 0.9, 0.999, 0.001),
        'life_decay': (0.004, 0.001, 0.02, 0.001),
        'size_scale': (1.0, 0.2, 3.0, 0.1),
    }

    PARAMS_GROUPS = {
        "Partikel": ["particle_count", "size_scale", "life_decay"],
        "Bewegung": ["speed_scale", "center_force", "friction", "explosion_threshold"],
        "Erscheinungsbild": ["glow_size", "glow_strength", "depth_enabled"],
        "Trail": ["trail_length"],
    }

    def _setup(self):
        """Initialisiere Shader, VBOs und Partikel-System."""
        self._prog = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;

            in vec2 in_vertex_pos;
            in vec2 in_particle_pos;
            in vec3 in_particle_color;
            in float in_particle_size;
            in float in_particle_alpha;

            out vec3 v_color;
            out float v_alpha;
            out vec2 v_local_pos;

            void main() {
                // Zentrum und Offset getrennt in NDC umrechnen, damit Kreise
                // bei nicht-quadratischer Aufloesung kreisrund bleiben.
                vec2 center_ndc = (in_particle_pos / u_resolution) * 2.0 - 1.0;
                center_ndc.y = -center_ndc.y;

                vec2 offset_ndc = in_vertex_pos * in_particle_size / u_resolution * 2.0;
                // X-Offset an Pixel-Aspekt anpassen (1 Pixel in X = height/width Pixel in Y)
                offset_ndc.x *= u_resolution.x / u_resolution.y;

                gl_Position = vec4(center_ndc + offset_ndc, 0.0, 1.0);

                v_color = in_particle_color;
                v_alpha = in_particle_alpha;
                v_local_pos = in_vertex_pos;
            }
            """,
            fragment_shader="""
            #version 330
            uniform float u_brightness;
            uniform float u_glow_strength;
            in vec3 v_color;
            in float v_alpha;
            in vec2 v_local_pos;
            out vec4 f_color;

            void main() {
                float dist = length(v_local_pos);
                if (dist > 1.0) discard;

                // Kern: fester Kreis
                float core = 1.0 - smoothstep(0.0, 0.65, dist);
                // Glow: exponentieller Abfall
                float glow = exp(-dist * dist * 3.5);

                vec3 final_color = v_color * (core + glow * u_glow_strength) * u_brightness;
                float alpha = (core * 0.95 + glow * 0.45) * v_alpha;

                f_color = vec4(final_color, alpha);
            }
            """,
        )

        # Ein einziges Quad (-1,-1) .. (1,1) als Basis-Geometrie
        quad = np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._quad_vbo = self.ctx.buffer(quad.tobytes())

        # Maximale Instanzen: Partikel + Trails + Zentrumspuls-Ringe
        max_particles = 500
        max_trail = 10
        max_rings = 4
        self._max_instances = max_particles * (1 + max_trail) + max_rings

        # Instanz-Daten: pos_x, pos_y, r, g, b, size, alpha
        self._instance_data = np.zeros((self._max_instances, 7), dtype=np.float32)
        self._instance_vbo = self.ctx.buffer(
            reserve=self._max_instances * 7 * 4, dynamic=True
        )

        # VAO: Quad-Vertex (non-instanced) + Instanz-Attribute (instanced via /i)
        self._vao = self.ctx.vertex_array(
            self._prog,
            [
                (self._quad_vbo, "2f", "in_vertex_pos"),
                (
                    self._instance_vbo,
                    "2f 3f 1f 1f /i",
                    "in_particle_pos",
                    "in_particle_color",
                    "in_particle_size",
                    "in_particle_alpha",
                ),
            ],
        )

        self._init_particles()

    def _on_params_changed(self):
        """Re-initialisiere Partikel wenn sich die Anzahl aendert."""
        self._init_particles()

    # Maximale Trail-Historie (entspricht PARAMS['trail_length'] Maximum)
    _MAX_TRAIL = 10

    def _init_particles(self):
        """Initialisiere Partikel-Array und Trail-Historie (vektorisiert)."""
        count = int(self.params["particle_count"])
        # Spalten: x, y, vx, vy, life, max_life, size, hue, depth
        self._particles = np.zeros((count, 9), dtype=np.float32)
        # Trail-Historie als Ringpuffer: (Slot, Partikel, [x, y, life])
        self._trail_hist = np.zeros((self._MAX_TRAIL, count, 3), dtype=np.float32)
        self._trail_valid = np.zeros(count, dtype=np.int32)

        self._spawn(np.arange(count), explode=False, chroma=None)

    def _spawn(self, idx: np.ndarray, explode: bool, chroma: np.ndarray = None):
        """(Re-)Initialisiert die Partikel an den angegebenen Indizes.

        Args:
            idx: Array von Partikel-Indizes.
            explode: True = Explosion vom Zentrum, False = zufaellige Startposition.
        """
        n = idx.size
        if n == 0:
            return
        p = self._particles
        cx, cy = self.width / 2.0, self.height / 2.0
        angle = np.random.random(n).astype(np.float32) * np.pi * 2

        if explode:
            # ease_out_expo Geschwindigkeit
            t = np.random.random(n).astype(np.float32)
            speed = (1.0 - np.power(2.0, -10.0 * t)) * 12.0 + 3.0
            speed *= self.params["speed_scale"]
            p[idx, 0] = cx
            p[idx, 1] = cy
            p[idx, 2] = np.cos(angle) * speed
            p[idx, 3] = np.sin(angle) * speed
        else:
            dist = np.random.random(n).astype(np.float32) * 80.0
            p[idx, 0] = cx + np.cos(angle) * dist
            p[idx, 1] = cy + np.sin(angle) * dist
            p[idx, 2] = np.cos(angle) * np.random.random(n) * 1.5
            p[idx, 3] = np.sin(angle) * np.random.random(n) * 1.5

        p[idx, 4] = 1.0  # life
        p[idx, 5] = 0.5 + np.random.random(n) * 1.0  # max_life
        p[idx, 6] = 2.0 + np.random.random(n) * 5.0  # size
        p[idx, 7] = self._new_hue(chroma) + np.random.random(n) * 0.1  # hue
        p[idx, 8] = np.random.random(n)  # depth
        self._trail_valid[idx] = 0

    @staticmethod
    def _hsv_to_rgb_array(h: np.ndarray, s, v: np.ndarray) -> np.ndarray:
        """Vektorisierte HSV->RGB-Konvertierung.

        Args:
            h: Hue-Array (0-1), s: Saettigung (Skalar oder Array), v: Value-Array.

        Returns:
            Array der Shape (N, 3).
        """
        h = np.mod(h, 1.0)
        i = (h * 6.0).astype(np.int32) % 6
        f = h * 6.0 - np.floor(h * 6.0)
        s = np.broadcast_to(np.float32(s), h.shape)
        pp = v * (1.0 - s)
        qq = v * (1.0 - s * f)
        tt = v * (1.0 - s * (1.0 - f))

        rgb = np.empty(h.shape + (3,), dtype=np.float32)
        conds = [
            (v, tt, pp), (qq, v, pp), (pp, v, tt),
            (pp, qq, v), (tt, pp, v), (v, pp, qq),
        ]
        for k, (r, g, b) in enumerate(conds):
            mask = i == k
            rgb[mask, 0] = r[mask]
            rgb[mask, 1] = g[mask]
            rgb[mask, 2] = b[mask]
        return rgb

    def _new_hue(self, chroma: np.ndarray = None) -> float:
        """Gibt einen neuen Farbton basierend auf color_mode zurueck."""
        mode = self.params.get('color_mode', 'chroma')
        if mode == 'chroma':
            if chroma is not None and chroma.size > 0:
                return self._color_to_hue(self._chroma_to_color(chroma))
            return 0.55
        if mode == 'fixed':
            primary = self.params.get('primary_color')
            if primary and isinstance(primary, str) and primary.startswith('#'):
                return self._color_to_hue(self._hex_to_rgb(primary))
            return float(self.params.get('base_hue', 0.55))
        if mode == 'warm':
            return 0.08 + np.random.random() * 0.06
        if mode == 'cool':
            return 0.55 + np.random.random() * 0.1
        # monochrome
        return 0.0

    def render(self, features: dict, time: float):
        """Rendert einen Frame mit Partikeln, Trails und Zentrumspuls.

        Physik, Farben und Instanz-Aufbau sind komplett vektorisiert
        (NumPy) — keine Python-Schleife ueber Partikel mehr.
        """
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        rms = float(f["rms"])
        onset = float(f["onset"])
        chroma = f["chroma"]

        cx, cy = self.width / 2.0, self.height / 2.0
        count = int(self.params["particle_count"])
        threshold = self.params["explosion_threshold"]
        trail_len = int(self.params["trail_length"])
        glow_size = self.params["glow_size"]
        glow_strength = self.params["glow_strength"]
        depth_enabled = self.params["depth_enabled"] > 0.5
        center_force = self.params["center_force"]
        friction = self.params["friction"]
        life_decay = self.params["life_decay"]
        size_scale = self.params["size_scale"]
        trail_decay = self.params.get("trail_decay", 0.7)

        p = self._particles

        # Trail-Historie aufzeichnen (Positionen VOR dem Physik-Update)
        if trail_len > 0:
            self._trail_hist[:-1] = self._trail_hist[1:]
            self._trail_hist[-1, :, 0] = p[:, 0]
            self._trail_hist[-1, :, 1] = p[:, 1]
            self._trail_hist[-1, :, 2] = p[:, 4]
            np.minimum(self._trail_valid + 1, trail_len, out=self._trail_valid)
        else:
            self._trail_valid[:] = 0

        # Beat-Explosion
        if onset > threshold:
            explode_count = int(count * onset * 0.3)
            if explode_count > 0:
                idx = np.random.randint(0, count, explode_count)
                self._spawn(np.unique(idx), explode=True, chroma=chroma)

        # === Physik-Update (vektorisiert) ===
        p[:, 0] += p[:, 2]
        p[:, 1] += p[:, 3]
        dx = cx - p[:, 0]
        dy = cy - p[:, 1]
        dist = np.sqrt(dx * dx + dy * dy) + 1.0
        force = center_force * rms
        p[:, 2] = (p[:, 2] + dx / dist * force) * friction
        p[:, 3] = (p[:, 3] + dy / dist * force) * friction
        p[:, 4] -= life_decay * (1.0 + rms)

        dead = np.where(p[:, 4] <= 0)[0]
        if dead.size > 0:
            self._spawn(dead, explode=False, chroma=chroma)

        # === Farben & Groessen (vektorisiert) ===
        base_color = self._chroma_to_color(chroma)
        main_color = tuple(c * 0.7 for c in base_color)
        color_mode = self.params.get('color_mode', 'chroma')
        base_saturation = 0.0 if color_mode == 'monochrome' else float(self.params.get('color_saturation', 0.7))
        base_hue = self._color_to_hue(base_color)

        life_ratio = np.where(p[:, 5] > 0, p[:, 4] / np.maximum(p[:, 5], 1e-6), 0.0)
        value = life_ratio * (0.5 + rms * 0.3)
        hue = (base_hue + p[:, 7] * 0.15) % 1.0
        rgb = self._hsv_to_rgb_array(hue, base_saturation * (0.5 + rms * 0.2), value)

        depth_scale = (0.6 + p[:, 8] * 0.4) if depth_enabled else np.ones(count, dtype=np.float32)
        current_size = p[:, 6] * life_ratio * (0.8 + rms * 0.4) * depth_scale * size_scale
        total_size = current_size * 1.5 + glow_size * rms

        instance_parts = []

        # === Trail-Instanzen ((Slot, Partikel)-Gitter, aeltester Slot zuerst) ===
        if trail_len > 0:
            hist = self._trail_hist[self._MAX_TRAIL - trail_len:]  # (T, N, 3)
            valid = self._trail_valid  # (N,)
            slots = np.arange(trail_len, dtype=np.int32)[:, None]  # (T, 1)
            ti = slots - (trail_len - valid[None, :])  # Index innerhalb der Partikel-Liste
            valid_mask = (ti >= 0) & (hist[:, :, 2] > 0)
            if valid_mask.any():
                valid_safe = np.maximum(valid[None, :], 1)
                t_ratio = (ti + 1) / valid_safe
                trail_dist = np.maximum(valid[None, :] - 1 - ti, 0)
                trail_fade = np.power(trail_decay, trail_dist)
                t_alpha = 0.35 * t_ratio * hist[:, :, 2] * trail_fade
                t_size = np.maximum(1.0, current_size * 0.4)

                trails = np.empty((trail_len, count, 7), dtype=np.float32)
                trails[:, :, 0:2] = hist[:, :, 0:2]
                trails[:, :, 2:5] = rgb[None, :, :]
                trails[:, :, 5] = t_size[None, :]
                trails[:, :, 6] = t_alpha
                instance_parts.append(trails[valid_mask])

        # === Partikel-Instanzen ===
        part_mask = current_size > 0
        if part_mask.any():
            parts = np.empty((count, 7), dtype=np.float32)
            parts[:, 0] = p[:, 0]
            parts[:, 1] = p[:, 1]
            parts[:, 2:5] = rgb
            parts[:, 5] = total_size
            parts[:, 6] = life_ratio
            instance_parts.append(parts[part_mask])

        # === Zentrumspuls-Ringe ===
        pulse_radius = 15.0 + rms * 25.0
        rings = np.empty((4, 7), dtype=np.float32)
        for j in range(4):
            rings[j] = [cx, cy, main_color[0], main_color[1], main_color[2],
                        pulse_radius + j * 8.0, (1.0 - j / 4.0) * rms * 0.35]
        instance_parts.append(rings)

        instances = np.concatenate(instance_parts, axis=0)
        if instances.shape[0] > self._max_instances:
            instances = instances[:self._max_instances]
        instance_count = instances.shape[0]

        # Aufloesung und Brightness an Shader uebergeben
        self._prog["u_resolution"].value = (self.width, self.height)
        self._prog["u_brightness"].value = self.params.get("brightness", 1.0)
        self._prog["u_glow_strength"].value = glow_strength

        # Rendern
        if instance_count > 0:
            self._instance_vbo.write(np.ascontiguousarray(instances).tobytes())
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._vao.render(mode=moderngl.TRIANGLE_STRIP, instances=instance_count)
            self.ctx.disable(moderngl.BLEND)
