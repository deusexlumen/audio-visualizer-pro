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
        'trail_length': (5, 0, 10, 1),
        'depth_enabled': (1, 0, 1, 1),
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
                // Quad-Vertex skalieren und auf Partikel-Position addieren
                vec2 pixel_pos = in_particle_pos + in_vertex_pos * in_particle_size;
                // Pixel -> Normalized Device Coordinates (-1 .. 1)
                vec2 ndc = (pixel_pos / u_resolution) * 2.0 - 1.0;
                // OpenGL-Y zeigt nach oben, unsere Pixel-Koordinaten nach unten
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);

                v_color = in_particle_color;
                v_alpha = in_particle_alpha;
                v_local_pos = in_vertex_pos;
            }
            """,
            fragment_shader="""
            #version 330
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

                vec3 final_color = v_color * (core + glow * 0.7);
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

    def _init_particles(self):
        """Initialisiere Partikel-Array und Trails."""
        count = int(self.params["particle_count"])
        self._particles = np.zeros((count, 9), dtype=np.float32)
        self._trails = [[] for _ in range(count)]

        cx, cy = self.width / 2.0, self.height / 2.0

        for i in range(count):
            angle = np.random.random() * np.pi * 2
            dist = np.random.random() * 80.0
            self._particles[i, 0] = cx + np.cos(angle) * dist
            self._particles[i, 1] = cy + np.sin(angle) * dist
            self._particles[i, 2] = np.cos(angle) * np.random.random() * 1.5
            self._particles[i, 3] = np.sin(angle) * np.random.random() * 1.5
            self._particles[i, 4] = 1.0  # life
            self._particles[i, 5] = 0.5 + np.random.random() * 1.0  # max_life
            self._particles[i, 6] = 2.0 + np.random.random() * 5.0  # size
            self._particles[i, 7] = np.random.random()  # hue
            self._particles[i, 8] = np.random.random()  # depth
            self._trails[i] = []

    def _explode_particle(self, idx: int, chroma: np.ndarray):
        """Explodiert ein Partikel vom Zentrum aus."""
        cx, cy = self.width / 2.0, self.height / 2.0
        angle = np.random.random() * np.pi * 2
        # ease_out_expo
        t = np.random.random()
        speed = (1.0 - pow(2.0, -10.0 * t)) * 12.0 + 3.0

        self._particles[idx, 0] = cx
        self._particles[idx, 1] = cy
        self._particles[idx, 2] = np.cos(angle) * speed
        self._particles[idx, 3] = np.sin(angle) * speed
        self._particles[idx, 4] = 1.0
        self._particles[idx, 5] = 0.5 + np.random.random() * 1.0
        self._particles[idx, 6] = 2.0 + np.random.random() * 5.0
        if chroma is not None and chroma.size > 0:
            self._particles[idx, 7] = float(np.argmax(chroma)) / 12.0
        else:
            self._particles[idx, 7] = np.random.random()
        self._particles[idx, 8] = np.random.random()
        self._trails[idx] = []

    def _reset_particle(self, idx: int):
        """Setzt ein Partikel auf zufaellige Startposition zurueck."""
        cx, cy = self.width / 2.0, self.height / 2.0
        angle = np.random.random() * np.pi * 2
        dist = np.random.random() * 80.0

        self._particles[idx, 0] = cx + np.cos(angle) * dist
        self._particles[idx, 1] = cy + np.sin(angle) * dist
        self._particles[idx, 2] = np.cos(angle) * np.random.random() * 1.5
        self._particles[idx, 3] = np.sin(angle) * np.random.random() * 1.5
        self._particles[idx, 4] = 1.0
        self._particles[idx, 5] = 0.5 + np.random.random() * 1.0
        self._particles[idx, 6] = 2.0 + np.random.random() * 5.0
        self._particles[idx, 7] = np.random.random()
        self._particles[idx, 8] = np.random.random()
        self._trails[idx] = []

    def render(self, features: dict, time: float):
        """Rendert einen Frame mit Partikeln, Trails und Zentrumspuls."""
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        rms = f["rms"]
        onset = f["onset"]
        chroma = f["chroma"]

        cx, cy = self.width / 2.0, self.height / 2.0
        count = int(self.params["particle_count"])
        threshold = self.params["explosion_threshold"]
        trail_len = int(self.params["trail_length"])
        glow_size = self.params["glow_size"]
        depth_enabled = self.params["depth_enabled"] > 0.5

        # Beat-Explosion
        if onset > threshold:
            explode_count = int(count * onset * 0.3)
            for _ in range(explode_count):
                idx = np.random.randint(0, count)
                self._explode_particle(idx, chroma)

        # Hauptfarbe aus Chroma fuer Zentrumspuls
        if chroma is not None and chroma.size > 0:
            main_hue = float(np.argmax(chroma)) / 12.0
        else:
            main_hue = 0.5
        main_color = self._hsv_to_rgb(main_hue, 0.35, 0.7)

        # Instance-Array fuellen
        instance_idx = 0

        for i in range(count):
            x = self._particles[i, 0]
            y = self._particles[i, 1]
            vx = self._particles[i, 2]
            vy = self._particles[i, 3]
            life = self._particles[i, 4]
            max_life = self._particles[i, 5]
            size = self._particles[i, 6]
            hue = self._particles[i, 7]
            depth = self._particles[i, 8]

            # Trail speichern
            self._trails[i].append((float(x), float(y), float(life)))
            if len(self._trails[i]) > trail_len:
                self._trails[i].pop(0)

            # Physik-Update
            x += vx
            y += vy

            dx = cx - x
            dy = cy - y
            dist = np.sqrt(dx * dx + dy * dy) + 1.0
            force = 0.04 * rms
            vx += (dx / dist) * force
            vy += (dy / dist) * force
            vx *= 0.985
            vy *= 0.985

            life -= 0.004 * (1.0 + rms)

            if life <= 0:
                self._reset_particle(i)
                x = self._particles[i, 0]
                y = self._particles[i, 1]
                vx = self._particles[i, 2]
                vy = self._particles[i, 3]
                life = self._particles[i, 4]
                max_life = self._particles[i, 5]
                size = self._particles[i, 6]
                hue = self._particles[i, 7]
                depth = self._particles[i, 8]

            # State zurueckschreiben
            self._particles[i, 0] = x
            self._particles[i, 1] = y
            self._particles[i, 2] = vx
            self._particles[i, 3] = vy
            self._particles[i, 4] = life

            # Farbe und Groesse berechnen
            life_ratio = life / max_life if max_life > 0 else 0.0
            saturation = 0.3 + rms * 0.15
            value = life_ratio * (0.4 + rms * 0.3)
            rgb = self._hsv_to_rgb(float(hue), saturation, value)

            depth_scale = 0.6 + depth * 0.4 if depth_enabled else 1.0
            current_size = size * life_ratio * (0.8 + rms * 0.4) * depth_scale
            total_size = current_size * 1.5 + glow_size * rms

            # Trail-Punkte als Instanzen
            if trail_len > 0:
                for ti, (tx, ty, tl) in enumerate(self._trails[i]):
                    if tl > 0 and instance_idx < self._max_instances:
                        t_ratio = (
                            (ti + 1) / len(self._trails[i])
                            if len(self._trails[i]) > 0
                            else 0
                        )
                        t_alpha = 0.35 * t_ratio * tl
                        t_size = max(1.0, current_size * 0.4)
                        self._instance_data[instance_idx] = [
                            tx,
                            ty,
                            rgb[0],
                            rgb[1],
                            rgb[2],
                            t_size,
                            t_alpha,
                        ]
                        instance_idx += 1

            # Partikel als Instanz
            if current_size > 0 and instance_idx < self._max_instances:
                alpha = life_ratio
                self._instance_data[instance_idx] = [
                    x,
                    y,
                    rgb[0],
                    rgb[1],
                    rgb[2],
                    total_size,
                    alpha,
                ]
                instance_idx += 1

        # Zentrumspuls-Ringe (als grosse, schwache Glow-Kreise)
        pulse_radius = 15.0 + rms * 25.0
        for j in range(4):
            if instance_idx >= self._max_instances:
                break
            ring_alpha = (1.0 - j / 4.0) * rms * 0.35
            ring_size = pulse_radius + j * 8.0
            self._instance_data[instance_idx] = [
                cx,
                cy,
                main_color[0],
                main_color[1],
                main_color[2],
                ring_size,
                ring_alpha,
            ]
            instance_idx += 1

        # Aufloesung an Shader uebergeben
        self._prog["u_resolution"].value = (self.width, self.height)

        # Rendern
        if instance_idx > 0:
            self._instance_vbo.write(self._instance_data[:instance_idx].tobytes())
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._vao.render(mode=moderngl.TRIANGLE_STRIP, instances=instance_idx)
            self.ctx.disable(moderngl.BLEND)
