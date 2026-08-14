"""
Particle Swarm - Galaxy/Vortex Visualizer.

Hunderte Gluehpartikel auf Spiralbahnen um ein gemeinsames Zentrum:
langsamer Einfall nach innen, Wobble, logarithmische Spiralarme und ein
leicht gekippter Galaxien-Blickwinkel. Alles deterministisch aus
Partikel-Index + festem Seed (kein Laufzeit-Zufallszustand) — der
Offline-Render ist dadurch exakt reproduzierbar.

Audio-Reaktionen (Musik-Modus):
- Bass/Onset/Transient: Schockwelle, die Partikel nach aussen schleudert,
  sichtbarer expandierender Ring + Kern-Aufblitzen.
- spectral_centroid (Treble): Funkeln (Groessen-/Alpha-Jitter pro Partikel).
- RMS/Energy: Rotationsgeschwindigkeit + sichtbare Partikelmenge/Helligkeit.
- Chroma: Farbverlauf ueber Radius und Lebensdauer (Hue-Spread).

Sprach-Modus (mode == "speech"): gleiche Optik, andere Empfindlichkeit —
Partikel folgen voice_band/voice_clarity, Betonungen (Anstieg im Voice-Band)
loesen sanfte Wirbel-Pulse aus, Pausen lassen die Galaxie fast ruhen.

Rendering: instanzierte Quads mit weichem Glow-Fragment-Shader, additive
Mischung (ONE, ONE), HDR-Ausgabe ohne clamp — das Tonemapping uebernimmt
zentral der Renderer. Der Hintergrund bleibt transparent (keine
Vollflaechen-Fuellung), damit Hintergrundbilder sichtbar bleiben.
"""

import numpy as np
import moderngl
from .base import (
    BaseGPUVisualizer,
    SHADER_COMMON_GLSL,
    compose_fragment,
)


class ParticleSwarmGPU(BaseGPUVisualizer):
    """
    Galaxy/Vortex: Gluehpartikel auf Spiralbahnen mit Einfall, Schockwellen
    und Kern-Leuchten. Additive HDR-Mischung, vollstaendig deterministisch.
    """

    PARAMS = {
        'particle_count': (550, 100, 1000, 20),
        'vortex_speed': (1.0, 0.0, 3.0, 0.05),
        'spiral_arms': (2, 1, 4, 1),
        'spiral_twist': (2.4, 0.0, 6.0, 0.1),
        'infall_speed': (0.35, 0.0, 1.5, 0.05),
        'wobble': (0.5, 0.0, 2.0, 0.05),
        'shockwave_strength': (1.0, 0.0, 2.5, 0.05),
        'sparkle': (0.6, 0.0, 2.0, 0.05),
        'hue_spread': (0.45, 0.0, 1.0, 0.05),
        'point_size': (1.0, 0.3, 3.0, 0.05),
        'core_glow': (1.0, 0.0, 3.0, 0.05),
        'galaxy_tilt': (0.55, 0.2, 1.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Galaxie": ["particle_count", "spiral_arms", "spiral_twist", "galaxy_tilt"],
        "Bewegung": ["vortex_speed", "infall_speed", "wobble"],
        "Reaktion": ["shockwave_strength", "sparkle", "hue_spread"],
        "Erscheinungsbild": ["point_size", "core_glow"],
    }

    # Feste Seed fuer alle statischen Partikel-Eigenschaften (Determinismus)
    _SEED = 0xA17C1E
    # Anzahl der Punkte, aus denen der Schockwellen-Ring gezeichnet wird
    _RING_POINTS = 56

    def _setup(self):
        """Initialisiere Shader, VBOs und Partikel-System."""
        vertex_shader = """
        #version 330
        uniform vec2 u_resolution;

        in vec2 in_vertex_pos;
        in vec2 in_pos;
        in vec3 in_color;
        in float in_size;
        in float in_alpha;
        in float in_seed;

        out vec3 v_color;
        out float v_alpha;
        out vec2 v_local;
        out float v_seed;

        void main() {
            // Zentrum und Offset getrennt in NDC umrechnen, damit die
            // Gluehpunkte bei nicht-quadratischer Aufloesung rund bleiben.
            vec2 center_ndc = (in_pos / u_resolution) * 2.0 - 1.0;
            center_ndc.y = -center_ndc.y;

            vec2 offset_ndc = in_vertex_pos * in_size / u_resolution * 2.0;
            offset_ndc.x *= u_resolution.x / u_resolution.y;

            gl_Position = vec4(center_ndc + offset_ndc, 0.0, 1.0);

            v_color = in_color;
            v_alpha = in_alpha;
            v_local = in_vertex_pos;
            v_seed = in_seed;
        }
        """

        fragment = compose_fragment(
            """
            uniform float u_brightness;

            in vec3 v_color;
            in float v_alpha;
            in vec2 v_local;
            in float v_seed;
            out vec4 f_color;

            void main() {
                float d = length(v_local);
                if (d > 1.0) discard;

                // Weicher Kern + weit auslaufender Halo (Glow-Punkt)
                float core = exp(-d * d * 10.0);
                float halo = exp(-d * 3.2) * 0.30;
                float glow = core + halo;

                // HDR: heisser Kern darf ueber 1.0 hinaus —
                // das Tonemapping uebernimmt zentral der Renderer.
                vec3 col = v_color * glow + v_color * core * core * 1.6;
                col *= u_brightness * v_alpha;

                // Triangular-Dithering gegen Farb-Banding im Halo
                col += ditherTriangular(gl_FragCoord.xy, v_seed);

                f_color = vec4(col, clamp(glow * v_alpha, 0.0, 1.0));
            }
            """,
            includes=(SHADER_COMMON_GLSL,),
        )

        self._prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment,
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

        # Maximale Instanzen: Partikel + Schockwellen-Ring + Kern-Schichten
        max_particles = int(self.PARAMS['particle_count'][2])
        self._max_instances = max_particles + self._RING_POINTS + 4

        # Instanz-Daten: pos_x, pos_y, r, g, b, size, alpha, seed
        self._instance_data = np.zeros((self._max_instances, 8), dtype=np.float32)
        self._instance_vbo = self.ctx.buffer(
            reserve=self._max_instances * 8 * 4, dynamic=True
        )

        # VAO: Quad-Vertex (non-instanced) + Instanz-Attribute (instanced via /i)
        self._vao = self.ctx.vertex_array(
            self._prog,
            [
                (self._quad_vbo, "2f", "in_vertex_pos"),
                (
                    self._instance_vbo,
                    "2f 3f 1f 1f 1f /i",
                    "in_pos",
                    "in_color",
                    "in_size",
                    "in_alpha",
                    "in_seed",
                ),
            ],
        )

        self._init_particles()
        self._reset_motion_state()

    def _on_params_changed(self):
        """Re-initialisiere Partikel wenn sich Struktur-Parameter aendern."""
        self._init_particles()

    def _reset_motion_state(self):
        """Setzt die (deterministischen) Bewegungs-Akkumulatoren zurueck."""
        self._last_time = None     # letzte Frame-Zeit (fuer dt)
        self._rot_phase = 0.0      # akkumulierte Rotationsphase des Vortex
        self._shock_age = 99.0     # Alter der letzten Schockwelle (Sekunden)
        self._energy_s = 0.0       # EMA-geglaettete Energie
        self._treble_s = 0.0       # EMA-geglaettetes Treble (Funkeln)
        self._flow_s = 0.0         # EMA-geglaetteter Sprach-Flow
        self._prev_voice = 0.0     # Voice-Band des letzten Frames (Betonung)

    def _init_particles(self):
        """Initialisiert die statischen Partikel-Eigenschaften (vektorisiert).

        Alle Zufallswerte stammen aus einem fest gesaeten Generator —
        identische Ergebnisse bei jedem Lauf, kein globaler RNG-Zustand.
        """
        n = int(self.params["particle_count"])
        arms = max(1, int(self.params["spiral_arms"]))
        rng = np.random.default_rng(self._SEED)

        # Radial-Verteilung: Dichte faellt nach aussen ab (Galaxien-Profil)
        self._rad0 = (0.05 + 0.95 * rng.random(n) ** 1.2).astype(np.float32)
        # Spiralarme: gleichmaessige Winkelverteilung + Streuung um den Arm
        arm_idx = (np.arange(n) % arms).astype(np.float32)
        self._arm_angle = (
            arm_idx * (2.0 * np.pi / arms) + rng.normal(0.0, 0.55, n)
        ).astype(np.float32)
        # Leichte Differentialrotation: innen schneller, Arme bleiben aber
        # ueber lange Zeit als Struktur lesbar (kein Verwischen zum Ring)
        self._orbit = (1.0 / (0.7 + 0.6 * self._rad0)).astype(np.float32)
        # Lebensdauer-Zyklus (Einfall + Respawn), rein zeitanalytisch
        self._lifespan = (7.0 + rng.random(n) * 9.0).astype(np.float32)
        self._life_off = rng.random(n).astype(np.float32)
        # Groesse: viele kleine, wenige grosse Partikel (Basis @720p)
        self._size0 = (1.6 + rng.random(n) ** 3 * 6.0).astype(np.float32)
        # Wobble: individuelle Frequenz/Phase/Amplitude
        self._wob_freq = (0.4 + rng.random(n) * 1.6).astype(np.float32)
        self._wob_phase = (rng.random(n) * 2.0 * np.pi).astype(np.float32)
        self._wob_amp = rng.random(n).astype(np.float32)
        # Farb-Jitter um den Chroma-Basisfarbton
        self._hue_jit = ((rng.random(n) - 0.5) * 0.12).astype(np.float32)
        # Sichtbarkeits-Schwelle: steuert die sichtbare Partikelmenge
        self._vis_thresh = rng.random(n).astype(np.float32)
        # Seed fuer Funkeln/Dithering im Shader
        self._seed = (rng.random(n) * 100.0).astype(np.float32)

    @staticmethod
    def _smoothstep(e0, e1, x):
        """Vektorisiertes smoothstep (wie GLSL)."""
        t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

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

    def render(self, features: dict, time: float):
        """Rendert einen Frame: Galaxien-Partikel, Schockwellen-Ring, Kern.

        Die Partikel-Positionen sind analytische Funktionen aus Index, Zeit
        und geglaetteten Audio-Features — kein Zufallszustand zur Laufzeit.
        """
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        u = self._map_features_to_uniforms(f, mode)
        is_speech = mode == "speech"

        # --- Frame-Delta (robust, falls nicht sequentiell gerendert wird) ---
        fps = float(features.get("fps", 30) or 30)
        if self._last_time is None or not (0.0 < time - self._last_time <= 0.25):
            dt = 1.0 / fps
            first_frame = True
        else:
            dt = time - self._last_time
            first_frame = False
        self._last_time = time

        # --- Geglaettete Feature-Werte (EMA) ---
        energy_raw = float(u["u_energy"])
        treble_raw = float(u["u_detail"])
        flow_raw = float(u["u_flow"])
        if first_frame:
            self._energy_s, self._treble_s, self._flow_s = (
                energy_raw, treble_raw, flow_raw,
            )
        else:
            k = 1.0 - np.exp(-dt * 8.0)
            self._energy_s += (energy_raw - self._energy_s) * k
            self._treble_s += (treble_raw - self._treble_s) * k
            self._flow_s += (flow_raw - self._flow_s) * k

        # --- Schockwelle: Musik = Beat/Transient, Sprache = Betonung ---
        if is_speech:
            # Anstieg im Voice-Band = Betonung -> sanfter Wirbel-Puls
            signal = max(0.0, flow_raw - self._prev_voice) * 3.0
            threshold = 0.18
            strength_scale = 0.45
        else:
            signal = max(float(u["u_impact"]), float(u.get("u_beat_intensity", 0.0)))
            threshold = 0.35
            strength_scale = 1.0
        self._prev_voice = flow_raw

        if signal > threshold and self._shock_age > 0.3:
            self._shock_age = 0.0
        self._shock_age += dt
        shock_env = float(np.exp(-self._shock_age * 2.8)) if self._shock_age < 3.0 else 0.0
        shock_radius = self._shock_age * 1.6  # normierte Front-Position

        # --- Rotation: Musik = Energie, Sprache = ruhiger Flow ---
        vortex_speed = float(self.params["vortex_speed"])
        if is_speech:
            rot_rate = vortex_speed * (0.10 + 0.55 * self._flow_s)
        else:
            rot_rate = vortex_speed * (0.25 + 1.6 * self._energy_s)
        self._rot_phase += rot_rate * dt

        # --- Geometrie-Konstanten ---
        cx, cy = self.width / 2.0, self.height / 2.0
        radius = min(self.width, self.height) * 0.55
        tilt = float(self.params["galaxy_tilt"])
        tilt_angle = -0.42  # feste Kipp-Rotation der Galaxien-Ebene
        cos_t, sin_t = np.cos(tilt_angle), np.sin(tilt_angle)

        twist = float(self.params["spiral_twist"])
        infall = max(float(self.params["infall_speed"]), 1e-4)
        wobble = float(self.params["wobble"])
        shock_str = float(self.params["shockwave_strength"]) * strength_scale
        sparkle = float(self.params["sparkle"]) * (0.3 if is_speech else 1.0)
        hue_spread = float(self.params["hue_spread"])
        point_size = float(self.params["point_size"])
        core_glow = float(self.params["core_glow"])

        # Aktiver Pegel: Musik = Energie, Sprache = Voice-Flow
        level = self._flow_s if is_speech else self._energy_s

        # === Partikel-Positionen (analytisch, vektorisiert) ===
        # Lebenszyklus: Einfall von aussen nach innen, dann Respawn
        life = np.mod(time * infall / self._lifespan + self._life_off, 1.0)
        r_norm = self._rad0 * (1.0 - 0.80 * life ** 1.6) + 0.04
        # Radial-Wobble
        r_norm = r_norm + wobble * 0.015 * self._wob_amp * np.sin(
            time * self._wob_freq + self._wob_phase
        )
        # Spiralwinkel: Arm + Twist + Vortex-Rotation + Wobble
        angle = (
            self._arm_angle
            + self._rad0 * twist
            + self._rot_phase * self._orbit
            + wobble * 0.05 * np.sin(time * self._wob_freq * 0.7 + self._wob_phase * 1.3)
        )

        # Schockwelle: Partikel nahe der Front werden nach aussen geschleudert
        band = np.exp(-((r_norm - shock_radius) ** 2) / 0.012)
        r_disp = r_norm + shock_str * 0.30 * shock_env * band

        # Gekippte Galaxien-Ebene -> Pixelkoordinaten
        gx = np.cos(angle) * r_disp
        gy = np.sin(angle) * r_disp * tilt
        px = cx + (gx * cos_t - gy * sin_t) * radius
        py = cy + (gx * sin_t + gy * cos_t) * radius

        # === Sichtbarkeit & Alpha ===
        # Sichtbare Menge skaliert mit dem Pegel (Pausen = fast ruhend)
        vis = (self._vis_thresh < (0.42 + 0.62 * level)).astype(np.float32)
        fade = self._smoothstep(0.0, 0.08, life) * self._smoothstep(1.0, 0.90, life)
        # Zentrum-nahe Partikel leuchten staerker (Dichte-Eindruck)
        lum = np.clip(1.25 - r_disp, 0.25, 1.25)

        # Funkeln: deterministischer Groessen-/Alpha-Jitter aus Hash
        tw_h = np.mod(
            np.sin(self._seed * 127.1 + np.floor(time * 14.0) * 311.7) * 43758.5453,
            1.0,
        )
        size_jit = 1.0 + sparkle * self._treble_s * (tw_h - 0.5) * 1.8
        alpha_jit = 1.0 + sparkle * self._treble_s * (np.mod(tw_h * 7.13, 1.0) - 0.5) * 1.2

        alpha = fade * vis * lum * alpha_jit * (0.45 + 0.75 * level)
        size_px = (
            self._size0 * size_jit * point_size
            * (self.height / 720.0)
            * (0.70 + 0.60 * level)
            * (1.0 + shock_env * band * 1.5)
        )

        # === Farben: Chroma-Basiston + Verlauf ueber Radius ===
        base_color = self._chroma_to_color(f["chroma"])
        base_hue = self._color_to_hue(base_color)
        color_mode = self.params.get('color_mode', 'chroma')
        sat = 0.0 if color_mode == 'monochrome' else (
            float(self.params.get('color_saturation', 0.7)) * (0.75 + 0.25 * level)
        )
        hue = base_hue + (r_disp - 0.45) * hue_spread + self._hue_jit + life * 0.15
        val = (0.50 + 0.60 * level) * lum * (1.0 + shock_env * band)
        rgb = self._hsv_to_rgb_array(hue, sat, val)

        # === Instanz-Puffer fuellen ===
        parts = []
        n = len(self._rad0)
        main = np.empty((n, 8), dtype=np.float32)
        main[:, 0] = px
        main[:, 1] = py
        main[:, 2:5] = rgb
        main[:, 5] = size_px * 2.4  # Quad halb so gross, Halo hat Platz
        main[:, 6] = np.clip(alpha, 0.0, 1.5)
        main[:, 7] = self._seed
        parts.append(main[size_px > 0.05])

        # === Schockwellen-Ring ===
        if shock_env > 0.02 and shock_radius < 1.3:
            ring_ang = np.linspace(0.0, 2.0 * np.pi, self._RING_POINTS,
                                   endpoint=False, dtype=np.float32)
            rgx = np.cos(ring_ang) * shock_radius
            rgy = np.sin(ring_ang) * shock_radius * tilt
            ring_px = cx + (rgx * cos_t - rgy * sin_t) * radius
            ring_py = cy + (rgx * sin_t + rgy * cos_t) * radius
            ring_col = tuple(min(1.0, c * 0.6 + 0.4) for c in base_color)
            ring = np.empty((self._RING_POINTS, 8), dtype=np.float32)
            ring[:, 0] = ring_px
            ring[:, 1] = ring_py
            ring[:, 2:5] = ring_col
            ring[:, 5] = (5.0 + 10.0 * shock_env) * (self.height / 720.0)
            ring[:, 6] = shock_env * 0.5 * float(self.params["shockwave_strength"])
            ring[:, 7] = 0.5
            parts.append(ring)

        # === Zentraler Kern (drei Schichten) ===
        flash = 1.0 + shock_env * 2.2
        res_scale = self.height / 720.0
        core_white = (1.0, 1.0, 1.0)

        def _mix(a, b, t):
            return tuple(a[i] * (1.0 - t) + b[i] * t for i in range(3))

        core_layers = [
            # Weiter Halo: sehr dezent, haelt den Hintergrund sichtbar
            (radius * 0.55, 0.05 + 0.10 * level,
             tuple(c * 0.30 * core_glow for c in base_color), 0.1),
            # Mittlerer Glow
            ((50.0 + 90.0 * level) * res_scale, 0.30 * core_glow,
             _mix(base_color, core_white, 0.30), 0.3),
            # Heisser Kern (HDR, blitzt bei Schockwelle auf)
            ((14.0 + 26.0 * level) * res_scale, 0.85 * core_glow,
             tuple(c * 1.4 * flash for c in _mix(core_white, base_color, 0.35)), 0.7),
        ]
        core = np.empty((len(core_layers), 8), dtype=np.float32)
        for j, (size, a, col, seed) in enumerate(core_layers):
            core[j] = [cx, cy, col[0], col[1], col[2], size * 2.4, a, seed]
        parts.append(core)

        instances = np.concatenate(parts, axis=0)
        if instances.shape[0] > self._max_instances:
            instances = instances[: self._max_instances]
        instance_count = instances.shape[0]

        self._prog["u_resolution"].value = (self.width, self.height)
        self._prog["u_brightness"].value = float(self.params.get("brightness", 1.0))

        if instance_count > 0:
            self._instance_vbo.write(np.ascontiguousarray(instances).tobytes())
            # Additive Mischung: Gluehpartikel akkumulieren im HDR-Buffer
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            self._vao.render(mode=moderngl.TRIANGLE_STRIP, instances=instance_count)
            self.ctx.disable(moderngl.BLEND)
