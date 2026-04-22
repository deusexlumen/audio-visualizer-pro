"""
GPU-beschleunigter Neon Oscilloscope mit leuchtenden Linien, Trail und Grid.

Die Wellenform wird als Triangle-Strip-Ribbon gerendert:
- Jeder Punkt erzeugt 2 Vertices (oben + unten)
- Der Fragment-Shader berechnet weichen Rand + exponentiellen Glow
- Trail-Effekt durch mehrere, abgeschwaechte Wellenformen
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class NeonOscilloscopeGPU(BaseGPUVisualizer):
    """
    Retro-futuristischer GPU-Oscilloscope mit Neon-Linien und Glow.
    """

    PARAMS = {
        'line_thickness': (4, 1, 12, 1),
        'trail_length': (8, 0, 16, 1),
        'num_points': (200, 50, 500, 10),
        'glow_radius': (12, 4, 40, 2),
        'grid_enabled': (1, 0, 1, 1),
    }

    def _setup(self):
        """Initialisiere Shader, VBOs und Waveform-History."""
        # --- Ribbon Shader (Wellenform als dickes, leuchtendes Band) ---
        self._ribbon_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;
            uniform float u_thickness;

            in vec2 in_pos;      // Mittelpunkt des Bands (Pixel)
            in float in_t;       // -1.0 = untere Kante, +1.0 = obere Kante
            in vec3 in_color;
            in float in_alpha;

            out vec3 v_color;
            out float v_alpha;
            out float v_t;

            void main() {
                vec2 pixel_pos = in_pos + vec2(0.0, in_t * u_thickness);
                vec2 ndc = (pixel_pos / u_resolution) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);

                v_color = in_color;
                v_alpha = in_alpha;
                v_t = in_t;
            }
            """,
            fragment_shader="""
            #version 330
            uniform float u_brightness;
            in vec3 v_color;
            in float v_alpha;
            in float v_t;
            out vec4 f_color;

            void main() {
                float dist = abs(v_t);  // 0.0 = Mitte, 1.0 = Rand

                // Kern: scharfe Mitte
                float core = 1.0 - smoothstep(0.0, 0.35, dist);
                // Glow: weicher, exponentieller Abfall
                float glow = exp(-dist * dist * 5.0);

                vec3 final_color = v_color * (core + glow * 0.8) * u_brightness;
                float alpha = (core * 0.95 + glow * 0.5) * v_alpha;

                f_color = vec4(final_color, alpha);
            }
            """,
        )

        # --- Line Shader (Grid, Rahmen, Scan-Linie) ---
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
            void main() {
                f_color = vec4(v_color * u_brightness, v_alpha);
            }
            """,
        )

        # --- Fullscreen-Quad Shader (Beat-Flash) ---
        self._flash_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            void main() {
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            uniform vec3 u_color;
            uniform float u_alpha;
            uniform float u_brightness;
            out vec4 f_color;
            void main() {
                f_color = vec4(u_color * u_brightness, u_alpha);
            }
            """,
        )

        # Buffer fuer Ribbon (Wellenform) – max 500 Punkte * 2 Vertices
        max_ribbon_verts = 500 * 2
        self._ribbon_vbo = self.ctx.buffer(reserve=max_ribbon_verts * 7 * 4, dynamic=True)
        self._ribbon_vao = self.ctx.vertex_array(
            self._ribbon_prog,
            [(self._ribbon_vbo, "2f 1f 3f 1f", "in_pos", "in_t", "in_color", "in_alpha")],
        )

        # Buffer fuer Lines (Grid, Rahmen, Scan) – ausreichend gross
        max_line_verts = 2000
        self._line_vbo = self.ctx.buffer(reserve=max_line_verts * 6 * 4, dynamic=True)
        self._line_vao = self.ctx.vertex_array(
            self._line_prog,
            [(self._line_vbo, "2f 3f 1f", "in_pos", "in_color", "in_alpha")],
        )

        # Fullscreen-Quad fuer Flash
        flash_quad = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [-1.0,  1.0],
            [ 1.0,  1.0],
        ], dtype=np.float32)
        self._flash_vbo = self.ctx.buffer(flash_quad.tobytes())
        self._flash_vao = self.ctx.vertex_array(
            self._flash_prog, [(self._flash_vbo, "2f", "in_pos")]
        )

        # History fuer Trail-Effekt
        self._history = []

    def _on_params_changed(self):
        """History zuruecksetzen wenn sich Parameter aendern."""
        self._history = []

    def _generate_waveform(self, frame_idx: float, rms: float, spectral: float) -> np.ndarray:
        """Generiert die Wellenform als Array von (x, y) Pixel-Koordinaten."""
        num_points = int(self.params["num_points"])
        cx = self.width / 2.0
        cy = self.height / 2.0
        points = np.zeros((num_points, 2), dtype=np.float32)

        phase = frame_idx * 0.1
        freq1 = 0.02 * (1.0 + spectral)
        freq2 = 0.05 * (1.0 + rms)

        for i in range(num_points):
            x = (i / (num_points - 1)) * self.width
            wave1 = np.sin(i * freq1 + phase) * rms * self.height * 0.25
            wave2 = np.sin(i * freq2 + phase * 1.5) * rms * self.height * 0.15
            wave3 = np.sin(i * 0.01 + phase * 0.5) * spectral * self.height * 0.1
            y = cy + wave1 + wave2 + wave3
            y = max(50.0, min(self.height - 50.0, y))
            points[i] = [x, y]

        return points

    def _build_ribbon_vertices(self, waveform: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
        """Baut Ribbon-Vertices fuer ein Triangle Strip aus der Wellenform."""
        n = len(waveform)
        # 2 Vertices pro Punkt: (pos, t, color, alpha)
        verts = np.zeros((n * 2, 7), dtype=np.float32)
        for i, (x, y) in enumerate(waveform):
            # Unterer Vertex
            verts[i * 2] = [x, y, -1.0, color[0], color[1], color[2], alpha]
            # Oberer Vertex
            verts[i * 2 + 1] = [x, y, 1.0, color[0], color[1], color[2], alpha]
        return verts

    def _build_grid_lines(self) -> np.ndarray:
        """Baut Grid-Vertices als LINES."""
        if not self.params["grid_enabled"]:
            return np.zeros((0, 6), dtype=np.float32)

        spacing = 50.0
        grid_color = (20 / 255.0, 20 / 255.0, 40 / 255.0)
        alpha = 1.0

        lines = []
        # Vertikale Linien
        x = 0.0
        while x < self.width:
            lines.append([x, 0.0, *grid_color, alpha])
            lines.append([x, self.height, *grid_color, alpha])
            x += spacing
        # Horizontale Linien
        y = 0.0
        while y < self.height:
            lines.append([0.0, y, *grid_color, alpha])
            lines.append([self.width, y, *grid_color, alpha])
            y += spacing

        return np.array(lines, dtype=np.float32)

    def _build_border_lines(self, color: tuple) -> np.ndarray:
        """Baut Rahmen-Vertices als LINE_LOOP."""
        alpha = 1.0
        return np.array([
            [2.0, 2.0, *color, alpha],
            [self.width - 3, 2.0, *color, alpha],
            [self.width - 3, self.height - 3, *color, alpha],
            [2.0, self.height - 3, *color, alpha],
        ], dtype=np.float32)

    def _build_scan_line(self, x: float, color: tuple) -> np.ndarray:
        """Baut Scan-Linien-Vertices als LINES."""
        alpha = 1.0
        return np.array([
            [x, 0.0, *color, alpha],
            [x, self.height, *color, alpha],
        ], dtype=np.float32)

    def render(self, features: dict, time: float):
        """Rendert einen Frame mit Wellenform, Trail, Grid und Effekten."""
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        rms = f["rms"]
        onset = f["onset"]
        spectral = f["spectral_centroid"]
        chroma = f["chroma"]

        # Farben aus Chroma
        if chroma is not None and chroma.size > 0:
            dominant = int(np.argmax(chroma))
            base_hue = dominant / 12.0
        else:
            base_hue = 0.5
        neon_color = self._hsv_to_rgb(base_hue, 0.35, 0.7)
        secondary_hue = (base_hue + 0.5) % 1.0
        secondary_color = self._hsv_to_rgb(secondary_hue, 0.35, 0.7)

        # --- Wellenform generieren und in History speichern ---
        waveform = self._generate_waveform(float(frame_idx), rms, spectral)
        self._history.append(waveform)
        max_hist = int(self.params["trail_length"])
        if len(self._history) > max_hist + 1:
            self._history.pop(0)

        # --- Ribbon rendern (Trail + Hauptlinie) ---
        self._ribbon_prog["u_resolution"].value = (self.width, self.height)
        thickness_scale = 0.5 + (self.params.get("line_width", 0.003) - 0.001) / 0.019 * 1.5
        self._ribbon_prog["u_thickness"].value = float(self.params["line_thickness"]) * thickness_scale

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Brightness-Uniforms binden
        brightness = self.params.get("brightness", 1.0)
        self._ribbon_prog["u_brightness"].value = brightness
        self._line_prog["u_brightness"].value = brightness
        self._flash_prog["u_brightness"].value = brightness

        # Trail-Linien (aeltere Wellenformen)
        trail_decay = self.params.get("trail_decay", 0.7)
        for i, hist_wave in enumerate(self._history[:-1]):
            trail_idx = len(self._history) - 1 - i
            trailFade = pow(1.0 - trail_decay, trail_idx)
            trail_alpha = trailFade * 0.5
            trail_color = (
                secondary_color[0] * trail_alpha,
                secondary_color[1] * trail_alpha,
                secondary_color[2] * trail_alpha,
            )
            verts = self._build_ribbon_vertices(hist_wave, trail_color, trail_alpha)
            self._ribbon_vbo.write(verts.tobytes())
            self._ribbon_vao.render(mode=moderngl.TRIANGLE_STRIP)

        # Haupt-Wellenform (helle Neon-Linie)
        main_verts = self._build_ribbon_vertices(waveform, neon_color, 1.0)
        self._ribbon_vbo.write(main_verts.tobytes())
        self._ribbon_vao.render(mode=moderngl.TRIANGLE_STRIP)

        # --- Grid, Rahmen, Scan-Linie (LINES) ---
        line_verts = []

        # Grid
        grid_verts = self._build_grid_lines()
        if len(grid_verts) > 0:
            line_verts.append(grid_verts)

        # Rahmen
        border_color = (max(0.0, c - 0.2) for c in neon_color)
        border_verts = self._build_border_lines(tuple(border_color))
        line_verts.append(border_verts)

        # Scan-Linie
        scan_x = frame_idx % self.width
        scan_color = (neon_color[0] * 0.25, neon_color[1] * 0.25, neon_color[2] * 0.25)
        scan_verts = self._build_scan_line(float(scan_x), scan_color)
        line_verts.append(scan_verts)

        if line_verts:
            all_lines = np.concatenate(line_verts, axis=0)
            self._line_prog["u_resolution"].value = (self.width, self.height)
            self._line_vbo.write(all_lines.tobytes())
            self._line_vao.render(mode=moderngl.LINES)

        # --- Beat-Flash ---
        if onset > 0.5:
            flash_alpha = (onset - 0.5) * 0.25
            flash_color = (neon_color[0] * 0.12, neon_color[1] * 0.12, neon_color[2] * 0.12)
            self._flash_prog["u_color"].value = flash_color
            self._flash_prog["u_alpha"].value = flash_alpha
            self._flash_vao.render(mode=moderngl.TRIANGLE_STRIP)

        self.ctx.disable(moderngl.BLEND)
