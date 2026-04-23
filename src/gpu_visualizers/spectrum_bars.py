"""
GPU-beschleunigter Spectrum-Bars-Visualizer mit ModernGL.

Rendert N vertikale Balken als farbige Quads. Die Hoehe skaliert dynamisch
mit RMS und Spectral-Centroid. Alle Balken befinden sich in einem einzigen
VBO fuer maximale Effizienz.
"""

import numpy as np
import moderngl

from .base import BaseGPUVisualizer


_VERTEX_SHADER = """
#version 330
uniform vec2 u_resolution;
in vec2 in_position;
in vec3 in_color;
out vec3 v_color;
void main() {
    vec2 pos = in_position;
    pos.x = (pos.x / u_resolution.x) * 2.0 - 1.0;
    pos.y = (pos.y / u_resolution.y) * 2.0 - 1.0;
    gl_Position = vec4(pos, 0.0, 1.0);
    v_color = in_color;
}
"""

_FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""


class SpectrumBarsGPU(BaseGPUVisualizer):
    """Spectrum-Bars-Visualizer mit GPU-beschleunigtem Rendering.

    Jeder Balken besteht aus 4 Vertices (2 Dreiecke). Die Vertex-Daten
    werden pro Frame neu in das VBO geschrieben, um die Balkenhoehen
    anzupassen. Farbverlaeufe werden pro Vertex interpoliert.
    """

    PARAMS = {
        'bar_count': (40, 10, 100, 5),
        'height_scale': (1.0, 0.2, 3.0, 0.1),
        'spacing': (0.25, 0.0, 0.8, 0.05),
        'color_shift': (0.0, 0.0, 1.0, 0.05),
    }

    def _setup(self):
        """Initialisiert Shader, VBO und VAO fuer die Balken."""
        self._init_geometry()

    def _on_params_changed(self):
        """Reinitialisiert Geometrie wenn sich bar_count aendert."""
        self._init_geometry()

    def _init_geometry(self):
        """Erstellt/aktualisiert Shader, VBO und VAO."""
        self.bar_count = int(self.params['bar_count'])
        self.bar_spacing_ratio = self.params['spacing']

        self.prog = self.ctx.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )
        self.prog["u_resolution"].value = (self.width, self.height)

        # Pro Balken 6 Vertices (2 Dreiecke), je 5 Floats (x, y, r, g, b)
        self._vertex_dtype = np.dtype([
            ("in_position", np.float32, 2),
            ("in_color", np.float32, 3),
        ])
        self._vertices_per_bar = 6
        self._max_vertices = self.bar_count * self._vertices_per_bar

        self.vbo = self.ctx.buffer(reserve=self._max_vertices * self._vertex_dtype.itemsize)

        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f 3f", "in_position", "in_color")],
        )

    def render(self, features: dict, time: float):
        """Rendert einen Frame mit aktualisierten Balkenhoehen.

        Args:
            features: Dictionary mit Audio-Features fuer alle Frames.
            time: Aktuelle Zeit in Sekunden.
        """
        frame_idx = int(time * features.get("fps", 30))
        frame_idx = max(0, min(frame_idx, features.get("frame_count", 0) - 1))

        f = self._get_feature_at_frame(features, frame_idx)

        rms = f["rms"]
        spectral_centroid = f["spectral_centroid"]
        chroma = f["chroma"]

        # Dynamische Hoehe basierend auf RMS und Spectral-Centroid
        base_height = 0.1
        height_scale = (rms * 0.7 + spectral_centroid * 0.3) * self.params['height_scale']
        max_height = self.height * (base_height + height_scale * 0.85)

        # Farbe aus Chroma ableiten
        dominant_note = int(np.argmax(chroma))
        hue = dominant_note / 12.0

        # Neue Vertex-Daten generieren
        vertices = self._build_bar_vertices(max_height, hue)
        self.vbo.write(vertices.tobytes())

        # Zeichnen
        self.vao.render(mode=moderngl.TRIANGLES)

    def _build_bar_vertices(self, max_height: float, hue: float) -> np.ndarray:
        """Baut das VBO-Array fuer alle Balken.

        Args:
            max_height: Maximale Balkenhoehe in Pixeln.
            hue: Grund-Farbton (0.0-1.0).

        Returns:
            Numpy-Array mit allen Vertex-Daten.
        """
        vertices = np.zeros(self._max_vertices, dtype=self._vertex_dtype)

        usable_width = self.width
        total_bar_width = usable_width / self.bar_count
        bar_width = total_bar_width * (1.0 - self.bar_spacing_ratio)
        spacing = total_bar_width * self.bar_spacing_ratio

        for i in range(self.bar_count):
            # Individuelle Hoehe pro Balken leicht variieren fuer visuelle Dynamik
            bar_height = max_height * (0.4 + 0.6 * np.sin(i * 0.3 + hue * 6.28) ** 2)
            bar_height = max(2.0, min(bar_height, self.height))

            x_left = i * total_bar_width + spacing / 2.0
            x_right = x_left + bar_width
            y_bottom = 0.0
            y_top = bar_height

            # Farbverlauf von unten (dunkel) nach oben (hell)
            color_shift = self.params['color_shift']
            color_bottom = self._hue_to_rgb(hue + i * 0.02 + color_shift, 0.8, 0.4)
            color_top = self._hue_to_rgb(hue + i * 0.02 + color_shift, 0.9, 1.0)

            idx = i * self._vertices_per_bar

            # Erstes Dreieck (links-unten, rechts-unten, rechts-oben)
            vertices[idx + 0]["in_position"] = (x_left, y_bottom)
            vertices[idx + 0]["in_color"] = color_bottom

            vertices[idx + 1]["in_position"] = (x_right, y_bottom)
            vertices[idx + 1]["in_color"] = color_bottom

            vertices[idx + 2]["in_position"] = (x_right, y_top)
            vertices[idx + 2]["in_color"] = color_top

            # Zweites Dreieck (links-unten, rechts-oben, links-oben)
            vertices[idx + 3]["in_position"] = (x_left, y_bottom)
            vertices[idx + 3]["in_color"] = color_bottom

            vertices[idx + 4]["in_position"] = (x_right, y_top)
            vertices[idx + 4]["in_color"] = color_top

            vertices[idx + 5]["in_position"] = (x_left, y_top)
            vertices[idx + 5]["in_color"] = color_top

        return vertices

    @staticmethod
    def _hue_to_rgb(h: float, s: float, v: float) -> tuple:
        """Konvertiert Hue-Saturation-Value nach RGB-Tupel.

        Args:
            h: Hue im Bereich 0.0-1.0.
            s: Saturation im Bereich 0.0-1.0.
            v: Value im Bereich 0.0-1.0.

        Returns:
            Tuple (r, g, b) im Bereich 0.0-1.0.
        """
        h = h % 1.0
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        i = i % 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        return (v, p, q)
