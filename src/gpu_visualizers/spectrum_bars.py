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
uniform float u_brightness;
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color * u_brightness, 1.0);
}
"""


class SpectrumBarsGPU(BaseGPUVisualizer):
    """Spectrum-Bars-Visualizer mit GPU-beschleunigtem Rendering.

    Jeder Balken besteht aus 4 Vertices (2 Dreiecke). Die Vertex-Daten
    werden pro Frame neu in das VBO geschrieben, um die Balkenhoehen
    anzupassen. Farbverlaeufe werden pro Vertex interpoliert und
    respektieren den gemeinsamen color_mode.
    """

    PARAMS = {
        'bar_count': (40, 10, 100, 5),
        'height_scale': (1.0, 0.2, 3.0, 0.1),
        'spacing': (0.25, 0.0, 0.8, 0.05),
        'color_shift': (0.0, 0.0, 1.0, 0.05),
        'base_height': (0.1, 0.0, 0.5, 0.05),
        'height_boost': (0.85, 0.0, 1.2, 0.05),
        'wave_count': (0.3, 0.0, 2.0, 0.1),
        'color_spread': (0.02, 0.0, 0.1, 0.005),
    }

    PARAMS_GROUPS = {
        "Balken": ["bar_count", "spacing", "base_height", "height_scale", "height_boost"],
        "Farbe": ["color_shift", "color_spread"],
        "Welle": ["wave_count"],
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
        base_height = self.params['base_height']
        height_boost = self.params['height_boost']
        height_scale = (rms * 0.7 + spectral_centroid * 0.3) * self.params['height_scale']
        max_height = self.height * (base_height + height_scale * height_boost)

        # Basisfarbe ueber den gemeinsamen color_mode ermitteln
        base_rgb = self._chroma_to_color(chroma)
        hue = self._color_to_hue(base_rgb)
        saturation = float(self.params.get('color_saturation', 0.7))
        brightness = float(self.params.get('brightness', 1.0))

        # Neue Vertex-Daten generieren
        vertices = self._build_bar_vertices(max_height, hue, saturation, brightness)
        self.vbo.write(vertices.tobytes())

        # Brightness binden und zeichnen
        self.prog["u_brightness"].value = self.params.get("brightness", 1.0)
        self.vao.render(mode=moderngl.TRIANGLES)

    def _build_bar_vertices(self, max_height: float, hue: float,
                            saturation: float, brightness: float) -> np.ndarray:
        """Baut das VBO-Array fuer alle Balken.

        Args:
            max_height: Maximale Balkenhoehe in Pixeln.
            hue: Grund-Farbton (0.0-1.0).
            saturation: Saettigung aus den gemeinsamen Farb-Parametern.
            brightness: Helligkeit aus den gemeinsamen Effekt-Parametern.

        Returns:
            Numpy-Array mit allen Vertex-Daten.
        """
        vertices = np.zeros(self._max_vertices, dtype=self._vertex_dtype)

        usable_width = self.width
        total_bar_width = usable_width / self.bar_count
        bar_width = total_bar_width * (1.0 - self.bar_spacing_ratio)
        spacing = total_bar_width * self.bar_spacing_ratio

        wave_count = self.params['wave_count']
        color_spread = self.params['color_spread']
        color_shift = self.params['color_shift']

        for i in range(self.bar_count):
            # Individuelle Hoehe pro Balken leicht variieren fuer visuelle Dynamik
            bar_height = max_height * (0.4 + 0.6 * np.sin(i * wave_count + hue * 6.28) ** 2)
            bar_height = max(2.0, min(bar_height, self.height))

            x_left = i * total_bar_width + spacing / 2.0
            x_right = x_left + bar_width
            y_bottom = 0.0
            y_top = bar_height

            # Farbverlauf von unten (dunkel) nach oben (hell)
            local_hue = hue + i * color_spread + color_shift
            color_bottom = self._hsv_to_rgb(local_hue, saturation, 0.45 * brightness)
            color_top = self._hsv_to_rgb(local_hue, saturation, 0.95 * brightness)

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


