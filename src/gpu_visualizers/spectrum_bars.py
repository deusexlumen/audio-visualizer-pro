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
        f = self._features_at_time(features, time)

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

    @staticmethod
    def _hsv_to_rgb_array(h: np.ndarray, s: float, v: np.ndarray) -> np.ndarray:
        """Vektorisierte HSV->RGB-Konvertierung fuer Balken-Farben."""
        h = np.mod(h, 1.0)
        i = (h * 6.0).astype(np.int32) % 6
        f = h * 6.0 - np.floor(h * 6.0)
        v = np.broadcast_to(np.asarray(v, dtype=np.float32), h.shape)
        s_arr = np.full(h.shape, s, dtype=np.float32)
        pp = v * (1.0 - s_arr)
        qq = v * (1.0 - s_arr * f)
        tt = v * (1.0 - s_arr * (1.0 - f))

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

    def _build_bar_vertices(self, max_height: float, hue: float,
                            saturation: float, brightness: float) -> np.ndarray:
        """Baut das VBO-Array fuer alle Balken (vektorisiert via NumPy).

        Args:
            max_height: Maximale Balkenhoehe in Pixeln.
            hue: Grund-Farbton (0.0-1.0).
            saturation: Saettigung aus den gemeinsamen Farb-Parametern.
            brightness: Helligkeit aus den gemeinsamen Effekt-Parametern.

        Returns:
            Numpy-Array mit allen Vertex-Daten.
        """
        n = self.bar_count
        total_bar_width = self.width / n
        bar_width = total_bar_width * (1.0 - self.bar_spacing_ratio)
        spacing = total_bar_width * self.bar_spacing_ratio

        wave_count = self.params['wave_count']
        color_spread = self.params['color_spread']
        color_shift = self.params['color_shift']

        i = np.arange(n, dtype=np.float32)

        # Individuelle Hoehe pro Balken (Wellen-Variation wie bisher)
        bar_height = max_height * (0.4 + 0.6 * np.sin(i * wave_count + hue * 6.28) ** 2)
        bar_height = np.clip(bar_height, 2.0, self.height)

        x_left = i * total_bar_width + spacing / 2.0
        x_right = x_left + bar_width

        # Farbverlauf von unten (dunkel) nach oben (hell)
        local_hue = hue + i * color_spread + color_shift
        color_bottom = self._hsv_to_rgb_array(local_hue, saturation, np.float32(0.45 * brightness))
        color_top = self._hsv_to_rgb_array(local_hue, saturation, np.float32(0.95 * brightness))

        # 6 Vertices pro Balken: (lu, ru, ro) + (lu, ro, lo)
        vertices = np.zeros(self._max_vertices, dtype=self._vertex_dtype)
        pos = vertices["in_position"].reshape(n, 6, 2)
        col = vertices["in_color"].reshape(n, 6, 3)

        pos[:, 0, 0] = x_left;  pos[:, 0, 1] = 0.0
        pos[:, 1, 0] = x_right; pos[:, 1, 1] = 0.0
        pos[:, 2, 0] = x_right; pos[:, 2, 1] = bar_height
        pos[:, 3, 0] = x_left;  pos[:, 3, 1] = 0.0
        pos[:, 4, 0] = x_right; pos[:, 4, 1] = bar_height
        pos[:, 5, 0] = x_left;  pos[:, 5, 1] = bar_height

        col[:, 0] = color_bottom
        col[:, 1] = color_bottom
        col[:, 2] = color_top
        col[:, 3] = color_bottom
        col[:, 4] = color_top
        col[:, 5] = color_top

        return vertices


