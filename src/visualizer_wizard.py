"""
Visualizer Wizard - Erstellung neuer GPU-Visualizer aus Templates.

Bietet reichhaltige Start-Templates fuer:
- shader:    Full-Screen Fragment-Shader mit Lygia-Snippets
- geometry:  CPU-generierte Geometrie (Grid/Quads) mit Vertex-Shader
- particles: CPU-aktualisiertes Partikel-System (Points + PointCoord)

Zusaetzlich enthaelt das Modul eine oeffentliche Hilfsfunktion fuer die GUI,
um einen "Neuen Visualizer erstellen"-Button in ein beliebiges Layout
einzufuegen (ohne das ParamsPanel direkt zu modifizieren).
"""

import re
from pathlib import Path
from typing import Literal

from src.gpu_visualizers.base import BaseGPUVisualizer


SUPPORTED_TYPES = ("shader", "geometry", "particles")


def _to_snake(name: str) -> str:
    """Normalisiert einen Namen zu snake_case fuer Datei- und Registry-Namen."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    s3 = re.sub(r"[^a-zA-Z0-9]+", "_", s2)
    return s3.lower().strip("_")


def _to_class_name(name: str) -> str:
    """Erzeugt einen Python-Klassennamen aus einem beliebigen Eingabenamen."""
    snake = _to_snake(name)
    if not snake.endswith("gpu"):
        snake += "_gpu"
    return "".join(part.capitalize() for part in snake.split("_") if part)


def _indent(text: str, spaces: int = 4) -> str:
    """Hilfsfunktion zum Einruecken von mehrzeiligen Strings."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _append_block(lines: list[str], text: str, indent: int = 0):
    """Fuegt einen mehrzeiligen Block mit Einrueckung zu einer Zeilenliste hinzu."""
    prefix = " " * indent
    for line in text.splitlines():
        lines.append(prefix + line)


class VisualizerWizard:
    """Wizard zum Generieren neuer GPU-Visualizer-Module."""

    def __init__(self, name: str, viz_type: Literal["shader", "geometry", "particles"] = "shader"):
        """Initialisiert den Wizard.

        Args:
            name: Name des neuen Visualizers (snake_case oder CamelCase).
            viz_type: Template-Typ: shader, geometry oder particles.
        """
        if viz_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unbekannter Typ: {viz_type}. Erlaubt: {SUPPORTED_TYPES}")

        self.name = name.strip()
        if not self.name:
            raise ValueError("Name darf nicht leer sein")

        self.viz_type = viz_type
        self.module_name = _to_snake(self.name)
        self.class_name = _to_class_name(self.name)

    @staticmethod
    def list_types() -> list[str]:
        """Gibt die verfuegbaren Template-Typen zurueck."""
        return list(SUPPORTED_TYPES)

    def generate(self) -> str:
        """Generiert den vollstaendigen Python-Quelltext des neuen Visualizers."""
        if self.viz_type == "shader":
            return self._template_shader()
        if self.viz_type == "geometry":
            return self._template_geometry()
        if self.viz_type == "particles":
            return self._template_particles()
        raise RuntimeError("Unbekannter Template-Typ")

    def write(self, target_dir: Path | str) -> Path:
        """Schreibt das generierte Template in eine Python-Datei.

        Args:
            target_dir: Zielverzeichnis (normalerweise src/gpu_visualizers).

        Returns:
            Pfad zur erstellten Datei.
        """
        target = Path(target_dir) / f"{self.module_name}.py"
        if target.exists():
            raise FileExistsError(f"Datei existiert bereits: {target}")
        target.write_text(self.generate(), encoding="utf-8")
        return target

    # ------------------------------------------------------------------
    # Template-Generatoren
    # ------------------------------------------------------------------

    def _template_shader(self) -> str:
        """Full-Screen Fragment-Shader Template."""
        lines: list[str] = []
        lines.extend(self._file_header_lines())

        lines.append(f"class {self.class_name}(BaseGPUVisualizer):")
        _append_block(lines, self._class_docstring("Full-Screen Shader Visualizer.\n\n"
                                                     "Reagiert auf Energie, Beats und Chroma-Farben.\n"
                                                     "Bearbeite den Fragment-Shader, um das Erscheinungsbild anzupassen."),
                      indent=4)
        lines.append("")
        lines.append("    PARAMS = {")
        lines.append("        'intensity': (1.0, 0.0, 3.0, 0.1),")
        lines.append("        'speed': (1.0, 0.0, 5.0, 0.1),")
        lines.append("        'zoom': (1.0, 0.1, 3.0, 0.1),")
        lines.append("        'noise_scale': (2.0, 0.5, 5.0, 0.1),")
        lines.append("    }")
        lines.append("")
        lines.append("    def _setup(self):")
        lines.append("        self._build_program()")
        lines.append("        self._setup_quad()")
        lines.append("")
        lines.append("    def _build_program(self):")
        lines.append("        self._prog = self.ctx.program(")
        lines.append("            vertex_shader=\"\"\"")
        lines.append("            #version 330")
        lines.append("            in vec2 in_pos;")
        lines.append("            void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }")
        lines.append("            \"\"\",")
        lines.append("            fragment_shader=\"\"\"")
        lines.append("            #version 330")
        lines.append("            uniform vec2 u_resolution;")
        lines.append("            uniform float u_time;")
        lines.append("            uniform float u_energy;")
        lines.append("            uniform float u_beat;")
        lines.append("            uniform float u_impact;")
        lines.append("            uniform float u_detail;")
        lines.append("            uniform float u_flow;")
        lines.append("            uniform float u_beat_intensity;")
        lines.append("            uniform vec3 u_color;")
        lines.append("            uniform vec3 u_secondary_color;")
        lines.append("            uniform float u_intensity;")
        lines.append("            uniform float u_speed;")
        lines.append("            uniform float u_zoom;")
        lines.append("            uniform float u_noise_scale;")
        lines.append("            uniform float u_brightness;")
        lines.append("")
        lines.append("            out vec4 f_color;")
        lines.append("")
        _append_block(lines, self._lygia_math(), indent=0)
        lines.append("")
        _append_block(lines, self._lygia_noise(), indent=0)
        lines.append("")
        _append_block(lines, self._lygia_sdf(), indent=0)
        lines.append("")
        _append_block(lines, self._lygia_color(), indent=0)
        lines.append("")
        lines.append("            void main() {")
        lines.append("                vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;")
        lines.append("                uv.x *= u_resolution.x / u_resolution.y;")
        lines.append("                uv *= u_zoom;")
        lines.append("")
        lines.append("                vec3 col = vec3(0.03, 0.03, 0.05);")
        lines.append("")
        lines.append("                float t = u_time * u_speed + u_flow * 2.0;")
        lines.append("                float n = fbm(uv * u_noise_scale * (0.5 + u_detail) + t * 0.1, 4);")
        lines.append("                col += vec3(0.05, 0.07, 0.12) * n;")
        lines.append("")
        lines.append("                float d = sdCircle(uv, 0.25 + u_energy * 0.15);")
        lines.append("                vec3 shapeColor = mix(u_color, u_secondary_color, u_beat);")
        lines.append("                col += shapeColor * exp(-d * d * 80.0) * u_intensity;")
        lines.append("")
        lines.append("                // Beat-Flash und Transienten-Explosion")
        lines.append("                col += u_color * (u_impact * 0.4 + u_beat_intensity * 0.15);")
        lines.append("")
        lines.append("                col *= u_brightness;")
        lines.append("                col = col / (1.0 + col);")
        lines.append("")
        lines.append("                f_color = vec4(col, 1.0);")
        lines.append("            }")
        lines.append("            \"\"\",")
        lines.append("        )")
        lines.append("")
        lines.append("    def _setup_quad(self):")
        lines.append("        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)")
        lines.append("        vbo = self.ctx.buffer(quad.tobytes())")
        lines.append("        self._vao = self.ctx.vertex_array(self._prog, [(vbo, \"2f\", \"in_pos\")])")
        lines.append("")
        lines.append("    def render(self, features: dict, time: float):")
        lines.append("        frame_idx = int(time * features.get(\"fps\", 30))")
        lines.append("        f = self._get_feature_at_frame(features, frame_idx)")
        lines.append("")
        lines.append("        # Features je nach Modus (music/speech/hybrid) auf Uniforms mappen")
        lines.append("        uniforms = self._map_features_to_uniforms(f)")
        lines.append("")
        lines.append("        # Chroma-Vektor in RGB-Farbe umwandeln")
        lines.append('        color = self._chroma_to_color(uniforms["u_chroma"])')
        lines.append("        secondary = (0.0, 0.8, 1.0)")
        lines.append("")
        lines.append('        self._prog["u_resolution"].value = (self.width, self.height)')
        lines.append('        self._prog["u_time"].value = time')
        lines.append('        self._prog["u_energy"].value = uniforms["u_energy"]')
        lines.append('        self._prog["u_beat"].value = uniforms["u_beat"]')
        lines.append('        self._prog["u_impact"].value = uniforms["u_impact"]')
        lines.append('        self._prog["u_detail"].value = uniforms["u_detail"]')
        lines.append('        self._prog["u_flow"].value = uniforms["u_flow"]')
        lines.append('        self._prog["u_beat_intensity"].value = uniforms.get("u_beat_intensity", uniforms["u_beat"])')
        lines.append('        self._prog["u_color"].value = color')
        lines.append('        self._prog["u_secondary_color"].value = secondary')
        lines.append('        self._prog["u_intensity"].value = self.params["intensity"]')
        lines.append('        self._prog["u_speed"].value = self.params["speed"]')
        lines.append('        self._prog["u_zoom"].value = self.params["zoom"]')
        lines.append('        self._prog["u_noise_scale"].value = self.params["noise_scale"]')
        lines.append('        self._prog["u_brightness"].value = self.params["brightness"]')
        lines.append("")
        lines.append("        self._vao.render(mode=moderngl.TRIANGLE_STRIP)")

        return "\n".join(lines)

    def _template_geometry(self) -> str:
        """Geometrie-Template mit CPU-generiertem Grid."""
        lines: list[str] = []
        lines.extend(self._file_header_lines())

        lines.append(f"class {self.class_name}(BaseGPUVisualizer):")
        _append_block(lines, self._class_docstring("Geometrie-basierter Visualizer.\n\n"
                                                     "Rendert ein dynamisches Grid aus Rechtecken, dessen Groesse und Farbe\n"
                                                     "auf Audio-Features reagieren."), indent=4)
        lines.append("")
        lines.append("    PARAMS = {")
        lines.append("        'grid_cols': (8, 2, 20, 1),")
        lines.append("        'grid_rows': (6, 2, 15, 1),")
        lines.append("        'cell_size': (0.8, 0.1, 1.5, 0.05),")
        lines.append("        'spacing': (0.15, 0.0, 0.5, 0.05),")
        lines.append("        'color_shift': (0.0, 0.0, 1.0, 0.05),")
        lines.append("    }")
        lines.append("")
        lines.append("    def _setup(self):")
        lines.append("        self._build_program()")
        lines.append("        self._init_geometry()")
        lines.append("")
        lines.append("    def _on_params_changed(self):")
        lines.append("        self._init_geometry()")
        lines.append("")
        lines.append("    def _build_program(self):")
        lines.append("        self._prog = self.ctx.program(")
        lines.append("            vertex_shader=\"\"\"")
        lines.append("            #version 330")
        lines.append("            in vec2 in_pos;")
        lines.append("            in vec3 in_color;")
        lines.append("            uniform vec2 u_resolution;")
        lines.append("            out vec3 v_color;")
        lines.append("            void main() {")
        lines.append("                vec2 pos = in_pos;")
        lines.append("                pos.x = (pos.x / u_resolution.x) * 2.0 - 1.0;")
        lines.append("                pos.y = (pos.y / u_resolution.y) * 2.0 - 1.0;")
        lines.append("                gl_Position = vec4(pos, 0.0, 1.0);")
        lines.append("                v_color = in_color;")
        lines.append("            }")
        lines.append("            \"\"\",")
        lines.append("            fragment_shader=\"\"\"")
        lines.append("            #version 330")
        lines.append("            in vec3 v_color;")
        lines.append("            uniform float u_brightness;")
        lines.append("            out vec4 f_color;")
        lines.append("            void main() {")
        lines.append("                f_color = vec4(v_color * u_brightness, 1.0);")
        lines.append("            }")
        lines.append("            \"\"\",")
        lines.append("        )")
        lines.append("")
        lines.append("    def _init_geometry(self):")
        lines.append("        self.grid_cols = int(self.params['grid_cols'])")
        lines.append("        self.grid_rows = int(self.params['grid_rows'])")
        lines.append("        self.cell_count = self.grid_cols * self.grid_rows")
        lines.append("        self._vertex_dtype = np.dtype([")
        lines.append('            ("in_pos", np.float32, 2),')
        lines.append('            ("in_color", np.float32, 3),')
        lines.append("        ])")
        lines.append("        self._vertices_per_cell = 6")
        lines.append("        self._max_vertices = self.cell_count * self._vertices_per_cell")
        lines.append("        self._vbo = self.ctx.buffer(reserve=self._max_vertices * self._vertex_dtype.itemsize)")
        lines.append("        self._vao = self.ctx.vertex_array(")
        lines.append("            self._prog,")
        lines.append('            [(self._vbo, "2f 3f", "in_pos", "in_color")],')
        lines.append("        )")
        lines.append("")
        lines.append("    def render(self, features: dict, time: float):")
        lines.append("        frame_idx = int(time * features.get(\"fps\", 30))")
        lines.append("        f = self._get_feature_at_frame(features, frame_idx)")
        lines.append("")
        lines.append("        uniforms = self._map_features_to_uniforms(f)")
        lines.append('        base_rgb = self._chroma_to_color(uniforms["u_chroma"])')
        lines.append("        hue = self._color_to_hue(base_rgb)")
        lines.append("        saturation = float(self.params.get('color_saturation', 0.7))")
        lines.append("        brightness = float(self.params.get('brightness', 1.0))")
        lines.append("")
        lines.append("        vertices = self._build_grid_vertices(")
        lines.append('            uniforms["u_energy"],')
        lines.append('            uniforms["u_beat"],')
        lines.append("            hue,")
        lines.append("            saturation,")
        lines.append("            brightness,")
        lines.append("        )")
        lines.append("        self._vbo.write(vertices.tobytes())")
        lines.append("")
        lines.append('        self._prog["u_resolution"].value = (self.width, self.height)')
        lines.append('        self._prog["u_brightness"].value = brightness')
        lines.append("        self._vao.render(mode=moderngl.TRIANGLES)")
        lines.append("")
        lines.append("    def _build_grid_vertices(self, energy: float, beat: float, hue: float,")
        lines.append("                             saturation: float, brightness: float) -> np.ndarray:")
        lines.append("        vertices = np.zeros(self._max_vertices, dtype=self._vertex_dtype)")
        lines.append("        spacing = self.params['spacing']")
        lines.append("        cell_size = self.params['cell_size']")
        lines.append("        color_shift = self.params['color_shift']")
        lines.append("")
        lines.append("        margin_x = self.width * 0.05")
        lines.append("        margin_y = self.height * 0.05")
        lines.append("        usable_w = self.width - 2.0 * margin_x")
        lines.append("        usable_h = self.height - 2.0 * margin_y")
        lines.append("")
        lines.append("        cell_w = usable_w / self.grid_cols")
        lines.append("        cell_h = usable_h / self.grid_rows")
        lines.append("        gap_x = cell_w * spacing")
        lines.append("        gap_y = cell_h * spacing")
        lines.append("")
        lines.append("        idx = 0")
        lines.append("        for row in range(self.grid_rows):")
        lines.append("            for col in range(self.grid_cols):")
        lines.append("                cx = margin_x + col * cell_w + cell_w * 0.5")
        lines.append("                cy = margin_y + row * cell_h + cell_h * 0.5")
        lines.append("")
        lines.append("                scale = cell_size * (0.5 + 0.5 * energy + 0.3 * beat)")
        lines.append("                scale *= (0.8 + 0.2 * np.sin(col * 0.5 + row * 0.3))")
        lines.append("                half_w = (cell_w - gap_x) * 0.5 * scale")
        lines.append("                half_h = (cell_h - gap_y) * 0.5 * scale")
        lines.append("")
        lines.append("                local_hue = hue + (col + row) * color_shift * 0.1")
        lines.append("                color = self._hsv_to_rgb(local_hue, saturation, brightness)")
        lines.append("")
        lines.append("                # Zwei Dreiecke pro Zelle")
        lines.append("                x0, x1 = cx - half_w, cx + half_w")
        lines.append("                y0, y1 = cy - half_h, cy + half_h")
        lines.append("")
        lines.append("                for pos in [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]:")
        lines.append('                    vertices[idx]["in_pos"] = pos')
        lines.append('                    vertices[idx]["in_color"] = color')
        lines.append("                    idx += 1")
        lines.append("")
        lines.append("        return vertices")

        return "\n".join(lines)

    def _template_particles(self) -> str:
        """Partikel-Template mit CPU-aktualisierten Punkt-Sprites."""
        lines: list[str] = []
        lines.extend(self._file_header_lines())

        lines.append(f"class {self.class_name}(BaseGPUVisualizer):")
        _append_block(lines, self._class_docstring("Partikel-basierter Visualizer.\n\n"
                                                     "Aktualisiert Partikel-Positionen und -Groessen auf der CPU und rendert\n"
                                                     "weiche Point-Sprites im Fragment-Shader."), indent=4)
        lines.append("")
        lines.append("    PARAMS = {")
        lines.append("        'particle_count': (128, 16, 512, 16),")
        lines.append("        'base_size': (8.0, 1.0, 32.0, 1.0),")
        lines.append("        'speed': (1.0, 0.0, 3.0, 0.1),")
        lines.append("        'spread': (0.5, 0.1, 2.0, 0.1),")
        lines.append("        'turbulence': (0.3, 0.0, 1.0, 0.05),")
        lines.append("    }")
        lines.append("")
        lines.append("    def _setup(self):")
        lines.append("        self._build_program()")
        lines.append("        self._init_particles()")
        lines.append("")
        lines.append("    def _on_params_changed(self):")
        lines.append("        self._init_particles()")
        lines.append("")
        lines.append("    def _build_program(self):")
        lines.append("        self._prog = self.ctx.program(")
        lines.append("            vertex_shader=\"\"\"")
        lines.append("            #version 330")
        lines.append("            in vec2 in_pos;")
        lines.append("            in float in_size;")
        lines.append("            uniform vec2 u_resolution;")
        lines.append("            out float v_size;")
        lines.append("            void main() {")
        lines.append("                vec2 pos = (in_pos / u_resolution) * 2.0 - 1.0;")
        lines.append("                gl_Position = vec4(pos, 0.0, 1.0);")
        lines.append("                gl_PointSize = in_size;")
        lines.append("                v_size = in_size;")
        lines.append("            }")
        lines.append("            \"\"\",")
        lines.append("            fragment_shader=\"\"\"")
        lines.append("            #version 330")
        lines.append("            in float v_size;")
        lines.append("            uniform vec3 u_color;")
        lines.append("            uniform float u_brightness;")
        lines.append("            out vec4 f_color;")
        lines.append("            void main() {")
        lines.append("                vec2 uv = gl_PointCoord - 0.5;")
        lines.append("                float d = length(uv);")
        lines.append("                float alpha = 1.0 - smoothstep(0.35, 0.5, d);")
        lines.append("                f_color = vec4(u_color * u_brightness, alpha);")
        lines.append("            }")
        lines.append("            \"\"\",")
        lines.append("        )")
        lines.append("")
        lines.append("    def _init_particles(self):")
        lines.append("        self.particle_count = int(self.params['particle_count'])")
        lines.append("        self._positions = np.zeros((self.particle_count, 2), dtype=np.float32)")
        lines.append("        self._sizes = np.ones(self.particle_count, dtype=np.float32) * self.params['base_size']")
        lines.append("")
        lines.append("        # Initial zufaellig im Bild verteilen")
        lines.append("        self._positions[:, 0] = np.random.rand(self.particle_count) * self.width")
        lines.append("        self._positions[:, 1] = np.random.rand(self.particle_count) * self.height")
        lines.append("")
        lines.append("        self._pos_vbo = self.ctx.buffer(self._positions.tobytes())")
        lines.append("        self._size_vbo = self.ctx.buffer(self._sizes.tobytes())")
        lines.append("        self._vao = self.ctx.vertex_array(")
        lines.append("            self._prog,")
        lines.append('            [(self._pos_vbo, "2f", "in_pos"), (self._size_vbo, "1f", "in_size")],')
        lines.append("        )")
        lines.append("")
        lines.append("    def render(self, features: dict, time: float):")
        lines.append("        frame_idx = int(time * features.get(\"fps\", 30))")
        lines.append("        f = self._get_feature_at_frame(features, frame_idx)")
        lines.append("")
        lines.append("        uniforms = self._map_features_to_uniforms(f)")
        lines.append('        color = self._chroma_to_color(uniforms["u_chroma"])')
        lines.append("")
        lines.append("        self._update_particles(uniforms, time)")
        lines.append("")
        lines.append("        self._pos_vbo.write(self._positions.tobytes())")
        lines.append("        self._size_vbo.write(self._sizes.tobytes())")
        lines.append("")
        lines.append('        self._prog["u_resolution"].value = (self.width, self.height)')
        lines.append('        self._prog["u_color"].value = color')
        lines.append('        self._prog["u_brightness"].value = self.params["brightness"]')
        lines.append("")
        lines.append("        self._vao.render(mode=moderngl.POINTS)")
        lines.append("")
        lines.append("    def _update_particles(self, uniforms: dict, time: float):")
        lines.append('        energy = uniforms["u_energy"]')
        lines.append('        beat = uniforms["u_beat"]')
        lines.append('        flow = uniforms["u_flow"]')
        lines.append("        spread = self.params['spread']")
        lines.append("        speed = self.params['speed']")
        lines.append("        turbulence = self.params['turbulence']")
        lines.append("        base_size = self.params['base_size']")
        lines.append("")
        lines.append("        n = self.particle_count")
        lines.append("        t = time * speed")
        lines.append("")
        lines.append("        # Jede Partikelgruppe bewegt sich leicht unterschiedlich")
        lines.append("        idx = np.arange(n, dtype=np.float32)")
        lines.append("        angle = idx * 0.37 + t * (0.5 + energy * 2.0)")
        lines.append("")
        lines.append("        vx = np.cos(angle) * spread * (0.5 + flow * 0.5)")
        lines.append("        vy = np.sin(angle * 0.71) * spread * (0.5 + energy)")
        lines.append("")
        lines.append("        # Turbulenz hinzufuegen")
        lines.append("        vx += np.sin(idx * 0.13 + t) * turbulence * 20.0")
        lines.append("        vy += np.cos(idx * 0.19 + t) * turbulence * 20.0")
        lines.append("")
        lines.append("        self._positions[:, 0] += vx")
        lines.append("        self._positions[:, 1] += vy")
        lines.append("")
        lines.append("        # Wrap-around am Bildrand")
        lines.append("        self._positions[:, 0] = np.mod(self._positions[:, 0], self.width)")
        lines.append("        self._positions[:, 1] = np.mod(self._positions[:, 1], self.height)")
        lines.append("")
        lines.append("        # Groesse reagiert auf Beat")
        lines.append("        self._sizes[:] = base_size * (0.7 + 0.6 * energy + 0.8 * beat)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Hilfsmethoden fuer gemeinsame Template-Teile
    # ------------------------------------------------------------------

    def _file_header_lines(self) -> list[str]:
        return [
            f'"""',
            f'{self.module_name}.py - {self.class_name} GPU-Visualisierung',
            '',
            f'Auto-generiert durch VisualizerWizard (Typ: {self.viz_type}).',
            'Bearbeite PARAMS, Shader und render()-Logik nach Bedarf.',
            f'"""',
            '',
            'import numpy as np',
            'import moderngl',
            '',
            'from .base import BaseGPUVisualizer',
            '',
            '',
        ]

    def _class_docstring(self, text: str) -> str:
        return f'"""\n{text}\n"""'

    def _lygia_math(self) -> str:
        return _indent(BaseGPUVisualizer.LYGIA_MATH.strip(), spaces=12)

    def _lygia_noise(self) -> str:
        return _indent(BaseGPUVisualizer.LYGIA_NOISE.strip(), spaces=12)

    def _lygia_sdf(self) -> str:
        return _indent(BaseGPUVisualizer.LYGIA_SDF.strip(), spaces=12)

    def _lygia_color(self) -> str:
        return _indent(BaseGPUVisualizer.LYGIA_COLOR.strip(), spaces=12)


# ----------------------------------------------------------------------
# Oeffentliche GUI-Hilfsfunktion
# ----------------------------------------------------------------------

def add_create_visualizer_button(parent_layout, state, parent_window=None):
    """Fuegt einem Layout einen Button hinzu, der den Visualizer-Wizard oeffnet.

    Diese Funktion ist als oeffentliche Schnittstelle gedacht und kann spaeter
    vom ParamsPanel (oder einem anderen GUI-Panel) aufgerufen werden, ohne
    dass dieses Modul das Panel direkt modifizieren muss.

    Args:
        parent_layout: QLayout, in den der Button eingefuegt wird.
        state: AppState-Instanz der GUI.
        parent_window: Optionales Eltern-Fenster (QWidget) fuer Modal-Dialoge.

    Returns:
        QPushButton: Der erzeugte Button.
    """
    from PyQt6.QtWidgets import QPushButton, QMessageBox, QInputDialog

    btn = QPushButton("Neuen Visualizer erstellen...")
    btn.setToolTip("Erstellt ein neues GPU-Visualizer-Template mit Wizard.")

    def _on_clicked():
        types = VisualizerWizard.list_types()
        type_text, ok = QInputDialog.getItem(
            parent_window,
            "Visualizer Typ",
            "Waehle einen Template-Typ:",
            types,
            0,
            False,
        )
        if not ok:
            return

        name, ok = QInputDialog.getText(
            parent_window,
            "Visualizer Name",
            "Name des neuen Visualizers (snake_case empfohlen):",
        )
        if not ok or not name.strip():
            return

        try:
            wizard = VisualizerWizard(name.strip(), viz_type=type_text)
            target = wizard.write("src/gpu_visualizers")

            # Auto-Discovery Registry aktualisieren
            from src.gpu_visualizers import refresh_registry
            refresh_registry()

            QMessageBox.information(
                parent_window,
                "Visualizer erstellt",
                f"Neuer Visualizer gespeichert unter:\n{target}\n\n"
                "Er ist nun in der Registry verfuegbar.",
            )
        except Exception as e:
            QMessageBox.critical(
                parent_window,
                "Fehler",
                f"Konnte Visualizer nicht erstellen:\n{e}",
            )

    btn.clicked.connect(_on_clicked)
    parent_layout.addWidget(btn)
    return btn
