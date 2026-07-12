"""
GPU-beschleunigte typografische Visualisierung fuer Podcasts.

Signature-Ueberarbeitung: kinetisches Type-Grid aus SDF-Bloecken (glyphen-artig)
mit beat-quantisierter Bewegung, u_detail-gesteuerter Blockdichte und einer
ruhigen Mittellinie plus Fortschrittsbalken. Sprach-optimiert (speech-Mapping),
Text-Overlays uebernimmt das Post-Processing.
"""

import moderngl
from .base import (
    BaseGPUVisualizer,
    FULLSCREEN_VERTEX_SHADER,
    LYGIA_MATH_GLSL,
    LYGIA_NOISE_GLSL,
    LYGIA_SDF_GLSL,
    SHADER_COMMON_GLSL,
    compose_fragment,
    create_fullscreen_quad,
)


class TypographicGPU(BaseGPUVisualizer):
    """Kinetisches Type-Grid fuer Podcasts (SDF-Bloecke, beat-quantisiert)."""

    PARAMS = {
        'grid_columns': (32, 8, 64, 1),
        'block_height': (0.10, 0.03, 0.30, 0.01),
        'density': (0.6, 0.1, 1.0, 0.05),
        'animation_speed': (0.6, 0.0, 2.0, 0.05),
        'beat_jump': (0.5, 0.0, 1.5, 0.05),
        'baseline_glow': (0.6, 0.0, 1.5, 0.05),
        'progress_enabled': (1.0, 0.0, 1.0, 1.0),
        'bg_brightness': (0.12, 0.0, 0.5, 0.01),
    }

    PARAMS_GROUPS = {
        "Raster": ["grid_columns", "block_height", "density"],
        "Bewegung": ["animation_speed", "beat_jump"],
        "Erscheinungsbild": ["baseline_glow", "progress_enabled", "bg_brightness"],
    }

    def _setup(self):
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_energy;
            uniform float u_beat;
            uniform float u_flow;
            uniform float u_detail;
            uniform float u_progress;
            uniform vec3 u_color;
            uniform vec3 u_secondary_color;
            uniform vec3 u_background_color;
            uniform float u_grid_columns;
            uniform float u_block_height;
            uniform float u_density;
            uniform float u_animation_speed;
            uniform float u_beat_jump;
            uniform float u_baseline_glow;
            uniform float u_progress_enabled;
            uniform float u_bg_brightness;
            uniform float u_brightness;
            out vec4 f_color;

            // Gefuellter, weichkantiger SDF-Block
            float block(vec2 p, vec2 half_size) {
                vec2 d = abs(p) - half_size;
                float sd = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
                return 1.0 - aastep(0.0, sd);
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                vec2 p = uv * 2.0 - 1.0;
                p.x *= u_resolution.x / u_resolution.y;

                vec3 col = u_background_color * u_bg_brightness;

                float cols = u_grid_columns;
                float colWidth = 2.0 * (u_resolution.x / u_resolution.y) / cols;

                // Spaltenindex (glyphen-artige Bloecke pro Spalte)
                float span = u_resolution.x / u_resolution.y;
                float xNorm = (p.x + span) / (2.0 * span);   // 0..1
                float ci = floor(xNorm * cols);
                float cx = (ci + 0.5) / cols * 2.0 * span - span;

                // Pseudozufaellige Charakteristik je Spalte
                float rnd = hash12(vec2(ci, 1.0));
                // Beat-quantisierter Sprung: Bloecke rasten in Stufen ein
                float step_t = floor(u_time * (1.0 + u_animation_speed * 3.0));
                float jump = hash12(vec2(ci, step_t)) * u_beat_jump * u_beat;

                // Aktivierte Spalte? -> Dichte + Sprachband. Hohe Detailwerte
                // (helle Zischlaute) erhoehen die effektive Blockdichte.
                float density = clamp(u_density + u_detail * 0.25, 0.0, 1.0);
                float colActive = step(1.0 - density, rnd);
                float amp = (0.15 + u_flow * 0.7 + jump) * colActive;

                // Symmetrische Bloecke ober-/unterhalb der Mittellinie
                float hh = u_block_height * (0.4 + amp * 2.0);
                float bx = abs(p.x - cx);
                float colMask = 1.0 - aastep(colWidth * 0.42, bx);

                float upper = block(vec2(0.0, p.y - hh), vec2(colWidth * 0.42, hh)) * colMask;
                float lower = block(vec2(0.0, p.y + hh), vec2(colWidth * 0.42, hh)) * colMask;

                vec3 blockCol = mix(u_color, u_secondary_color, rnd);
                col += blockCol * (upper + lower) * (0.7 + u_energy * 0.8);

                // Ruhige, leuchtende Mittellinie
                float baseline = exp(-p.y * p.y * 900.0);
                col += mix(u_color, vec3(1.0), 0.3) * baseline * u_baseline_glow;

                // Fortschrittsbalken unten
                if (u_progress_enabled > 0.5) {
                    float by = uv.y;                       // 0 unten .. 1 oben
                    float barBand = smoothstep(0.045, 0.04, by) * smoothstep(0.02, 0.025, by);
                    float filled = step(uv.x, u_progress);
                    col += u_color * barBand * (0.2 + filled * 0.8);
                }

                col = max(col, 0.0) * u_brightness;
                f_color = vec4(col, 1.0);
            }
            """,
            includes=(LYGIA_MATH_GLSL, LYGIA_NOISE_GLSL, LYGIA_SDF_GLSL, SHADER_COMMON_GLSL),
        )
        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        uniforms = self._map_features_to_uniforms(f, mode="speech")

        color = self._chroma_to_color(uniforms["u_chroma"])
        h, s, v = self._rgb_to_hsv(*color)
        secondary = self._hsv_to_rgb((h + 0.5) % 1.0, s, v)

        bg = self.params.get("background_color")
        if isinstance(bg, str) and bg.startswith("#"):
            try:
                bg_rgb = self._hex_to_rgb(bg)
            except Exception:
                bg_rgb = (0.03, 0.03, 0.05)
        else:
            bg_rgb = (0.03, 0.03, 0.05)

        duration = features.get("duration", 1.0) or 1.0
        progress = f.get("progress", min(1.0, time / duration))

        self.prog["u_resolution"].value = (self.width, self.height)
        self.prog["u_time"].value = time
        self.prog["u_energy"].value = uniforms["u_energy"]
        self.prog["u_beat"].value = uniforms["u_beat"]
        self.prog["u_flow"].value = uniforms["u_flow"]
        self.prog["u_detail"].value = uniforms["u_detail"]
        self.prog["u_progress"].value = float(progress)
        self.prog["u_color"].value = color
        self.prog["u_secondary_color"].value = secondary
        self.prog["u_background_color"].value = bg_rgb
        self.prog["u_grid_columns"].value = float(self.params["grid_columns"])
        self.prog["u_block_height"].value = float(self.params["block_height"])
        self.prog["u_density"].value = float(self.params["density"])
        self.prog["u_animation_speed"].value = float(self.params["animation_speed"])
        self.prog["u_beat_jump"].value = float(self.params["beat_jump"])
        self.prog["u_baseline_glow"].value = float(self.params["baseline_glow"])
        self.prog["u_progress_enabled"].value = float(self.params["progress_enabled"])
        self.prog["u_bg_brightness"].value = float(self.params["bg_brightness"])
        self.prog["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
