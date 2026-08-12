"""
Spirograph - GPU-Visualizer "Spirograph".

Eine einzige durchgehende Kurve (Hypotrochoide), die sich selbst
ueberlagert — Archetyp Kurvenzeichnung. In der Sammlung gab es bisher
Partikel, Felder, Balken, Kreise und Wellenlinien, aber keine Figur, die
aus einer parametrischen Kurve entsteht.

Design:
- Hypotrochoide: ein Kreis rollt im Inneren eines groesseren; ein Punkt
  im Abstand `d` zeichnet die Bahn. Das Verhaeltnis der Radien bestimmt
  die Zackenzahl der Figur.
- Das Verhaeltnis kommt aus dem staerksten Chroma-Ton: jede Tonart hat
  ihre eigene Figur, ein Tonartwechsel formt die Kurve sichtbar um.
- rms = Auslenkung `d` (von fast kreisrund bis stark gezackt),
  spectral_centroid = Feinheit, transient = Dreh-Ruck,
  beat_intensity = Aufleuchten, zero_crossing_rate = Zittern der Linie.
- Drei Echos der Kurve mit Phasenversatz erzeugen eine Nachzieh-Spur.
- Sprach-Modus: langsame Verformung auf voice_band, ruhige Linie,
  Betonungen geben kurze Impulse. Gleiche Optik, andere Empfindlichkeit.

Die Kurve wird im Fragment-Shader ueber Abstandsberechnung zu
Streckenzuegen gezeichnet — duenne Linien mit viel Schwarz dazwischen,
ein Hintergrundbild bleibt sichtbar.
HDR-Ausgabe ohne clamp, Tonemapping macht zentral der Renderer.
"""

import numpy as np
import moderngl

from .base import (
    BaseGPUVisualizer,
    FULLSCREEN_VERTEX_SHADER,
    LYGIA_MATH_GLSL,
    LYGIA_SDF_GLSL,
    SHADER_COMMON_GLSL,
    compose_fragment,
    create_fullscreen_quad,
)

# Stuetzstellen entlang der Kurve. Hoeher = glatter, aber teurer:
# die Distanz wird pro Pixel gegen jedes Segment gerechnet.
CURVE_SAMPLES = 96


class SpirographGPU(BaseGPUVisualizer):
    """Hypotrochoide, deren Form aus der Tonart entsteht."""

    COLOR_PARAMS = {
        'color_mode': 'chroma',
        'base_hue': 0.58,
        'color_saturation': 0.8,
    }

    PARAMS = {
        'scale': (0.34, 0.10, 0.48, 0.01),
        'ratio_base': (5.0, 2.0, 12.0, 1.0),
        'offset_response': (0.55, 0.0, 1.2, 0.05),
        'line_width': (0.0028, 0.0008, 0.012, 0.0002),
        'echo_count': (3, 1, 5, 1),
        'echo_spread': (0.12, 0.0, 0.5, 0.01),
        'spin_speed': (0.25, 0.0, 1.5, 0.05),
        'spin_kick': (1.1, 0.0, 3.0, 0.05),
        'beat_flash': (0.9, 0.0, 2.5, 0.05),
        'jitter': (0.35, 0.0, 1.5, 0.05),
        'glow': (0.6, 0.0, 2.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Figur": ["scale", "ratio_base", "offset_response"],
        "Linie": ["line_width", "glow", "jitter"],
        "Echo": ["echo_count", "echo_spread"],
        "Bewegung": ["spin_speed", "spin_kick"],
        "Reaktion": ["beat_flash"],
    }

    def _setup(self):
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_energy;
            uniform float u_beat;
            uniform float u_impact;
            uniform float u_centroid;
            uniform float u_zcr;
            uniform float u_speech;
            uniform float u_spin;
            uniform float u_ratio;        // Radienverhaeltnis (Zackenzahl)
            uniform float u_offset;       // Auslenkung d
            uniform float u_scale;
            uniform float u_line_width;
            uniform float u_echo_count;
            uniform float u_echo_spread;
            uniform float u_beat_flash;
            uniform float u_jitter;
            uniform float u_glow;
            uniform vec3 u_color_a;
            uniform vec3 u_color_b;
            uniform float u_brightness;
            out vec4 f_color;

            const float TAU = 6.28318530718;

            // Punkt auf der Hypotrochoide bei Parameter t (0..1 = eine Runde
            // des aeusseren Kreises), mit Drehung und Auslenkung.
            vec2 curvePoint(float t, float spin, float offset) {
                float a = t * TAU + spin;
                float k = u_ratio;
                vec2 p = vec2(cos(a), sin(a)) * (1.0 - offset)
                       + vec2(cos(-a * k), sin(-a * k)) * offset;
                return p * u_scale;
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                float aspect = u_resolution.x / max(u_resolution.y, 1.0);
                vec2 p = vec2((uv.x - 0.5) * aspect, uv.y - 0.5);

                // Ausserhalb der Figur gar nicht erst rechnen
                float bound = u_scale * 1.15 + 0.05;
                if (length(p) > bound) {
                    vec3 dark = vec3(ditherTriangular(gl_FragCoord.xy, 1.0));
                    f_color = vec4(dark, 1.0);
                    return;
                }

                // Feines Zittern der Linie mit dem Rauschanteil des Signals
                float wob = (hash12(floor(gl_FragCoord.xy * 0.5)) - 0.5)
                            * u_zcr * u_jitter * 0.006;

                vec3 col = vec3(0.0);
                int echoes = int(u_echo_count);

                for (int e = 0; e < 5; e++) {
                    if (e >= echoes) break;
                    float fe = float(e);
                    // Jedes Echo liegt zeitlich zurueck: leicht andere
                    // Drehung und Auslenkung ergeben die Nachzieh-Spur.
                    float lag = fe * u_echo_spread;
                    float spin = u_spin - lag;
                    float offset = clamp(u_offset - lag * 0.10, 0.05, 0.95);
                    float weight = 1.0 / (1.0 + fe * 1.6);

                    float best = 1e9;
                    vec2 prev = curvePoint(0.0, spin, offset);
                    for (int i = 1; i <= """ + str(CURVE_SAMPLES) + """; i++) {
                        float t = float(i) / float(""" + str(CURVE_SAMPLES) + """);
                        vec2 cur = curvePoint(t, spin, offset);
                        best = min(best, sdSegment(p, prev, cur));
                        prev = cur;
                    }
                    best += wob;

                    // Schaerfe der Linie folgt dem spektralen Schwerpunkt
                    float lw = u_line_width * mix(1.5, 0.75, clamp(u_centroid, 0.0, 1.0));
                    float core = 1.0 - aastep(lw, best);
                    float halo = exp(-best / max(lw * 7.0, 1e-5)) * 0.5 * u_glow;

                    vec3 tint = mix(u_color_a, u_color_b, fe / max(u_echo_count, 1.0));
                    col += tint * (core + halo) * weight
                           * (0.45 + 1.0 * u_energy)
                           * (1.0 + u_beat * u_beat_flash * 0.8);
                }

                // Betonungen zuenden kurz den Mittelpunkt
                float center = exp(-pow(length(p) * 22.0, 2.0));
                col += mix(u_color_a, u_color_b, 0.5) * center
                       * u_impact * 0.35 * (0.5 + 0.5 * u_speech);

                col = max(col, 0.0) * u_brightness;
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time) * 9.0);
                f_color = vec4(col, 1.0);
            }
            """,
            includes=(LYGIA_MATH_GLSL, LYGIA_SDF_GLSL, SHADER_COMMON_GLSL),
        )
        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

        self._spin = 0.0
        self._ratio = 5.0
        self._last_time = None

    def _advance(self, f: dict, time: float, speech: float, ratio_target: float):
        """Integriert Drehung und glaettet den Formwechsel.

        Das Radienverhaeltnis springt sonst bei jedem Tonwechsel hart um —
        die Figur soll sich verformen, nicht umschalten.
        """
        base_speed = float(self.params["spin_speed"])
        if self._last_time is None or time < self._last_time - 1e-6 \
                or (time - self._last_time) > 0.5:
            self._spin = time * base_speed
            self._ratio = ratio_target
            self._last_time = time
            return self._spin, self._ratio

        dt = max(time - self._last_time, 1e-4)
        self._last_time = time

        impact = f.get("transient", f["onset"])
        kick = impact * float(self.params["spin_kick"]) * (1.0 - 0.6 * speech)
        self._spin += (base_speed * (0.35 + 0.9 * f["rms"]) + kick) * dt

        # Zeitkonstante ~0.6 s, im Sprach-Modus traeger
        tau = 0.6 * (1.0 + speech)
        alpha = 1.0 - float(np.exp(-dt / tau))
        self._ratio += (ratio_target - self._ratio) * alpha

        return self._spin, self._ratio

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        speech = 1.0 if mode == "speech" else (0.5 if mode == "hybrid" else 0.0)

        chroma = np.asarray(f["chroma"], dtype=np.float32).flatten()
        if chroma.size < 12:
            chroma = np.pad(chroma, (0, 12 - chroma.size))
        dominant = int(np.argmax(chroma[:12])) if chroma.size else 0

        # Jede Tonart bekommt ihre eigene Zackenzahl. Der Sprach-Modus
        # bleibt bei der Grundform, weil Chroma dort nichts aussagt.
        base = float(self.params["ratio_base"])
        ratio_target = base + (dominant % 7) * (1.0 - speech)

        spin, ratio = self._advance(f, time, speech, ratio_target)

        energy = f["rms"] * (1.0 - speech) + float(f.get("voice_band", f["rms"])) * speech
        offset = 0.20 + float(self.params["offset_response"]) * float(np.clip(energy, 0.0, 1.0))
        offset = float(np.clip(offset, 0.05, 0.9))

        color_a = self._chroma_to_color(f["chroma"])
        hue = self._color_to_hue(color_a)
        sat = float(self.params.get("color_saturation", 0.8))
        color_b = self._hsv_to_rgb((hue + 0.28) % 1.0, min(1.0, sat), 1.15)

        beat = f.get("beat_intensity", f["onset"]) * (1.0 - 0.65 * speech)
        impact = f.get("transient", f["onset"]) * (1.0 - 0.4 * speech)

        p = self.prog
        p["u_resolution"].value = (self.width, self.height)
        p["u_time"].value = float(time)
        p["u_energy"].value = float(energy)
        p["u_beat"].value = float(beat)
        p["u_impact"].value = float(impact)
        p["u_centroid"].value = float(f["spectral_centroid"])
        p["u_zcr"].value = float(f.get("zero_crossing_rate", 0.0))
        p["u_speech"].value = float(speech)
        p["u_spin"].value = float(spin)
        p["u_ratio"].value = float(ratio)
        p["u_offset"].value = offset
        p["u_scale"].value = float(self.params["scale"])
        p["u_line_width"].value = float(self.params["line_width"])
        p["u_echo_count"].value = float(self.params["echo_count"])
        p["u_echo_spread"].value = float(self.params["echo_spread"])
        p["u_beat_flash"].value = float(self.params["beat_flash"])
        p["u_jitter"].value = float(self.params["jitter"])
        p["u_glow"].value = float(self.params["glow"])
        p["u_color_a"].value = tuple(color_a)
        p["u_color_b"].value = tuple(color_b)
        p["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
