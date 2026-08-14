"""
Retro Sun - GPU-Visualizer "Retro-Sonne".

Ein Horizont-Bild statt einer zentrierten Form: tief stehende Sonne mit
waagerechten Schlitzen, darunter ein perspektivisches Gitter, das auf den
Fluchtpunkt zulaeuft. Archetyp Landschaft/Horizont — in der Sammlung
bisher nicht vorhanden.

Design:
- Sonnenscheibe mit Farbverlauf, von waagerechten Schlitzen zerschnitten.
  Die Schlitze wandern langsam nach oben; ihre Breite folgt der Energie,
  bei Beats reissen sie kurz auf.
- Gitter unter dem Horizont: Querlinien laufen auf den Betrachter zu
  (Tempo/Beat), Laengslinien fluchten auf den Fluchtpunkt. Nur Linien,
  keine Flaeche — das Hintergrundbild bleibt sichtbar.
- Horizontlinie als schmaler Lichtstreifen, Helligkeit aus RMS.
- Chroma faerbt Sonne (oben/unten) und Gitter.
- transient = kurzer heller Scan, der ueber die Sonne laeuft.
- spectral_centroid = Feinheit/Anzahl der Schlitze.
- Sprach-Modus: Sonne atmet mit voice_band, das Gitter steht fast still,
  Schlitze bleiben ruhig — dieselbe Optik, nur andere Empfindlichkeit.

Alles additiv auf Schwarz, keine Vollflaechen — hintergrundbild-freundlich.
HDR-Ausgabe ohne clamp, Tonemapping macht zentral der Renderer.
"""

import numpy as np
import moderngl

from .base import (
    BaseGPUVisualizer,
    FULLSCREEN_VERTEX_SHADER,
    LYGIA_MATH_GLSL,
    LYGIA_NOISE_GLSL,
    SHADER_COMMON_GLSL,
    compose_fragment,
    create_fullscreen_quad,
)


class RetroSunGPU(BaseGPUVisualizer):
    """Retro-Sonne am Horizont mit perspektivischem Gitter."""

    COLOR_PARAMS = {
        'color_mode': 'chroma',
        'base_hue': 0.95,
        'color_saturation': 0.85,
    }

    PARAMS = {
        'horizon_y': (0.42, 0.20, 0.70, 0.01),
        'sun_radius': (0.26, 0.10, 0.45, 0.01),
        'slit_count': (22, 3, 40, 1),
        'slit_width': (0.34, 0.05, 0.9, 0.02),
        'slit_drift': (0.35, 0.0, 1.5, 0.05),
        'grid_speed': (0.6, 0.0, 3.0, 0.05),
        'grid_density': (9, 3, 24, 1),
        'grid_brightness': (0.95, 0.0, 2.5, 0.05),
        'beat_bloom': (0.8, 0.0, 2.0, 0.05),
        'scan_strength': (0.7, 0.0, 2.0, 0.05),
        'haze': (0.15, 0.0, 1.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Sonne": ["sun_radius", "slit_count", "slit_width", "slit_drift"],
        "Gitter": ["grid_speed", "grid_density", "grid_brightness"],
        "Reaktion": ["beat_bloom", "scan_strength"],
        "Bild": ["horizon_y", "haze"],
    }

    def _setup(self):
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_energy;          // RMS
            uniform float u_beat;            // Beat-Decay-Envelope
            uniform float u_impact;          // Transiente
            uniform float u_centroid;        // Spectral Centroid
            uniform float u_voice;           // Voice-Band
            uniform float u_speech;          // 1.0 = Sprache
            uniform float u_scan_pos;        // Position des Scan-Streifens 0..1
            uniform float u_grid_phase;      // aufsummierte Gitter-Bewegung
            uniform vec3 u_sun_top;
            uniform vec3 u_sun_bottom;
            uniform vec3 u_grid_color;
            uniform float u_horizon_y;
            uniform float u_sun_radius;
            uniform float u_slit_count;
            uniform float u_slit_width;
            uniform float u_slit_drift;
            uniform float u_grid_density;
            uniform float u_grid_brightness;
            uniform float u_beat_bloom;
            uniform float u_scan_strength;
            uniform float u_haze;
            uniform float u_brightness;
            out vec4 f_color;

            // Eine Gitterlinie pro ganzzahligem Schritt von x.
            // Wichtig: wird die Linienfolge dichter als das Pixelraster,
            // blendet sie aus. Ohne diese Bremse laeuft die Perspektive nahe
            // am Horizont in eine geschlossene Flaeche statt in Linien.
            // `step` ist die Schrittweite von x pro Pixel, aussen berechnet.
            // fwidth() waere hier falsch: die Funktion laeuft nur unterhalb
            // des Horizonts, also in divergentem Kontrollfluss, wo
            // Ableitungen laut GLSL undefiniert sind.
            float gridLine(float x, float glow, float step) {
                float w = step * 1.5 + 1e-6;
                float f = fract(x);
                float dist = min(f, 1.0 - f);
                float line = 1.0 - smoothstep(0.0, w, dist);
                // Weicher Saum: aus Haarlinien wird ein Leuchtstreifen
                line += exp(-dist / max(w * 3.5, 1e-5)) * glow;
                return line * smoothstep(0.45, 0.10, w);
            }

            // Sonnenscheibe mit Schlitzen. p ist bereits horizontzentriert.
            vec3 sunLayer(vec2 p, float aspect, float radius, out float mask) {
                vec2 q = vec2(p.x * aspect, p.y);
                float d = length(q) - radius;
                float disc = aafill(d);

                // Waagerechte Schlitze: nach unten hin dichter und breiter,
                // damit die Scheibe unten "aufgeloest" wirkt (Retro-Look).
                float depth = clamp((radius - p.y) / (2.0 * radius), 0.0, 1.0);
                float phase = p.y * u_slit_count - u_time * u_slit_drift;
                float saw = abs(fract(phase) - 0.5) * 2.0;
                float gap = u_slit_width * mix(0.12, 0.95, pow(depth, 1.6));
                // Bei Beats reissen die Schlitze kurz auf
                gap *= 1.0 + u_beat * 0.5 * u_beat_bloom;
                float slit = aastep(gap, saw);

                mask = disc * slit;
                float grad = clamp((p.y + radius) / (2.0 * radius), 0.0, 1.0);
                vec3 col = mix(u_sun_bottom, u_sun_top, grad);

                // Randglut: heller Saum entlang der Scheibenkante
                float rim = exp(-abs(d) * 60.0) * disc;
                col += u_sun_top * rim * (0.35 + 0.65 * u_energy);
                return col;
            }

            void main() {
                // Der Renderer liest den Framebuffer vertikal gespiegelt aus —
                // hier spiegeln, damit unten auch unten ist.
                vec2 uv = gl_FragCoord.xy / u_resolution;
                uv.y = 1.0 - uv.y;
                float aspect = u_resolution.x / max(u_resolution.y, 1.0);

                vec3 col = vec3(0.0);
                float hy = u_horizon_y;

                // --- Sonne (nur oberhalb des Horizonts sichtbar) ---
                float radius = u_sun_radius
                    * (1.0 + 0.18 * mix(u_energy, u_voice, u_speech))
                    * (1.0 + 0.10 * u_beat * u_beat_bloom);
                vec2 p = vec2(uv.x - 0.5, uv.y - hy);
                float sun_mask = 0.0;
                vec3 sun_col = sunLayer(p, aspect, radius, sun_mask);
                float above = aastep(hy - 0.002, uv.y);
                col += sun_col * sun_mask * above * (0.55 + 0.75 * u_energy);

                // Scan-Streifen: kurzer heller Balken, von Transienten geworfen
                float scan_y = hy + (u_scan_pos * 2.0 - 0.2) * radius;
                float scan = exp(-pow((uv.y - scan_y) * 90.0, 2.0));
                col += u_sun_top * scan * sun_mask * above
                       * u_impact * u_scan_strength * 1.6;

                // --- Horizontlinie ---
                float line = exp(-pow((uv.y - hy) * 420.0, 2.0));
                col += mix(u_sun_bottom, u_sun_top, 0.35) * line
                       * (0.30 + 0.9 * u_energy);

                // --- Gitter unterhalb des Horizonts ---
                if (uv.y < hy) {
                    // Perspektive: je weiter unten, desto naeher am Betrachter
                    float depth = (hy - uv.y) / max(hy, 1e-3);   // 0 = Horizont
                    float z = 1.0 / max(depth, 1e-3);            // Tiefe
                    float fade = smoothstep(0.0, 0.22, depth)
                                 * smoothstep(1.10, 0.40, depth);

                    // Laengslinien fluchten auf den Fluchtpunkt bei x = 0.5
                    // Schrittweite analytisch: d/dx von (x-0.5)*z*s ist z*s,
                    // umgerechnet auf einen Pixel in x-Richtung.
                    float px_x = 1.0 / max(u_resolution.x, 1.0);
                    float sx = abs(z * 0.45 * u_grid_density) * px_x;
                    float lines_x = gridLine((uv.x - 0.5) * z * 0.45 * u_grid_density,
                                             0.30, sx);

                    // Querlinien laufen auf den Betrachter zu. Der Faktor
                    // bestimmt, wie viele Reihen zwischen Horizont und
                    // Bildunterkante liegen — zu klein und man sieht nur eine.
                    // d/dy von z*2.2 mit z = 1/depth, depth = (hy-uv.y)/hy
                    float py = 1.0 / max(u_resolution.y, 1.0);
                    float sz = 2.2 * z * z / max(hy, 1e-3) * py;
                    float lines_z = gridLine(z * 2.2 - u_grid_phase * 2.0, 0.45, sz);

                    // Naeher = heller. Ohne diese Staffelung wirkt der Boden
                    // flach, weil alle Linien gleich stark leuchten.
                    float near = mix(0.45, 1.0, smoothstep(0.0, 0.8, depth));
                    float grid = max(lines_x, lines_z) * fade * near;
                    // Querlinien pulsen auf dem Beat mit
                    float pulse = 1.0 + u_beat * u_beat_bloom * 0.9 * lines_z;
                    col += u_grid_color * grid * u_grid_brightness
                           * (0.35 + 0.9 * u_energy) * pulse;

                    // Spiegelung der Sonne auf dem Boden: schmale Saeule unter
                    // der Scheibe, in Streifen zerlegt wie das Original.
                    float refl_x = exp(-pow((uv.x - 0.5) * (3.4 / max(radius, 0.05)), 2.0));
                    float refl_stripes = step(0.35,
                        abs(fract(depth * 26.0 - u_time * 0.6) - 0.5) * 2.0);
                    col += u_sun_bottom * refl_x * refl_stripes
                           * exp(-depth * 3.2) * 0.55 * (0.3 + u_energy);

                    // Sehr dezenter Bodennebel direkt unter dem Horizont —
                    // bewusst schwach, damit ein Hintergrundbild durchkommt.
                    col += u_grid_color * u_haze * 0.06
                           * exp(-depth * 9.0) * (0.3 + u_energy);
                }

                // Feiner Rauschanteil in den hellen Bereichen (Roehren-Look)
                float grain = (hash12(gl_FragCoord.xy + u_time * 60.0) - 0.5);
                col += col * grain * 0.05 * (0.3 + u_centroid);

                col = max(col, 0.0) * u_brightness;
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time) * 11.0);
                f_color = vec4(col, 1.0);
            }
            """,
            includes=(LYGIA_MATH_GLSL, LYGIA_NOISE_GLSL, SHADER_COMMON_GLSL),
        )
        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

        # Zustand fuer die zeitlich integrierte Gitterbewegung und den Scan
        self._grid_phase = 0.0
        self._scan_pos = 1.5
        self._last_time = None

    def _advance(self, f: dict, time: float, speech: float) -> tuple:
        """Fuehrt Gitter-Phase und Scan-Position deterministisch fort.

        Beides muss zeitlich integriert werden (Geschwindigkeit statt
        Position), sonst springt das Bild beim Scrubben in der Vorschau.
        """
        if self._last_time is None or time < self._last_time - 1e-6 \
                or (time - self._last_time) > 0.5:
            # Sprung in der Zeitachse: Zustand aus der absoluten Zeit ableiten
            speed = float(self.params["grid_speed"])
            self._grid_phase = (time * speed * 0.6) % 1.0
            self._scan_pos = 1.5
            self._last_time = time
            return self._grid_phase, self._scan_pos

        dt = max(time - self._last_time, 1e-4)
        self._last_time = time

        # Gitter: Grundtempo plus Beat-Schub, im Sprach-Modus stark gedaempft
        beat = f.get("beat_intensity", f["onset"])
        speed = float(self.params["grid_speed"]) * (0.25 + 0.75 * (1.0 - speech))
        speed *= 0.6 + 0.9 * f["rms"] + 0.7 * beat
        self._grid_phase = (self._grid_phase + speed * dt) % 1.0

        # Scan: Transiente startet einen Streifen, der ueber die Sonne laeuft
        impact = f.get("transient", f["onset"])
        if impact > 0.45 and self._scan_pos > 1.2:
            self._scan_pos = -0.15
        if self._scan_pos <= 1.2:
            self._scan_pos += dt * 1.6

        return self._grid_phase, self._scan_pos

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        speech = 1.0 if mode == "speech" else (0.5 if mode == "hybrid" else 0.0)

        grid_phase, scan_pos = self._advance(f, time, speech)

        # Farben: Chroma gibt den Grundton, Sonne oben warm / unten kalt-pink
        chroma_color = self._chroma_to_color(f["chroma"])
        hue = self._color_to_hue(chroma_color)
        sat = float(self.params.get("color_saturation", 0.85))
        # Der Chroma-Ton faerbt, aber die Palette bleibt erkennbar ein
        # Sonnenuntergang: oben warmes Gelb-Orange, unten Magenta.
        # Ohne diese Bindung wird aus der Retro-Sonne je nach Tonart ein
        # blauer Klecks.
        def _toward(target, weight=0.85):
            delta = ((target - hue + 0.5) % 1.0) - 0.5
            return (hue + delta * weight) % 1.0

        sun_top = self._hsv_to_rgb(_toward(0.08), min(1.0, sat * 0.95), 1.30)
        sun_bottom = self._hsv_to_rgb(_toward(0.92), min(1.0, sat), 1.05)
        grid_color = self._hsv_to_rgb(_toward(0.60, 0.55), min(1.0, sat * 0.9), 1.15)

        # Modus = Empfindlichkeit: Sprache faehrt auf Stimme statt auf Beats
        beat = f.get("beat_intensity", f["onset"])
        impact = f.get("transient", f["onset"])
        if speech > 0.0:
            beat *= 1.0 - 0.7 * speech
            impact *= 1.0 - 0.5 * speech

        p = self.prog
        p["u_resolution"].value = (self.width, self.height)
        p["u_time"].value = float(time)
        p["u_energy"].value = float(f["rms"])
        p["u_beat"].value = float(beat)
        p["u_impact"].value = float(impact)
        p["u_centroid"].value = float(f["spectral_centroid"])
        p["u_voice"].value = float(f.get("voice_band", f["rms"]))
        p["u_speech"].value = float(speech)
        p["u_scan_pos"].value = float(np.clip(scan_pos, -0.2, 1.4))
        p["u_grid_phase"].value = float(grid_phase)
        p["u_sun_top"].value = tuple(sun_top)
        p["u_sun_bottom"].value = tuple(sun_bottom)
        p["u_grid_color"].value = tuple(grid_color)
        p["u_horizon_y"].value = float(self.params["horizon_y"])
        p["u_sun_radius"].value = float(self.params["sun_radius"])
        p["u_slit_count"].value = float(self.params["slit_count"])
        p["u_slit_width"].value = float(self.params["slit_width"])
        p["u_slit_drift"].value = float(self.params["slit_drift"])
        p["u_grid_density"].value = float(self.params["grid_density"])
        p["u_grid_brightness"].value = float(self.params["grid_brightness"])
        p["u_beat_bloom"].value = float(self.params["beat_bloom"])
        p["u_scan_strength"].value = float(self.params["scan_strength"])
        p["u_haze"].value = float(self.params["haze"])
        p["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
