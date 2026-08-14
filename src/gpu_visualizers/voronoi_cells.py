"""
Voronoi Cells - GPU-Visualizer "Zellfeld".

Der Bildraum wird in Zellen zerlegt: jeder Punkt gehoert zum naechsten
Keim. Sichtbar sind nur die Grenzen — ein wanderndes Netz aus Kanten.
Archetyp Zerlegung/Mosaik, in der Sammlung bisher nicht vorhanden.

Design:
- Bis zu 24 Keime driften langsam; die Zellen gehoeren reihum je
  einem der zwoelf Chroma-Toene.
  Klingt der Ton, faerbt und hebt sich seine Zelle. Ein Akkord zeichnet
  damit eine Gruppe zusammenhaengender Zellen.
- Gezeichnet werden nur die Kanten (Abstand zwischen naechstem und
  zweitnaechstem Keim). Die Flaechen bleiben nahezu schwarz, ein
  Hintergrundbild bleibt sichtbar.
- rms = Weite des Feldes (Zellen wachsen/schrumpfen),
  transient = Keime springen kurz auseinander,
  beat_intensity = Welle, die von der Bildmitte durch die Kanten laeuft,
  spectral_centroid = Schaerfe der Kanten,
  zero_crossing_rate = Flackern einzelner Kanten.
- Sprach-Modus: langsames Driften, weiche Kanten, Helligkeit folgt
  voice_band. Gleiche Optik, andere Empfindlichkeit.

HDR-Ausgabe ohne clamp, Tonemapping macht zentral der Renderer.
"""

import numpy as np
import moderngl

from .base import (
    BaseGPUVisualizer,
    FULLSCREEN_VERTEX_SHADER,
    LYGIA_MATH_GLSL,
    SHADER_COMMON_GLSL,
    compose_fragment,
    create_fullscreen_quad,
)

# Feste Obergrenze fuer die Keim-Schleife im Shader
MAX_SITES = 24


class VoronoiCellsGPU(BaseGPUVisualizer):
    """Wanderndes Zellnetz, dessen Zellen den Chroma-Toenen gehoeren."""

    COLOR_PARAMS = {
        'color_mode': 'chroma',
        'base_hue': 0.48,
        'color_saturation': 0.75,
    }

    PARAMS = {
        'site_count': (18, 4, MAX_SITES, 1),
        'spread': (0.80, 0.25, 1.2, 0.02),
        'drift_speed': (0.16, 0.0, 0.8, 0.02),
        'edge_width': (0.011, 0.003, 0.10, 0.001),
        'edge_glow': (0.45, 0.0, 2.0, 0.05),
        'cell_tint': (0.05, 0.0, 0.6, 0.01),
        'zoom_response': (0.25, 0.0, 1.0, 0.05),
        'jitter': (0.7, 0.0, 2.0, 0.05),
        'wave_strength': (1.0, 0.0, 2.5, 0.05),
        'flicker': (0.35, 0.0, 1.5, 0.05),
    }

    PARAMS_GROUPS = {
        "Feld": ["site_count", "spread", "zoom_response"],
        "Kanten": ["edge_width", "edge_glow", "flicker"],
        "Zellen": ["cell_tint"],
        "Bewegung": ["drift_speed", "jitter"],
        "Reaktion": ["wave_strength"],
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
            uniform float u_drift_phase;   // aufsummierte Driftbewegung
            uniform float u_wave_pos;      // Radius der Kantenwelle
            uniform float u_chroma[12];
            uniform vec3 u_color_a;
            uniform vec3 u_color_b;
            uniform float u_site_count;
            uniform float u_spread;
            uniform float u_edge_width;
            uniform float u_edge_glow;
            uniform float u_cell_tint;
            uniform float u_zoom_response;
            uniform float u_jitter;
            uniform float u_wave_strength;
            uniform float u_flicker;
            uniform float u_brightness;
            out vec4 f_color;

            // Lage eines Keims: fester Startpunkt aus dem Hash plus langsame
            // Drift plus kurzer Ausschlag bei Betonungen.
            vec2 sitePos(float fi) {
                vec2 seed = vec2(fi * 13.71, fi * 7.31);
                vec2 base = vec2(hash12(seed), hash12(seed + 11.0)) * 2.0 - 1.0;
                float sp = 0.6 + 0.9 * hash12(seed + 3.0);
                vec2 drift = vec2(sin(u_drift_phase * sp + fi * 2.1),
                                  cos(u_drift_phase * sp * 0.83 + fi * 1.7)) * 0.16;
                // Transienten schleudern die Keime kurz nach aussen
                vec2 out_dir = normalize(base + 1e-4);
                return (base + drift) * u_spread
                       + out_dir * u_impact * u_jitter * 0.09;
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                float aspect = u_resolution.x / max(u_resolution.y, 1.0);
                vec2 p = vec2((uv.x - 0.5) * aspect, uv.y - 0.5) * 2.0;

                // Lautere Stellen dehnen das Feld — Zellen werden groesser
                p /= (1.0 + u_zoom_response * u_energy);

                int count = int(u_site_count);
                float f1 = 1e9, f2 = 1e9;
                float nearest = 0.0;

                for (int i = 0; i < """ + str(MAX_SITES) + """; i++) {
                    if (i >= count) break;
                    float fi = float(i);
                    float d = length(p - sitePos(fi));
                    if (d < f1) {
                        f2 = f1;
                        f1 = d;
                        nearest = fi;
                    } else if (d < f2) {
                        f2 = d;
                    }
                }

                // Kante: dort, wo naechster und zweitnaechster Keim
                // gleich weit weg sind.
                float border = f2 - f1;
                float width = u_edge_width * mix(1.6, 0.7, clamp(u_centroid, 0.0, 1.0));
                width *= 1.0 + 0.5 * u_speech;   // Sprache: weichere Kanten
                float edge = 1.0 - smoothstep(0.0, width, border);
                float halo = exp(-border / max(width * 1.6, 1e-5)) * 0.32 * u_edge_glow;

                // Flackern einzelner Kanten mit dem Rauschanteil
                float flick = 1.0 - u_flicker * u_zcr * 0.5
                    * step(0.55, hash12(vec2(nearest, floor(u_time * 9.0))));

                // Welle, die von der Mitte durch die Kanten laeuft
                float rad = length(p);
                float wave = exp(-pow((rad - u_wave_pos) * 4.5, 2.0))
                             * u_beat * u_wave_strength;

                int ci = int(mod(nearest, 12.0));
                float chroma = u_chroma[ci];
                float lit = 0.30 + 1.2 * smoothstep(0.12, 0.8, chroma);
                vec3 cell = mix(u_color_a, u_color_b, chroma);

                vec3 col = vec3(0.0);
                col += cell * (edge + halo) * flick * lit
                       * (0.40 + 1.0 * u_energy) * (1.0 + wave * 1.6);

                // Sehr zarte Flaechentoenung der aktiven Zellen: gibt dem Netz
                // Tiefe, bleibt aber weit unter der Schwelle, ab der ein
                // Hintergrundbild uebermalt wuerde.
                float inner = smoothstep(0.0, width * 3.0, border);
                col += cell * inner * u_cell_tint
                       * smoothstep(0.35, 0.95, chroma) * (0.3 + 0.7 * u_energy);

                // Sprach-Modus faehrt insgesamt leiser — Netz bleibt, draengt
                // sich aber nicht auf.
                col *= 1.0 - 0.30 * u_speech;

                col = max(col, 0.0) * u_brightness;
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time) * 13.0);
                f_color = vec4(col, 1.0);
            }
            """,
            includes=(LYGIA_MATH_GLSL, SHADER_COMMON_GLSL),
        )
        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

        self._drift = 0.0
        self._wave_pos = 3.0
        self._last_time = None

    def _advance(self, f: dict, time: float, speech: float) -> tuple:
        """Integriert Drift und Kantenwelle ueber die Zeit."""
        base_speed = float(self.params["drift_speed"])
        if self._last_time is None or time < self._last_time - 1e-6 \
                or (time - self._last_time) > 0.5:
            self._drift = time * base_speed
            self._wave_pos = 3.0
            self._last_time = time
            return self._drift, self._wave_pos

        dt = max(time - self._last_time, 1e-4)
        self._last_time = time

        # Sprache driftet ruhiger als Musik
        speed = base_speed * (0.35 + 0.9 * f["rms"]) * (1.0 - 0.5 * speech)
        self._drift += speed * dt

        beat = f.get("beat_intensity", f["onset"])
        if beat > 0.55 and self._wave_pos > 2.2:
            self._wave_pos = 0.0
        if self._wave_pos <= 2.2:
            self._wave_pos += dt * (1.0 + 1.8 * f["rms"])

        return self._drift, self._wave_pos

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        speech = 1.0 if mode == "speech" else (0.5 if mode == "hybrid" else 0.0)

        drift, wave_pos = self._advance(f, time, speech)

        chroma = np.asarray(f["chroma"], dtype=np.float32).flatten()
        if chroma.size < 12:
            chroma = np.pad(chroma, (0, 12 - chroma.size))
        chroma = chroma[:12]
        peak = float(chroma.max()) if chroma.size else 0.0
        if peak > 1e-6:
            chroma = chroma / peak
        if speech > 0.0:
            # Sprache traegt keine Tonart — die Zellen leuchten dann
            # gemeinsam mit der Stimme, nur gedaempft.
            voice = float(f.get("voice_band", f["rms"]))
            chroma = chroma * (1.0 - speech) + (0.20 + 0.50 * voice) * speech

        color_a = self._chroma_to_color(f["chroma"])
        hue = self._color_to_hue(color_a)
        sat = float(self.params.get("color_saturation", 0.75))
        color_b = self._hsv_to_rgb((hue + 0.47) % 1.0, min(1.0, sat), 1.1)

        beat = f.get("beat_intensity", f["onset"]) * (1.0 - 0.6 * speech)
        impact = f.get("transient", f["onset"]) * (1.0 - 0.5 * speech)
        energy = f["rms"] * (1.0 - speech) + float(f.get("voice_band", f["rms"])) * speech

        p = self.prog
        p["u_resolution"].value = (self.width, self.height)
        p["u_time"].value = float(time)
        p["u_energy"].value = float(energy)
        p["u_beat"].value = float(beat)
        p["u_impact"].value = float(impact)
        p["u_centroid"].value = float(f["spectral_centroid"])
        p["u_zcr"].value = float(f.get("zero_crossing_rate", 0.0))
        p["u_speech"].value = float(speech)
        p["u_drift_phase"].value = float(drift)
        p["u_wave_pos"].value = float(np.clip(wave_pos, 0.0, 2.4))
        p["u_chroma"].write(chroma.astype(np.float32).tobytes())
        p["u_color_a"].value = tuple(color_a)
        p["u_color_b"].value = tuple(color_b)
        p["u_site_count"].value = float(self.params["site_count"])
        p["u_spread"].value = float(self.params["spread"])
        p["u_edge_width"].value = float(self.params["edge_width"])
        p["u_edge_glow"].value = float(self.params["edge_glow"])
        p["u_cell_tint"].value = float(self.params["cell_tint"])
        p["u_zoom_response"].value = float(self.params["zoom_response"])
        p["u_jitter"].value = float(self.params["jitter"])
        p["u_wave_strength"].value = float(self.params["wave_strength"])
        p["u_flicker"].value = float(self.params["flicker"])
        p["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
