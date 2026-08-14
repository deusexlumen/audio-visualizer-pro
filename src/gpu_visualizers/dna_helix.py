"""
DNA Helix - GPU-Visualizer "Doppelhelix".

Zwei umeinander laufende Straenge mit Querstreben — Archetyp Struktur/
Gitter, in der Sammlung bisher nicht vorhanden (keine Kreise, keine
Wellenlinie, keine Partikel).

Design:
- Zwei Straenge winden sich um eine senkrechte Achse. Die Tiefe (vorne/
  hinten) steuert Dicke und Helligkeit, dadurch entsteht Raum ohne 3D.
- Querstreben verbinden die Straenge. Jede Strebe gehoert zu einem
  Chroma-Ton: klingt der Ton, leuchtet die Strebe auf. Die Helix wird
  damit zur laufenden Tonart-Anzeige.
- rms = Weite der Windung, transient = kurzer Dreh-Ruck (Torsion),
  beat_intensity = Lichtpuls, der die Helix hinaufwandert,
  spectral_centroid = Feinheit/Anzahl der Streben,
  zero_crossing_rate = Flimmern der Strebenenden.
- Sprach-Modus: die Helix atmet mit voice_band, dreht langsam, Streben
  leuchten sanft auf Betonungen. Gleiche Optik, andere Empfindlichkeit.

Nur Linien und Punkte auf Schwarz — ein Hintergrundbild bleibt sichtbar.
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

# Feste Obergrenze fuer die Streben-Schleife im Shader
MAX_RUNGS = 28


class DnaHelixGPU(BaseGPUVisualizer):
    """Doppelhelix mit chroma-gesteuerten Querstreben."""

    COLOR_PARAMS = {
        'color_mode': 'chroma',
        'base_hue': 0.52,
        'color_saturation': 0.75,
    }

    PARAMS = {
        'turns': (2.4, 0.5, 6.0, 0.1),
        'helix_width': (0.20, 0.05, 0.45, 0.01),
        'strand_thickness': (0.0035, 0.001, 0.012, 0.0005),
        'rung_count': (16, 4, MAX_RUNGS, 1),
        'rung_thickness': (0.0022, 0.0005, 0.008, 0.0005),
        'spin_speed': (0.30, 0.0, 1.5, 0.05),
        'twist_kick': (0.8, 0.0, 2.5, 0.05),
        'pulse_strength': (1.0, 0.0, 2.5, 0.05),
        'depth_contrast': (0.7, 0.0, 1.0, 0.05),
        'glow': (0.55, 0.0, 2.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Helix": ["turns", "helix_width", "strand_thickness", "depth_contrast"],
        "Streben": ["rung_count", "rung_thickness"],
        "Bewegung": ["spin_speed", "twist_kick"],
        "Reaktion": ["pulse_strength", "glow"],
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
            uniform float u_spin;            // aufsummierter Drehwinkel
            uniform float u_pulse_pos;       // Position des Lichtpulses 0..1
            uniform float u_chroma[12];
            uniform vec3 u_color_a;
            uniform vec3 u_color_b;
            uniform float u_turns;
            uniform float u_helix_width;
            uniform float u_strand_thickness;
            uniform float u_rung_count;
            uniform float u_rung_thickness;
            uniform float u_pulse_strength;
            uniform float u_depth_contrast;
            uniform float u_glow;
            uniform float u_brightness;
            out vec4 f_color;

            const float TAU = 6.28318530718;

            // Lage eines Strangs auf Hoehe y: x-Auslenkung und Tiefe (-1..1)
            void strand(float y, float side, float amp, out float x, out float z) {
                float phase = y * u_turns * TAU + u_spin + side * 3.14159265;
                x = sin(phase) * amp;
                z = cos(phase);
            }

            // Weiche Linie um eine Distanz, mit Tiefen-Abhaengigkeit.
            // Heller Klang (hoher Centroid) gibt einen engeren, schaerferen
            // Schein, dumpfer Klang einen breiten weichen.
            float lineGlow(float dist, float thickness, float depth01) {
                float t = thickness * mix(0.55, 1.35, depth01);
                float core = 1.0 - aastep(t, dist);
                float spread = mix(8.0, 3.5, clamp(u_centroid, 0.0, 1.0));
                float halo = exp(-dist / max(t * spread, 1e-5))
                             * mix(0.55, 0.32, clamp(u_centroid, 0.0, 1.0));
                return core + halo;
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                uv.y = 1.0 - uv.y;
                float aspect = u_resolution.x / max(u_resolution.y, 1.0);

                // Aspektkorrigierter Raum um die Bildmitte
                vec2 p = vec2((uv.x - 0.5) * aspect, uv.y - 0.5);

                float amp = u_helix_width * aspect
                          * (0.75 + 0.55 * u_energy);

                vec3 col = vec3(0.0);

                // --- Lichtpuls, der die Helix hinaufwandert ---
                float pulse = exp(-pow((uv.y - u_pulse_pos) * 6.0, 2.0))
                              * u_beat * u_pulse_strength;

                // --- Straenge ---
                for (int s = 0; s < 2; s++) {
                    float side = float(s);
                    float x, z;
                    strand(uv.y, side, amp, x, z);
                    float depth01 = z * 0.5 + 0.5;          // 0 = hinten, 1 = vorne
                    float dist = abs(p.x - x);
                    float g = lineGlow(dist, u_strand_thickness * aspect, depth01);
                    // Hintere Haelfte wird abgedunkelt statt verdeckt —
                    // so bleibt die Helix auf Schwarz lesbar
                    float shade = mix(1.0 - u_depth_contrast, 1.0, depth01);
                    vec3 base = mix(u_color_a, u_color_b, side);
                    col += base * g * shade * (0.45 + 0.9 * u_energy)
                           * (1.0 + pulse * 1.8);
                }

                // --- Querstreben ---
                int rungs = int(u_rung_count);
                for (int i = 0; i < """ + str(MAX_RUNGS) + """; i++) {
                    if (i >= rungs) break;
                    float fi = float(i);
                    float ry = (fi + 0.5) / u_rung_count;
                    float dy = abs(uv.y - ry);
                    // Nur in der Naehe der Strebe weiterrechnen
                    if (dy > 0.02) continue;

                    float xa, za, xb, zb;
                    strand(ry, 0.0, amp, xa, za);
                    strand(ry, 1.0, amp, xb, zb);

                    // Chroma dieses Tons bestimmt Leuchtkraft und Farbe
                    int ci = int(mod(fi, 12.0));
                    float chroma = u_chroma[ci];
                    float lit = smoothstep(0.15, 0.75, chroma);

                    float d = sdSegment(vec2(p.x, uv.y), vec2(xa, ry), vec2(xb, ry));
                    float depth01 = (za * 0.5 + 0.5);
                    float t = u_rung_thickness * aspect * mix(0.6, 1.3, depth01);
                    float core = 1.0 - aastep(t, d);
                    float halo = exp(-d / max(t * 6.0, 1e-5)) * 0.35;

                    // Flimmern der Enden mit dem Rauschanteil des Signals
                    float flick = 1.0 - u_zcr * 0.35
                        * step(0.5, hash12(vec2(fi, floor(u_time * 12.0))));

                    vec3 rc = mix(u_color_a, u_color_b, chroma);
                    col += rc * (core + halo) * (0.15 + 1.15 * lit) * flick
                           * (0.4 + 0.8 * u_energy) * (1.0 + pulse * 2.2);

                    // Verbindungspunkte auf den Straengen
                    float knot = exp(-pow((p.x - xa) * 90.0, 2.0))
                               + exp(-pow((p.x - xb) * 90.0, 2.0));
                    col += rc * knot * exp(-dy * 260.0) * (0.3 + 0.9 * lit);
                }

                // Weicher Schein um die gesamte Struktur (nur wo schon Licht ist)
                col += col * u_glow * 0.35;

                // Sprach-Modus faehrt insgesamt eine Stufe leiser — die
                // Struktur bleibt, das Bild draengt sich nicht auf.
                col *= 1.0 - 0.28 * u_speech;

                // Betonungen zuenden kurz die Achse
                float axis = exp(-pow(p.x * 26.0, 2.0));
                col += mix(u_color_a, u_color_b, 0.5) * axis * u_impact * 0.22
                       * (0.4 + 0.6 * u_speech);

                col = max(col, 0.0) * u_brightness;
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time) * 5.0);
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
        self._pulse_pos = 1.5
        self._last_time = None

    def _advance(self, f: dict, time: float, speech: float) -> tuple:
        """Integriert Drehung und Lichtpuls ueber die Zeit.

        Beides sind Geschwindigkeiten, keine Positionen — bei einem Sprung
        in der Zeitachse (Vorschau-Scrubbing) wird aus der absoluten Zeit
        neu aufgesetzt statt weiterzulaufen.
        """
        base_speed = float(self.params["spin_speed"])
        if self._last_time is None or time < self._last_time - 1e-6 \
                or (time - self._last_time) > 0.5:
            self._spin = time * base_speed
            self._pulse_pos = 1.5
            self._last_time = time
            return self._spin, self._pulse_pos

        dt = max(time - self._last_time, 1e-4)
        self._last_time = time

        impact = f.get("transient", f["onset"])
        # Torsion: Transienten geben der Helix einen kurzen Dreh-Ruck
        kick = impact * float(self.params["twist_kick"]) * (1.0 - 0.6 * speech)
        speed = base_speed * (0.4 + 0.9 * f["rms"]) + kick
        self._spin += speed * dt

        beat = f.get("beat_intensity", f["onset"])
        if beat > 0.55 and self._pulse_pos > 1.15:
            self._pulse_pos = -0.1
        if self._pulse_pos <= 1.15:
            self._pulse_pos += dt * (0.8 + 1.6 * f["rms"])

        return self._spin, self._pulse_pos

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        speech = 1.0 if mode == "speech" else (0.5 if mode == "hybrid" else 0.0)

        spin, pulse_pos = self._advance(f, time, speech)

        chroma = np.asarray(f["chroma"], dtype=np.float32).flatten()
        if chroma.size < 12:
            chroma = np.pad(chroma, (0, 12 - chroma.size))
        chroma = chroma[:12]
        peak = float(chroma.max()) if chroma.size else 0.0
        if peak > 1e-6:
            chroma = chroma / peak

        # Im Sprach-Modus traegt Chroma wenig Information — dann uebernimmt
        # die Stimme die Ansteuerung der Streben.
        if speech > 0.0:
            voice = float(f.get("voice_band", f["rms"]))
            # Grundwert nicht zu tief ansetzen: sonst verschwinden die Streben
            # im Sprach-Modus ganz und die Helix verliert ihren Charakter.
            chroma = chroma * (1.0 - speech) + (0.30 + 0.55 * voice) * speech

        color_a = self._chroma_to_color(f["chroma"])
        hue = self._color_to_hue(color_a)
        sat = float(self.params.get("color_saturation", 0.75))
        color_b = self._hsv_to_rgb((hue + 0.42) % 1.0, min(1.0, sat), 1.1)

        beat = f.get("beat_intensity", f["onset"]) * (1.0 - 0.6 * speech)
        impact = f.get("transient", f["onset"])
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
        p["u_spin"].value = float(spin)
        p["u_pulse_pos"].value = float(np.clip(pulse_pos, -0.15, 1.2))
        p["u_chroma"].write(chroma.astype(np.float32).tobytes())
        p["u_color_a"].value = tuple(color_a)
        p["u_color_b"].value = tuple(color_b)
        p["u_turns"].value = float(self.params["turns"])
        p["u_helix_width"].value = float(self.params["helix_width"])
        p["u_strand_thickness"].value = float(self.params["strand_thickness"])
        # Feinheit der Streben folgt dem spektralen Schwerpunkt
        rung_count = float(self.params["rung_count"]) * (0.7 + 0.6 * f["spectral_centroid"])
        p["u_rung_count"].value = float(np.clip(round(rung_count), 4, MAX_RUNGS))
        p["u_rung_thickness"].value = float(self.params["rung_thickness"])
        p["u_pulse_strength"].value = float(self.params["pulse_strength"])
        p["u_depth_contrast"].value = float(self.params["depth_contrast"])
        p["u_glow"].value = float(self.params["glow"])
        p["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
