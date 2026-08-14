"""
Kaleidoscope - GPU-Visualizer "Kaleidoskop".

Winkel-Spiegelung statt Zentralform: der Bildraum wird in N Sektoren
gefaltet, jeder Sektor zeigt dasselbe Muster gespiegelt. Archetyp
Symmetrie/Spiegelung — in der Sammlung bisher nicht vorhanden.

Design:
- Faltung in N Sektoren; N kommt aus dem Parameter, wird aber von der
  Tonhoehe leicht moduliert (heller Klang = mehr Sektoren).
- Muster im Sektor: Speichen, Ringe und ein feiner fbm-Schimmer. Alles
  als Linien mit dunklen Zwischenraeumen, keine Vollflaeche.
- Jeder Sektor traegt die Farbe eines Chroma-Tons; klingt der Ton, wird
  der Sektor heller. Die Symmetrie wird damit hoerbar zugeordnet.
- rms = Zoom (Muster kommt auf den Betrachter zu), beat_intensity =
  Ring-Welle nach aussen, transient = Aufblitzen der Spiegel-Nahtstellen,
  spectral_centroid = Feinheit der Speichen, zero_crossing_rate = Koernung.
- Sprach-Modus: langsame Drehung, wenige Sektoren wirken beruhigt, die
  Helligkeit folgt voice_band statt den Beats.

Alles additiv auf Schwarz — ein Hintergrundbild bleibt sichtbar.
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


class KaleidoscopeGPU(BaseGPUVisualizer):
    """Kaleidoskop: Winkel-Faltung mit chroma-gefaerbten Sektoren."""

    COLOR_PARAMS = {
        'color_mode': 'chroma',
        'base_hue': 0.62,
        'color_saturation': 0.8,
    }

    PARAMS = {
        'segments': (8, 3, 16, 1),
        'spoke_count': (7, 2, 20, 1),
        'ring_count': (9, 2, 24, 1),
        'line_width': (0.19, 0.02, 0.6, 0.02),
        'spin_speed': (0.12, 0.0, 1.0, 0.02),
        'spin_kick': (0.9, 0.0, 3.0, 0.05),
        'zoom_response': (0.35, 0.0, 1.5, 0.05),
        'wave_strength': (1.0, 0.0, 2.5, 0.05),
        'seam_flash': (0.8, 0.0, 2.0, 0.05),
        'shimmer': (0.30, 0.0, 1.5, 0.05),
        'center_glow': (0.5, 0.0, 2.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Faltung": ["segments", "line_width", "center_glow"],
        "Muster": ["spoke_count", "ring_count", "shimmer"],
        "Bewegung": ["spin_speed", "spin_kick", "zoom_response"],
        "Reaktion": ["wave_strength", "seam_flash"],
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
            uniform float u_spin;         // aufsummierter Drehwinkel
            uniform float u_wave_pos;     // Radius der laufenden Ringwelle
            uniform float u_chroma[12];
            uniform vec3 u_color_a;
            uniform vec3 u_color_b;
            uniform float u_segments;
            uniform float u_spoke_count;
            uniform float u_ring_count;
            uniform float u_line_width;
            uniform float u_zoom_response;
            uniform float u_wave_strength;
            uniform float u_seam_flash;
            uniform float u_shimmer;
            uniform float u_center_glow;
            uniform float u_brightness;
            out vec4 f_color;

            const float TAU = 6.28318530718;

            // Periodische Linie: 1.0 auf der Linie, 0.0 dazwischen.
            // `width` ist der Anteil einer Periode, den die Linie einnimmt
            // (0.16 = schmale Linie mit viel Schwarz dazwischen). Wird die
            // Folge dichter als das Pixelraster, blendet sie aus — sonst
            // wird aus dem Muster eine geschlossene Flaeche.
            float periodicLine(float x, float width) {
                float saw = abs(fract(x) - 0.5) * 2.0;
                float w = fwidth(x) * 2.2 + 1e-5;
                float line = 1.0 - smoothstep(0.0, w, saw - width);
                return line * smoothstep(0.8, 0.2, w);
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                float aspect = u_resolution.x / max(u_resolution.y, 1.0);
                vec2 p = vec2((uv.x - 0.5) * aspect, uv.y - 0.5);

                float r = length(p);
                float a = atan(p.y, p.x) + u_spin;

                // --- Winkel-Faltung ---
                float seg = TAU / max(u_segments, 1.0);
                float idx = floor(a / seg);                 // Sektor-Nummer
                float af = mod(a, seg);
                af = abs(af - seg * 0.5);                   // Spiegelung

                // Zoom: lautere Stellen holen das Muster nach vorne
                float zoom = 1.0 + u_zoom_response * u_energy;
                float rz = r * zoom;

                // --- Muster im Sektor ---
                // Speichen: Feinheit folgt dem spektralen Schwerpunkt
                float spokes_n = u_spoke_count * (0.6 + 0.8 * u_centroid);
                float spokes = periodicLine(af / seg * spokes_n, u_line_width);
                // Nach aussen ausduennen, damit die Mitte die Struktur traegt
                spokes *= exp(-rz * 2.9);

                float rings = periodicLine(rz * u_ring_count - u_time * 0.15, u_line_width);
                rings *= smoothstep(0.02, 0.12, rz) * exp(-rz * 2.6);

                // Feiner Schimmer, bricht die strenge Symmetrie leicht auf
                float shim = fbm(vec2(af * 6.0, rz * 5.0 - u_time * 0.4), 3);
                // Nur die Spitzen des Rauschens leuchten — sonst legt sich ein
                // geschlossener Schleier ueber das ganze Bild.
                shim = smoothstep(0.68, 1.0, shim) * u_shimmer * exp(-rz * 3.0);

                // Ringwelle nach aussen (Beat)
                float wave = exp(-pow((rz - u_wave_pos) * 9.0, 2.0))
                             * u_beat * u_wave_strength;

                // --- Farbe pro Sektor aus Chroma ---
                int ci = int(mod(abs(idx), 12.0));
                float chroma = u_chroma[ci];
                float lit = 0.35 + 1.15 * smoothstep(0.1, 0.8, chroma);
                vec3 sector = mix(u_color_a, u_color_b, chroma);

                // Im Sprach-Modus faellt die Struktur ruhiger aus: weniger
                // Schimmer, weichere Naehte — gleiche Optik, andere Dosis.
                float calm = 1.0 - 0.45 * u_speech;
                shim *= calm;

                vec3 col = vec3(0.0);
                col += sector * spokes * lit * (0.55 + 1.35 * u_energy);
                col += mix(u_color_b, u_color_a, 0.35) * rings
                       * (0.45 + 1.2 * u_energy) * (0.6 + 0.6 * lit);
                col += sector * shim * (0.4 + 0.8 * u_energy);
                col += mix(u_color_a, u_color_b, 0.5) * wave * 0.9;

                // Spiegel-Nahtstellen blitzen bei Betonungen auf
                float seam_d = min(af, seg * 0.5 - af);
                float seam = exp(-seam_d * 90.0) * exp(-rz * 1.3);
                col += u_color_b * seam * u_impact * u_seam_flash * 0.9 * calm;

                // Kern in der Mitte
                float core = exp(-pow(rz * 18.0, 2.0));
                col += mix(u_color_a, u_color_b, 0.5) * core
                       * u_center_glow * (0.4 + 1.2 * u_energy);

                // Koernung aus dem Rauschanteil des Signals (nur wo Licht ist)
                float grain = hash12(gl_FragCoord.xy + floor(u_time * 24.0)) - 0.5;
                col += col * grain * 0.18 * u_zcr;

                col = max(col, 0.0) * u_brightness;
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time) * 3.0);
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

        self._spin = 0.0
        self._wave_pos = 2.0
        self._last_time = None

    def _advance(self, f: dict, time: float, speech: float) -> tuple:
        """Integriert Drehung und Ringwelle ueber die Zeit."""
        base_speed = float(self.params["spin_speed"])
        if self._last_time is None or time < self._last_time - 1e-6 \
                or (time - self._last_time) > 0.5:
            self._spin = time * base_speed
            self._wave_pos = 2.0
            self._last_time = time
            return self._spin, self._wave_pos

        dt = max(time - self._last_time, 1e-4)
        self._last_time = time

        impact = f.get("transient", f["onset"])
        kick = impact * float(self.params["spin_kick"]) * (1.0 - 0.65 * speech)
        # Richtungswechsel je nach Tonlage: hell dreht vorwaerts, dumpf zurueck
        direction = 1.0 if f["spectral_centroid"] >= 0.45 else -1.0
        self._spin += (base_speed * (0.4 + 0.8 * f["rms"]) + kick) * direction * dt

        beat = f.get("beat_intensity", f["onset"])
        if beat > 0.55 and self._wave_pos > 1.1:
            self._wave_pos = 0.02
        if self._wave_pos <= 1.1:
            self._wave_pos += dt * (0.5 + 1.1 * f["rms"])

        return self._spin, self._wave_pos

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        speech = 1.0 if mode == "speech" else (0.5 if mode == "hybrid" else 0.0)

        spin, wave_pos = self._advance(f, time, speech)

        chroma = np.asarray(f["chroma"], dtype=np.float32).flatten()
        if chroma.size < 12:
            chroma = np.pad(chroma, (0, 12 - chroma.size))
        chroma = chroma[:12]
        peak = float(chroma.max()) if chroma.size else 0.0
        if peak > 1e-6:
            chroma = chroma / peak
        if speech > 0.0:
            # Sprache traegt keine Tonart — dann leuchten die Sektoren
            # gemeinsam mit der Stimme statt einzeln mit Toenen.
            voice = float(f.get("voice_band", f["rms"]))
            chroma = chroma * (1.0 - speech) + voice * speech * 0.8

        color_a = self._chroma_to_color(f["chroma"])
        hue = self._color_to_hue(color_a)
        sat = float(self.params.get("color_saturation", 0.8))
        color_b = self._hsv_to_rgb((hue + 0.33) % 1.0, min(1.0, sat), 1.15)

        beat = f.get("beat_intensity", f["onset"]) * (1.0 - 0.65 * speech)
        impact = f.get("transient", f["onset"]) * (1.0 - 0.4 * speech)
        energy = f["rms"] * (1.0 - speech) + float(f.get("voice_band", f["rms"])) * speech

        # Sektorzahl leicht von der Tonlage moduliert, im Sprach-Modus ruhiger
        segments = float(self.params["segments"])
        segments *= 1.0 + 0.25 * (f["spectral_centroid"] - 0.5) * (1.0 - speech)
        segments = float(np.clip(round(segments), 3, 16))

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
        p["u_wave_pos"].value = float(np.clip(wave_pos, 0.0, 1.2))
        p["u_chroma"].write(chroma.astype(np.float32).tobytes())
        p["u_color_a"].value = tuple(color_a)
        p["u_color_b"].value = tuple(color_b)
        p["u_segments"].value = segments
        p["u_spoke_count"].value = float(self.params["spoke_count"])
        p["u_ring_count"].value = float(self.params["ring_count"])
        p["u_line_width"].value = float(self.params["line_width"])
        p["u_zoom_response"].value = float(self.params["zoom_response"])
        p["u_wave_strength"].value = float(self.params["wave_strength"])
        p["u_seam_flash"].value = float(self.params["seam_flash"])
        p["u_shimmer"].value = float(self.params["shimmer"])
        p["u_center_glow"].value = float(self.params["center_glow"])
        p["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
