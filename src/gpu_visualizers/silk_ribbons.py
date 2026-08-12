"""
Silk Ribbons - GPU-Visualizer "Seidenbaender".

Baender aus Stoff, die quer durchs Bild wehen: gewellte Streifen mit
Glanzkante, die sich verdrehen und ueberlagern. Archetyp Tuch/Band — in
der Sammlung bisher nicht vorhanden (voice_flow zeichnet Linien, keine
Flaechen mit Materialwirkung).

## Der Trick: Glanz statt Geometrie

Ein Band wirkt nur dann wie Stoff, wenn es eine Oberflaeche hat. Eine
echte Simulation (Massepunkte, Federn) braucht Zustand ueber Frames —
dasselbe Problem wie beim Fluid. Stattdessen wird aus der Ableitung der
Mittellinie eine Pseudo-Normale gebaut:

    Mittellinie   yc(x) = Summe aus Sinusanteilen
    Steigung      dy/dx  (analytisch, nicht numerisch)
    Normale       n = normalize(vec2(-dy, 1))

Mit einer festen Lichtrichtung ergibt das einen Glanzstreifen, der ueber
die Welle wandert — die Faltung wird sichtbar, ohne dass eine einzige
Flaeche berechnet wird. Zustandslos und deterministisch.

Audio:
- rms = Amplitude der Wellen (ruhig haengend bis stark wehend)
- transient = Peitschenschlag (kurzer Phasenversatz, laeuft durchs Band)
- beat_intensity = Glanz-Blitz, der ueber die Baender zieht
- spectral_centroid = Anzahl der Wellen pro Band
- chroma = Farbe je Band; jedes Band gehoert einem Ton
- zero_crossing_rate = feines Kraeuseln der Kante
- Sprach-Modus: langsames Wogen auf voice_band, wenig Glanz

Zwischen den Baendern bleibt es schwarz — ein Hintergrundbild bleibt
sichtbar. HDR-Ausgabe ohne clamp, Tonemapping macht zentral der Renderer.
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

# Feste Obergrenze fuer die Band-Schleife im Shader
MAX_RIBBONS = 8


class SilkRibbonsGPU(BaseGPUVisualizer):
    """Wehende Baender mit Glanzkante aus einer Pseudo-Normalen."""

    COLOR_PARAMS = {
        'color_mode': 'chroma',
        'base_hue': 0.88,
        'color_saturation': 0.8,
    }

    PARAMS = {
        'ribbon_count': (5, 1, MAX_RIBBONS, 1),
        'band_width': (0.030, 0.008, 0.09, 0.002),
        'wave_amplitude': (0.10, 0.01, 0.30, 0.005),
        'wave_count': (2.2, 0.5, 6.0, 0.1),
        'flow_speed': (0.35, 0.0, 1.5, 0.05),
        'whip_strength': (1.0, 0.0, 3.0, 0.05),
        'sheen_power': (9.0, 2.0, 48.0, 1.0),
        'sheen_strength': (1.6, 0.0, 4.0, 0.05),
        'beat_flash': (0.9, 0.0, 2.5, 0.05),
        'crinkle': (0.5, 0.0, 2.0, 0.05),
        'depth_fade': (0.55, 0.0, 1.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Baender": ["ribbon_count", "band_width", "depth_fade"],
        "Welle": ["wave_amplitude", "wave_count", "flow_speed"],
        "Material": ["sheen_power", "sheen_strength", "crinkle"],
        "Reaktion": ["whip_strength", "beat_flash"],
    }

    def _setup(self):
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_phase;         // aufsummierte Wehbewegung
            uniform float u_whip;          // aufsummierter Peitschen-Versatz
            uniform float u_energy;
            uniform float u_beat;
            uniform float u_impact;
            uniform float u_centroid;
            uniform float u_zcr;
            uniform float u_speech;
            uniform float u_flash_pos;     // x-Position des Glanz-Blitzes
            uniform float u_chroma[12];
            uniform float u_base_hue;
            uniform float u_saturation;
            uniform float u_ribbon_count;
            uniform float u_band_width;
            uniform float u_wave_amplitude;
            uniform float u_wave_count;
            uniform float u_sheen_power;
            uniform float u_sheen_strength;
            uniform float u_beat_flash;
            uniform float u_crinkle;
            uniform float u_depth_fade;
            uniform float u_brightness;
            out vec4 f_color;

            const float TAU = 6.28318530718;

            // Mittellinie eines Bandes und ihre Steigung.
            // Die Ableitung wird analytisch gebildet — numerisch waere sie
            // bei hohen Wellenzahlen sichtbar verrauscht.
            void ribbon(float x, float fi, float amp, float waves,
                        out float yc, out float dy) {
                float ph = u_phase * (0.7 + 0.35 * fi) + fi * 2.4;
                float w1 = waves * TAU;
                float w2 = waves * TAU * 1.7;
                float a1 = amp;
                float a2 = amp * 0.42;
                // Peitschenschlag laeuft mit x versetzt durchs Band
                float whip = u_whip * (0.6 + 0.5 * fi);

                yc = a1 * sin(w1 * x + ph + whip * x)
                   + a2 * sin(w2 * x - ph * 1.3 + whip);
                dy = a1 * w1 * cos(w1 * x + ph + whip * x)
                   + a2 * w2 * cos(w2 * x - ph * 1.3 + whip);
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                uv.y = 1.0 - uv.y;
                float aspect = u_resolution.x / max(u_resolution.y, 1.0);
                float x = (uv.x - 0.5) * aspect;

                float amp = u_wave_amplitude * (0.45 + 1.1 * u_energy);
                float waves = u_wave_count * (0.65 + 0.7 * u_centroid);

                // Feste Lichtrichtung — der Glanz wandert dann mit der
                // Faltung, nicht mit der Zeit.
                vec2 light = normalize(vec2(-0.45, 1.0));

                int count = int(u_ribbon_count);
                vec3 col = vec3(0.0);

                for (int i = 0; i < """ + str(MAX_RIBBONS) + """; i++) {
                    if (i >= count) break;
                    float fi = float(i);

                    // Baender gleichmaessig ueber die Bildhoehe verteilen
                    float base_y = (fi + 0.5) / u_ribbon_count;
                    // Tiefe: hintere Baender duenner und dunkler
                    float depth = hash12(vec2(fi, 4.2));
                    float dscale = mix(1.0, 0.55, depth * u_depth_fade);

                    float yc, dy;
                    ribbon(x, fi, amp * dscale, waves, yc, dy);
                    float y = base_y + yc;

                    // Kraeuseln der Kante mit dem Rauschanteil des Signals
                    float crink = (noise(vec2(x * 26.0 + fi * 10.0,
                                              u_phase * 2.0)) - 0.5)
                                  * u_crinkle * u_zcr * 0.012;

                    float half_w = u_band_width * dscale
                                 * (0.7 + 0.5 * sin(x * 3.1 + fi * 1.9));
                    float dist = abs(uv.y - y) + crink;
                    float band = 1.0 - smoothstep(half_w * 0.72, half_w, dist);
                    if (band <= 0.001) continue;

                    // Pseudo-Normale aus der Steigung der Mittellinie
                    vec2 n = normalize(vec2(-dy, 1.0));
                    float lambert = clamp(dot(n, light) * 0.5 + 0.5, 0.0, 1.0);
                    float sheen = pow(lambert, u_sheen_power) * u_sheen_strength;

                    // Quer ueber das Band abdunkeln — gibt Woelbung
                    float across = 1.0 - pow(clamp(dist / max(half_w, 1e-5), 0.0, 1.0), 2.0);

                    // Glanz-Blitz, der auf dem Beat ueber die Baender zieht
                    float flash = exp(-pow((uv.x - u_flash_pos) * 5.0, 2.0))
                                  * u_beat * u_beat_flash;

                    int ci = int(mod(fi, 12.0));
                    float chroma = u_chroma[ci];
                    // Eigener Farbton je Band, um die Grundfarbe herum
                    // gefaechert — sonst sehen alle Baender gleich aus.
                    float h = fract(u_base_hue + fi * 0.11 + chroma * 0.05);
                    vec3 tint = hsv2rgb(vec3(h, u_saturation, 1.0));

                    float lit = (0.22 + 0.9 * chroma) * (0.4 + 0.9 * u_energy);
                    col += tint * band * across * lit * dscale;
                    col += mix(tint, vec3(1.0), 0.55) * band * sheen
                           * dscale * (0.5 + 0.8 * u_energy);
                    col += mix(tint, vec3(1.0), 0.7) * band * flash * 0.7;
                }

                // Betonungen heben alle Baender kurz an
                col *= 1.0 + u_impact * 0.35 * (1.0 - 0.5 * u_speech);
                col *= 1.0 - 0.26 * u_speech;

                col = max(col, 0.0) * u_brightness;
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time) * 23.0);
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

        self._phase = 0.0
        self._whip = 0.0
        self._flash_pos = 1.6
        self._last_time = None

    def _advance(self, f: dict, time: float, speech: float) -> tuple:
        """Integriert Wehbewegung, Peitschenschlag und Glanz-Blitz."""
        base = float(self.params["flow_speed"])
        if self._last_time is None or time < self._last_time - 1e-6 \
                or (time - self._last_time) > 0.5:
            self._phase = time * base
            self._whip = 0.0
            self._flash_pos = 1.6
            self._last_time = time
            return self._phase, self._whip, self._flash_pos

        dt = max(time - self._last_time, 1e-4)
        self._last_time = time

        self._phase += base * (0.45 + 1.0 * f["rms"]) * (1.0 - 0.4 * speech) * dt

        # Peitschenschlag: Transiente gibt einen Stoss, der wieder abklingt
        impact = f.get("transient", f["onset"])
        self._whip += impact * float(self.params["whip_strength"]) * 2.5 * dt
        self._whip *= float(np.exp(-dt / 0.45))

        beat = f.get("beat_intensity", f["onset"])
        if beat > 0.55 and self._flash_pos > 1.3:
            self._flash_pos = -0.2
        if self._flash_pos <= 1.3:
            self._flash_pos += dt * (1.0 + 1.4 * f["rms"])

        return self._phase, self._whip, self._flash_pos

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        speech = 1.0 if mode == "speech" else (0.5 if mode == "hybrid" else 0.0)

        phase, whip, flash_pos = self._advance(f, time, speech)

        chroma = np.asarray(f["chroma"], dtype=np.float32).flatten()
        if chroma.size < 12:
            chroma = np.pad(chroma, (0, 12 - chroma.size))
        chroma = chroma[:12]
        peak = float(chroma.max()) if chroma.size else 0.0
        if peak > 1e-6:
            chroma = chroma / peak
        if speech > 0.0:
            voice = float(f.get("voice_band", f["rms"]))
            chroma = chroma * (1.0 - speech) + (0.30 + 0.55 * voice) * speech

        # Grundfarbton aus Chroma; die Baender faechern im Shader darum herum
        hue = self._color_to_hue(self._chroma_to_color(f["chroma"]))
        sat = float(self.params.get("color_saturation", 0.8))

        beat = f.get("beat_intensity", f["onset"]) * (1.0 - 0.6 * speech)
        impact = f.get("transient", f["onset"])
        energy = f["rms"] * (1.0 - speech) + float(f.get("voice_band", f["rms"])) * speech

        p = self.prog
        p["u_resolution"].value = (self.width, self.height)
        p["u_time"].value = float(time)
        p["u_phase"].value = float(phase)
        p["u_whip"].value = float(whip)
        p["u_energy"].value = float(energy)
        p["u_beat"].value = float(beat)
        p["u_impact"].value = float(impact)
        p["u_centroid"].value = float(f["spectral_centroid"])
        p["u_zcr"].value = float(f.get("zero_crossing_rate", 0.0))
        p["u_speech"].value = float(speech)
        p["u_flash_pos"].value = float(np.clip(flash_pos, -0.25, 1.4))
        p["u_chroma"].write(chroma.astype(np.float32).tobytes())
        p["u_base_hue"].value = float(hue)
        p["u_saturation"].value = float(min(1.0, sat))
        p["u_ribbon_count"].value = float(self.params["ribbon_count"])
        p["u_band_width"].value = float(self.params["band_width"])
        p["u_wave_amplitude"].value = float(self.params["wave_amplitude"])
        p["u_wave_count"].value = float(self.params["wave_count"])
        p["u_sheen_power"].value = float(self.params["sheen_power"])
        p["u_sheen_strength"].value = float(self.params["sheen_strength"])
        p["u_beat_flash"].value = float(self.params["beat_flash"])
        p["u_crinkle"].value = float(self.params["crinkle"])
        p["u_depth_fade"].value = float(self.params["depth_fade"])
        p["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
