"""
Ink Bloom - GPU-Visualizer "Tintenfahne".

Tinte, die sich in Wasser ausbreitet: Schlieren, Faeden, aufsteigende
Wolken. Archetyp Fluid/Diffusion — in der Sammlung bisher nicht
vorhanden (nebula_drift ist ein Nebelfeld ohne Stroemung, liquid_blobs
sind geschlossene Metaballs).

## Der Trick: kein Fluid-Solver

Echte Advektion braucht ein Dichtefeld, das von Frame zu Frame
weitergereicht wird (Ping-Pong-Framebuffer). Das haette zwei Nachteile:
der Zustand haengt am Aufrufverlauf — beim Scrubben in der Vorschau
saehe derselbe Zeitpunkt jedes Mal anders aus — und der Offline-Render
waere nicht mehr reproduzierbar.

Stattdessen Domain Warping: fbm-Rauschen wird mit sich selbst verzerrt.

    q = fbm(p)              erste Verzerrung
    r = fbm(p + 4*q)        zweite Verzerrung auf der ersten
    d = fbm(p + 4*r)        Dichte im doppelt verzerrten Raum

Die Faeden und Wirbel entstehen aus der Verschachtelung, nicht aus einer
Simulation. Fuenf Rauschabfragen pro Pixel, zustandslos, deterministisch:
derselbe Zeitpunkt ergibt immer dasselbe Bild.

Audio:
- rms = Staerke der Verzerrung (ruhig = glatte Schlieren, laut = wilde Faeden)
- beat_intensity = Tintenstoss, der als Front nach aussen laeuft
- transient = kurzes Aufreissen der Faeden (zusaetzliche Verzerrung)
- spectral_centroid = Feinheit der Struktur (Oktaven-Skalierung)
- chroma = Farbe der Tinte, zweiter Ton gibt die Gegenfarbe
- zero_crossing_rate = Koernung in den dichten Bereichen
- Sprach-Modus: langsames Quellen auf voice_band, wenig Turbulenz

Nur die dichten Faeden leuchten, dazwischen bleibt es schwarz — ein
Hintergrundbild bleibt sichtbar.
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


class InkBloomGPU(BaseGPUVisualizer):
    """Tintenfahne aus verschachteltem Domain Warping."""

    COLOR_PARAMS = {
        'color_mode': 'chroma',
        'base_hue': 0.56,
        'color_saturation': 0.8,
    }

    PARAMS = {
        'warp_strength': (3.2, 0.5, 8.0, 0.1),
        'detail': (1.0, 0.3, 2.5, 0.05),
        'flow_speed': (0.10, 0.0, 0.6, 0.01),
        'rise': (0.35, -0.5, 1.0, 0.05),
        'density_gate': (0.56, 0.30, 0.85, 0.01),
        'filament': (0.7, 0.0, 2.0, 0.05),
        'puff_strength': (1.0, 0.0, 2.5, 0.05),
        'turbulence': (0.8, 0.0, 2.5, 0.05),
        'glow': (0.8, 0.0, 2.5, 0.05),
        'grain': (0.3, 0.0, 1.5, 0.05),
    }

    PARAMS_GROUPS = {
        "Tinte": ["warp_strength", "detail", "density_gate", "filament"],
        "Stroemung": ["flow_speed", "rise", "turbulence"],
        "Reaktion": ["puff_strength", "glow", "grain"],
    }

    def _setup(self):
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_flow;          // aufsummierte Stroemungszeit
            uniform float u_energy;
            uniform float u_beat;
            uniform float u_impact;
            uniform float u_centroid;
            uniform float u_zcr;
            uniform float u_speech;
            uniform float u_puff_pos;      // Radius der Tintenstoss-Front
            uniform vec3 u_ink_a;
            uniform vec3 u_ink_b;
            uniform float u_warp_strength;
            uniform float u_detail;
            uniform float u_rise;
            uniform float u_density_gate;
            uniform float u_filament;
            uniform float u_puff_strength;
            uniform float u_turbulence;
            uniform float u_glow;
            uniform float u_grain;
            uniform float u_brightness;
            out vec4 f_color;

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                uv.y = 1.0 - uv.y;
                float aspect = u_resolution.x / max(u_resolution.y, 1.0);
                vec2 p = vec2((uv.x - 0.5) * aspect, uv.y - 0.5) * 2.2;

                float t = u_flow;
                // Auftrieb: die Fahne zieht nach oben, unabhaengig von der
                // Stroemungsrichtung des Rauschens
                p.y -= t * u_rise;

                float scale = mix(1.6, 3.4, clamp(u_centroid, 0.0, 1.0)) * u_detail;
                vec2 sp = p * scale;

                // --- Domain Warping (der eigentliche Trick) ---
                float warp = u_warp_strength * (0.45 + 0.85 * u_energy);
                // Betonungen reissen die Faeden kurz auf
                warp += u_impact * u_turbulence * 1.6 * (1.0 - 0.6 * u_speech);

                vec2 q = vec2(fbm(sp + vec2(0.0, t * 0.7), 3),
                              fbm(sp + vec2(5.2, 1.3) + vec2(0.0, t * 0.6), 3));
                vec2 r = vec2(fbm(sp + warp * q + vec2(1.7, 9.2), 3),
                              fbm(sp + warp * q + vec2(8.3, 2.8), 3));
                float d = fbm(sp + warp * r + vec2(0.0, t * 0.4), 4);

                // Dichte in Faeden verwandeln: nur oberhalb des Tors leuchtet
                // es. Ohne dieses Tor legt sich ein geschlossener Schleier
                // ueber das Bild und ein Hintergrund waere weg.
                float gate = u_density_gate - 0.10 * u_energy;
                float ink = smoothstep(gate, gate + 0.22, d);

                // Faserstruktur: die Ableitung des Warp-Feldes betont Kanten
                float edge = abs(r.x - r.y);
                ink += smoothstep(0.55, 0.05, edge) * ink * u_filament * 0.8;

                // Tintenstoss: helle Front, die vom Zentrum nach aussen laeuft
                float rad = length(p);
                float puff = exp(-pow((rad - u_puff_pos) * 3.2, 2.0))
                             * u_beat * u_puff_strength;
                ink += puff * 0.55 * smoothstep(gate - 0.25, gate + 0.1, d);

                // Randabfall, damit die Fahne nicht am Bildrand abgeschnitten
                // wirkt
                ink *= smoothstep(2.4, 0.6, rad);

                // Farbe: dichte Kerne in der Grundfarbe, Raender in der
                // Gegenfarbe — wie Tinte, die sich im Wasser aufhellt
                vec3 col = mix(u_ink_b, u_ink_a, smoothstep(gate - 0.04, gate + 0.16, d));
                col *= ink * (0.45 + 1.1 * u_energy);
                col += col * u_glow * 0.4;

                // Koernung nur in den dichten Bereichen
                float g = hash12(gl_FragCoord.xy + floor(u_time * 20.0)) - 0.5;
                col += col * g * u_grain * 0.35 * u_zcr;

                col *= 1.0 - 0.25 * u_speech;
                col = max(col, 0.0) * u_brightness;
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time) * 17.0);
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

        self._flow = 0.0
        self._puff_pos = 3.0
        self._last_time = None

    def _advance(self, f: dict, time: float, speech: float) -> tuple:
        """Integriert Stroemungszeit und Tintenstoss.

        Die Stroemung laeuft mit der Energie schneller — deshalb reicht die
        absolute Zeit nicht, sie muss aufsummiert werden. Bei einem Sprung
        in der Zeitachse wird aus der absoluten Zeit neu aufgesetzt.
        """
        base = float(self.params["flow_speed"])
        if self._last_time is None or time < self._last_time - 1e-6 \
                or (time - self._last_time) > 0.5:
            self._flow = time * base
            self._puff_pos = 3.0
            self._last_time = time
            return self._flow, self._puff_pos

        dt = max(time - self._last_time, 1e-4)
        self._last_time = time

        speed = base * (0.4 + 1.3 * f["rms"]) * (1.0 - 0.45 * speech)
        self._flow += speed * dt

        beat = f.get("beat_intensity", f["onset"])
        if beat > 0.55 and self._puff_pos > 2.3:
            self._puff_pos = 0.05
        if self._puff_pos <= 2.3:
            self._puff_pos += dt * (0.8 + 1.6 * f["rms"])

        return self._flow, self._puff_pos

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        speech = 1.0 if mode == "speech" else (0.5 if mode == "hybrid" else 0.0)

        flow, puff_pos = self._advance(f, time, speech)

        ink_a = self._chroma_to_color(f["chroma"])
        hue = self._color_to_hue(ink_a)
        sat = float(self.params.get("color_saturation", 0.8))
        # Gegenfarbe fuer die aufgehellten Raender
        ink_b = self._hsv_to_rgb((hue + 0.52) % 1.0, min(1.0, sat * 0.55), 1.05)

        beat = f.get("beat_intensity", f["onset"]) * (1.0 - 0.6 * speech)
        impact = f.get("transient", f["onset"])
        energy = f["rms"] * (1.0 - speech) + float(f.get("voice_band", f["rms"])) * speech

        p = self.prog
        p["u_resolution"].value = (self.width, self.height)
        p["u_time"].value = float(time)
        p["u_flow"].value = float(flow)
        p["u_energy"].value = float(energy)
        p["u_beat"].value = float(beat)
        p["u_impact"].value = float(impact)
        p["u_centroid"].value = float(f["spectral_centroid"])
        p["u_zcr"].value = float(f.get("zero_crossing_rate", 0.0))
        p["u_speech"].value = float(speech)
        p["u_puff_pos"].value = float(np.clip(puff_pos, 0.0, 2.5))
        p["u_ink_a"].value = tuple(ink_a)
        p["u_ink_b"].value = tuple(ink_b)
        p["u_warp_strength"].value = float(self.params["warp_strength"])
        p["u_detail"].value = float(self.params["detail"])
        p["u_rise"].value = float(self.params["rise"])
        p["u_density_gate"].value = float(self.params["density_gate"])
        p["u_filament"].value = float(self.params["filament"])
        p["u_puff_strength"].value = float(self.params["puff_strength"])
        p["u_turbulence"].value = float(self.params["turbulence"])
        p["u_glow"].value = float(self.params["glow"])
        p["u_grain"].value = float(self.params["grain"])
        p["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
