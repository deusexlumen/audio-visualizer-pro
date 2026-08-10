"""
GPU-beschleunigter Metropolis-Skyline-Visualizer (Registrierungsname: typographic).

Naechtliche Stadt-Skyline in leichter Zentralperspektive mit 2-3 Tiefen-Reihen:
- Gebaeudehoehen folgen Frequenz-Baendern (Bass = grosse Tuerme mittig,
  Hoehen = feine Gebaeude aussen), geglaettet mit Peak-Hold-Abfall (CPU-seitig).
- Fenster sind kleine leuchtende Rechtecke (hash12, deterministisch), die bei
  Beats wellenartig aufleuchten und mit spectral_centroid flimmern.
- Onset = Skyline-Flash am Horizont, RMS = Stadt-Grundhelligkeit,
  Chroma = warm/kalte Farbstimmung der Fenster.
- Speech-Modus: Fenster atmen mit voice_band, Betonungen (transient) zuenden
  Fenster-Gruppen, Pausen lassen eine ruhige Nachtstadt mit wenigen Lichtern.

Himmel bleibt schwarz (hintergrundbild-freundlich, kein Vollflaechen-Gradient).
Unteres Drittel: Fenster werden zum Bildrand hin abgedunkelt, damit
Zitat-Overlays lesbar bleiben. HDR-Ausgabe ohne clamp, Dithering aus
SHADER_COMMON_GLSL, pixelgenaues AA via aastep/aafill.
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

# Feste Array-Groesse fuer das Band-Uniform (Shader-Limit)
MAX_BUILDINGS = 32


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """Kleines smoothstep-Helferlein fuer numpy-Arrays."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class TypographicGPU(BaseGPUVisualizer):
    """Metropolis-Skyline: naechtliche Stadt, deren Gebaeude und Fenster
    auf Frequenzbaender, Beats und Sprache reagieren."""

    PARAMS = {
        'building_count': (24, 8, MAX_BUILDINGS, 1),
        'depth_layers': (3, 1, 3, 1),
        'height_response': (1.0, 0.2, 2.5, 0.05),
        'peak_hold': (0.90, 0.70, 0.98, 0.01),
        'window_density': (0.55, 0.1, 1.0, 0.05),
        'window_flicker': (0.5, 0.0, 1.0, 0.05),
        'beat_flash': (0.8, 0.0, 2.0, 0.05),
        'horizon_glow': (0.6, 0.0, 1.5, 0.05),
    }

    PARAMS_GROUPS = {
        "Skyline": ["building_count", "depth_layers", "height_response", "peak_hold"],
        "Fenster": ["window_density", "window_flicker"],
        "Reaktion": ["beat_flash", "horizon_glow"],
    }

    def _setup(self):
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_energy;          // RMS (Stadt-Grundhelligkeit)
            uniform float u_onset;           // Onset (Skyline-Flash)
            uniform float u_beat_intensity;  // Beat-Decay-Envelope (Fenster-Welle)
            uniform float u_impact;          // Transiente (Betonung / Bass-Schlag)
            uniform float u_centroid;        // Spectral Centroid (Fenster-Flimmern)
            uniform float u_voice;           // Voice-Band (Atmen im Speech-Modus)
            uniform float u_speech;          // 1.0 = Speech-Modus, 0.0 = Musik
            uniform float u_bands[""" + str(MAX_BUILDINGS) + """];
            uniform float u_bands_avg;
            uniform float u_building_count;
            uniform float u_depth_layers;
            uniform float u_window_density;
            uniform float u_height_response;
            uniform float u_beat_flash;
            uniform float u_window_flicker;
            uniform float u_horizon_glow;
            uniform vec3 u_window_color;     // warm/kalt aus Chroma
            uniform vec3 u_flash_color;      // Horizont-Flash-Farbe (Chroma)
            uniform float u_brightness;
            out vec4 f_color;

            // Fenster eines Gebaeudes. p_local: Pixel relativ zur Gebaeude-
            // Unterkante-links, bid: Gebaeude-ID. Liefert Fuellung 0..1 und
            // einen deterministischen Seed pro Fenster (fuer Helligkeit/Flimmern).
            float windows(vec2 p_local, float bid, float density_eff,
                          out float win_seed) {
                vec2 pitch = vec2(15.0, 19.0) * (u_resolution.x / 1280.0);
                vec2 cell = floor(p_local / pitch);
                // Fenster-ID: eindeutig pro Gebaeude und Zelle (deterministisch)
                float cid = bid * 977.0 + cell.y * 57.0 + cell.x;
                win_seed = hash12(vec2(cid, 3.7));
                // Fenster-Rechteck innerhalb der Zelle (mit AA ueber aafill)
                vec2 fr = fract(p_local / pitch) - 0.5;
                vec2 d = abs(fr) - vec2(0.26, 0.30);
                float rect = aafill(max(d.x, d.y) * 18.0);
                float on = step(hash12(vec2(cid, 11.3)), density_eff);
                // Weicher Halo um jedes Fenster (Leuchten statt harter Pixel)
                float halo = exp(-max(max(d.x, d.y), 0.0) * 5.0) * 0.35;
                return (rect + halo) * on;
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                // Der Renderer liest den Framebuffer vertikal gespiegelt aus —
                // hier vorab spiegeln, damit die Skyline unten steht.
                uv.y = 1.0 - uv.y;

                // --- Himmel: schwarz bleibt schwarz (hintergrundfreundlich) ---
                vec3 col = vec3(0.0);

                // Wenige, sehr dunkle Sterne weit oben (Punkte, kein Gradient)
                if (uv.y > 0.55) {
                    vec2 sp = floor(gl_FragCoord.xy / 3.0);
                    float star = step(0.9965, hash12(sp));
                    float tw = 0.6 + 0.4 * sin(u_time * 0.7 + hash12(sp + 5.0) * 40.0);
                    col += vec3(0.05, 0.055, 0.07) * star * tw
                           * smoothstep(0.55, 0.75, uv.y);
                }

                // --- Horizont: Stadt-Grundglut + Beat-Flash ---
                float horizon = 0.20;
                float glow_fall = exp(-max(uv.y - 0.05, 0.0) * 9.0);
                col += u_flash_color * glow_fall * 0.13 * (0.25 + u_energy) * u_horizon_glow;
                float flash = max(u_beat_intensity, u_onset * 0.8) * u_beat_flash;
                col += u_flash_color * exp(-pow((uv.y - horizon) * 13.0, 2.0))
                       * flash * 0.55;

                // --- Gebaeude-Reihen von hinten nach vorne (Painter) ---
                int count = int(u_building_count);
                int layers = int(u_depth_layers);
                float slot_w = 1.0 / u_building_count;   // Slotbreite in uv.x

                for (int layer = 2; layer >= 0; layer--) {
                    if (layer >= layers) continue;
                    float fl = float(layer);
                    // Tiefenstaffelung: hinten dunkler, kleiner, Basis hoeher
                    float depth = 1.0 - fl * 0.32;                 // 1.0 / 0.68 / 0.36
                    float base_y = 0.045 + fl * 0.075;
                    float h_scale = 1.0 - fl * 0.30;
                    float sil_brightness = 0.030 * depth + 0.006;
                    // Fenster: vorne voll, Mitte schwach, hinten keine
                    float win_layer = (layer == 0) ? 1.0 : (layer == 1 ? 0.30 : 0.0);

                    for (int i = 0; i < """ + str(MAX_BUILDINGS) + """; i++) {
                        if (i >= count) break;
                        float fi = float(i);
                        // Hintere Reihen um einen halben Slot versetzt
                        float slot_x = (fi + 0.5 + 0.5 * fl) * slot_w;
                        // Gebaeudebreite: stark variierend (hash), hintere breiter
                        float bid = fi + fl * 131.0;
                        float bw = slot_w * (0.55 + 0.55 * hash12(vec2(bid, 1.0)))
                                   * (1.0 + fl * 0.25);
                        float x0 = slot_x - bw * 0.5;

                        if (uv.x < x0 || uv.x > x0 + bw) continue;

                        // Bandwert: hintere Reihen geglaettet (Nachbar-Mix)
                        float band = u_bands[i];
                        if (layer == 1) {
                            int ip = max(i - 1, 0);
                            int in_ = min(i + 1, count - 1);
                            band = (u_bands[ip] + band + u_bands[in_]) / 3.0;
                        } else if (layer == 2) {
                            band = mix(band, u_bands_avg, 0.6);
                        }
                        // Zentralperspektive-Anmutung: Mitte hoeher
                        float center_boost = 1.0 + 0.30 * (1.0 - pow(uv.x * 2.0 - 1.0, 2.0));
                        float h = (0.10 + band * 0.50 * u_height_response)
                                  * h_scale * center_boost;

                        // Gebaeude-SDF (Box ab Basis) mit pixelgenauer Kante
                        vec2 gp = vec2(uv.x - (x0 + bw * 0.5), uv.y - base_y);
                        vec2 half_size = vec2(bw * 0.5, h);
                        // sdBox inline (oben offen: Boxmittelpunkt bei h)
                        vec2 dq = abs(gp - vec2(0.0, h)) - half_size;
                        float sd = length(max(dq, 0.0)) + min(max(dq.x, dq.y), 0.0);
                        float fill = aafill(sd);

                        // Silhouette: dunkel, aber sichtbar; leicht blaeulich
                        vec3 sil_col = vec3(0.75, 0.85, 1.15) * sil_brightness
                                       * (0.8 + 0.4 * u_energy);
                        // Kanten-Rim: dezenter Farbschein, beat-gepulst
                        sil_col += u_flash_color * exp(-abs(sd) * 160.0)
                                   * 0.05 * (0.4 + flash);

                        col = mix(col, sil_col, fill);

                        // --- Fenster (nur wo Gebaeude gefuellt) ---
                        if (win_layer > 0.0 && fill > 0.001) {
                            vec2 p_local = vec2((uv.x - x0) * u_resolution.x,
                                                (uv.y - base_y) * u_resolution.y);
                            // Beat-Welle: laeuft von links nach rechts ueber die Stadt
                            float wavefront = (1.0 - u_beat_intensity) * 1.25;
                            float wave = exp(-pow((uv.x - wavefront) * 5.0, 2.0))
                                         * u_beat_intensity * 2.2;
                            // Dichte: Musik = Energie, Speech = Stimme (Pausen = dunkel)
                            float density_eff = u_window_density
                                * mix(0.55 + 0.45 * u_energy,
                                      0.15 + 0.85 * u_voice, u_speech);
                            float win_seed;
                            float w = windows(p_local, bid, density_eff, win_seed);
                            // Flimmern mit spectral_centroid (Zeit-Quantisierung)
                            float flick_rate = 2.0 + u_centroid * 9.0;
                            float flick = mix(1.0,
                                step(0.45, hash12(vec2(win_seed * 913.0,
                                                       floor(u_time * flick_rate)))),
                                u_window_flicker * u_centroid);
                            // Speech: Atmen mit der Stimme + Betonungs-Gruppen
                            float breathe = 1.0 + u_speech * u_voice * 0.7
                                * sin(u_time * 3.0 + win_seed * 6.2831);
                            float accent = u_speech
                                * step(1.0 - u_impact * 0.9,
                                       hash12(vec2(floor(win_seed * 64.0), 17.0)))
                                * 1.6;
                            // Grundhelligkeit pro Fenster + Reaktionen
                            float lum = (0.30 + 0.70 * hash12(vec2(win_seed * 517.0, 7.7)))
                                        * (1.0 + wave) * flick * breathe + accent;
                            // Zum Boden hin abdunkeln (Zitat-Zone lesbar halten)
                            lum *= 0.35 + 0.65 * smoothstep(0.02, 0.30, uv.y);
                            vec3 win_col = u_window_color
                                           * (0.85 + 0.30 * win_seed);
                            col += win_col * w * lum * win_layer * fill;
                        }

                        // --- Antenne mit Blinklicht auf manchen Tuermen ---
                        if (layer == 0 && hash12(vec2(bid, 23.0)) > 0.72) {
                            float roof = base_y + 2.0 * h;
                            float ant_h = 0.05 + 0.05 * hash12(vec2(bid, 29.0));
                            float ax = abs(uv.x - (x0 + bw * 0.5));
                            float in_ant = (1.0 - aastep(0.0018, ax))
                                * step(roof, uv.y) * step(uv.y, roof + ant_h);
                            col += vec3(0.10, 0.10, 0.12) * in_ant;
                            // Rotes Blinklicht (deterministische Phase)
                            float blink = pow(0.5 + 0.5 * sin(u_time * 2.2
                                + hash12(vec2(bid, 31.0)) * 6.2831), 3.0);
                            float tip = exp(-pow((uv.y - roof - ant_h) * 400.0, 2.0))
                                * (1.0 - aastep(0.0035, ax));
                            col += vec3(1.2, 0.08, 0.05) * tip * (0.25 + blink);
                        }
                    }
                }

                // HDR-Ausgabe: kein clamp — Tonemapping macht zentral der Renderer.
                col = max(col, 0.0) * u_brightness;
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time) * 7.0);
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

        # Peak-Hold-Zustand fuer die Gebaeudehoehen (Band-Envelope)
        self._peaks = np.zeros(MAX_BUILDINGS, dtype=np.float32)
        self._last_time = None

    def _compute_bands(self, f: dict, time: float, count: int) -> np.ndarray:
        """Berechnet geglaettete Band-Hoehen (Peak-Hold) fuer die Gebaeude.

        Bass-Energie steuert die grossen Tuerme (Mitte), Mitten die
        mittleren Gebaeude, Hoehen (Centroid/ZCR) die feinen Randbauten.
        """
        rms = f["rms"]
        transient = f.get("transient", f["onset"])
        centroid = f["spectral_centroid"]
        zcr = f.get("zero_crossing_rate", 0.0)
        voice_band = f.get("voice_band", rms)
        voice_clarity = f.get("voice_clarity", rms)
        chroma = np.asarray(f["chroma"], dtype=np.float32).flatten()
        chroma_peak = float(np.max(chroma)) if chroma.size else 0.0
        speech = 1.0 if f.get("mode") == "speech" else 0.0

        pos = (np.arange(count, dtype=np.float32) + 0.5) / count
        dist = np.abs(pos - 0.5) * 2.0  # 0 = Mitte (Bass), 1 = Rand (Hoehen)

        # Band-Gewichtung ueber die Breite: Bass mittig, Hoehen aussen
        w_low = np.exp(-((dist * 2.1) ** 2))
        w_high = _smoothstep(0.45, 1.0, dist)
        w_mid = np.clip(1.0 - w_low - w_high, 0.0, 1.0)

        # Energien je nach Modus unterschiedlich empfindlich mischen
        low_e = (0.55 * rms + 0.85 * transient) * (1.0 - 0.6 * speech) \
            + speech * 0.35 * rms
        mid_e = (1.0 - speech) * (0.45 * rms + 0.55 * chroma_peak) \
            + speech * 1.15 * voice_band
        high_e = (1.0 - speech) * (0.75 * centroid + 0.35 * zcr) \
            + speech * (0.5 * voice_clarity + 0.3 * centroid)

        e = w_low * low_e + w_mid * mid_e + w_high * high_e

        # Statisches Hoehenprofil pro Gebaeude (deterministischer Hash)
        hv = np.mod(np.sin(np.arange(count, dtype=np.float32) * 12.9898
                           + 78.233) * 43758.5453, 1.0)
        profile = (0.35 + 0.65 * hv) * (1.0 + 0.35 * (1.0 - dist ** 2))

        target = np.clip(profile * e * float(self.params["height_response"]),
                         0.0, 1.2).astype(np.float32)

        # Peak-Hold: schneller Anstieg, langsamer Abfall (Zeit-dt-basiert)
        if self._last_time is None or time < self._last_time - 1e-6 \
                or (time - self._last_time) > 0.5:
            self._peaks[:count] = target
            dt = 1.0 / 30.0
        else:
            dt = max(time - self._last_time, 1e-4)
            decay = float(self.params["peak_hold"]) ** (dt * 30.0)
            self._peaks[:count] = np.maximum(target, self._peaks[:count] * decay)
        self._last_time = time

        bands = np.zeros(MAX_BUILDINGS, dtype=np.float32)
        bands[:count] = self._peaks[:count]
        return bands

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        speech = 1.0 if mode == "speech" else (0.5 if mode == "hybrid" else 0.0)

        count = int(self.params["building_count"])
        bands = self._compute_bands(f, time, count)

        # Chroma -> warm/kalte Fenster-Farbstimmung
        chroma_color = self._chroma_to_color(f["chroma"])
        hue = self._color_to_hue(chroma_color)
        warmth = 0.5 + 0.5 * float(np.cos(2.0 * np.pi * (hue - 0.09)))
        warm = np.array([1.0, 0.72, 0.42], dtype=np.float32)
        cold = np.array([0.55, 0.72, 1.0], dtype=np.float32)
        window_color = tuple((cold * (1.0 - warmth) + warm * warmth).tolist())

        p = self.prog
        p["u_resolution"].value = (self.width, self.height)
        p["u_time"].value = float(time)
        p["u_energy"].value = float(f["rms"])
        p["u_onset"].value = float(f["onset"])
        p["u_beat_intensity"].value = float(f.get("beat_intensity", f["onset"]))
        p["u_impact"].value = float(f.get("transient", f["onset"]))
        p["u_centroid"].value = float(f["spectral_centroid"])
        p["u_voice"].value = float(f.get("voice_band", f["rms"]))
        p["u_speech"].value = float(speech)
        p["u_bands"].write(bands.tobytes())
        p["u_bands_avg"].value = float(np.mean(bands[:count]))
        p["u_building_count"].value = float(count)
        p["u_depth_layers"].value = float(self.params["depth_layers"])
        p["u_window_density"].value = float(self.params["window_density"])
        p["u_height_response"].value = float(self.params["height_response"])
        p["u_beat_flash"].value = float(self.params["beat_flash"])
        p["u_window_flicker"].value = float(self.params["window_flicker"])
        p["u_horizon_glow"].value = float(self.params["horizon_glow"])
        p["u_window_color"].value = window_color
        p["u_flash_color"].value = tuple(chroma_color)
        p["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
