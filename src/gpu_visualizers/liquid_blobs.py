"""
Liquid Blobs — Plasma-Metaballs (GPU, SDF-basiert).

6-10 organische Metaballs, die ueber smooth-min verschmelzen und sich wieder
trennen — wie fluessige Lichttropfen. Transluzent/leuchtend, additiver
Charakter, dunkler Hintergrund (keine Vollflaeche, Hintergrundbild-freundlich).

Feature-Mapping:
- Bass/Onset/Beat: Pulsation + beat-synchron wachsender smooth-min-Radius
  (staerkeres Verschmelzen)
- spectral_centroid (Treble): feine Oberflaechen-Textur/Shimmer (FBM-Amplitude)
- RMS/Energy: Bewegungsgeschwindigkeit + Blob-Groesse + Kern-Helligkeit
- Chroma: Farbbasis, Verlauf Primaer -> Sekundaer pro Blob

Sprach-Modus: gleiche Optik, andere Empfindlichkeit — langsames Atmen
gesteuert von voice_band, Betonungen blühen sanft auf, Pausen fast reglos.
"""

import numpy as np
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


class LiquidBlobsGPU(BaseGPUVisualizer):
    """
    Plasma-Metaballs: verschmelzende, transluzente Licht-Blobs (SDF/smin).
    """

    PARAMS = {
        'blob_count': (7, 4, 10, 1),
        'blob_size': (0.17, 0.06, 0.35, 0.01),
        'fluidity': (0.8, 0.1, 2.0, 0.05),
        'merge_strength': (1.0, 0.2, 3.0, 0.05),
        'pulse_strength': (0.7, 0.0, 1.5, 0.05),
        'shimmer': (0.6, 0.0, 1.5, 0.05),
        'glow_strength': (0.8, 0.0, 2.0, 0.05),
        'color_spread': (0.6, 0.0, 1.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Blobs": ["blob_count", "blob_size", "fluidity"],
        "Verschmelzung & Puls": ["merge_strength", "pulse_strength"],
        "Textur & Glow": ["shimmer", "glow_strength"],
        "Farben": ["color_spread"],
    }

    # Maximale Blob-Anzahl (muss mit GLSL-Loop-Grenze uebereinstimmen)
    MAX_BLOBS = 10

    def _setup(self):
        """Fullscreen-Quad + Metaball-Shader erstellen."""
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;          // akkumulierte Simulationszeit (CPU)
            uniform float u_energy;        // Musik: RMS | Sprache: voice_band
            uniform float u_beat;          // Beat-/Betonungs-Envelope
            uniform float u_detail;        // Treble (spectral_centroid)
            uniform vec3 u_color;          // Primaerfarbe (Chroma)
            uniform vec3 u_secondary_color;
            uniform vec3 u_background_color;
            uniform float u_blob_count;
            uniform float u_blob_size;
            uniform float u_merge;         // smooth-min-Basisradius
            uniform float u_pulse;         // Staerke der Beat-Pulsation
            uniform float u_shimmer;       // FBM-Oberflaechentextur
            uniform float u_glow;          // Halo-/Glow-Staerke
            uniform float u_color_spread;  // Farbmischung Primaer->Sekundaer
            uniform float u_brightness;

            out vec4 f_color;

            // Polynomielles smooth-min (IQ): k = Verschmelzungsradius
            float smin(float a, float b, float k) {
                float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
                return mix(b, a, h) - k * h * (1.0 - h);
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
                float aspect = u_resolution.x / u_resolution.y;
                uv.x *= aspect;

                // Hintergrund: sehr dunkel, transparent-freundlich (Luma-Alpha)
                vec3 col = u_background_color * 0.15;

                // smooth-min-Radius waechst beat-synchron -> staerkeres Verschmelzen
                float k = u_merge * (0.06 + 0.05 * u_energy + 0.22 * u_beat * u_pulse);

                // === Metaball-Feld aufbauen ===
                float d = 1e5;
                vec3 wcol = vec3(0.0);
                float wsum = 0.0;
                int n = int(u_blob_count + 0.5);
                for (int i = 0; i < 10; i++) {
                    if (i >= n) break;
                    float fi = float(i);
                    // Deterministische, pro Blob stabile Zufaellswerte
                    vec3 h = vec3(hash12(vec2(fi, 1.7)),
                                  hash12(vec2(fi, 5.3)),
                                  hash12(vec2(fi, 9.9)));

                    // Organische Orbits (goldener Winkel streut die Phasen)
                    float spd = 0.20 + 0.45 * h.x;
                    float ph1 = fi * 2.39996 + h.y * 6.2831;
                    float ph2 = fi * 1.73205 + h.z * 6.2831;
                    vec2 c = vec2(sin(u_time * spd + ph1) * (0.28 + 0.34 * h.y) * aspect,
                                  cos(u_time * spd * 0.83 + ph2) * (0.28 + 0.34 * h.z));

                    // Groesse: Basis * Energie * individueller Beat-Versatz
                    float pulse = 1.0 + u_pulse * u_beat
                                  * (0.5 + 0.5 * sin(fi * 1.93 + u_time * 2.0));
                    float r = u_blob_size * (0.55 + 0.55 * h.z)
                              * (0.75 + 0.5 * u_energy) * pulse;

                    float di = sdCircle(uv - c, r);
                    d = smin(d, di, k);

                    // Farbe einflussgewichtet mischen (weiche Farbverlaeufe)
                    float w = (r * r) / (di * di + 0.02);
                    wcol += mix(u_color, u_secondary_color, h.x * u_color_spread) * w;
                    wsum += w;
                }
                vec3 blobCol = wcol / max(wsum, 1e-4);

                // === Treble-Shimmer: FBM verzerrt die Oberflaeche ===
                float shAmp = u_shimmer * (0.015 + 0.09 * u_detail);
                float nz = fbm(uv * 4.0 + vec2(u_time * 0.4, -u_time * 0.3), 4) - 0.5;
                float dd = d + nz * shAmp;

                // === Koerper: weiche Kante, innerer Verlauf (HDR) ===
                float body = aafill(dd);
                // Kern-Normierung knapp unter Blob-Radius: der Mittelpunkt
                // erreicht inside ~1 und ist damit die hellste Stelle
                float inside = clamp(-dd / (u_blob_size * 0.75 + k), 0.0, 1.0);
                vec3 inner = blobCol * (0.45 + 2.6 * pow(inside, 1.1)
                                        * (0.55 + 0.65 * u_energy
                                           + 0.55 * u_beat * u_pulse));
                // Feine innere Textur (Treble-sichtbar)
                inner *= 0.85 + 0.30 * fbm(uv * 9.0 - u_time * 0.2, 3);
                col += inner * body;

                // === Leuchtende Kante / Glow an Verschmelzungsstellen ===
                // Die smin-Kontur hellt bei wachsendem k auf -> Beat-Glow
                // Bewusst schwaecher als der Kern — sonst wirken die Blobs hohl
                float rim = exp(-abs(dd) * 26.0);
                col += blobCol * rim * (0.15 + 0.35 * u_beat * u_pulse)
                       * (0.5 + 4.0 * k);

                // === Aeusserer Halo (additiv, transluzent) ===
                float halo = exp(-max(dd, 0.0) * 5.0) * u_glow;
                col += blobCol * halo * (0.20 + 0.45 * u_energy);

                // HDR-Ausgabe: kein clamp — ACES-Tonemapping macht der Renderer.
                col = max(col, 0.0) * u_brightness;
                // Triangular-Dithering gegen Banding bei 8-Bit-Quantisierung
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time));

                f_color = vec4(col, 1.0);
            }
            """,
            includes=(LYGIA_MATH_GLSL, LYGIA_NOISE_GLSL, LYGIA_SDF_GLSL,
                      SHADER_COMMON_GLSL),
        )

        self._prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self._vao, self._vbo = create_fullscreen_quad(self.ctx, self._prog)

        # Simulationszustand (deterministisch: nur aus Frame-Sequenz abgeleitet)
        self._sim_time = 0.0      # akkumulierte, geschwindigkeitsgesteuerte Zeit
        self._last_time = None    # letzter Zeitstempel fuer dt
        self._energy_s = 0.0      # EMA-geglaettete Energie

    def _on_params_changed(self):
        # Alle Parameter werden pro Frame gelesen — kein Rebuild noetig.
        pass

    def render(self, features: dict, time: float):
        """Rendert einen Frame: Features -> Modus-Empfindlichkeit -> Uniforms."""
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")

        # === Modus = Empfindlichkeit, nicht Visualisierung ===
        if mode == "speech":
            # Sprache: ruhiges Atmen via voice_band, Betonung = sanftes Aufblühen
            energy = f.get("voice_band", f["rms"])
            beat = f.get("beat_intensity", f["onset"]) * 0.35
            detail = f["spectral_centroid"] * 0.5
            speed_base = 0.45          # deutlich langsamer
            energy_gain = 0.8          # Pausen -> kleine, fast reglose Blobs
        else:
            # Musik: Beats, Onset, volle Treble-Textur
            energy = f["rms"]
            beat = max(f["onset"], f.get("beat_intensity", f["onset"]))
            detail = f["spectral_centroid"]
            speed_base = 1.0
            energy_gain = 1.0

        # Leichte EMA-Glaettung gegen Frame-Jitter (deterministisch)
        energy = energy * energy_gain
        self._energy_s += (energy - self._energy_s) * 0.25

        # Simulationszeit akkumulieren: Geschwindigkeit folgt der Energie,
        # ohne dass Phasenspruenge im Shader entstehen.
        if self._last_time is None:
            dt = 0.0
        else:
            dt = max(0.0, min(time - self._last_time, 0.25))
        self._last_time = time
        speed = speed_base * self.params["fluidity"] * (0.35 + 0.85 * self._energy_s)
        self._sim_time += dt * speed

        color = self._chroma_to_color(f["chroma"])

        def _rgb_from_hex(value, default):
            if isinstance(value, str) and value.startswith('#'):
                try:
                    return self._hex_to_rgb(value)
                except Exception:
                    pass
            return default

        secondary = _rgb_from_hex(self.params.get("secondary_color"), (0.0, 0.8, 1.0))
        background = _rgb_from_hex(self.params.get("background_color"), (0.02, 0.02, 0.04))

        p = self._prog
        p["u_resolution"].value = (self.width, self.height)
        p["u_time"].value = self._sim_time
        p["u_energy"].value = float(self._energy_s)
        p["u_beat"].value = float(min(beat, 1.0))
        p["u_detail"].value = float(detail)
        p["u_color"].value = color
        p["u_secondary_color"].value = secondary
        p["u_background_color"].value = background
        p["u_blob_count"].value = float(int(self.params["blob_count"]))
        p["u_blob_size"].value = float(self.params["blob_size"])
        p["u_merge"].value = float(self.params["merge_strength"])
        p["u_pulse"].value = float(self.params["pulse_strength"])
        p["u_shimmer"].value = float(self.params["shimmer"])
        p["u_glow"].value = float(self.params["glow_strength"])
        p["u_color_spread"].value = float(self.params["color_spread"])
        p["u_brightness"].value = float(self.params["brightness"])

        self._vao.render(mode=moderngl.TRIANGLE_STRIP)
