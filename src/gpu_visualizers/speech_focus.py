"""
Speech Focus - Podcast-Flaggschiff-Visualizer (Redesign v2).

Eine edle, ruhige Speech-first-Visualisierung:
- Eine praezise, duenne horizontale Stimm-Linie in der Bildmitte,
  deren Amplitude aus dem Sprachband (voice_band) gespeist wird.
- Betonungen (Transiente oder steigende voice_clarity) loesen weiche,
  goldene Lichtpulse aus, die entlang der Linie verlaufen.
- Bei Silenz (voice_band -> 0) faellt die Linie zu fast perfekter Ruhe ab.
- Das untere Bilddrittel bleibt bewusst ruhig/dunkel (Zitat-Overlays).

Musik-Modus (features["mode"] == "music"):
- Dieselbe Linie wird zum feinen Spektrum-Band: Beats/Onset pulsen die
  Amplitude und Helligkeit, spectral_centroid moduliert Feinheit/Textur,
  die Chroma-Farbe tont die Linie dezent.

Bewusst NICHT: VU-Meter, dicke Balken, Vollflaechen. Fast alles bleibt
schwarz (hintergrundbild-freundlich), nur Linie und Pulse leuchten additiv.
HDR-Ausgabe ohne clamp — das Tonemapping uebernimmt der Renderer.
"""

import moderngl
from .base import (
    BaseGPUVisualizer,
    FULLSCREEN_VERTEX_SHADER,
    LYGIA_NOISE_GLSL,
    SHADER_COMMON_GLSL,
    compose_fragment,
    create_fullscreen_quad,
)


class SpeechFocusGPU(BaseGPUVisualizer):
    """Ruhiger Speech-First-Visualizer: duenne Stimm-Linie + goldene Pulse."""

    PARAMS = {
        # Dicke der Kern-Linie in Pixeln
        'line_thickness': (1.6, 0.5, 5.0, 0.1),
        # Maximale Amplitude der Linie (Bildhoehen-Anteil)
        'wave_amplitude': (0.075, 0.0, 0.2, 0.005),
        # Staerke der goldenen Betonungs-Pulse
        'pulse_strength': (1.4, 0.0, 3.0, 0.05),
        # Wie tief die Linie bei Silenz abfaellt (1.0 = fast perfekte Ruhe)
        'calm_factor': (0.85, 0.0, 1.0, 0.05),
        # 0 = kuehles Silber, 1 = warmes Gold (Linien-Grundfarbe)
        'accent_warmth': (0.7, 0.0, 1.0, 0.05),
        # Staerke des weichen Halos um die Linie
        'glow_strength': (0.8, 0.0, 2.0, 0.05),
        # Zusaetzliche Feinheit/Textur der Linienbewegung
        'detail_texture': (0.5, 0.0, 1.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Linie": ["line_thickness", "wave_amplitude", "detail_texture", "glow_strength"],
        "Pulse": ["pulse_strength", "accent_warmth"],
        "Ruhe": ["calm_factor"],
    }

    def _setup(self):
        """Shader-Programm und Fullscreen-Quad einmalig aufbauen."""
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_voice;        // geglaettete Sprach-/Energie-Huellkurve 0..1
            uniform float u_beat;         // Beat-Huellkurve (Musik-Modus)
            uniform float u_detail;       // spectral_centroid -> Feinheit
            uniform float u_mode;         // 0.0 = Sprache, 1.0 = Musik
            uniform float u_pulse_time;   // Startzeit des letzten Pulses (<0 = keiner)
            uniform float u_pulse_seed;   // Zufalls-Seed fuer die Puls-Position
            uniform vec3 u_line_color;    // Grundfarbe der Linie
            uniform vec3 u_pulse_color;   // Gold der Betonungs-Pulse
            uniform vec3 u_music_tint;    // Chroma-Toenung (nur Musik-Modus)
            uniform float u_line_thickness;
            uniform float u_wave_amplitude;
            uniform float u_pulse_strength;
            uniform float u_calm_factor;
            uniform float u_glow_strength;
            uniform float u_detail_texture;
            uniform float u_brightness;
            out vec4 f_color;

            // Organische, mehrschichtige Auslenkung der Stimm-Linie (~[-1, 1]).
            float lineShape(float x, float t, float detail) {
                float w = 0.0;
                w += sin(x * 6.0 * detail + t * 1.7) * 0.55;
                w += sin(x * 11.0 * detail - t * 1.1 + 1.7) * 0.30;
                w += (fbm(vec2(x * 4.0 * detail + t * 0.6, t * 0.25), 3) - 0.5) * 1.2;
                return w;
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                float x = uv.x;

                // Feinheit: Musik + hoher Centroid -> feinere Textur
                float detail = mix(1.0, 2.4,
                    clamp(u_detail * (0.35 + 0.65 * u_mode) + u_detail_texture * 0.5, 0.0, 1.0));

                // Amplituden-Huellkurve: Bildmitte voller, Raender ruhiger
                float edge = smoothstep(0.02, 0.14, x) * smoothstep(0.98, 0.86, x);
                float env = pow(max(u_voice, 0.0), 1.2);
                float amp = u_wave_amplitude * env * (0.30 + 0.70 * edge);
                // Musik: Beats druecken die Amplitude kurz hoch
                amp *= 1.0 + u_beat * 1.0 * u_mode;

                float yc = 0.5 + lineShape(x, u_time, detail) * amp;
                float d = abs(uv.y - yc);

                // Kern-Linie: pixelgenau anti-aliasiert (aafill nutzt fwidth)
                float thick = u_line_thickness / u_resolution.y;
                float core = aafill(d - thick * 0.5);
                // Weicher Halo um die Linie
                float glowR = thick * 5.0 + 0.010 + env * 0.030;
                float halo = exp(-d * d / (glowR * glowR)) * u_glow_strength;

                // Ruhe-Helligkeit: bei Silenz faellt die Linie fast auf Null
                float rest = 1.0 - u_calm_factor * 0.94;
                float bright = (rest + (1.0 - rest) * env) * u_brightness;
                // Musik: Beat laesst die Linie aufleuchten
                bright *= 1.0 + u_beat * 1.3 * u_mode;

                // Linienfarbe: Sprache streng warm/golden, Musik dezent chroma-getoent
                vec3 lineCol = mix(u_line_color, u_music_tint, u_mode * 0.45);

                vec3 col = vec3(0.0);
                col += lineCol * core * bright * 1.7;
                col += lineCol * halo * bright * 0.5;

                // --- Goldener Betonungs-Puls entlang der Linie ---
                float age = u_time - u_pulse_time;
                if (age > 0.0 && age < 1.6) {
                    // Position aus Seed: Puls entsteht an wechselnder Stelle
                    float ppos = 0.15 + 0.7 * hash12(vec2(u_pulse_seed, 7.31));
                    // Puls laeuft mit zunehmendem Alter sanft auseinander
                    float spread = 0.025 + age * 0.22;
                    float spatial = exp(-pow((x - ppos) / spread, 2.0));
                    float fade = exp(-age * 2.4);
                    float pd = abs(uv.y - yc);
                    float onLine = exp(-pd * pd / (0.018 * 0.018));
                    float around = exp(-pd * pd / (0.075 * 0.075));
                    col += u_pulse_color * spatial * fade * onLine
                           * u_pulse_strength * (0.6 + env);
                    col += u_pulse_color * spatial * fade * around
                           * u_pulse_strength * 0.30;
                }

                // Untere Bildzone ruhig/dunkel halten (dort liegen Zitat-Overlays)
                float calmMask = smoothstep(0.10, 0.38, uv.y);
                col *= mix(0.15, 1.0, calmMask);

                // HDR-Ausgabe: kein clamp, Tonemapping macht der Renderer
                f_color = vec4(col, 1.0);
            }
            """,
            includes=(LYGIA_NOISE_GLSL, SHADER_COMMON_GLSL),
        )
        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

        # Laufzeit-Zustand: Huellkurve, Clarity-Flanke, Puls-Zeit/Seed
        self._env = 0.0
        self._prev_clarity = 0.0
        self._pulse_time = -100.0
        self._pulse_seed = 0.0

    def render(self, features: dict, time: float):
        """Rendert einen Frame mit aktuellen Audio-Features.

        Args:
            features: Dictionary mit Audio-Feature-Arrays.
            time: Aktuelle Zeit in Sekunden.
        """
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        is_music = 1.0 if mode == "music" else 0.0

        voice = f.get("voice_band", f["rms"])
        clarity = f.get("voice_clarity", f["rms"])
        transient = f.get("transient", f["onset"])
        beat = f.get("beat_intensity", f["onset"])

        # Huellkurve: Sprache = Sprachband, Musik = Energie + Beat-Anteil
        target = voice if not is_music else min(1.0, f["rms"] * 0.9 + beat * 0.4)

        # Anstieg schnell, Abfall langsam -> weiche, edle Bewegung
        if target > self._env:
            self._env += (target - self._env) * 0.35
        else:
            self._env += (target - self._env) * 0.06

        # Betonungs-Erkennung: Transiente oder steigende Sprach-Klarheit
        clarity_rise = max(0.0, clarity - self._prev_clarity)
        self._prev_clarity = clarity
        if is_music:
            emphasis = max(transient, f["onset"] * 0.9)
            threshold = 0.38
        else:
            emphasis = max(transient * 0.8, clarity_rise * 4.0)
            threshold = 0.30
        # Rate-Limit, damit Pulse einzeln lesbar bleiben
        if emphasis > threshold and (time - self._pulse_time) > 0.35:
            self._pulse_time = time
            self._pulse_seed += 1.0

        # Farben: warme Linie (Silber<->Gold ueber accent_warmth),
        # Puls immer sattes Gold, Musik-Toenung aus der Chroma-Farbe
        warmth = float(self.params["accent_warmth"])
        silver = (0.72, 0.76, 0.85)
        gold = (1.0, 0.84, 0.52)
        line_rgb = tuple(silver[i] + (gold[i] - silver[i]) * warmth for i in range(3))
        pulse_rgb = (1.0, 0.76, 0.32)
        music_rgb = self._chroma_to_color(f["chroma"])

        self.prog["u_resolution"].value = (self.width, self.height)
        self.prog["u_time"].value = float(time)
        self.prog["u_voice"].value = float(self._env)
        self.prog["u_beat"].value = float(beat)
        self.prog["u_detail"].value = float(f["spectral_centroid"])
        self.prog["u_mode"].value = float(is_music)
        self.prog["u_pulse_time"].value = float(self._pulse_time)
        self.prog["u_pulse_seed"].value = float(self._pulse_seed)
        self.prog["u_line_color"].value = line_rgb
        self.prog["u_pulse_color"].value = pulse_rgb
        self.prog["u_music_tint"].value = music_rgb
        self.prog["u_line_thickness"].value = float(self.params["line_thickness"])
        self.prog["u_wave_amplitude"].value = float(self.params["wave_amplitude"])
        self.prog["u_pulse_strength"].value = float(self.params["pulse_strength"])
        self.prog["u_calm_factor"].value = float(self.params["calm_factor"])
        self.prog["u_glow_strength"].value = float(self.params["glow_strength"])
        self.prog["u_detail_texture"].value = float(self.params["detail_texture"])
        self.prog["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
