"""
Orchestral Swell - GPU-Visualizer "Swell-Vorhaenge".

Breite, vertikale Licht-Vorhaenge, die wie Buehnenlicht von unten aufsteigen
und langsam anschwellen/verklingen — majestaetisch, langsam, orchestrisch.

Design:
- Mehrere ueberlagerte vertikale Farb-Saeulen mit weichen Kanten und
  fbm-Textur, tiefengestaffelt (vordere Vorhaenge kraeftiger, hintere diffuser)
- Stark geglaettete RMS-/Voice-Huellkurve steuert Hoehe + Leuchtdichte
  (langsames Anschwellen, kein Zucken)
- Beats = breites, sanftes Aufhellen (Swell, kein Strobe)
- spectral_centroid = Feinheit der Vorhang-Textur
- Chroma = Warm/Kalt-Farbstimmung
- transient = kurzer Glanzstreifen, der nach oben laeuft
- Sprach-Modus: sehr ruhiges Schwellen auf Phrasen (voice_band), Pausen
  lassen die Vorhaenge sanft in sich zusammensinken

HDR-Ausgabe (kein clamp) — Tonemapping macht zentral der Renderer.
Deterministisch: alle Huellkurven werden kausal aus den Feature-Arrays
berechnet und pro Feature-Satz gecacht.
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


class OrchestralSwellGPU(BaseGPUVisualizer):
    """Swell-Vorhaenge: vertikale Licht-Saeulen mit orchestrischem Schwellen."""

    COLOR_PARAMS = {
        'color_mode': 'warm',     # Orchestral-Look: warme Toene als Default
        'base_hue': 0.09,         # 0.0-1.0, nur fuer 'fixed'
        'color_saturation': 0.7,  # 0.0-1.0
    }

    PARAMS = {
        'curtain_count': (6, 2, 10, 1),
        'swell_response': (1.0, 0.2, 2.5, 0.05),
        'rise_speed': (0.25, 0.0, 1.0, 0.05),
        'texture_detail': (1.0, 0.2, 3.0, 0.1),
        'warmth': (0.6, 0.0, 1.0, 0.05),
        'curtain_softness': (0.70, 0.1, 1.5, 0.05),
        'height_max': (0.80, 0.3, 0.95, 0.02),
        'beat_glow': (0.5, 0.0, 1.5, 0.05),
        'glint_strength': (0.8, 0.0, 2.0, 0.05),
        'bg_brightness': (0.10, 0.0, 0.6, 0.01),
    }

    PARAMS_GROUPS = {
        "Vorhaenge": ["curtain_count", "curtain_softness", "height_max"],
        "Bewegung & Textur": ["rise_speed", "texture_detail"],
        "Reaktion": ["swell_response", "beat_glow", "glint_strength"],
        "Farbe & Hintergrund": ["warmth", "bg_brightness"],
    }

    def _setup(self):
        self._env_cache = None  # Huellkurven-Cache (pro Feature-Satz)
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_swell;        // stark geglaettete Dynamik-Huellkurve
            uniform float u_beat;         // sanftes Beat-Aufhellen (Decay-Envelope)
            uniform float u_detail;       // spectral_centroid (geglaettet)
            uniform float u_glint_pos;    // Position des Glanzstreifens (0..1.2)
            uniform float u_glint;        // Intensitaet des Glanzstreifens
            uniform vec3 u_warm_color;
            uniform vec3 u_cold_color;
            uniform vec3 u_background_color;
            uniform float u_curtain_count;
            uniform float u_swell_response;
            uniform float u_rise_speed;
            uniform float u_texture_detail;
            uniform float u_warmth;
            uniform float u_curtain_softness;
            uniform float u_height_max;
            uniform float u_beat_glow;
            uniform float u_glint_strength;
            uniform float u_bg_brightness;
            uniform float u_brightness;

            out vec4 f_color;

            const int MAX_CURTAINS = 10;

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                // Im Renderer liegt gl_FragCoord.y = 0 OBEN im fertigen Bild.
                // Fuer "von unten aufsteigend" muss y gespiegelt werden.
                uv.y = 1.0 - uv.y;
                float aspect = u_resolution.x / u_resolution.y;
                // p.x in [0, aspect], p.y in [0, 1] — Vorhaenge am Boden verankert
                vec2 p = vec2(uv.x * aspect, uv.y);

                // Sehr dunkler, warmer Grund — obere Bildhaelfte und Raender
                // bleiben dunkel (Hintergrundbild-Harmonie, Schwarz = transparent)
                vec3 col = u_background_color * u_bg_brightness
                         * (0.55 + 0.45 * uv.y) * (1.0 - 0.35 * uv.y * uv.y);

                // Anschwellen: Hoehe und Leuchtdichte folgen der Huellkurve,
                // Beats hellen breit und sanft auf (kein Strobe)
                float swell = clamp(u_swell * u_swell_response, 0.0, 1.5);
                float glowAll = 1.0 + u_beat * u_beat_glow;

                float curtainSum = 0.0;  // Maske aller Vorhaenge (fuer Glanzstreifen)

                for (int i = 0; i < MAX_CURTAINS; i++) {
                    if (float(i) >= u_curtain_count) break;
                    float fi = float(i);
                    float seed = fi * 7.31 + 1.7;

                    // Tiefenstaffelung: 0 = ganz hinten (diffus), 1 = vorn (kraeftig)
                    float depth = 0.25 + 0.75 * hash(seed * 3.17);

                    // Gleichmaessige Verteilung mit organischem Jitter
                    float cx = (fi + 0.5) / u_curtain_count;
                    cx += (hash(seed * 5.03) - 0.5) * 0.35 / u_curtain_count;
                    cx = clamp(cx, 0.04, 0.96) * aspect;
                    // Langsame horizontale Atembewegung
                    cx += sin(u_time * 0.07 + seed) * 0.02;

                    // Breite: hintere Vorhaenge breiter und diffuser
                    float w = mix(0.30, 0.14, depth) * u_curtain_softness;

                    // Hoehe: Basis + Schwellen, pro Vorhang variiert,
                    // sehr langsames eigenes Auf und Ab
                    // Deutlich unterschiedliche Grundhoehen -> keine flache Kante
                    float hBase = 0.22 + 0.34 * hash(seed * 9.11);
                    float hSwell = (u_height_max - hBase) * clamp(swell * (0.75 + 0.5 * hash(seed * 4.7)), 0.0, 1.3);
                    float h = hBase + hSwell + sin(u_time * 0.11 + seed * 2.0) * 0.03;

                    // Ausgefranste Oberkante via fbm (langsam wandernd)
                    float edge = (fbm(vec2(p.x * 2.5 + seed * 10.0, u_time * 0.04), 3) - 0.5) * 0.18;
                    float hEdge = h + edge * (0.4 + 0.6 * swell);

                    // Vertikale Maske: von unten bis zur weichen Oberkante
                    float topSoft = 0.10 + 0.10 * (1.0 - depth);
                    float vmask = 1.0 - smoothstep(hEdge - topSoft, hEdge, p.y);
                    // Unteres Drittel ruhig halten (Zitat-Zone): sanft abgedimmt
                    vmask *= mix(0.35, 1.0, smoothstep(0.0, 0.28, p.y));
                    // Lichtstrahl-Charakter: nach oben hin duenner/schwaecher,
                    // sonst wirkt der Vorhang wie eine flache Nebelwand
                    vmask *= mix(1.0, 0.30, clamp(p.y / max(hEdge, 0.05), 0.0, 1.0));

                    // Horizontale Maske: weiche Gauß-Kanten
                    float dx = (p.x - cx) / w;
                    float hmask = exp(-dx * dx * 2.0);

                    // Aufsteigende fbm-Textur: anisotrop gestreckt = vertikale
                    // Streifen wie Licht im Bühnenhaze; Feinheit via Centroid
                    float texScale = (2.0 + u_detail * 6.0) * u_texture_detail;
                    float rise = u_rise_speed * (0.5 + 0.5 * depth);
                    float tex = fbm(vec2(p.x * texScale * 3.0 + seed * 20.0,
                                         p.y * texScale * 0.35 - u_time * rise), 4);
                    float body = 0.45 + 1.1 * tex;

                    float curtain = hmask * vmask * body;
                    curtainSum += hmask * vmask;

                    // Farbe: Warm/Kalt-Verlauf nach Param + Tiefenlage,
                    // vordere Vorhaenge saettiger und heller
                    float warmMix = clamp(u_warmth + (hash(seed * 6.7) - 0.5) * 0.5, 0.0, 1.0);
                    vec3 cCol = mix(u_cold_color, u_warm_color, warmMix);
                    float bright = mix(0.40, 1.0, depth) * (0.45 + 0.95 * swell);

                    col += cCol * curtain * bright * glowAll;
                }

                // === Glanzstreifen: laeuft bei Transienten nach oben ===
                if (u_glint > 0.001 && u_glint_pos < 1.2) {
                    float dy = (p.y - u_glint_pos) / 0.035;
                    float band = exp(-dy * dy);
                    // horizontal breit, innerhalb der Vorhaenge kraeftiger
                    float spread = 0.25 + 0.75 * clamp(curtainSum, 0.0, 1.5);
                    vec3 glintCol = mix(vec3(1.0, 0.97, 0.9), u_warm_color, 0.35);
                    col += glintCol * band * spread * u_glint * u_glint_strength;
                }

                // HDR-Ausgabe: kein clamp, kein lokales Tonemapping —
                // das uebernimmt zentral der ACES-Pass des Renderers.
                col = max(col, 0.0) * u_brightness;
                // Triangular-Dithering gegen Banding in den weichen Verlaeufen
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time * 0.5) * 100.0);

                f_color = vec4(col, 1.0);
            }
            """,
            includes=(LYGIA_MATH_GLSL, LYGIA_NOISE_GLSL, SHADER_COMMON_GLSL),
        )
        self._prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self._prog["u_resolution"].value = (self.width, self.height)
        self._vao, self._vbo = create_fullscreen_quad(self.ctx, self._prog)

    # === Huellkurven (kausal, gecacht, deterministisch) ===

    @staticmethod
    def _ema_envelope(x: np.ndarray, attack: float, release: float) -> np.ndarray:
        """Asymmetrische EMA: schneller Anstieg, langsames Abklingen.

        Erzeugt das organische 'Anschwellen und Verklingen' ohne Zucken.
        """
        out = np.empty(len(x), dtype=np.float64)
        prev = 0.0
        for i, v in enumerate(x):
            a = attack if v > prev else release
            prev += a * (float(v) - prev)
            out[i] = prev
        return out

    @staticmethod
    def _normalize_envelope(x: np.ndarray) -> np.ndarray:
        """Spreizt eine Huellkurve auf den vollen Wertebereich des Clips.

        Leise gemasterte Stuecke liefern sonst dauerhaft nur ~0.2 und die
        Vorhaenge blieben immer kurz und dunkel. Bezug ist das 95%-Perzentil
        (robust gegen einzelne Spitzen), das Ergebnis ist deterministisch,
        weil die Feature-Arrays offline vollstaendig vorliegen.
        """
        if len(x) == 0:
            return x
        ref = float(np.percentile(x, 95))
        if ref < 1e-6:
            return x
        return np.clip(x / ref, 0.0, 1.15)

    def _envelopes(self, features: dict) -> dict:
        """Berechnet alle geglaetteten Huellkurven einmal pro Feature-Satz."""
        fid = id(features)
        if self._env_cache is not None and self._env_cache.get("id") == fid:
            return self._env_cache

        fps = float(features.get("fps", 30))
        n = int(features.get("frame_count", 0))

        def _arr(key):
            a = features.get(key)
            if a is None or not hasattr(a, "__len__") or len(a) == 0:
                return np.zeros(n, dtype=np.float64)
            return np.asarray(a, dtype=np.float64).reshape(-1)[:n] if len(a) >= n else np.pad(
                np.asarray(a, dtype=np.float64).reshape(-1), (0, n - len(a)))

        rms = _arr("rms")
        voice = _arr("voice_band")
        if not np.any(voice):
            voice = rms
        centroid = _arr("spectral_centroid")
        transient = _arr("transient")

        # Anschwell-Huellkurven: Musik etwas agiler, Sprache sehr ruhig/traege.
        # Koeffizienten auf ~0.4s Anstieg / ~1.5s Abfall (Musik) bzw.
        # ~0.8s / ~3s (Sprache) bei 30 fps ausgelegt, fps-skalierend.
        a_mus, r_mus = 2.5 / fps, 0.65 / fps
        a_sp, r_sp = 1.2 / fps, 0.33 / fps
        swell_music = self._normalize_envelope(self._ema_envelope(rms, a_mus, r_mus))
        swell_voice = self._normalize_envelope(self._ema_envelope(voice, a_sp, r_sp))
        # Textur-Feinheit folgt dem Klang, aber ohne Flackern
        detail = self._ema_envelope(centroid, 1.5 / fps, 0.5 / fps)

        # Glanzstreifen: bei Transienten-Impuls startet ein Streifen unten
        # und laeuft in ~0.7s nach oben, Intensitaet klingt ab.
        glint_pos = np.zeros(n, dtype=np.float64)
        glint_int = np.zeros(n, dtype=np.float64)
        pos = 2.0
        inten = 0.0
        prev = 0.0
        speed = 1.3 / (0.7 * fps)      # Bildhoehe pro Frame
        decay = 1.0 - 2.2 / fps        # ~0.45s Halbwertszeit
        for i, v in enumerate(transient):
            if v > 0.3 and v > prev * 1.15 and inten < 0.35:
                pos = 0.0
                inten = min(1.0, 0.4 + float(v))
            prev = float(v)
            pos += speed
            inten *= decay
            glint_pos[i] = pos
            glint_int[i] = inten if pos < 1.2 else 0.0

        self._env_cache = {
            "id": fid,
            "swell_music": swell_music,
            "swell_voice": swell_voice,
            "detail": np.clip(detail, 0.0, 1.0),
            "glint_pos": glint_pos,
            "glint_int": glint_int,
        }
        return self._env_cache

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        frame_idx = int(time * features.get("fps", 30))
        frame_idx = max(0, min(frame_idx, features.get("frame_count", 1) - 1))

        env = self._envelopes(features)
        mode = f.get("mode", "music")

        # Modus = Empfindlichkeit, nicht andere Optik:
        # Sprache fahrt auf der Voice-Huellkurve, Beats/Glanz stark gedämpft.
        if mode == "speech":
            swell = float(env["swell_voice"][frame_idx])
            beat = f.get("beat_intensity", f["onset"]) * 0.3
            glint = float(env["glint_int"][frame_idx]) * 0.3
        else:
            swell = float(env["swell_music"][frame_idx])
            beat = f.get("beat_intensity", f["onset"])
            glint = float(env["glint_int"][frame_idx])

        # Farbstimmung: Primaerfarbe aus Chroma, dazu analoger Kalt-Partner.
        # warmth-Param mischt im Shader zwischen beiden.
        warm_color = self._chroma_to_color(f["chroma"])
        h, s, v = self._rgb_to_hsv(*warm_color)
        cold_color = self._hsv_to_rgb((h + 0.55) % 1.0, min(1.0, s * 0.85), v * 0.9)

        bg = self.params.get("background_color")
        if isinstance(bg, str) and bg.startswith("#"):
            try:
                bg_rgb = self._hex_to_rgb(bg)
            except Exception:
                bg_rgb = (0.03, 0.018, 0.01)
        else:
            bg_rgb = (0.03, 0.018, 0.01)

        self._prog["u_resolution"].value = (self.width, self.height)
        self._prog["u_time"].value = time
        self._prog["u_swell"].value = float(swell)
        self._prog["u_beat"].value = float(beat)
        self._prog["u_detail"].value = float(env["detail"][frame_idx])
        self._prog["u_glint_pos"].value = float(env["glint_pos"][frame_idx])
        self._prog["u_glint"].value = float(glint)
        self._prog["u_warm_color"].value = warm_color
        self._prog["u_cold_color"].value = cold_color
        self._prog["u_background_color"].value = bg_rgb
        self._prog["u_curtain_count"].value = float(self.params["curtain_count"])
        self._prog["u_swell_response"].value = float(self.params["swell_response"])
        self._prog["u_rise_speed"].value = float(self.params["rise_speed"])
        self._prog["u_texture_detail"].value = float(self.params["texture_detail"])
        self._prog["u_warmth"].value = float(self.params["warmth"])
        self._prog["u_curtain_softness"].value = float(self.params["curtain_softness"])
        self._prog["u_height_max"].value = float(self.params["height_max"])
        self._prog["u_beat_glow"].value = float(self.params["beat_glow"])
        self._prog["u_glint_strength"].value = float(self.params["glint_strength"])
        self._prog["u_bg_brightness"].value = float(self.params["bg_brightness"])
        self._prog["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self._vao.render(mode=moderngl.TRIANGLE_STRIP)
