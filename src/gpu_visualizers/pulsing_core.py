"""
Pulsing Core - Neon-Tunnel-Visualizer (Redesign).

Konzept: Flug durch einen Neon-Tunnel. Konzentrische Ring-/Polygon-Linien
(im Log-Radius-Raum => echte Tiefenperspektive) wandern aus der Bildmitte
nach aussen. Abwechselnde Primaer-/Sekundaerfarben, additive Leuchtlinien
auf nahezu schwarzem Grund (Hintergrundbild-Harmonie: dunkle Bereiche
bleiben dunkel, das Compositing legt die Visualisierung ueber das Bild).

Audio-Reaktionen (Musik):
- Bass/Onset      = Expansions-Stoss (Tunnel-Kick) + Helligkeits-Sweep + Hue-Kick
- Treble (Centroid) = feine Funken/Glitzer zwischen den Ringen
- RMS/Energy      = Wandergeschwindigkeit und Linien-Helligkeit
- Chroma          = Farbbasis (rotiert langsam mit der Tonart)

Sprach-Modus (features["mode"] == "speech"):
- Geschwindigkeit/Helligkeit folgen voice_band (Pausen => fast Stillstand)
- Statt Beat-Stoesssen ein sanftes Atmen ueber voice_clarity
- Gleiche Optik, andere Empfindlichkeit.

HDR-Ausgabe ohne lokales Clamp (Tonemapping macht zentral der Renderer).
"""

import math

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


class PulsingCoreGPU(BaseGPUVisualizer):
    """Neon-Tunnel: konzentrische Ring-/Polygon-Linien mit Tiefenflug."""

    PARAMS = {
        'tunnel_speed': (1.0, 0.0, 4.0, 0.05),
        'ring_density': (6.0, 2.0, 16.0, 0.5),
        'ring_width': (0.05, 0.01, 0.3, 0.01),
        'bass_boost': (1.2, 0.0, 3.0, 0.1),
        'treble_sparkle': (0.8, 0.0, 2.0, 0.1),
        'hue_shift': (0.0, 0.0, 1.0, 0.01),
        'polygon_sides': (0, 0, 8, 1),
        'rotation_speed': (0.08, -1.0, 1.0, 0.02),
        'depth_fade': (1.0, 0.2, 2.5, 0.1),
        'center_glow': (0.45, 0.0, 2.0, 0.05),
        'bg_brightness': (0.10, 0.0, 0.5, 0.01),
    }

    PARAMS_GROUPS = {
        "Tunnel": ["tunnel_speed", "ring_density", "ring_width", "rotation_speed"],
        "Reaktion": ["bass_boost", "treble_sparkle", "hue_shift"],
        "Form": ["polygon_sides", "depth_fade"],
        "Erscheinungsbild": ["center_glow", "bg_brightness"],
    }

    def __init__(self, ctx, width, height):
        # Laufzeit-Zustand (akkumulierte Tunnel-Phase, Beat-Huellen)
        self._phase = 0.0        # akkumulierte Tunnel-Wanderung
        self._shock = 0.0        # abklingender Bass-Stoss
        self._sweep = 0.0        # Position des Helligkeits-Sweeps (0..1)
        self._hue_kick = 0.0     # kurzzeitige Farbrotation auf Beats
        self._last_time = 0.0
        super().__init__(ctx, width, height)

    def _setup(self):
        """Initialisiert Shader, VBO und VAO fuer den Fullscreen-Quad."""
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_phase;          // akkumulierte Tunnel-Wanderung
            uniform float u_energy;         // RMS (Musik) bzw. voice_band (Sprache)
            uniform float u_shock;          // abklingender Bass-Stoss (0..1+)
            uniform float u_sweep;          // Sweep-Position (0..1, wrapt)
            uniform float u_treble;         // spectral_centroid
            uniform float u_breath;         // Atem-Huelle (Sprache) / Energie (Musik)
            uniform vec3 u_color;           // Primaerfarbe
            uniform vec3 u_secondary_color; // Sekundaerfarbe
            uniform vec3 u_background_color;
            uniform float u_ring_density;
            uniform float u_ring_width;
            uniform float u_bass_boost;
            uniform float u_treble_sparkle;
            uniform float u_polygon_sides;  // 0 = Kreis, sonst N-Eck
            uniform float u_rotation_speed;
            uniform float u_depth_fade;
            uniform float u_center_glow;
            uniform float u_bg_brightness;
            uniform float u_brightness;
            out vec4 f_color;

            void main() {
                // Zentriert, aspektkorrigiert -> Kreise bleiben rund
                vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
                uv.x *= u_resolution.x / u_resolution.y;
                float r = length(uv);
                float ang = atan(uv.y, uv.x) + u_time * u_rotation_speed;

                // Optionale Polygon-Form: radialer Rueckzug auf N-Eck-Kanten
                float rr = r;
                if (u_polygon_sides >= 3.0) {
                    float seg = 6.2831853 / u_polygon_sides;
                    rr = r * cos(mod(ang, seg) - seg * 0.5) / cos(seg * 0.5);
                }

                // Log-Radius => gleiche Abstaende wirken wie Tiefenperspektive
                float lr = log(max(rr, 1e-4));
                float rid = lr * u_ring_density - u_phase;
                float fr = fract(rid);
                float dr = min(fr, 1.0 - fr);            // 0 auf der Ringlinie
                float ringIndex = floor(rid + 0.5);

                // Tiefen-Staffelung: Ferne (Mitte) dunkel, Naehe hell
                float depthFade = smoothstep(-2.4, -0.3, lr * u_depth_fade);
                float edgeFade = 1.0 - smoothstep(1.15, 1.95, r);

                vec3 col = u_background_color * u_bg_brightness;

                // === Ring-Linien (pixelgenaues Anti-Aliasing) ===
                float line = aafill(dr - u_ring_width);
                // Abwechselnde Primaer-/Sekundaerfarbe pro Ring
                float parity = mod(ringIndex, 2.0);
                vec3 ringColor = mix(u_color, u_secondary_color, parity);
                // Dezentes Eigenleben der Ringe
                float flicker = 0.9 + 0.1 * sin(ringIndex * 1.7 + u_time * 0.7);
                // Farb-Basis bleibt gesaettigt; der Bass-Stoss kommt als
                // separater weisslicher Flash dazu (kein Ausbleichen via ACES)
                float baseGain = (0.30 + u_energy * 0.8)
                               * (0.55 + 0.45 * u_breath) * flicker;
                float flash = u_shock * u_bass_boost * 0.55;
                col += ringColor * line * depthFade * edgeFade * baseGain;
                col += mix(ringColor, vec3(1.0), 0.5)
                     * line * depthFade * edgeFade * flash;

                // === Bass-Sweep: heller Puls, der nach aussen laeuft ===
                float bp = fract(u_sweep);
                float bd = min(abs(fr - bp), 1.0 - abs(fr - bp));
                float sweep = exp(-bd * bd * 160.0) * u_shock;
                col += mix(u_color, u_secondary_color, 0.5)
                     * sweep * depthFade * edgeFade * u_bass_boost * 1.2;

                // === Treble-Funken zwischen den Ringen ===
                // Punktfoermige Glitzer: Zelle in (Ring, Winkel), lokaler Abfall
                // um die Zellmitte => kleine Punkte statt Flaechen.
                float af = ang / 6.2831853 * 96.0;
                float h = hash12(vec2(ringIndex, floor(af)));
                vec2 cp = vec2(fr, fract(af)) - 0.5;   // Mitte = zwischen den Linien
                float loc = exp(-dot(cp, cp) * 22.0);
                float tw = 0.5 + 0.5 * sin(u_time * (3.0 + h * 7.0) + h * 40.0);
                float spark = pow(h, 22.0) * loc * tw;
                col += mix(u_secondary_color, vec3(1.0), 0.6)
                     * spark * u_treble_sparkle * (0.15 + u_treble * 1.6)
                     * depthFade * edgeFade * 2.5;

                // === Fernlicht am Fluchtpunkt (dezent, kein Glow-Blob) ===
                col += mix(u_color, vec3(1.0), 0.5)
                     * exp(-r * r * 40.0) * u_center_glow
                     * (0.35 + u_energy * 0.8 + u_shock * 0.5);

                // HDR-Ausgabe: zentrales ACES-Tonemapping im Renderer
                col = max(col, 0.0) * u_brightness;
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

    def render(self, features: dict, time: float):
        """Rendert einen Frame; Modus steuert nur die Empfindlichkeit."""
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        rms = f["rms"]
        onset = f["onset"]
        beat = f.get("beat_intensity", onset)
        centroid = f["spectral_centroid"]
        voice_band = f.get("voice_band", rms)
        voice_clarity = f.get("voice_clarity", rms)

        # Zeitdelta (robust gegen Spruenge/Reset)
        dt = time - self._last_time
        self._last_time = time
        dt = max(0.0, min(dt, 0.1))

        p = self.params

        if mode == "speech":
            # Sprache: sanftes Atmen, Pausen beruhigen den Tunnel fast zum Stillstand
            drive = 0.03 + voice_band * 0.40
            breath = min(1.0, voice_clarity * 1.2)
            energy = voice_band * 0.85
            shock_target = 0.0
            kick_target = 0.0
        else:
            # Musik: RMS treibt Geschwindigkeit, Beats stossen den Tunnel an
            drive = 0.25 + rms * 0.85
            breath = 0.4 + rms * 0.6
            energy = rms
            shock_target = max(onset, beat)
            kick_target = max(0.0, shock_target - 0.5) * 0.25

        # Akkumulierte Zustaende (Huellkurven)
        self._phase += dt * float(p["tunnel_speed"]) * drive * 2.0
        self._shock = max(shock_target, self._shock * math.exp(-dt * 3.5))
        self._sweep = (self._sweep + dt * 2.5) % 1.0
        self._hue_kick = max(kick_target, self._hue_kick * math.exp(-dt * 1.5))

        # Primaerfarbe aus Chroma, plus manuellem Hue-Shift und Beat-Hue-Kick
        color = self._chroma_to_color(f["chroma"])
        h, s, v = self._rgb_to_hsv(*color)
        h = (h + float(p["hue_shift"]) + self._hue_kick) % 1.0
        # Mindest-Saettigung/-Helligkeit, damit Linien nicht grau wirken
        s = max(s, 0.55)
        v = max(v, 0.75)
        primary = self._hsv_to_rgb(h, s, v)

        # Sekundaerfarbe: Hex-Param oder komplementaer versetzter Hue
        secondary_param = p.get("secondary_color")
        if isinstance(secondary_param, str) and secondary_param.startswith('#'):
            try:
                sr, sg, sb = self._hex_to_rgb(secondary_param)
                sh, ss, sv = self._rgb_to_hsv(sr, sg, sb)
                secondary = self._hsv_to_rgb(
                    (sh + float(p["hue_shift"]) + self._hue_kick) % 1.0, ss, sv
                )
            except Exception:
                secondary = self._hsv_to_rgb((h + 0.45) % 1.0, s, v)
        else:
            secondary = self._hsv_to_rgb((h + 0.45) % 1.0, s, v)

        background_param = p.get("background_color")
        if isinstance(background_param, str) and background_param.startswith('#'):
            try:
                background = self._hex_to_rgb(background_param)
            except Exception:
                background = (0.02, 0.02, 0.04)
        else:
            background = (0.02, 0.02, 0.04)

        prog = self.prog
        prog["u_resolution"].value = (self.width, self.height)
        prog["u_time"].value = float(time)
        prog["u_phase"].value = float(self._phase)
        prog["u_energy"].value = float(energy)
        prog["u_shock"].value = float(self._shock)
        prog["u_sweep"].value = float(self._sweep)
        prog["u_treble"].value = float(centroid)
        prog["u_breath"].value = float(breath)
        prog["u_color"].value = primary
        prog["u_secondary_color"].value = secondary
        prog["u_background_color"].value = background
        prog["u_ring_density"].value = float(p["ring_density"])
        prog["u_ring_width"].value = float(p["ring_width"])
        prog["u_bass_boost"].value = float(p["bass_boost"])
        prog["u_treble_sparkle"].value = float(p["treble_sparkle"])
        prog["u_polygon_sides"].value = float(p["polygon_sides"])
        prog["u_rotation_speed"].value = float(p["rotation_speed"])
        prog["u_depth_fade"].value = float(p["depth_fade"])
        prog["u_center_glow"].value = float(p["center_glow"])
        prog["u_bg_brightness"].value = float(p["bg_brightness"])
        prog["u_brightness"].value = float(p.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
