"""
Scissor Lattice - GPU-Visualizer "Scherengitter".

Gekreuzte Streben, die in Reihen haengen und sich alle gemeinsam oeffnen
und schliessen — wie ein Scherengitter oder ein Nuernberger Schere.
Archetyp Mechanik/Zwangsfuehrung: zum ersten Mal in der Sammlung bewegt
sich etwas nicht frei, sondern gefuehrt.

## Was das Bild traegt: ein einziger Freiheitsgrad

Ein Scherengitter hat genau **einen** Freiheitsgrad. Aus dem
Oeffnungswinkel theta folgt alles andere zwangslaeufig:

    Zellenbreite  b = 2 * L * sin(theta)
    Zellenhoehe   h = 2 * L * cos(theta)

Oeffnet sich das Gitter, wird es breiter UND flacher — beides zugleich,
weil die Strebenlaenge L fest ist. Genau diese Kopplung liest das Auge
als Mechanik. Ohne sie waeren es nur zappelnde Striche.

Der Audio-Pegel steuert theta ueber eine Feder mit Daempfung: auf einem
Schlag schnappt das Gitter auf und schwingt kurz nach, statt dem Pegel
traege zu folgen. Das ist der mechanische Eindruck.

Gezeichnet werden nur Streben und Gelenke — dazwischen bleibt es
schwarz, ein Hintergrundbild bleibt sichtbar.

Audio:
- rms = Oeffnungswinkel (geschlossen und hoch bis weit und flach)
- beat_intensity = Stoss auf die Feder (Aufschnappen mit Nachschwingen)
- transient = Aufblitzen der Gelenke
- chroma = Farbe je Zelle; die Faerbung wandert mit dem Gitter
- spectral_centroid = Strebendicke
- zero_crossing_rate = Zittern in den Gelenken
- Sprach-Modus: ruhiges Atmen ohne Schnappen

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

# Grenzen des Oeffnungswinkels in Radiant (0 = ganz zu, pi/2 = flach)
THETA_MIN = 0.22
THETA_MAX = 1.28


class ScissorLatticeGPU(BaseGPUVisualizer):
    """Scherengitter mit einem Freiheitsgrad, gefedert auf den Beat."""

    COLOR_PARAMS = {
        'color_mode': 'chroma',
        'base_hue': 0.45,
        'color_saturation': 0.75,
    }

    PARAMS = {
        'row_count': (3, 1, 6, 1),
        'bar_length': (0.16, 0.06, 0.32, 0.01),
        'bar_width': (0.0032, 0.0008, 0.010, 0.0002),
        'joint_size': (0.55, 0.0, 2.5, 0.05),
        'open_response': (0.75, 0.0, 1.5, 0.05),
        'spring': (1.0, 0.0, 2.5, 0.05),
        'damping': (0.45, 0.10, 1.0, 0.05),
        'drift_speed': (0.10, 0.0, 0.8, 0.02),
        'row_offset': (0.5, 0.0, 1.0, 0.05),
        'glow': (0.40, 0.0, 2.0, 0.05),
    }

    PARAMS_GROUPS = {
        "Gitter": ["row_count", "bar_length", "row_offset"],
        "Streben": ["bar_width", "joint_size", "glow"],
        "Mechanik": ["open_response", "spring", "damping"],
        "Bewegung": ["drift_speed"],
    }

    def _setup(self):
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_theta;         // Oeffnungswinkel (der Freiheitsgrad)
            uniform float u_drift;         // seitliches Wandern des Gitters
            uniform float u_energy;
            uniform float u_beat;
            uniform float u_impact;
            uniform float u_centroid;
            uniform float u_zcr;
            uniform float u_speech;
            uniform float u_chroma[12];
            uniform vec3 u_color_a;
            uniform vec3 u_color_b;
            uniform float u_row_count;
            uniform float u_bar_length;
            uniform float u_bar_width;
            uniform float u_joint_size;
            uniform float u_row_offset;
            uniform float u_glow;
            uniform float u_brightness;
            out vec4 f_color;

            // WICHTIG: kein fwidth/aastep hier. Die Funktion wird innerhalb
            // einer Schleife mit pixelabhaengigem `continue` aufgerufen — in
            // divergentem Kontrollfluss sind Ableitungen laut GLSL undefiniert
            // und liefern je nach Durchlauf andere Werte. Die Kantenbreite
            // kommt deshalb aus der Aufloesung: der Raum ist auf die Bildhoehe
            // normiert, ein Pixel ist also genau 1.0 / u_resolution.y.
            float lineAt(float dist, float width, float px) {
                float core = 1.0 - smoothstep(width - px, width + px, dist);
                float halo = exp(-dist / max(width * 4.0, 1e-5)) * 0.26;
                return core + halo;
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                uv.y = 1.0 - uv.y;
                float aspect = u_resolution.x / max(u_resolution.y, 1.0);
                vec2 p = vec2((uv.x - 0.5) * aspect, uv.y - 0.5);

                float L = u_bar_length;
                // Der eine Freiheitsgrad bestimmt Breite UND Hoehe zugleich
                float half_b = L * sin(u_theta);
                float half_h = L * cos(u_theta);

                float lw = u_bar_width * mix(1.5, 0.8, clamp(u_centroid, 0.0, 1.0));
                float px = 1.0 / max(u_resolution.y, 1.0);
                int rows = int(u_row_count);
                vec3 col = vec3(0.0);

                for (int rIdx = 0; rIdx < 6; rIdx++) {
                    if (rIdx >= rows) break;
                    float fr = float(rIdx);

                    // Reihen mittig um die Bildmitte stapeln
                    float row_y = (fr - (u_row_count - 1.0) * 0.5) * (2.0 * half_h);
                    float dy = p.y - row_y;
                    if (abs(dy) > half_h + lw * 8.0) continue;

                    // Benachbarte Reihen versetzt, damit ein Netz entsteht
                    float shift = u_drift + fr * u_row_offset * half_b;
                    float xs = p.x - shift;

                    float cw = 2.0 * half_b;
                    float k = floor(xs / cw + 0.5);       // Zellen-Index
                    float lx = xs - k * cw;               // Lage in der Zelle

                    // Zittern in den Gelenken
                    float jitter = (hash12(vec2(k, fr)) - 0.5) * u_zcr * 0.004;

                    // Die beiden gekreuzten Streben der Zelle
                    vec2 q = vec2(lx, dy + jitter);
                    float d1 = sdSegment(q, vec2(-half_b, -half_h), vec2(half_b, half_h));
                    float d2 = sdSegment(q, vec2(-half_b, half_h), vec2(half_b, -half_h));
                    float bars = lineAt(min(d1, d2), lw, px);

                    // Farbe wandert mit dem Zellen-Index durch die Tonleiter
                    int ci = int(mod(abs(k) + fr, 12.0));
                    float chroma = u_chroma[ci];
                    float lit = 0.30 + 1.15 * smoothstep(0.12, 0.8, chroma);
                    vec3 tint = mix(u_color_a, u_color_b, chroma);

                    // Der Schlag hellt die Streben mit auf — sonst sieht man
                    // das Aufschnappen nur an der Geometrie, nicht am Licht.
                    col += tint * bars * lit * (0.40 + 1.0 * u_energy)
                           * (1.0 + u_beat * 0.55);

                    // Gelenke: Kreuzungspunkt und die vier Enden
                    float joints = exp(-pow(length(q) * 90.0, 2.0));
                    joints += exp(-pow(length(q - vec2(-half_b, -half_h)) * 80.0, 2.0));
                    joints += exp(-pow(length(q - vec2(-half_b, half_h)) * 80.0, 2.0));
                    joints += exp(-pow(length(q - vec2(half_b, -half_h)) * 80.0, 2.0));
                    joints += exp(-pow(length(q - vec2(half_b, half_h)) * 80.0, 2.0));
                    col += mix(tint, vec3(1.0), 0.35) * joints * u_joint_size
                           * (0.30 + 0.9 * u_energy)
                           * (1.0 + u_impact * 1.6 * (1.0 - 0.5 * u_speech));
                }

                col += col * u_glow * 0.35;
                col *= 1.0 - 0.24 * u_speech;
                col = max(col, 0.0) * u_brightness;
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time) * 31.0);
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

        self._theta = THETA_MIN
        self._theta_vel = 0.0
        self._drift = 0.0
        self._last_time = None

    def _target_theta(self, energy: float) -> float:
        response = float(self.params["open_response"])
        t = float(np.clip(energy * response, 0.0, 1.0))
        return THETA_MIN + (THETA_MAX - THETA_MIN) * t

    def _advance(self, f: dict, time: float, speech: float, energy: float) -> tuple:
        """Fuehrt Feder und Drift fort.

        Der Oeffnungswinkel folgt dem Pegel nicht direkt, sondern ueber
        eine gedaempfte Feder — auf einem Schlag schnappt das Gitter auf
        und schwingt nach. Bei einem Sprung in der Zeitachse wird auf die
        Ruhelage gesetzt, damit die Vorschau reproduzierbar bleibt.
        """
        target = self._target_theta(energy)
        drift_speed = float(self.params["drift_speed"])

        if self._last_time is None or time < self._last_time - 1e-6 \
                or (time - self._last_time) > 0.5:
            self._theta = target
            self._theta_vel = 0.0
            self._drift = time * drift_speed
            self._last_time = time
            return self._theta, self._drift

        dt = min(max(time - self._last_time, 1e-4), 0.1)
        self._last_time = time

        # Federkonstante und Daempfung; Sprache laeuft deutlich traeger
        k = 60.0 * float(self.params["spring"]) * (1.0 - 0.5 * speech)
        c = 2.0 * float(self.params["damping"]) * float(np.sqrt(max(k, 1e-6)))
        beat = f.get("beat_intensity", f["onset"])
        # Der Schlag gibt der Feder einen Stoss nach oben
        kick = beat * 7.0 * (1.0 - 0.75 * speech)

        accel = k * (target - self._theta) - c * self._theta_vel + kick
        self._theta_vel += accel * dt
        self._theta += self._theta_vel * dt
        self._theta = float(np.clip(self._theta, THETA_MIN * 0.6, THETA_MAX * 1.05))

        self._drift += drift_speed * (0.4 + 0.9 * f["rms"]) * dt
        return self._theta, self._drift

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        mode = f.get("mode", "music")
        speech = 1.0 if mode == "speech" else (0.5 if mode == "hybrid" else 0.0)

        energy = f["rms"] * (1.0 - speech) + float(f.get("voice_band", f["rms"])) * speech
        theta, drift = self._advance(f, time, speech, energy)

        chroma = np.asarray(f["chroma"], dtype=np.float32).flatten()
        if chroma.size < 12:
            chroma = np.pad(chroma, (0, 12 - chroma.size))
        chroma = chroma[:12]
        peak = float(chroma.max()) if chroma.size else 0.0
        if peak > 1e-6:
            chroma = chroma / peak
        if speech > 0.0:
            voice = float(f.get("voice_band", f["rms"]))
            chroma = chroma * (1.0 - speech) + (0.28 + 0.55 * voice) * speech

        color_a = self._chroma_to_color(f["chroma"])
        hue = self._color_to_hue(color_a)
        sat = float(self.params.get("color_saturation", 0.75))
        color_b = self._hsv_to_rgb((hue + 0.30) % 1.0, min(1.0, sat), 1.15)

        beat = f.get("beat_intensity", f["onset"]) * (1.0 - 0.6 * speech)
        impact = f.get("transient", f["onset"])

        p = self.prog
        p["u_resolution"].value = (self.width, self.height)
        p["u_time"].value = float(time)
        p["u_theta"].value = float(theta)
        p["u_drift"].value = float(drift)
        p["u_energy"].value = float(energy)
        p["u_beat"].value = float(beat)
        p["u_impact"].value = float(impact)
        p["u_centroid"].value = float(f["spectral_centroid"])
        p["u_zcr"].value = float(f.get("zero_crossing_rate", 0.0))
        p["u_speech"].value = float(speech)
        p["u_chroma"].write(chroma.astype(np.float32).tobytes())
        p["u_color_a"].value = tuple(color_a)
        p["u_color_b"].value = tuple(color_b)
        p["u_row_count"].value = float(self.params["row_count"])
        p["u_bar_length"].value = float(self.params["bar_length"])
        p["u_bar_width"].value = float(self.params["bar_width"])
        p["u_joint_size"].value = float(self.params["joint_size"])
        p["u_row_offset"].value = float(self.params["row_offset"])
        p["u_glow"].value = float(self.params["glow"])
        p["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
