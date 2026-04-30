"""
Abstrakte Basisklasse fuer GPU-beschleunigte Visualizer mit ModernGL.

v2.0 Features:
- Exponential Smoothing Support (EMA-gelättete Features)
- Musik/Speech/Hybrid Uniform Mapping
- Lygia-ähnliche Shader-Bibliothek (Noise, SDF, FBM)
- Chroma-Farb-Mapping mit Hue-Shift
"""

import abc
import numpy as np
import moderngl


# === Lygia-ähnliche Shader-Bibliothek ===
LYGIA_MATH_GLSL = """
float remap(float v, float i_min, float i_max, float o_min, float o_max) {
    return o_min + (v - i_min) * (o_max - o_min) / (i_max - i_min + 1e-8);
}

mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
"""

LYGIA_NOISE_GLSL = """
float hash(float n) { return fract(sin(n) * 43758.5453123); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float noise(float x) {
    float i = floor(x);
    float f = fract(x);
    return mix(hash(i), hash(i + 1.0), smoothstep(0.0, 1.0, f));
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p, int octaves) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < octaves; i++) {
        v += a * noise(p);
        p = rot * p * 2.0 + vec2(100.0);
        a *= 0.5;
    }
    return v;
}

float fbm(float x, int octaves) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < octaves; i++) {
        v += a * noise(x);
        x *= 2.1;
        a *= 0.5;
    }
    return v;
}
"""

LYGIA_SDF_GLSL = """
float sdCircle(vec2 p, float r) { return length(p) - r; }
float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
"""

LYGIA_COLOR_GLSL = """
vec3 chromaColor(vec3 chroma, float sat, float val) {
    float angles[12];
    angles[0] = 0.0; angles[1] = 0.5236; angles[2] = 1.0472;
    angles[3] = 1.5708; angles[4] = 2.0944; angles[5] = 2.6180;
    angles[6] = 3.1416; angles[7] = 3.6652; angles[8] = 4.1888;
    angles[9] = 4.7124; angles[10] = 5.2360; angles[11] = 5.7596;
    float x = 0.0, y = 0.0;
    for (int i = 0; i < 12; i++) {
        x += chroma[i] * cos(angles[i]);
        y += chroma[i] * sin(angles[i]);
    }
    float hue = atan(y, x) / 6.28318;
    if (hue < 0.0) hue += 1.0;
    float strength = max(max(chroma[0], chroma[1]), max(max(chroma[2], chroma[3]),
                      max(max(chroma[4], chroma[5]), max(max(chroma[6], chroma[7]),
                      max(max(chroma[8], chroma[9]), max(chroma[10], chroma[11]))))));
    // Gedämpfte, elegante Farben
    return hsv2rgb(vec3(hue, sat + 0.15 * strength, val + 0.2 * strength));
}

vec3 applyChromaticAberration(sampler2D tex, vec2 uv, float amount) {
    float r = texture(tex, uv + vec2(amount, 0.0)).r;
    float g = texture(tex, uv).g;
    float b = texture(tex, uv - vec2(amount, 0.0)).b;
    return vec3(r, g, b);
}
"""


class BaseGPUVisualizer(abc.ABC):
    """Basisklasse fuer GPU-beschleunigte Visualizer mit ModernGL.

    Jeder Visualizer erhaelt einen ModernGL-Context und die Zielaufloesung.
    Das Rendern erfolgt offscreen in ein vom Aufrufer bereitgestelltes
    Framebuffer-Objekt.

    Jeder Visualizer kann Parameter haben, die via GUI-Slider angepasst werden.
    """

    # Visuelle Effekt-Parameter, die von allen GPU-Visualizern unterstuetzt werden
    EFFECTS = {
        'line_width': (0.003, 0.001, 0.02, 0.001),
        'trail_length': (0, 0, 12, 1),
        'trail_decay': (0.7, 0.1, 0.95, 0.05),
        'brightness': (1.0, 0.5, 2.0, 0.05),
    }

    # Override in subclasses: {param_name: (default, min, max, step)}
    PARAMS = {}

    # Lygia Shader Snippets (in Subclasses via f-String einbinden)
    LYGIA_MATH = LYGIA_MATH_GLSL
    LYGIA_NOISE = LYGIA_NOISE_GLSL
    LYGIA_SDF = LYGIA_SDF_GLSL
    LYGIA_COLOR = LYGIA_COLOR_GLSL

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        # Merge EFFECTS und PARAMS (PARAMS ueberschreiben EFFECTS bei Duplikaten)
        self.params = {k: v[0] for k, v in self.EFFECTS.items()}
        self.params.update({k: v[0] for k, v in self.PARAMS.items()})
        self._setup()

    def set_params(self, params: dict):
        """Aktualisiert die Visualizer-Parameter."""
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
        self._on_params_changed()

    def _on_params_changed(self):
        """Wird aufgerufen, wenn sich Parameter aendern."""
        pass

    @abc.abstractmethod
    def _setup(self):
        """Einmalige Initialisierung: Shader, VAOs, Texturen erstellen."""
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, features: dict, time: float):
        """Rendert einen einzelnen Frame.

        Args:
            features: Dictionary mit Audio-Features fuer alle Frames.
            time: Aktuelle Zeit in Sekunden.
        """
        raise NotImplementedError

    def _get_feature_at_frame(self, features: dict, frame_idx: int) -> dict:
        """Hilfsmethode: Extrahiert die Features fuer einen bestimmten Frame.

        Args:
            features: Dictionary mit den gesamten Audio-Features.
            frame_idx: Index des gewuenschten Frames.

        Returns:
            Dictionary mit den skalaren Features fuer den angegebenen Frame.
        """
        frame_idx = max(0, min(frame_idx, features.get("frame_count", 0) - 1))

        chroma = features["chroma"]
        if len(chroma.shape) > 1 and chroma.shape[0] == 12:
            chroma_frame = chroma[:, frame_idx]
        elif len(chroma.shape) > 1 and chroma.shape[1] == 12:
            chroma_frame = chroma[frame_idx, :]
        else:
            chroma_frame = chroma[frame_idx]

        result = {
            "rms": float(features["rms"][frame_idx]),
            "onset": float(features["onset"][frame_idx]),
            "chroma": chroma_frame,
            "spectral_centroid": float(features["spectral_centroid"][frame_idx]),
            "mode": features.get("mode", "music"),
        }

        # Neue Features (falls vorhanden)
        if "transient" in features and len(features["transient"]) > 0:
            result["transient"] = float(features["transient"][frame_idx])
        else:
            result["transient"] = result["onset"] * 1.5  # Fallback

        if "voice_clarity" in features and len(features["voice_clarity"]) > 0:
            result["voice_clarity"] = float(features["voice_clarity"][frame_idx])
        else:
            result["voice_clarity"] = result["rms"]  # Fallback

        if "voice_band" in features and len(features["voice_band"]) > 0:
            result["voice_band"] = float(features["voice_band"][frame_idx])
        else:
            result["voice_band"] = result.get("voice_clarity", result["rms"])  # Fallback

        if "tempo" in features:
            result["tempo"] = float(features["tempo"])
        else:
            result["tempo"] = 120.0

        return result

    def _map_features_to_uniforms(self, f: dict, mode: str = None) -> dict:
        """Mappt Audio-Features auf Uniform-Werte basierend auf dem Modus.

        Musik-Modus: Fokus auf Transienten, Onset, Beat
        Podcast-Modus: Fokus auf Voice-Clarity, RMS, sanfte Uebergaenge
        Hybrid: Kombination beider

        Returns:
            Dictionary mit uniform-Namen und Werten.
        """
        if mode is None:
            mode = f.get("mode", "music")

        if mode == "music":
            return {
                "u_energy": f["rms"],
                "u_beat": f["onset"],
                "u_impact": f.get("transient", f["onset"]),
                "u_detail": f["spectral_centroid"],
                "u_flow": f["rms"] * 0.3,  # Musik: wenig Flow
                "u_chroma": f["chroma"],
            }
        elif mode == "speech":
            return {
                "u_energy": f["rms"] * 0.7,
                "u_beat": f["onset"] * 0.3,
                "u_impact": f.get("transient", f["onset"]) * 0.2,
                "u_detail": f["spectral_centroid"] * 0.5,
                "u_flow": f.get("voice_band", f.get("voice_clarity", f["rms"])),  # Podcast: Voice-Band > Voice-Clarity > RMS
                "u_chroma": f["chroma"],
            }
        else:  # hybrid
            return {
                "u_energy": f["rms"],
                "u_beat": f["onset"] * 0.7,
                "u_impact": f.get("transient", f["onset"]) * 0.7,
                "u_detail": f["spectral_centroid"],
                "u_flow": f.get("voice_clarity", f["rms"]) * 0.5,
                "u_chroma": f["chroma"],
            }

    @staticmethod
    def _chroma_to_color(chroma: np.ndarray) -> tuple:
        """Wandelt ein Chroma-Vektor in eine RGB-Farbe um."""
        chroma = np.asarray(chroma).flatten()
        if chroma.size < 12:
            chroma = np.pad(chroma, (0, 12 - chroma.size))

        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        x = np.sum(chroma * np.cos(angles))
        y = np.sum(chroma * np.sin(angles))

        hue = np.arctan2(y, x) / (2.0 * np.pi)
        if hue < 0:
            hue += 1.0

        strength = float(np.max(chroma))
        saturation = 0.7 + 0.3 * strength
        value = 0.6 + 0.4 * strength

        return BaseGPUVisualizer._hsv_to_rgb(hue, saturation, value)

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
        """Konvertiert HSV nach RGB."""
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        i = i % 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        return (v, p, q)
