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
vec3 chromaColor(float chroma[12], float sat, float val) {
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

# === Gemeinsame Shader-Bausteine fuer die HDR-Pipeline ===
# Anti-Aliasing (fwidth-basiert), Tonemapping, Gamma und Dithering.
# In Visualizern via f-String einbinden: f"...{self.SHADER_COMMON}..."
SHADER_COMMON_GLSL = """
// Pixelgenaue weiche Kante: 0..1-Uebergang mit Breite von ~1 Pixel.
// Ersetzt hartkodierte smoothstep-Breiten (die je Aufloesung anders aussehen).
float aastep(float threshold, float value) {
    float w = max(fwidth(value), 1e-6);
    return smoothstep(threshold - w, threshold + w, value);
}

// Weiche Fuellung fuer SDF-Werte: 1.0 innerhalb (d < 0), 0.0 ausserhalb.
float aafill(float d) {
    float w = max(fwidth(d), 1e-6);
    return clamp(0.5 - d / w, 0.0, 1.0);
}

// Kompakter 2D-Hash (bessere Verteilung als sin-Hash)
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// ACES-Tonemapping (Narkowicz-Fit): weiche Highlight-Kompression
// statt hartem Clipping, filmische S-Kurve.
vec3 tonemapACES(vec3 x) {
    x = max(x, 0.0);
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec3 linearToSrgb(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
vec3 srgbToLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }

// Triangular-Dithering (1/255-Amplitude) gegen Farb-Banding
// bei der Quantisierung von Float-Farben auf 8 Bit.
float ditherTriangular(vec2 pos, float seed) {
    float r1 = hash12(pos + seed * 337.0);
    float r2 = hash12(pos.yx * 1.371 + seed * 173.0 + 17.0);
    return (r1 + r2 - 1.0) / 255.0;
}
"""

# Standard-Vertex-Shader fuer Fullscreen-Quads (Position only)
FULLSCREEN_VERTEX_SHADER = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Standard-Vertex-Shader fuer texturierte Fullscreen-Quads
TEXTURED_VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""


def compose_fragment(body: str, includes: tuple = ()) -> str:
    """Baut einen Fragment-Shader aus #version-Header, Includes und Body zusammen.

    Args:
        body: Shader-Code ohne #version-Zeile.
        includes: GLSL-Bausteine (z.B. SHADER_COMMON_GLSL, LYGIA_NOISE_GLSL).
    """
    parts = ["#version 330"]
    parts.extend(includes)
    parts.append(body)
    return "\n".join(parts)


def create_fullscreen_quad(ctx: moderngl.Context, program, attr: str = "in_pos"):
    """Erzeugt VAO+VBO fuer einen Fullscreen-Quad (TRIANGLE_STRIP, Clip-Space).

    Returns:
        (vao, vbo) — der Aufrufer ist fuer release() verantwortlich.
    """
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype=np.float32)
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(program, [(vbo, "2f", attr)])
    return vao, vbo


def hex_to_rgb(hex_color: str) -> tuple:
    """Konvertiert 6-stelligen Hex-String nach RGB-Tupel (0.0-1.0).

    Zentrale Implementierung — frueher dreifach dupliziert in
    base.py, gpu_renderer.py und gpu_preview.py.
    """
    hex_color = hex_color.lstrip('#')
    return (
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
    )


def create_textured_quad(ctx: moderngl.Context, program,
                         pos_attr: str = "in_pos", uv_attr: str = "in_uv"):
    """Erzeugt VAO+VBO fuer einen texturierten Fullscreen-Quad (Position + UV).

    Returns:
        (vao, vbo) — der Aufrufer ist fuer release() verantwortlich.
    """
    vertices = np.array([
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype=np.float32)
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(program, [(vbo, "2f 2f", pos_attr, uv_attr)])
    return vao, vbo


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
    
    # Farb-Parameter, die von allen GPU-Visualizern unterstuetzt werden
    COLOR_PARAMS = {
        'color_mode': 'chroma',   # 'chroma' | 'fixed' | 'monochrome' | 'warm' | 'cool'
        'base_hue': 0.55,         # 0.0-1.0, nur fuer 'fixed'
        'color_saturation': 0.7,  # 0.0-1.0
    }

    # Override in subclasses: {param_name: (default, min, max, step)}
    PARAMS = {}

    # Lygia Shader Snippets (in Subclasses via f-String einbinden)
    LYGIA_MATH = LYGIA_MATH_GLSL
    LYGIA_NOISE = LYGIA_NOISE_GLSL
    LYGIA_SDF = LYGIA_SDF_GLSL
    LYGIA_COLOR = LYGIA_COLOR_GLSL
    SHADER_COMMON = SHADER_COMMON_GLSL

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        # Merge EFFECTS, COLOR_PARAMS und PARAMS (PARAMS ueberschreiben bei Duplikaten)
        self.params = {k: v[0] for k, v in self.EFFECTS.items()}
        self.params.update(self.COLOR_PARAMS)
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

    def _features_at_time(self, features: dict, time: float) -> dict:
        """Hilfsmethode: Features fuer einen Zeitpunkt (statt Frame-Index).

        Ersetzt den in jedem Visualizer duplizierten Boilerplate:
        Frame-Index aus Zeit berechnen, clampen, Features extrahieren.
        """
        frame_idx = int(time * features.get("fps", 30))
        frame_idx = max(0, min(frame_idx, features.get("frame_count", 0) - 1))
        return self._get_feature_at_frame(features, frame_idx)

    def _get_feature_at_frame(self, features: dict, frame_idx: int) -> dict:
        """Hilfsmethode: Extrahiert die Features fuer einen bestimmten Frame.

        Args:
            features: Dictionary mit den gesamten Audio-Features.
            frame_idx: Index des gewuenschten Frames.

        Returns:
            Dictionary mit den skalaren Features fuer den angegebenen Frame.
        """
        # Sicherer Clamp gegen frame_count UND tatsaechliche Array-Laengen
        max_idx = features.get("frame_count", 0) - 1
        for key in ("rms", "onset", "spectral_centroid", "chroma"):
            if key in features:
                arr = features[key]
                if hasattr(arr, "shape") and len(arr.shape) > 1:
                    # 2D-Array: laengste Achse bestimmen
                    arr_len = max(arr.shape)
                elif hasattr(arr, "__len__"):
                    arr_len = len(arr)
                else:
                    arr_len = 0
                max_idx = min(max_idx, arr_len - 1)
        frame_idx = max(0, min(frame_idx, max_idx))

        chroma = features.get("chroma")
        if chroma is not None and hasattr(chroma, "shape") and len(chroma.shape) > 1:
            if chroma.shape[0] == 12 and chroma.shape[1] > frame_idx >= 0:
                chroma_frame = chroma[:, frame_idx]
            elif chroma.shape[1] == 12 and chroma.shape[0] > frame_idx >= 0:
                chroma_frame = chroma[frame_idx, :]
            else:
                chroma_frame = np.zeros(12, dtype=np.float32)
        elif chroma is not None and hasattr(chroma, "__len__") and len(chroma) > frame_idx >= 0:
            chroma_frame = chroma[frame_idx]
        else:
            chroma_frame = np.zeros(12, dtype=np.float32)

        def _safe_float(arr, idx, default=0.0):
            if arr is None:
                return default
            if hasattr(arr, "__len__") and len(arr) > idx >= 0:
                return float(arr[idx])
            return default

        result = {
            "rms": _safe_float(features.get("rms"), frame_idx, 0.0),
            "onset": _safe_float(features.get("onset"), frame_idx, 0.0),
            "chroma": chroma_frame,
            "spectral_centroid": _safe_float(features.get("spectral_centroid"), frame_idx, 0.0),
            "mode": features.get("mode", "music"),
        }

        # Neue Features (falls vorhanden)
        if "transient" in features and len(features["transient"]) > 0:
            result["transient"] = _safe_float(features["transient"], frame_idx, 0.0)
        else:
            result["transient"] = min(result["onset"] * 1.5, 1.0)  # Fallback, clamped

        if "voice_clarity" in features and len(features["voice_clarity"]) > 0:
            result["voice_clarity"] = _safe_float(features["voice_clarity"], frame_idx, 0.0)
        else:
            result["voice_clarity"] = result["rms"]  # Fallback

        if "voice_band" in features and len(features["voice_band"]) > 0:
            result["voice_band"] = _safe_float(features["voice_band"], frame_idx, 0.0)
        else:
            result["voice_band"] = result.get("voice_clarity", result["rms"])  # Fallback

        if "beat_intensity" in features and len(features["beat_intensity"]) > 0:
            result["beat_intensity"] = _safe_float(features["beat_intensity"], frame_idx, 0.0)
        else:
            result["beat_intensity"] = result["onset"]  # Fallback

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
                "u_beat_intensity": f.get("beat_intensity", f["onset"]),
            }
        elif mode == "speech":
            return {
                "u_energy": f["rms"] * 0.7,
                "u_beat": f["onset"] * 0.3,
                "u_impact": f.get("transient", f["onset"]) * 0.2,
                "u_detail": f["spectral_centroid"] * 0.5,
                "u_flow": f.get("voice_band", f.get("voice_clarity", f["rms"])),  # Podcast: Voice-Band > Voice-Clarity > RMS
                "u_chroma": f["chroma"],
                "u_beat_intensity": f.get("beat_intensity", f["onset"]) * 0.3,
            }
        else:  # hybrid
            return {
                "u_energy": f["rms"],
                "u_beat": f["onset"] * 0.7,
                "u_impact": f.get("transient", f["onset"]) * 0.7,
                "u_detail": f["spectral_centroid"],
                "u_flow": f.get("voice_clarity", f["rms"]) * 0.5,
                "u_chroma": f["chroma"],
                "u_beat_intensity": f.get("beat_intensity", f["onset"]),
            }

    def _chroma_to_color(self, chroma: np.ndarray) -> tuple:
        """Wandelt ein Chroma-Vektor in eine RGB-Farbe um.

        Beruecksichtigt color_mode Param:
        - 'chroma': Dynamische Farbe aus Audio-Chroma (bunt)
        - 'fixed': Feste Farbe aus primary_color / base_hue
        - 'monochrome': Graustufen
        - 'warm': Warme Toene (Orange/Gelb)
        - 'cool': Kuehle Toene (Blau/Cyan)
        """
        mode = self.params.get('color_mode', 'chroma')
        saturation = float(self.params.get('color_saturation', 0.7))
        brightness = float(self.params.get('brightness', 1.0))

        chroma_arr = np.asarray(chroma).flatten()
        strength = float(np.max(chroma_arr)) if chroma_arr.size > 0 else 0.5

        if mode == 'monochrome':
            val = 0.55 * brightness
            return (val, val, val)
        elif mode == 'fixed':
            primary_color = self.params.get('primary_color')
            if primary_color and isinstance(primary_color, str) and primary_color.startswith('#'):
                rgb = self._hex_to_rgb(primary_color)
                h, s, v = self._rgb_to_hsv(*rgb)
                return self._hsv_to_rgb(h, saturation * (0.7 + 0.3 * s), v * brightness)
            hue = float(self.params.get('base_hue', 0.55))
            return self._hsv_to_rgb(hue, saturation, 0.85 * brightness)
        elif mode == 'warm':
            hue = 0.08 + 0.06 * strength  # Orange bis Gelb
            return self._hsv_to_rgb(hue, saturation, brightness * (0.7 + 0.3 * strength))
        elif mode == 'cool':
            hue = 0.55 + 0.1 * strength  # Cyan bis Blau
            return self._hsv_to_rgb(hue, saturation, brightness * (0.7 + 0.3 * strength))

        # Default: 'chroma' - dynamische Farbe aus Audio
        if chroma_arr.size < 12:
            chroma_arr = np.pad(chroma_arr, (0, 12 - chroma_arr.size))

        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        x = np.sum(chroma_arr * np.cos(angles))
        y = np.sum(chroma_arr * np.sin(angles))

        hue = np.arctan2(y, x) / (2.0 * np.pi)
        if hue < 0:
            hue += 1.0

        sat = saturation * (0.7 + 0.3 * strength)
        val = brightness * (0.55 + 0.45 * strength)

        return self._hsv_to_rgb(hue, sat, val)

    def _color_to_hue(self, rgb: tuple) -> float:
        """Extrahiert den Hue-Wert (0.0-1.0) aus einem RGB-Tupel."""
        if rgb is None:
            return 0.55
        return self._rgb_to_hsv(*rgb)[0]

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

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        """Konvertiert 6-stelligen Hex-String nach RGB-Tupel (0.0-1.0)."""
        return hex_to_rgb(hex_color)

    @staticmethod
    def _rgb_to_hsv(r: float, g: float, b: float) -> tuple:
        """Konvertiert RGB nach HSV."""
        mx = max(r, g, b)
        mn = min(r, g, b)
        diff = mx - mn
        if diff == 0:
            h = 0.0
        elif mx == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
        h = h / 360.0
        s = 0.0 if mx == 0 else diff / mx
        v = mx
        return (h, s, v)
