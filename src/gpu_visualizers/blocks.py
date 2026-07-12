"""
Baustein-Bibliothek fuer das Visualizer-Studio.

Jeder Baustein ist eine GLSL-Funktion, die einen Farbbeitrag (vec3, HDR) fuer
einen Bildpunkt liefert. Der `CompositeVisualizer` setzt aus mehreren Bausteinen
(Ebenen) einen kompletten Fragment-Shader zusammen. Kein Baustein-Code wird als
Python generiert — die Komposition passiert einmalig zur Ladezeit in GLSL.

Konventionen:
- Die Funktion erhaelt `vec2 p` (zentriert, aspektkorrigiert) und `vec3 col`
  (Basisfarbe der Ebene) als erste zwei Argumente.
- Danach folgen die in `arg_order` gelisteten Parameter (als float-Uniforms).
- Globale Uniforms (u_time, u_energy, ...) und Includes (fbm, hash12, aastep)
  stehen zur Verfuegung.
"""

# Jeder Eintrag:
#   display_name : deutscher Anzeigename
#   description  : kurze Erklaerung
#   glsl         : Funktionsdefinition (Name: block_<key>)
#   params       : {name: (default, min, max, step)}
#   arg_order    : Reihenfolge der Parameter in der Funktionssignatur

BLOCK_LIBRARY = {
    "ring": {
        "display_name": "Ring",
        "description": "Leuchtender Kreisring",
        "arg_order": ["radius", "width"],
        "params": {
            "radius": (0.35, 0.05, 1.0, 0.01),
            "width": (0.02, 0.004, 0.1, 0.002),
        },
        "glsl": """
        vec3 block_ring(vec2 p, vec3 col, float radius, float width) {
            float d = abs(length(p) - radius) - width;
            float g = exp(-d * d * 1500.0);
            return col * g;
        }
        """,
    },
    "core_glow": {
        "display_name": "Kern-Glow",
        "description": "Weicher, heller Kern in der Mitte",
        "arg_order": ["size", "intensity"],
        "params": {
            "size": (0.25, 0.05, 0.8, 0.01),
            "intensity": (1.4, 0.2, 3.0, 0.05),
        },
        "glsl": """
        vec3 block_core_glow(vec2 p, vec3 col, float size, float intensity) {
            float g = exp(-dot(p, p) / max(size * size, 1e-4));
            return mix(col, vec3(1.0), 0.5) * g * intensity;
        }
        """,
    },
    "wave": {
        "display_name": "Welle",
        "description": "Horizontale, schwingende Linie",
        "arg_order": ["amplitude", "frequency", "thickness"],
        "params": {
            "amplitude": (0.3, 0.0, 0.9, 0.01),
            "frequency": (6.0, 1.0, 30.0, 0.5),
            "thickness": (0.03, 0.005, 0.15, 0.005),
        },
        "glsl": """
        vec3 block_wave(vec2 p, vec3 col, float amplitude, float frequency, float thickness) {
            float wy = sin(p.x * frequency + u_time * 2.0) * amplitude;
            float d = abs(p.y - wy);
            float g = exp(-d * d / max(thickness * thickness, 1e-5));
            return col * g;
        }
        """,
    },
    "bars": {
        "display_name": "Balken",
        "description": "Vertikale, gespiegelte Balken",
        "arg_order": ["count", "height", "thickness"],
        "params": {
            "count": (16.0, 4.0, 64.0, 1.0),
            "height": (0.5, 0.1, 1.0, 0.05),
            "thickness": (0.6, 0.1, 1.0, 0.05),
        },
        "glsl": """
        vec3 block_bars(vec2 p, vec3 col, float count, float height, float thickness) {
            float span = u_resolution.x / u_resolution.y;
            float xn = (p.x + span) / (2.0 * span);
            float cell = fract(xn * count);
            float bar = 1.0 - aastep(thickness * 0.5, abs(cell - 0.5));
            float h = height * (0.3 + 0.7 * abs(sin(floor(xn * count) * 1.7 + u_time)));
            float mask = 1.0 - aastep(h, abs(p.y));
            return col * bar * mask;
        }
        """,
    },
    "particles": {
        "display_name": "Partikel",
        "description": "Funkelnde Punkte",
        "arg_order": ["count", "size"],
        "params": {
            "count": (18.0, 0.0, 48.0, 1.0),
            "size": (0.6, 0.1, 2.0, 0.05),
        },
        "glsl": """
        vec3 block_particles(vec2 p, vec3 col, float count, float size) {
            vec3 acc = vec3(0.0);
            for (int i = 0; i < 48; i++) {
                if (float(i) >= count) break;
                float fi = float(i);
                vec2 seed = vec2(fi * 0.137, fi * 0.319);
                float px = (hash12(seed) * 2.0 - 1.0) * (u_resolution.x / u_resolution.y);
                float py = hash12(seed + 3.0) * 2.0 - 1.0;
                py += sin(u_time * (0.5 + hash12(seed + 5.0)) + fi) * 0.05;
                float d = length(p - vec2(px, py));
                float tw = 0.5 + 0.5 * sin(u_time * 3.0 + fi * 1.7);
                acc += mix(col, vec3(1.0), 0.5) * exp(-d * d * (2200.0 / max(size, 0.1))) * tw;
            }
            return acc;
        }
        """,
    },
    "noise_field": {
        "display_name": "Nebelfeld",
        "description": "Weiche, treibende fbm-Wolken",
        "arg_order": ["scale", "density"],
        "params": {
            "scale": (2.0, 0.5, 6.0, 0.1),
            "density": (0.8, 0.1, 2.0, 0.05),
        },
        "glsl": """
        vec3 block_noise_field(vec2 p, vec3 col, float scale, float density) {
            float n = fbm(p * scale + u_time * 0.1, 5);
            return col * pow(max(n, 0.0), 1.8) * density;
        }
        """,
    },
}


# Erlaubte Blend-Modi (Name -> GLSL-Ausdruck, das `acc` und `c` kombiniert)
BLEND_MODES = {
    "add": "acc + c",
    "screen": "1.0 - (1.0 - acc) * (1.0 - clamp(c, 0.0, 1.0))",
    "max": "max(acc, c)",
}

# Audio-Quellen, die als Mapping-Ziel dienen koennen (Uniform-Namen)
AUDIO_SOURCES = [
    "u_energy", "u_beat", "u_impact", "u_flow", "u_detail",
    "u_beat_intensity", "u_texture", "u_warmth",
]


def block_names() -> list:
    """Liste aller verfuegbaren Baustein-Schluessel."""
    return list(BLOCK_LIBRARY.keys())


def get_block(name: str) -> dict:
    """Liefert die Definition eines Bausteins oder wirft KeyError."""
    return BLOCK_LIBRARY[name]
