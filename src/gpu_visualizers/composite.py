"""
CompositeVisualizer - datengetriebener Visualizer aus einem Studio-Rezept.

Nimmt ein deklaratives Rezept (Ebenen aus Bausteinen) und baut daraus EINMALIG
einen Fragment-Shader. Die Parameter jeder Ebene bekommen namespaced Keys
(l0_radius, l1_size, ...), sodass das bestehende Parameter-Panel sie ohne
Aenderung als Regler darstellt. Audio-Mappings werden in render() ausgewertet
und mit EMA geglaettet auf die Uniform-Werte aufaddiert.

Kein Python-Code wird generiert — nur GLSL wird zur Ladezeit zusammengesetzt.
"""

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
from .blocks import BLOCK_LIBRARY, BLEND_MODES


class CompositeVisualizer(BaseGPUVisualizer):
    """Basisklasse fuer rezeptbasierte Visualizer. RECIPE wird pro Klasse gesetzt."""

    # Wird von make_recipe_visualizer_class() gesetzt (dict aus RecipeSchema.model_dump()).
    RECIPE: dict = None

    def _setup(self):
        recipe = self.RECIPE or {"layers": []}
        self._layers = recipe.get("layers", [])
        self._mode_hint = recipe.get("mode_hint", "music")
        # EMA-Zustand pro (Ebene, Ziel-Param)
        self._smoothed = {}

        fragment = compose_fragment(
            self._build_fragment_body(),
            includes=(LYGIA_MATH_GLSL, LYGIA_NOISE_GLSL, LYGIA_SDF_GLSL, SHADER_COMMON_GLSL),
        )
        self.prog = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao, self.vbo = create_fullscreen_quad(self.ctx, self.prog)

    # ------------------------------------------------------------------
    # Shader-Aufbau
    # ------------------------------------------------------------------

    def _used_block_glsl(self) -> str:
        """GLSL-Definitionen der tatsaechlich genutzten Bausteine (dedupliziert)."""
        seen = []
        parts = []
        for layer in self._layers:
            b = layer.get("block")
            if b in BLOCK_LIBRARY and b not in seen:
                seen.append(b)
                parts.append(BLOCK_LIBRARY[b]["glsl"])
        return "\n".join(parts)

    def _layer_param_uniforms(self) -> str:
        """Uniform-Deklarationen fuer alle namespaced Ebenen-Parameter + Transform."""
        lines = []
        for i, layer in enumerate(self._layers):
            block = BLOCK_LIBRARY.get(layer.get("block"))
            if not block:
                continue
            for pname in block["params"]:
                lines.append(f"uniform float l{i}_{pname};")
            lines.append(f"uniform vec4 l{i}_transform;   // offx, offy, scale, rot_speed")
        return "\n".join(lines)

    def _build_fragment_body(self) -> str:
        uniforms = """
        uniform vec2 u_resolution;
        uniform float u_time;
        uniform float u_energy;
        uniform float u_beat;
        uniform float u_impact;
        uniform float u_flow;
        uniform float u_detail;
        uniform float u_beat_intensity;
        uniform float u_texture;
        uniform float u_warmth;
        uniform vec3 u_color;
        uniform vec3 u_secondary_color;
        uniform vec3 u_background_color;
        uniform float u_bg_brightness;
        uniform float u_brightness;
        """ + self._layer_param_uniforms()

        block_defs = self._used_block_glsl()

        # main(): fuer jede Ebene Transform anwenden, Baustein aufrufen, blenden
        layer_code = []
        for i, layer in enumerate(self._layers):
            block = BLOCK_LIBRARY.get(layer.get("block"))
            if not block:
                continue
            blend_expr = BLEND_MODES.get(layer.get("blend", "add"), BLEND_MODES["add"])
            args = ", ".join(f"l{i}_{p}" for p in block["arg_order"])
            layer_col = "u_color" if i % 2 == 0 else "u_secondary_color"
            fn = f"block_{layer['block']}"
            layer_code.append(f"""
            {{
                vec4 tf = l{i}_transform;
                vec2 lp = p - tf.xy;
                float ang = u_time * tf.w;
                float ca = cos(ang), sa = sin(ang);
                lp = mat2(ca, -sa, sa, ca) * lp;
                lp /= max(tf.z, 0.05);
                vec3 c = {fn}(lp, {layer_col}, {args});
                acc = {blend_expr};
            }}""")

        body = f"""
        {uniforms}

        {block_defs}

        out vec4 f_color;

        void main() {{
            vec2 uv = gl_FragCoord.xy / u_resolution;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= u_resolution.x / u_resolution.y;

            vec3 acc = u_background_color * u_bg_brightness;
            {''.join(layer_code)}
            acc = max(acc, 0.0) * u_brightness;
            f_color = vec4(acc, 1.0);
        }}
        """
        return body

    # ------------------------------------------------------------------
    # Rendern
    # ------------------------------------------------------------------

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        uniforms = self._map_features_to_uniforms(f, mode=self._mode_hint)

        color = self._chroma_to_color(uniforms["u_chroma"])
        h, s, v = self._rgb_to_hsv(*color)
        secondary = self._hsv_to_rgb((h + 0.5) % 1.0, s, v)

        bg = self.params.get("background_color")
        if isinstance(bg, str) and bg.startswith("#"):
            try:
                bg_rgb = self._hex_to_rgb(bg)
            except Exception:
                bg_rgb = (0.02, 0.02, 0.04)
        else:
            bg_rgb = (0.02, 0.02, 0.04)

        def _set(name, value):
            # Ungenutzte Uniforms werden vom GLSL-Compiler entfernt -> pruefen
            if name in self.prog._members:
                self.prog[name].value = value

        _set("u_resolution", (self.width, self.height))
        _set("u_time", time)
        for key in ("u_energy", "u_beat", "u_impact", "u_flow", "u_detail",
                    "u_beat_intensity", "u_texture", "u_warmth"):
            _set(key, float(uniforms.get(key, 0.0)))
        _set("u_color", color)
        _set("u_secondary_color", secondary)
        _set("u_background_color", bg_rgb)
        _set("u_bg_brightness", float(self.params.get("bg_brightness", 0.15)))
        _set("u_brightness", float(self.params.get("brightness", 1.0)))

        # Ebenen-Parameter + Audio-Mappings setzen
        for i, layer in enumerate(self._layers):
            block = BLOCK_LIBRARY.get(layer.get("block"))
            if not block:
                continue
            # Basiswerte aus namespaced Params (Panel) oder Rezept-Default
            values = {}
            for pname, spec in block["params"].items():
                key = f"l{i}_{pname}"
                values[pname] = float(self.params.get(key, spec[0]))
            # Audio-Mappings aufaddieren (mit EMA-Glaettung)
            for m in layer.get("mappings", []):
                target = m.get("target")
                if target not in values:
                    continue
                raw = float(uniforms.get(m.get("source"), 0.0)) * m.get("gain", 0.0) + m.get("offset", 0.0)
                sm = float(m.get("smooth", 0.0))
                skey = (i, target)
                prev = self._smoothed.get(skey, raw)
                cur = prev * sm + raw * (1.0 - sm)
                self._smoothed[skey] = cur
                values[target] += cur
            # Uniforms schreiben
            for pname, val in values.items():
                uname = f"l{i}_{pname}"
                if uname in self.prog._members:
                    self.prog[uname].value = float(val)
            tf = layer.get("transform", {}) or {}
            tfu = f"l{i}_transform"
            if tfu in self.prog._members:
                self.prog[tfu].value = (
                    float(tf.get("offset_x", 0.0)), float(tf.get("offset_y", 0.0)),
                    float(tf.get("scale", 1.0)), float(tf.get("rotation_speed", 0.0)),
                )

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)


def build_params_for_recipe(recipe: dict) -> tuple:
    """Erzeugt (PARAMS, PARAMS_GROUPS) fuer ein Rezept mit namespaced Keys."""
    params = {"bg_brightness": (0.15, 0.0, 0.6, 0.01)}
    groups = {}
    for i, layer in enumerate(recipe.get("layers", [])):
        block = BLOCK_LIBRARY.get(layer.get("block"))
        if not block:
            continue
        group_keys = []
        for pname, spec in block["params"].items():
            key = f"l{i}_{pname}"
            # Rezept-Default uebernehmen, falls gesetzt
            default = layer.get("params", {}).get(pname, spec[0])
            params[key] = (default, spec[1], spec[2], spec[3])
            group_keys.append(key)
        label = f"Ebene {i + 1}: {block['display_name']}"
        groups[label] = group_keys
    return params, groups


def make_recipe_visualizer_class(recipe: dict) -> type:
    """Baut dynamisch eine CompositeVisualizer-Subklasse mit gebackenem Rezept."""
    params, groups = build_params_for_recipe(recipe)
    name = recipe.get("name", "recipe")
    cls = type(
        f"Recipe_{name}",
        (CompositeVisualizer,),
        {
            "RECIPE": recipe,
            "PARAMS": params,
            "PARAMS_GROUPS": groups,
            "__doc__": recipe.get("description", "") or f"Studio-Rezept: {name}",
        },
    )
    return cls
