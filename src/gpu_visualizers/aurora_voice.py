"""
Aurora Voice - Signature Podcast-Visualizer.

Ruhige, langsam wogende Aurora-Baender, angetrieben von Sprachband und Flow.
Bewusst ohne Beat-Blitzen: fuer stundenlange Sprach-Inhalte (Podcasts,
Interviews, Hoerbuecher), die nicht flackern oder ermueden sollen.
Ein einzelner Fullscreen-Fragment-Shader, HDR-Ausgabe.
"""

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


class AuroraVoiceGPU(BaseGPUVisualizer):
    """Sanfte Aurora-Baender fuer Podcast/Sprache (kein Beat-Blitzen)."""

    PARAMS = {
        'band_count': (4, 1, 7, 1),
        'flow_speed': (0.15, 0.02, 0.6, 0.01),
        'wave_depth': (0.35, 0.1, 0.8, 0.05),
        'band_softness': (0.18, 0.05, 0.5, 0.01),
        'breathe_intensity': (0.4, 0.0, 1.0, 0.05),
        'breathe_speed': (0.25, 0.05, 1.0, 0.05),
        'voice_response': (0.7, 0.0, 1.5, 0.05),
        'glow_strength': (0.8, 0.2, 2.0, 0.05),
        'bg_brightness': (0.18, 0.0, 0.6, 0.01),
    }

    PARAMS_GROUPS = {
        "Baender": ["band_count", "wave_depth", "band_softness"],
        "Bewegung": ["flow_speed", "breathe_intensity", "breathe_speed"],
        "Reaktion": ["voice_response", "glow_strength", "bg_brightness"],
    }

    def _setup(self):
        fragment = compose_fragment(
            """
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_energy;
            uniform float u_flow;
            uniform vec3 u_color;
            uniform vec3 u_secondary_color;
            uniform vec3 u_background_color;
            uniform float u_band_count;
            uniform float u_flow_speed;
            uniform float u_wave_depth;
            uniform float u_band_softness;
            uniform float u_breathe_intensity;
            uniform float u_breathe_speed;
            uniform float u_voice_response;
            uniform float u_glow_strength;
            uniform float u_bg_brightness;
            uniform float u_brightness;
            out vec4 f_color;

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                vec2 p = uv * 2.0 - 1.0;
                p.x *= u_resolution.x / u_resolution.y;

                // Sanfter vertikaler Hintergrund-Gradient
                vec3 col = mix(u_background_color * 0.4,
                               u_background_color, uv.y) * (u_bg_brightness / 0.18);

                // Langsames "Atmen" der gesamten Szene
                float breathe = 1.0 + sin(u_time * u_breathe_speed) * u_breathe_intensity * 0.3;
                // Sprachband hebt die Baender sanft an (kein hartes Blitzen)
                float voice = u_flow * u_voice_response;

                int bands = int(u_band_count);
                for (int i = 0; i < 7; i++) {
                    if (i >= bands) break;
                    float fi = float(i);
                    float t = u_time * u_flow_speed * (1.0 + fi * 0.2);

                    // Wogende Mittellinie des Bandes aus fbm
                    float centerY = (fi / max(1.0, u_band_count - 1.0) - 0.5) * 1.2;
                    float wave = fbm(vec2(p.x * 0.8 + fi * 3.0, t), 4) - 0.5;
                    centerY += wave * u_wave_depth * breathe;
                    centerY += voice * 0.15 * sin(p.x * 2.0 + fi);

                    float dist = abs(p.y - centerY);
                    float soft = u_band_softness * (0.6 + voice * 0.8);
                    float band = exp(-dist * dist / (soft * soft));

                    // Farbe changiert zwischen Primaer und Sekundaer je Band
                    vec3 bandCol = mix(u_color, u_secondary_color, fract(fi * 0.37 + u_time * 0.02));
                    col += bandCol * band * u_glow_strength * (0.5 + u_energy * 0.5 + voice);
                }

                // Feiner Sternen-/Schleier-Glanz oben (dezent)
                float shimmer = fbm(vec2(p.x * 6.0, p.y * 6.0 - u_time * 0.1), 3);
                col += u_secondary_color * smoothstep(0.75, 1.0, shimmer) * 0.06 * (0.5 + uv.y);

                col = max(col, 0.0) * u_brightness;
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

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)
        uniforms = self._map_features_to_uniforms(f, mode="speech")

        color = self._chroma_to_color(uniforms["u_chroma"])
        h, s, v = self._rgb_to_hsv(*color)
        # Analoge, harmonische Sekundaerfarbe (nahe am Primaer, ruhiger Look)
        secondary = self._hsv_to_rgb((h + 0.12) % 1.0, min(1.0, s * 0.9), v)

        bg = self.params.get("background_color")
        if isinstance(bg, str) and bg.startswith("#"):
            try:
                bg_rgb = self._hex_to_rgb(bg)
            except Exception:
                bg_rgb = (0.02, 0.03, 0.06)
        else:
            bg_rgb = (0.02, 0.03, 0.06)

        self.prog["u_resolution"].value = (self.width, self.height)
        self.prog["u_time"].value = time
        self.prog["u_energy"].value = uniforms["u_energy"]
        self.prog["u_flow"].value = uniforms["u_flow"]
        self.prog["u_color"].value = color
        self.prog["u_secondary_color"].value = secondary
        self.prog["u_background_color"].value = bg_rgb
        self.prog["u_band_count"].value = float(self.params["band_count"])
        self.prog["u_flow_speed"].value = float(self.params["flow_speed"])
        self.prog["u_wave_depth"].value = float(self.params["wave_depth"])
        self.prog["u_band_softness"].value = float(self.params["band_softness"])
        self.prog["u_breathe_intensity"].value = float(self.params["breathe_intensity"])
        self.prog["u_breathe_speed"].value = float(self.params["breathe_speed"])
        self.prog["u_voice_response"].value = float(self.params["voice_response"])
        self.prog["u_glow_strength"].value = float(self.params["glow_strength"])
        self.prog["u_bg_brightness"].value = float(self.params["bg_brightness"])
        self.prog["u_brightness"].value = float(self.params.get("brightness", 1.0))

        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
