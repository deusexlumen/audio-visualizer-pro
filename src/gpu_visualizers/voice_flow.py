"""
Voice Flow - Signature Podcast-Visualizer v2.

Deutliche, leuchtende Wellen fuer Podcasts und Sprach-Inhalte.
- Sichtbare Wellen-Linien (wie ein sanftes Oszilloskop)
- Voice-Clarity steuert Amplitude klar und deutlich
- Bewegung auch bei Stille (Atem-Effekt)
- Helle, leuchtende Farben auf dunklem Hintergrund
- Nie ablenkend, aber immer sichtbar
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class VoiceFlowGPU(BaseGPUVisualizer):
    """
    Voice Flow v2 - Klar sichtbare Podcast-Wellen.
    Leuchtende Linien, die mit der Stimme tanzen.
    """

    PARAMS = {
        'flow_speed': (0.4, 0.1, 1.0, 0.05),
        'wave_depth': (0.5, 0.1, 1.0, 0.05),
        'color_saturation': (0.7, 0.3, 1.0, 0.05),
        'breathe_intensity': (0.4, 0.0, 0.8, 0.05),
        'line_count': (5, 3, 12, 1),
        'glow_strength': (0.6, 0.0, 1.0, 0.05),
    }

    def _setup(self):
        self._prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
            #version 330
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_voice;
            uniform vec3 u_color;
            uniform float u_flow_speed;
            uniform float u_wave_depth;
            uniform float u_color_saturation;
            uniform float u_breathe_intensity;
            uniform float u_line_count;
            uniform float u_glow_strength;

            out vec4 f_color;

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                // Normalisierte UV-Koordinaten (0..1)
                vec2 uv = gl_FragCoord.xy / u_resolution;
                
                // Aspect-ratio korrigierte x-Koordinate fuer symmetrische Wellen
                float x = uv.x;
                float y = uv.y;
                
                // Atem-Takt (langsam, immer sichtbar)
                float breathe = sin(u_time * 0.6) * 0.5 + 0.5;
                float breatheAmt = u_breathe_intensity * (0.3 + breathe * 0.7);
                
                // Voice-Amplitude (mindestens 0.15 damit immer etwas zu sehen ist)
                float voiceAmt = max(0.15, u_voice) * u_wave_depth;
                
                // Kombinierte Amplitude: Voice + Atem
                float totalAmp = voiceAmt + breatheAmt * 0.2;
                
                // Zeit mit Flow-Speed
                float t = u_time * u_flow_speed;
                
                // Hintergrund: sehr dunkel, leicht gefaerbt
                vec3 bgColor = u_color * 0.08;
                vec3 col = bgColor;
                
                // Base-Hue aus der Chroma-Farbe
                float baseHue = fract(u_color.x);
                
                // Mehrere Wellen-Linien
                int lines = int(u_line_count);
                for (int i = 0; i < 12; i++) {
                    if (i >= lines) break;
                    
                    float fi = float(i);
                    
                    // Jede Linie hat eigene Frequenz, Phase und Geschwindigkeit
                    float freq = 2.0 + fi * 1.5;
                    float phase = fi * 1.047; // 60-Grad-Offset
                    float speed = 1.0 + fi * 0.3;
                    
                    // Vertikale Position der Linie (verteilt ueber den Bildschirm)
                    float lineY = 0.2 + fi * 0.6 / max(float(lines - 1), 1.0);
                    
                    // Welle: Sinus mit Noise-Modulation
                    float wave = sin(x * freq * 3.14159 + t * speed + phase);
                    // Zusaetzliche Oberwelle fuer Detail
                    wave += sin(x * freq * 2.5 + t * speed * 1.3 + phase * 2.0) * 0.3;
                    wave *= 0.5 + 0.5; // Normalize to -1..1
                    
                    // Amplitude skalieren
                    float amp = totalAmp * (0.8 + fi * 0.1);
                    float yOffset = wave * amp * 0.15;
                    
                    // Distanz zur Wellen-Linie
                    float dist = abs(y - (lineY + yOffset));
                    
                    // Hue fuer diese Linie (leicht verschoben)
                    float hue = fract(baseHue + fi * 0.08 + sin(t * 0.1) * 0.02);
                    float sat = u_color_saturation * (0.8 + fi * 0.05);
                    vec3 lineColor = hsv2rgb(vec3(hue, sat, 1.0));
                    
                    // Gluehender Kern der Linie
                    float lineWidth = 0.003 + u_voice * 0.002;
                    float core = smoothstep(lineWidth, 0.0, dist);
                    
                    // Glow um die Linie
                    float glow = smoothstep(0.08, 0.0, dist) * u_glow_strength;
                    
                    // Voice reagiert auf Glow-Staerke
                    glow *= (0.5 + u_voice * 0.5);
                    
                    // Zur Farbe addieren
                    col += lineColor * core * 0.9;
                    col += lineColor * glow * 0.4;
                }
                
                // Horizontaler Scanline-Effekt (subtil, Voice-reaktiv)
                float scanline = sin(y * u_resolution.y * 0.5 + t * 2.0) * 0.5 + 0.5;
                col *= 0.95 + scanline * 0.05 * (0.5 + u_voice * 0.5);
                
                // Vignette (Raender abdunkeln)
                vec2 center = uv - 0.5;
                float vig = 1.0 - length(center) * 0.6;
                vig = smoothstep(0.0, 1.0, vig);
                col *= 0.7 + vig * 0.3;
                
                // Clamp
                col = clamp(col, 0.0, 1.0);
                
                f_color = vec4(col, 1.0);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self._vao = self.ctx.vertex_array(self._prog, [(vbo, "2f", "in_pos")])

    def render(self, features: dict, time: float):
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        uniforms = self._map_features_to_uniforms(f, mode="speech")

        color = self._chroma_to_color(uniforms["u_chroma"])

        self._prog["u_resolution"].value = (self.width, self.height)
        self._prog["u_time"].value = time
        self._prog["u_voice"].value = uniforms["u_flow"]
        self._prog["u_color"].value = color
        self._prog["u_flow_speed"].value = self.params["flow_speed"]
        self._prog["u_wave_depth"].value = self.params["wave_depth"]
        self._prog["u_color_saturation"].value = self.params["color_saturation"]
        self._prog["u_breathe_intensity"].value = self.params["breathe_intensity"]
        self._prog["u_line_count"].value = self.params["line_count"]
        self._prog["u_glow_strength"].value = self.params["glow_strength"]

        self._vao.render(mode=moderngl.TRIANGLE_STRIP)
