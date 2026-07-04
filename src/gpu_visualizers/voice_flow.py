"""
Voice Flow - Signature Podcast-Visualizer v3.

Deutliche, leuchtende Wellen fuer Podcasts und Sprach-Inhalte.
- Sichtbare Wellen-Linien (wie ein sanftes Oszilloskop)
- Voice-Clarity steuert Amplitude klar und deutlich
- Bewegung auch bei Stille (Atem-Effekt)
- Helle, leuchtende Farben auf dunklem Hintergrund
- NACHZIEH-EFFEKT (Trail/Schweif) wie eine Leuchtrakete
- Nie ablenkend, aber immer sichtbar
"""

import numpy as np
import moderngl
from .base import BaseGPUVisualizer


class VoiceFlowGPU(BaseGPUVisualizer):
    """
    Voice Flow v3 - Klar sichtbare Podcast-Wellen mit Trail-Effekt.
    Leuchtende Linien, die mit der Stimme tanzen und einen Schweif ziehen.
    """

    PARAMS = {
        'flow_speed': (0.4, 0.1, 1.0, 0.05),
        'wave_depth': (0.5, 0.1, 1.0, 0.05),
        'color_saturation': (0.7, 0.3, 1.0, 0.05),
        'breathe_intensity': (0.4, 0.0, 0.8, 0.05),
        'breathe_speed': (0.6, 0.1, 2.0, 0.1),
        'line_count': (5, 3, 12, 1),
        'line_spacing': (0.6, 0.2, 1.5, 0.05),
        'glow_strength': (0.6, 0.0, 1.0, 0.05),
        'glow_size': (0.08, 0.01, 0.3, 0.01),
        'line_width': (0.003, 0.001, 0.02, 0.001),     # Dicke der Wellen
        'trail_length': (3, 0, 12, 1),                   # Anzahl Echos (0 = kein Schweif)
        'trail_decay': (0.7, 0.1, 0.95, 0.05),           # Verblass-Geschwindigkeit
        'trail_time_offset': (0.04, 0.01, 0.1, 0.005),
        'scanline_intensity': (0.05, 0.0, 0.3, 0.01),
        'vignette_intensity': (0.6, 0.0, 1.5, 0.05),
        'brightness': (1.0, 0.5, 2.0, 0.05),             # Gesamthelligkeit
    }

    def _setup(self):
        self._prev_voice = 0.0
        self._base_alpha = 0.08
        
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
            uniform vec3 u_secondary_color;
            uniform vec3 u_background_color;
            uniform float u_hue;
            uniform float u_flow_speed;
            uniform float u_wave_depth;
            uniform float u_color_saturation;
            uniform float u_breathe_intensity;
            uniform float u_breathe_speed;
            uniform float u_line_count;
            uniform float u_line_spacing;
            uniform float u_glow_strength;
            uniform float u_glow_size;
            uniform float u_line_width;
            uniform float u_trail_length;
            uniform float u_trail_decay;
            uniform float u_trail_time_offset;
            uniform float u_scanline_intensity;
            uniform float u_vignette_intensity;
            uniform float u_brightness;

            out vec4 f_color;

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                float x = uv.x;
                float y = uv.y;
                
                float breathe = sin(u_time * u_breathe_speed) * 0.5 + 0.5;
                float breatheAmt = u_breathe_intensity * (0.3 + breathe * 0.7);
                float voiceAmt = max(0.15, u_voice) * u_wave_depth;
                float totalAmp = voiceAmt + breatheAmt * 0.2;
                float t = u_time * u_flow_speed;
                
                vec3 col = u_background_color * 0.08;
                float baseHue = fract(u_hue);
                
                int lines = int(u_line_count);
                int trails = int(u_trail_length);
                
                for (int i = 0; i < 12; i++) {
                    if (i >= lines) break;
                    
                    float fi = float(i);
                    float lineY = 0.2 + fi * u_line_spacing / max(float(lines - 1), 1.0);
                    float freq = 2.0 + fi * 1.5;
                    float phase = fi * 1.047;
                    float speed = 1.0 + fi * 0.3;
                    float hue = fract(baseHue + fi * 0.08 + sin(t * 0.1) * 0.02);
                    float sat = u_color_saturation * (0.8 + fi * 0.05);
                    vec3 lineColor = hsv2rgb(vec3(hue, sat, 1.0));
                    lineColor = mix(lineColor, u_secondary_color, clamp(fi * 0.12, 0.0, 1.0));
                    
                    // === HAUPT-WELLE + TRAIL-ECHOS ===
                    for (int trail = 0; trail <= 12; trail++) {
                        if (trail > trails) break;
                        
                        float trailFade = 1.0;
                        float trailTime = t;
                        float trailGlowStr = u_glow_strength;
                        float trailWidth = u_line_width;
                        
                        if (trail > 0) {
                            float trailF = float(trail);
                            // Echo ist in der Vergangenheit
                            trailTime = t - trailF * u_trail_time_offset;
                            // Verblasst mit trail_decay
                            trailFade = pow(u_trail_decay, trailF);
                            // Glow und Width auch reduzieren
                            trailGlowStr *= trailFade * 0.6;
                            trailWidth *= (0.5 + trailFade * 0.5);
                        }
                        
                        // Welle berechnen mit evtl. zeitversetztem t
                        float wave = sin(x * freq * 3.14159 + trailTime * speed + phase);
                        wave += sin(x * freq * 2.5 + trailTime * speed * 1.3 + phase * 2.0) * 0.3;
                        
                        float amp = totalAmp * (0.8 + fi * 0.1);
                        float yOffset = wave * amp * 0.15;
                        
                        float dist = abs(y - (lineY + yOffset));
                        
                        // Gluehender Kern
                        float core = smoothstep(trailWidth, 0.0, dist) * trailFade;
                        
                        // Glow um die Linie
                        float glow = smoothstep(u_glow_size, 0.0, dist) * trailGlowStr * trailFade;
                        glow *= (0.5 + u_voice * 0.5);
                        
                        // Zur Farbe addieren
                        col += lineColor * core * 0.9 * u_brightness;
                        col += lineColor * glow * 0.4 * u_brightness;
                    }
                }
                
                // Horizontaler Scanline-Effekt
                float scanline = sin(y * u_resolution.y * 0.5 + t * 2.0) * 0.5 + 0.5;
                col *= 0.95 + scanline * u_scanline_intensity * (0.5 + u_voice * 0.5);
                
                // Vignette
                vec2 center = uv - 0.5;
                float vig = 1.0 - length(center) * u_vignette_intensity;
                vig = smoothstep(0.0, 1.0, vig);
                col *= 0.7 + vig * 0.3;
                
                col = clamp(col, 0.0, 1.0);
                f_color = vec4(col, 1.0);
            }
            """,
        )

        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self._vao = self.ctx.vertex_array(self._prog, [(vbo, "2f", "in_pos")])

    def render(self, features: dict, time: float):
        f = self._features_at_time(features, time)

        # === DYNAMISCHES SMOOTHING ===
        # Roh-Wert aus dem Voice-Band (FFT Bins 4-20), Fallback auf voice_clarity / RMS
        voice_raw = f.get("voice_band", f.get("voice_clarity", f["rms"]))
        transient = f.get("transient", 0.0)

        # Delta zur letzten Frame
        delta = abs(voice_raw - self._prev_voice)

        # Adaptive Alpha: Bei schnellen Transienten / grossen Deltas steigt alpha
        # -> weniger Glättung, sofortige Reaktion, kein Lag
        # Basis: 0.08 (sehr geglättet)
        # Max: 0.95 (fast kein Smoothing)
        dynamic_alpha = self._base_alpha + min(
            0.95 - self._base_alpha,
            max(transient * 0.75, delta * 5.0)
        )

        voice_smoothed = dynamic_alpha * voice_raw + (1.0 - dynamic_alpha) * self._prev_voice
        self._prev_voice = voice_smoothed

        color = self._chroma_to_color(f["chroma"])

        def _rgb_from_hex(value, default):
            if isinstance(value, str) and value.startswith('#'):
                try:
                    return self._hex_to_rgb(value)
                except Exception:
                    pass
            return default

        secondary = _rgb_from_hex(self.params.get("secondary_color"), (0.0, 0.8, 1.0))
        background = _rgb_from_hex(self.params.get("background_color"), (0.02, 0.02, 0.04))

        self._prog["u_resolution"].value = (self.width, self.height)
        self._prog["u_time"].value = time
        self._prog["u_voice"].value = voice_smoothed
        self._prog["u_secondary_color"].value = secondary
        self._prog["u_background_color"].value = background
        self._prog["u_hue"].value = self._color_to_hue(color)
        self._prog["u_flow_speed"].value = self.params["flow_speed"]
        self._prog["u_wave_depth"].value = self.params["wave_depth"]
        self._prog["u_color_saturation"].value = self.params["color_saturation"]
        self._prog["u_breathe_intensity"].value = self.params["breathe_intensity"]
        self._prog["u_breathe_speed"].value = self.params["breathe_speed"]
        self._prog["u_line_count"].value = self.params["line_count"]
        self._prog["u_line_spacing"].value = self.params["line_spacing"]
        self._prog["u_glow_strength"].value = self.params["glow_strength"]
        self._prog["u_glow_size"].value = self.params["glow_size"]
        self._prog["u_line_width"].value = self.params["line_width"]
        self._prog["u_trail_length"].value = self.params["trail_length"]
        self._prog["u_trail_decay"].value = self.params["trail_decay"]
        self._prog["u_trail_time_offset"].value = self.params["trail_time_offset"]
        self._prog["u_scanline_intensity"].value = self.params["scanline_intensity"]
        self._prog["u_vignette_intensity"].value = self.params["vignette_intensity"]
        self._prog["u_brightness"].value = self.params["brightness"]

        self._vao.render(mode=moderngl.TRIANGLE_STRIP)
