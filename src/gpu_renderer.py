"""
GPU-beschleunigter Batch-Renderer mit ModernGL.

Rendert Video-Frames auf der GPU (offscreen, kein Fenster) und
piped rohe RGB-Pixel direkt in FFmpeg stdin fuer das Encoding.
Wesentlich schneller als der alte Python/PIL-basierte Renderer.
"""

import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path

import moderngl
import numpy as np

from .analyzer import AudioAnalyzer
from .types import AudioFeatures, Quote
from .gpu_visualizers import get_visualizer
from .gpu_text_renderer import SDFFontAtlas, GPUTextRenderer
from .quote_overlay import QuoteOverlayConfig, QuoteOverlayRenderer


class GPUBatchRenderer:
    """GPU-Renderer fuer Audio-Visualisierungen mit ModernGL.

    Erzeugt einen standalone OpenGL-Context, rendert offscreen in ein
    Framebuffer-Objekt und schreibt die Pixel-Daten direkt in einen
    FFmpeg-Subprozess zur Video-Erzeugung.
    """

    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        # Standalone OpenGL-Context erzeugen (Windows: default, Linux: ggf. egl)
        self.ctx = moderngl.create_standalone_context()

        # Offscreen-Framebuffer fuer das Rendering (RGBA fuer Alpha-Kanal-Erhalt)
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )
        
        # Temporaerer FBO fuer Hintergrundbild
        self.bg_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 3)]
        )
        
        # Temporaerer FBO fuer Visualizer (wird ueber Hintergrundbild composited)
        self.viz_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )
        
        # Dummy schwarze Textur fuer Composite ohne Hintergrundbild (verhindert Memory-Leak)
        self._dummy_black_texture = self.ctx.texture((1, 1), 3, b'\x00\x00\x00')
        
        # Post-Process FBO (zweiter Pass fuer Color-Grading, RGBA fuer Alpha-Erhalt)
        self.post_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )
        self._init_postprocess()
        self._init_composite_shader()

    def render(
        self,
        audio_path: str,
        visualizer_type: str,
        output_path: str,
        features: AudioFeatures = None,
        preview_mode: bool = False,
        preview_duration: float = 5.0,
        params: dict = None,
        background_image: str = None,
        background_blur: float = 0.0,
        background_vignette: float = 0.0,
        background_opacity: float = 0.3,
        quotes: list = None,
        quote_config: QuoteOverlayConfig = None,
        sync_quotes_to_beats: bool = False,
        codec: str = "h264",
        quality: str = "high",
        gpu_encode: bool = False,
        postprocess: dict = None,
        viz_offset_x: float = 0.0,
        viz_offset_y: float = 0.0,
        viz_scale: float = 1.0,
        progress_callback=None,
        cancel_event=None,
    ):
        """Rendert ein Video aus Audio-Analyse auf der GPU.

        Args:
            audio_path: Pfad zur Audiodatei.
            visualizer_type: Name des GPU-Visualizers (z.B. 'spectrum_bars').
            output_path: Pfad fuer die Ausgabe-MP4.
            features: Vorberechnete AudioFeatures (optional).
            preview_mode: Wenn True, nur Vorschau-Laenge rendern.
            preview_duration: Vorschau-Laenge in Sekunden.
            quotes: Liste von Quote-Objekten fuer Text-Overlays.
            quote_config: Konfiguration fuer Quote-Overlays.
            sync_quotes_to_beats: Wenn True, werden Quotes auf Beats synchronisiert.
            codec: Video-Codec ('h264', 'hevc', 'prores').
            quality: Qualitaet ('low', 'medium', 'high', 'lossless').
            postprocess: Color-Grading Parameter dict mit keys: contrast, saturation, brightness, warmth, film_grain.
            viz_offset_x: Horizontaler Offset in normalisierten Koordinaten (-1.0 bis 1.0).
            viz_offset_y: Vertikaler Offset in normalisierten Koordinaten (-1.0 bis 1.0).
            viz_scale: Skalierungsfaktor des Visualizers (0.5 bis 2.0).
            progress_callback: Optionaler Callback(frame, total_frames) fuer Fortschritts-Updates.
            cancel_event: Optional threading.Event. Wenn gesetzt, wird die Render-Schleife unterbrochen.
        """
        audio_path = str(audio_path)
        output_path = str(output_path)

        # Audio analysieren falls noetig
        if features is None:
            analyzer = AudioAnalyzer()
            features = analyzer.analyze(audio_path, fps=self.fps)

        # Frame-Anzahl bestimmen
        if preview_mode:
            frame_count = int(preview_duration * self.fps)
        else:
            frame_count = features.frame_count

        # Sicherstellen, dass die Feature-Arrays nicht kuerzer sind als frame_count
        # (librosa kann manchmal um 1 Frame abweichen)
        frame_count = min(
            frame_count,
            len(features.rms),
            len(features.onset),
            len(features.spectral_centroid),
        )
        if features.chroma.ndim > 1 and features.chroma.shape[0] == 12:
            frame_count = min(frame_count, features.chroma.shape[1])
        elif features.chroma.ndim > 1:
            frame_count = min(frame_count, features.chroma.shape[0])

        print(
            f"[GPU] Rendere {frame_count} Frames @ {self.fps}fps "
            f"({frame_count / self.fps:.1f}s)"
        )
        print(f"[GPU] Visualizer: {visualizer_type}")
        print(f"[GPU] Aufloesung: {self.width}x{self.height}")
        if gpu_encode:
            print("[GPU] GPU-Encoding aktiviert (NVENC/AMF/QSV)")

        # Quotes optional zu Beats synchronisieren
        if sync_quotes_to_beats and quotes and len(features.beat_frames) > 0:
            from .beat_sync import sync_quotes_to_beats as sync_fn
            quotes = sync_fn(quotes, features.beat_frames, self.fps)
            print(f"[GPU] {len(quotes)} Quotes auf Beats synchronisiert")
        
        # Beat-Intensitaet berechnen (vektorisiert, ~100x schneller als Python-Schleife)
        beat_intensity = np.zeros(frame_count, dtype=np.float32)
        if len(features.beat_frames) > 0:
            decay_frames = max(3, int(self.fps * 0.1))
            for bf in features.beat_frames:
                if bf >= frame_count:
                    continue
                end = min(bf + decay_frames + 1, frame_count)
                if end > bf:
                    dists = np.arange(end - bf, dtype=np.float32)
                    vals = 1.0 - dists / decay_frames
                    vals = np.clip(vals, 0.0, 1.0)
                    beat_intensity[bf:end] = np.maximum(beat_intensity[bf:end], vals)
        
        # Visualizer-Instanz erzeugen
        viz_cls = get_visualizer(visualizer_type)
        viz = viz_cls(self.ctx, self.width, self.height)
        if params:
            viz.set_params(params)
        
        # Hintergrundbild vorbereiten
        bg_texture = None
        if background_image and os.path.exists(background_image):
            bg_texture = self._load_background_texture(
                background_image, background_blur
            )

        # Feature-Dictionary fuer den Visualizer vorbereiten
        features_dict = {
            "rms": features.rms[:frame_count],
            "onset": features.onset[:frame_count],
            "beat_intensity": beat_intensity,
            "chroma": features.chroma,
            "spectral_centroid": features.spectral_centroid[:frame_count],
            "fps": self.fps,
            "frame_count": frame_count,
        }

        # Temporaere Videodatei fuer den Video-Stream (ohne Audio)
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_video.close()

        # FFmpeg fuer Video-Encoding starten
        ffmpeg_cmd = self._build_ffmpeg_cmd(
            temp_video.name, codec, quality, gpu_encode=gpu_encode
        )

        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=8 * 1024 * 1024,  # 8MB Buffer fuer schnelleres Schreiben
        )

        try:
            # Quote-Renderer initialisieren falls noetig
            if quotes and quote_config and quote_config.enabled:
                self._init_text_renderer()
            
            # Frame-Index fuer Quote-Buffer bauen (O(1) Lookups im Render-Loop)
            self._build_quote_frame_index(quotes, quote_config, frame_count)

            # === PRODUCER-CONSUMER: Render und Encode parallel ===
            # Der Render-Thread rendert Frames in eine Queue.
            # Ein separater Thread schreibt sie zu FFmpeg stdin.
            # Queue OHNE maxsize: put() blockiert nie, FFmpeg ist der einzige Engpass.
            frame_queue = queue.Queue()
            encode_done = threading.Event()
            encode_error = [None]
            _DEBUG = False  # Auf True setzen fuer Debug-Screenshots

            def _encode_worker():
                try:
                    while True:
                        item = frame_queue.get()
                        if item is None:
                            break
                        process.stdin.write(item)
                except Exception as e:
                    encode_error[0] = e
                    print(f"[GPU] Encode-Fehler: {e}")
                finally:
                    try:
                        process.stdin.close()
                    except Exception:
                        pass
                    encode_done.set()

            encode_thread = threading.Thread(target=_encode_worker, daemon=True)
            encode_thread.start()

            try:
                # Haupt-Render-Loop
                for i in range(frame_count):
                    if cancel_event is not None and cancel_event.is_set():
                        print("[GPU] Render abgebrochen durch User.")
                        break

                    time = i / self.fps
                    
                    self.fbo.use()
                    self.ctx.clear(0.05, 0.05, 0.05)
                    
                    if _DEBUG and i == 0:
                        self._save_debug(self.fbo, "debug_step1_after_clear.png")
                    
                    if bg_texture is not None:
                        self._render_background(bg_texture, background_opacity, background_vignette)
                        if _DEBUG and i == 0:
                            self._save_debug(self.fbo, "debug_step2_after_bg.png")
                    
                    self.viz_fbo.use()
                    self.ctx.clear(0.0, 0.0, 0.0, 0.0)
                    viz.render(features_dict, time)
                    if _DEBUG and i == 0:
                        self._save_debug(self.viz_fbo, "debug_step3_after_viz.png")
                    
                    self.fbo.use()
                    self._blit_viz_to_fbo(
                        self.viz_fbo.color_attachments[0],
                        offset_x=viz_offset_x,
                        offset_y=viz_offset_y,
                        scale=viz_scale,
                    )
                    if _DEBUG and i == 0:
                        self._save_debug(self.fbo, "debug_step3b_after_viz_blit.png")
                    
                    if postprocess:
                        self._apply_postprocess(
                            self.fbo.color_attachments[0],
                            contrast=postprocess.get("contrast", 1.0),
                            saturation=postprocess.get("saturation", 1.0),
                            brightness=postprocess.get("brightness", 0.0),
                            warmth=postprocess.get("warmth", 0.0),
                            film_grain=postprocess.get("film_grain", 0.0),
                            time=time,
                        )
                        if _DEBUG and i == 0:
                            self._save_debug(self.post_fbo, "debug_step4_after_postprocess.png")
                        target_fbo = self.post_fbo
                    else:
                        target_fbo = self.fbo
                    
                    if quotes and quote_config and quote_config.enabled:
                        target_fbo.use()
                        self._render_quotes_gpu(time, quotes, quote_config, frame_idx=i)
                        if _DEBUG and i == 0:
                            self._save_debug(target_fbo, "debug_step5_after_quotes.png")
                    
                    pixels = target_fbo.read(components=3)
                    if _DEBUG and i == 0:
                        self._save_debug(target_fbo, "debug_step6_final.png")
                    
                    # FFmpeg-Health-Check VOR dem put
                    if process.poll() is not None:
                        enc = ffmpeg_cmd[ffmpeg_cmd.index("-c:v") + 1] if "-c:v" in ffmpeg_cmd else "unknown"
                        raise RuntimeError(
                            f"FFmpeg ist unerwartet beendet (Code {process.returncode}). "
                            f"Pruefe ob der Encoder '{enc}' verfuegbar ist (ffmpeg -encoders)."
                        )
                    
                    if encode_error[0] is not None:
                        raise RuntimeError(f"Encode-Thread-Fehler: {encode_error[0]}")
                    
                    frame_queue.put(pixels)

                    if i % 30 == 0 or i == frame_count - 1:
                        if progress_callback:
                            progress_callback(i + 1, frame_count)
                        if i % 120 == 0 or i == frame_count - 1:
                            progress_pct = (i + 1) / frame_count * 100
                            print(
                                f"[GPU] {progress_pct:.1f}% ({i + 1}/{frame_count})",
                                flush=True,
                            )
            finally:
                frame_queue.put(None)
                # Warte bis der Encode-Thread fertig ist (stdin schliesst sich)
                # Bei 31k Frames kann das >10s dauern, besonders bei Software-Encoding
                encode_done.wait(timeout=300)
                encode_thread.join(timeout=120)
                # Sicherstellen dass FFmpeg stdin geschlossen ist, auch wenn
                # der Thread haengen geblieben ist
                try:
                    process.stdin.close()
                except Exception:
                    pass

            process.wait()

            if process.returncode != 0:
                raise RuntimeError("FFmpeg Video-Encoding fehlgeschlagen")

            # Audio mit dem Video muxen
            self._mux_audio(temp_video.name, audio_path, output_path)
            print(f"[GPU] Fertig: {output_path}")

        finally:
            # FFmpeg-Prozess sauber beenden falls noch aktiv
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

            # Temporaere Datei aufraeumen
            if os.path.exists(temp_video.name):
                os.unlink(temp_video.name)

    def _save_debug(self, fbo_obj, filename: str):
        """Speichert den aktuellen FBO-Inhalt als PNG fuer Debugging."""
        try:
            from PIL import Image
            raw = fbo_obj.read(components=3)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            # ModernGL fbo.read() gibt Daten top-to-bottom (PIL-kompatibel)
            Image.fromarray(arr, mode='RGB').save(filename)
            print(f"[GPU] DEBUG: {filename} gespeichert ({self.width}x{self.height})")
        except Exception as e:
            print(f"[GPU] DEBUG: Konnte {filename} nicht speichern: {e}")

    def _load_background_texture(self, image_path: str, blur: float):
        """Laedt ein Hintergrundbild als Textur.

        Args:
            image_path: Pfad zum Bild.
            blur: Gaussian-Blur Radius (CPU-seitig mit PIL).

        Returns:
            ModernGL Textur.
        """
        from PIL import Image, ImageFilter
        
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.width, self.height), Image.LANCZOS)
        
        # Blur nur anwenden wenn wirklich > 0 (robuster gegen Float-Rauschen)
        if blur > 0.01:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))
        
        data = np.array(img, dtype=np.uint8)
        # KEIN np.flipud mehr noetig — ModernGL Textur-Upload und Shader-UVs
        # sind konsistent mit PIL-TopDown-Orientierung
        texture = self.ctx.texture((self.width, self.height), 3, data.tobytes())
        return texture
    
    def _build_ffmpeg_cmd(self, output_path: str, codec: str, quality: str, gpu_encode: bool = False):
        """Baut den FFmpeg-Befehl basierend auf Codec und Qualitaet auf.
        
        Unterstuetzt GPU-Encoding (NVENC, AMF, QSV) fuer massiv schnelleres
        Encoding (~5-10x gegenueber Software-Encoding).
        
        NEU: 'high' und 'lossless' verwenden yuv444p (kein Chroma-Subsampling)
        fuer scharfe Kanten und knallige Farben. 'medium'/'low' bleiben bei
        yuv420p fuer bessere Kompatibilitaet.
        """
        
        quality_profiles = {
            "low": {"preset": "ultrafast", "crf": "28", "bitrate": "4M", "pix_fmt": "yuv420p"},
            "medium": {"preset": "fast", "crf": "23", "bitrate": "8M", "pix_fmt": "yuv420p"},
            "high": {"preset": "fast", "crf": "20", "bitrate": "16M", "pix_fmt": "yuv444p"},
            "lossless": {"preset": "slow", "crf": "0", "bitrate": "50M", "pix_fmt": "yuv444p"},
        }
        
        q = quality_profiles.get(quality, quality_profiles["high"])
        
        # GPU-Encoding: Automatisch besten verfuegbaren Encoder waehlen
        gpu_encoder = None
        if gpu_encode:
            gpu_encoder = self._detect_gpu_encoder(codec)
        
        if gpu_encoder:
            # GPU-Encoding Parameter
            video_codec = gpu_encoder
            pix_fmt = "yuv420p"  # GPU-Encoder unterstuetzen meist nur yuv420p
            extra_args = self._build_gpu_encoder_args(gpu_encoder, quality, q)
        elif codec == "hevc" or codec == "h265":
            video_codec = "libx265"
            pix_fmt = q.get("pix_fmt", "yuv420p")
            extra_args = ["-tag:v", "hvc1"]
        elif codec == "prores":
            video_codec = "prores_ks"
            pix_fmt = "yuv422p10le"
            extra_args = ["-profile:v", "3"]
        else:
            video_codec = "libx264"
            pix_fmt = q.get("pix_fmt", "yuv420p")
            extra_args = ["-movflags", "+faststart"]
        
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", video_codec,
            "-pix_fmt", pix_fmt,
        ]
        
        if not gpu_encoder:
            cmd.extend(["-preset", q["preset"]])
        
        if codec != "prores" and not gpu_encoder:
            cmd.extend(["-crf", q["crf"]])
        elif codec == "prores":
            cmd.extend(["-b:v", q["bitrate"]])
        
        cmd.extend(extra_args)
        cmd.append(output_path)
        
        return cmd
    
    def _detect_gpu_encoder(self, codec: str) -> str | None:
        """Erkennt den besten verfuegbaren GPU-Encoder.
        
        Reihenfolge: NVENC (NVIDIA) > AMF (AMD) > QSV (Intel)
        """
        # Cache: Einmalig pro Sitzung pruefen
        if not hasattr(self, '_gpu_encoder_cache'):
            self._gpu_encoder_cache = {}
        
        cache_key = f"{codec}_gpu"
        if cache_key in self._gpu_encoder_cache:
            return self._gpu_encoder_cache[cache_key]
        
        suffix = "hevc" if codec in ("hevc", "h265") else "h264"
        
        encoders_to_check = [
            f"{suffix}_nvenc",   # NVIDIA NVENC
            f"{suffix}_amf",     # AMD AMF
            f"{suffix}_qsv",     # Intel QuickSync
        ]
        
        detected = None
        for enc in encoders_to_check:
            if self._ffmpeg_has_encoder(enc) and self._test_encoder_works(enc):
                detected = enc
                print(f"[GPU] GPU-Encoder verifiziert: {enc}")
                break
        
        self._gpu_encoder_cache[cache_key] = detected
        return detected
    
    def _ffmpeg_has_encoder(self, encoder_name: str) -> bool:
        """Prueft ob FFmpeg einen bestimmten Encoder unterstuetzt."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True, text=True, timeout=10
            )
            return encoder_name in result.stdout
        except Exception:
            return False
    
    def _test_encoder_works(self, encoder_name: str) -> bool:
        """Echter Funktionstest: Versucht einen 1-Frame Encode mit dem Encoder.
        
        Manche Encoder sind in der Liste, funktionieren aber nicht weil
        die GPU fehlt (z.B. NVENC ohne nvcuda.dll).
        """
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                    "-c:v", encoder_name,
                    "-frames:v", "1",
                    "-f", "null", "-"
                ],
                capture_output=True, text=True, timeout=15
            )
            stderr_lower = result.stderr.lower()
            # GPU-Fehler sind eindeutig; andere Fehler (z.B. Access Violation
            # beim Prozess-Exit auf Windows) sind nicht aussagekraeftig
            gpu_error_indicators = [
                "cannot load nvcuda",
                "cannot load amfrt",
                "error creating a mfx",
                "error while opening encoder",
                "operation not permitted",
            ]
            for indicator in gpu_error_indicators:
                if indicator in stderr_lower:
                    return False
            # Wenn mindestens ein Frame geschrieben wurde, gilt der Encoder
            # als funktionstuechtig (auch bei komischen Exit-Codes)
            if "frame=" in stderr_lower or "frame=" in result.stdout.lower():
                return True
            # Fallback: nur bei sauberem Exit-Code akzeptieren
            return result.returncode == 0
        except Exception:
            return False
    
    def _build_gpu_encoder_args(self, encoder: str, quality: str, q: dict) -> list:
        """Baut GPU-spezifische FFmpeg-Argumente."""
        if "nvenc" in encoder:
            # NVIDIA NVENC: p1=schnellste, p7=langsamste
            nvenc_presets = {
                "low": "p1",
                "medium": "p4",
                "high": "p5",
                "lossless": "p7",
            }
            nvenc_cq = {
                "low": "32",
                "medium": "26",
                "high": "22",
                "lossless": "18",
            }
            preset = nvenc_presets.get(quality, "p4")
            cq = nvenc_cq.get(quality, "26")
            return [
                "-preset", preset,
                "-cq", cq,
                "-profile:v", "high",
                "-movflags", "+faststart",
            ]
        
        elif "amf" in encoder:
            # AMD AMF: speed, balanced, quality
            amf_quality = {
                "low": "speed",
                "medium": "balanced",
                "high": "quality",
                "lossless": "quality",
            }
            amf_qp = {
                "low": "32",
                "medium": "26",
                "high": "22",
                "lossless": "18",
            }
            q_setting = amf_quality.get(quality, "balanced")
            qp = amf_qp.get(quality, "26")
            return [
                "-quality", q_setting,
                "-qp_p", qp,
                "-movflags", "+faststart",
            ]
        
        elif "qsv" in encoder:
            # Intel QuickSync
            qsv_presets = {
                "low": "veryfast",
                "medium": "fast",
                "high": "medium",
                "lossless": "slow",
            }
            qsv_quality = {
                "low": "28",
                "medium": "23",
                "high": "20",
                "lossless": "18",
            }
            preset = qsv_presets.get(quality, "fast")
            global_q = qsv_quality.get(quality, "23")
            return [
                "-preset", preset,
                "-global_quality", global_q,
                "-movflags", "+faststart",
            ]
        
        return ["-movflags", "+faststart"]
    
    def _init_postprocess(self):
        """Initialisiert den Post-Process Shader fuer Color-Grading."""
        self._pp_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                gl_Position = vec4(in_pos, 0.0, 1.0);
                v_uv = in_uv;
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D u_texture;
            uniform float u_contrast;
            uniform float u_saturation;
            uniform float u_brightness;
            uniform float u_warmth;
            uniform float u_film_grain;
            uniform float u_time;
            in vec2 v_uv;
            out vec4 f_color;
            
            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }
            
            vec3 rgb2hsv(vec3 c) {
                vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
                vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }
            
            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }
            
            void main() {
                vec2 uv = v_uv;
                vec4 tex = texture(u_texture, uv);
                vec3 col = tex.rgb;
                float alpha = tex.a;
                
                // Brightness
                col += u_brightness;
                
                // Contrast (um 0.5 zentriert)
                col = (col - 0.5) * u_contrast + 0.5;
                
                // Saturation
                vec3 hsv = rgb2hsv(col);
                hsv.y *= u_saturation;
                col = hsv2rgb(hsv);
                
                // Warmth (positive = warm/gelb, negative = kalt/blau)
                if (u_warmth > 0.0) {
                    col.r += u_warmth * 0.08;
                    col.g += u_warmth * 0.03;
                    col.b -= u_warmth * 0.05;
                } else if (u_warmth < 0.0) {
                    col.r += u_warmth * 0.03;
                    col.g += u_warmth * 0.01;
                    col.b -= u_warmth * 0.08;
                }
                
                // Film Grain
                if (u_film_grain > 0.0) {
                    float grain = hash(gl_FragCoord.xy + fract(u_time * 100.0) * 100.0) * 2.0 - 1.0;
                    col += grain * u_film_grain * 0.05;
                }
                
                // Clamp
                col = clamp(col, 0.0, 1.0);
                
                // Alpha-Kanal der Original-Textur erhalten!
                f_color = vec4(col, alpha);
            }
            """,
        )
        
        quad = np.array([
            [-1.0, -1.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self._pp_vao = self.ctx.vertex_array(
            self._pp_prog, [(vbo, "2f 2f", "in_pos", "in_uv")]
        )
    
    def _apply_postprocess(self, texture, contrast=1.0, saturation=1.0, brightness=0.0, warmth=0.0, film_grain=0.0, time=0.0):
        """Wendet Color-Grading Post-Process auf die Textur an.
        
        Rendert das Ergebnis in self.post_fbo.
        """
        self.post_fbo.use()
        self._pp_prog["u_texture"].value = 0
        self._pp_prog["u_contrast"].value = contrast
        self._pp_prog["u_saturation"].value = saturation
        self._pp_prog["u_brightness"].value = brightness
        self._pp_prog["u_warmth"].value = warmth
        self._pp_prog["u_film_grain"].value = film_grain
        self._pp_prog["u_time"].value = time
        
        texture.use(location=0)
        self._pp_vao.render(mode=moderngl.TRIANGLE_STRIP)
    
    def _init_composite_shader(self):
        """Initialisiert einen Shader, der Visualizer (mit Alpha) ueber Hintergrundbild mischt."""
        self._composite_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                gl_Position = vec4(in_pos, 0.0, 1.0);
                v_uv = in_uv;
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D u_bg_texture;
            uniform sampler2D u_viz_texture;
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                vec3 bg = texture(u_bg_texture, v_uv).rgb;
                vec4 viz = texture(u_viz_texture, v_uv);
                // Visualizer-Alpha verwenden, mit Fallback fuer Shader die
                // keinen Alpha ausgeben (viz.a bleibt 0, aber Farbe ist sichtbar)
                float viz_alpha = viz.a;
                if (viz_alpha < 0.01 && length(viz.rgb) > 0.01) {
                    viz_alpha = 1.0;
                }
                vec3 col = mix(bg, viz.rgb, viz_alpha);
                // Alpha-Kanal des Visualizers erhalten (nicht hardcodieren)
                f_color = vec4(col, viz_alpha);
            }
            """
        )
        quad = np.array([
            [-1.0, -1.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self._composite_vao = self.ctx.vertex_array(
            self._composite_prog, [(vbo, "2f 2f", "in_pos", "in_uv")]
        )
    
    def _composite_viz_over_bg(self, bg_texture, viz_texture):
        """Mischt Visualizer-Textur (RGBA) ueber Hintergrund-Textur (RGB).
        
        Wenn bg_texture None, wird nur der Visualizer (auf schwarzem Hintergrund) gerendert.
        """
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        if bg_texture is not None:
            bg_texture.use(location=0)
            self._composite_prog["u_bg_texture"].value = 0
        else:
            # Dummy schwarze Textur wiederverwenden (kein Memory-Leak)
            self._dummy_black_texture.use(location=0)
            self._composite_prog["u_bg_texture"].value = 0
        viz_texture.use(location=1)
        self._composite_prog["u_viz_texture"].value = 1
        self._composite_vao.render(mode=moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
    
    def _init_blit_shader(self):
        """Initialisiert einen Shader zum Blitten einer Textur mit Offset und Skalierung."""
        self._blit_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                gl_Position = vec4(in_pos, 0.0, 1.0);
                v_uv = in_uv;
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D u_texture;
            uniform float u_opacity;
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                vec4 tex = texture(u_texture, v_uv);
                f_color = vec4(tex.rgb, tex.a * u_opacity);
            }
            """
        )
        quad = np.array([
            [-1.0, -1.0, 0.0, 0.0],
            [ 1.0, -1.0, 1.0, 0.0],
            [-1.0,  1.0, 0.0, 1.0],
            [ 1.0,  1.0, 1.0, 1.0],
        ], dtype=np.float32)
        self._blit_vbo = self.ctx.buffer(quad.tobytes())
        self._blit_vao = self.ctx.vertex_array(
            self._blit_prog, [(self._blit_vbo, "2f 2f", "in_pos", "in_uv")]
        )
    
    def _blit_viz_to_fbo(self, source_texture, offset_x=0.0, offset_y=0.0, scale=1.0, opacity=1.0):
        """Blittet die Visualizer-Textur auf den aktuellen FBO mit Offset und Skalierung."""
        if not hasattr(self, '_blit_prog'):
            self._init_blit_shader()
        
        # Quad-Vertices basierend auf Offset und Skalierung berechnen
        x1 = -1.0 * scale + offset_x
        x2 =  1.0 * scale + offset_x
        y1 = -1.0 * scale + offset_y
        y2 =  1.0 * scale + offset_y
        
        vertices = np.array([
            x1, y1, 0.0, 0.0,
            x2, y1, 1.0, 0.0,
            x1, y2, 0.0, 1.0,
            x2, y2, 1.0, 1.0,
        ], dtype=np.float32)
        self._blit_vbo.write(vertices.tobytes())
        
        self._blit_prog["u_texture"].value = 0
        self._blit_prog["u_opacity"].value = opacity
        source_texture.use(location=0)
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._blit_vao.render(mode=moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
    
    def _render_background(self, texture, opacity: float, vignette: float = 0.0):
        """Zeichnet das Hintergrundbild als Fullscreen-Quad mit Shader-Vignette.

        Args:
            texture: ModernGL-Textur mit dem Hintergrundbild.
            opacity: Deckkraft des Hintergrunds (0.0-1.0).
            vignette: Staerke der Vignette im Shader (0.0-1.0).
        """
        if not hasattr(self, '_bg_prog'):
            self._bg_prog = self.ctx.program(
                vertex_shader="""
                #version 330
                in vec2 in_position;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    v_uv = in_uv;
                }
                """,
                fragment_shader="""
                #version 330
                uniform sampler2D u_texture;
                uniform float u_opacity;
                uniform float u_vignette;
                in vec2 v_uv;
                out vec4 f_color;
                void main() {
                    vec4 tex = texture(u_texture, v_uv);
                    vec3 rgb = tex.rgb;
                    // Vignette: Abdunklung an den Raendern
                    vec2 center = v_uv - 0.5;
                    float dist = length(center) * 1.4142; // normalisiert auf 0..1
                    float vig = 1.0 - u_vignette * smoothstep(0.3, 1.0, dist);
                    rgb *= vig;
                    // Alpha-Kanal der Original-Textur erhalten
                    f_color = vec4(rgb * u_opacity, tex.a);
                }
                """
            )
            vertices = np.array([
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 1.0,
            ], dtype=np.float32)
            self._bg_vbo = self.ctx.buffer(vertices.tobytes())
            self._bg_vao = self.ctx.vertex_array(
                self._bg_prog,
                [(self._bg_vbo, '2f 2f', 'in_position', 'in_uv')]
            )
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        texture.use(location=0)
        self._bg_prog['u_texture'].value = 0
        self._bg_prog['u_opacity'].value = opacity
        self._bg_prog['u_vignette'].value = vignette
        self._bg_vao.render(mode=moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
    
    def _init_text_renderer(self):
        """Lazy-Initialisierung des SDF Text-Renderers.

        Gibt vorhandene Ressourcen explizit frei, bevor neue erzeugt werden,
        um VRAM-Leaks bei wiederholter Initialisierung zu vermeiden.
        """
        if hasattr(self, '_text_renderer') and self._text_renderer is not None:
            return

        # Defensives Cleanup: Falls vorherige Initialisierung abgebrochen wurde
        if hasattr(self, '_text_renderer') and self._text_renderer:
            self._text_renderer.release()
            self._text_renderer = None
        if hasattr(self, '_font_texture') and self._font_texture:
            self._font_texture.release()
            self._font_texture = None

        font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(font_path):
            # Fallback-Fonts probieren
            for fallback in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/calibri.ttf"]:
                if os.path.exists(fallback):
                    font_path = fallback
                    break
        
        self._font_atlas = SDFFontAtlas(font_path, font_size=64, sdf_size=64)
        self._font_texture = self._font_atlas.build(self.ctx)
        self._text_renderer = GPUTextRenderer(
            self.ctx, self._font_atlas, self._font_texture,
            width=self.width, height=self.height
        )
        
        # Box-Shader fuer Quote-Hintergruende (abgerundetes Rechteck)
        self._box_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform vec2 u_resolution;
            in vec2 in_pos;
            in vec2 in_center;
            in vec2 in_size;
            in vec4 in_color;
            out vec4 v_color;
            out vec2 v_local;
            out vec2 v_size;
            void main() {
                vec2 pixel = in_center + in_pos * in_size;
                vec2 ndc = (pixel / u_resolution) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
                v_color = in_color;
                v_local = in_pos * in_size;  // Pixel-Koordinaten innerhalb der Box
                v_size = in_size;
            }
            """,
            fragment_shader="""
            #version 330
            in vec4 v_color;
            in vec2 v_local;
            in vec2 v_size;
            out vec4 f_color;
            
            float sdRoundBox(vec2 p, vec2 b, float r) {
                vec2 d = abs(p) - b + r;
                return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - r;
            }
            
            uniform float u_radius;
            void main() {
                float radius = min(v_size.x, v_size.y) * u_radius;
                float d = sdRoundBox(v_local, v_size, radius);
                // Anti-Aliasing: 1 Pixel weicher Uebergang am Rand
                float edge_alpha = 1.0 - smoothstep(0.0, 1.0, d);
                if (edge_alpha < 0.01) discard;
                float alpha = edge_alpha * v_color.a;
                f_color = vec4(v_color.rgb, alpha);
            }
            """,
        )
        quad = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        box_quad_vbo = self.ctx.buffer(quad.tobytes())
        self._box_vbo = self.ctx.buffer(reserve=10 * 8 * 4, dynamic=True)
        self._box_vao = self.ctx.vertex_array(
            self._box_prog,
            [
                (box_quad_vbo, "2f", "in_pos"),
                (self._box_vbo, "2f 2f 4f /i", "in_center", "in_size", "in_color"),
            ],
        )

    def _build_quote_frame_index(self, quotes, quote_config, frame_count):
        """Baut einen vorberechneten Frame-Index fuer O(1) Quote-Lookups."""
        self._quote_frame_index = None
        if not quotes or not quote_config:
            return
        
        latency_offset = getattr(quote_config, 'latency_offset', 0.0)
        display_duration = getattr(quote_config, 'display_duration', 8.0)
        
        self._quote_frame_index = [[] for _ in range(frame_count)]
        self._quote_config_display_duration = display_duration
        self._quote_latency_offset = latency_offset
        
        for quote in quotes:
            adj_start = quote.start_time + latency_offset
            effective_end = min(quote.end_time, quote.start_time + display_duration)
            adj_end = effective_end + latency_offset
            start_frame = max(0, min(int(adj_start * self.fps), frame_count - 1))
            end_frame = max(0, min(int(adj_end * self.fps), frame_count - 1))
            
            for f in range(start_frame, end_frame + 1):
                self._quote_frame_index[f].append(quote)

    def _get_active_quote(self, time_seconds: float, quotes: list, display_duration: float, frame_idx: int = None):
        """Findet das aktuell aktive Zitat. Nutzt Frame-Index wenn verfuegbar."""
        if frame_idx is not None and hasattr(self, '_quote_frame_index') and self._quote_frame_index is not None:
            if 0 <= frame_idx < len(self._quote_frame_index):
                candidates = self._quote_frame_index[frame_idx]
                latency_offset = getattr(self, '_quote_latency_offset', 0.0)
                for quote in candidates:
                    effective_end = min(quote.end_time, quote.start_time + display_duration)
                    adj_start = quote.start_time + latency_offset
                    adj_end = effective_end + latency_offset
                    if adj_start <= time_seconds <= adj_end:
                        return quote
                return None
        
        # Fallback: lineare Suche
        for quote in quotes:
            effective_end = min(quote.end_time, quote.start_time + display_duration)
            if quote.start_time <= time_seconds <= effective_end:
                return quote
        return None

    def _calculate_fade_alpha(self, time_seconds: float, quote, fade_duration: float, display_duration: float):
        """Berechnet Fade-Alpha (0.0 - 1.0)."""
        effective_end = min(quote.end_time, quote.start_time + display_duration)
        
        if time_seconds < quote.start_time + fade_duration:
            progress = (time_seconds - quote.start_time) / fade_duration
            return max(0.0, min(1.0, progress))
        elif time_seconds > effective_end - fade_duration:
            progress = (effective_end - time_seconds) / fade_duration
            return max(0.0, min(1.0, progress))
        return 1.0

    def _render_quotes_gpu(self, time: float, quotes: list, config, frame_idx: int = None):
        """Rendert Quote-Overlays via PIL-basiertem QuoteOverlayRenderer.
        
        Liest den aktuellen FBO, wendet den bewaehrten PIL-Renderer an
        und schreibt das Ergebnis zurueck. Robuster als der SDF-Ansatz.
        """
        quote = self._get_active_quote(time, quotes, config.display_duration, frame_idx)
        if quote is None:
            return
        
        alpha = self._calculate_fade_alpha(time, quote, config.fade_duration, config.display_duration)
        if alpha <= 0.01:
            return
        
        # FBO in RGBA-Array lesen (ModernGL gibt bottom-up, PIL braucht top-down)
        pixels = self.fbo.read(components=4)
        arr_rgba = np.frombuffer(pixels, dtype=np.uint8).reshape((self.height, self.width, 4)).copy()
        arr_rgba = np.flipud(arr_rgba)  # bottom-up -> top-down
        
        # PIL-Renderer auf RGB-Teil anwenden
        arr_rgb = arr_rgba[:, :, :3].copy()
        renderer = QuoteOverlayRenderer(quotes=[quote], config=config)
        arr_rgb = renderer.apply(arr_rgb, time, frame_idx)
        
        # Zurueck in RGBA einfuegen und in OpenGL bottom-up konvertieren
        arr_rgba[:, :, :3] = arr_rgb
        arr_rgba = np.flipud(arr_rgba)
        self.fbo.color_attachments[0].write(arr_rgba.tobytes())

    def _mux_audio(self, video_path: str, audio_path: str, output_path: str):
        """Kombiniert Video-Stream mit Original-Audio.

        Args:
            video_path: Pfad zur temporaren Videodatei (ohne Ton).
            audio_path: Pfad zur Original-Audiodatei.
            output_path: Pfad fuer die finale Ausgabedatei.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",                 # Video kopieren (kein Re-Encode)
            "-c:a", "aac",
            "-b:a", "320k",
            "-shortest",                    # Kuerzeste Datei bestimmt Laenge
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Audio-Muxing fehlgeschlagen: {result.stderr}"
            )

    def release(self):
        """Gibt GPU-Ressourcen explizit frei."""
        try:
            if hasattr(self, "post_fbo") and self.post_fbo:
                self.post_fbo.release()
                self.post_fbo = None
            if hasattr(self, "fbo") and self.fbo:
                self.fbo.release()
                self.fbo = None
            if hasattr(self, "quad_vao") and self.quad_vao:
                self.quad_vao.release()
                self.quad_vao = None
            if hasattr(self, "quad_vbo") and self.quad_vbo:
                self.quad_vbo.release()
                self.quad_vbo = None
            if hasattr(self, "bg_texture") and self.bg_texture:
                self.bg_texture.release()
                self.bg_texture = None
            if hasattr(self, "text_renderer") and self.text_renderer:
                self.text_renderer.release()
                self.text_renderer = None
            if hasattr(self, "_font_texture") and self._font_texture:
                self._font_texture.release()
                self._font_texture = None
            if hasattr(self, "ctx") and self.ctx:
                self.ctx.release()
                self.ctx = None
        except Exception:
            pass

    def __del__(self):
        """Gibt GPU-Ressourcen beim Zerstoeren der Instanz frei."""
        self.release()


class GPUPreviewRenderer(GPUBatchRenderer):
    """Schneller Vorschau-Renderer mit reduzierter Aufloesung."""

    def __init__(self, width: int = 854, height: int = 480, fps: int = 30):
        super().__init__(width, height, fps)
