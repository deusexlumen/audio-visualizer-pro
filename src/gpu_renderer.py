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
from .app_logging import get_logger
from .ffmpeg_locator import get_ffmpeg_path, get_ffprobe_path
from .gpu_bloom import BloomPass, load_cube_lut
from .render_common import build_features_dict
from .types import AudioFeatures, Quote
from .gpu_visualizers import get_visualizer
from .gpu_visualizers.base import hex_to_rgb as _hex_to_rgb
from .gpu_text_renderer import SDFFontAtlas, GPUTextRenderer
from .quote_overlay import QuoteOverlayConfig, QuoteOverlayRenderer

logger = get_logger(__name__)

# Video-Erweiterungen für automatische Erkennung im Hintergrund
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.gif'}


class GPUNichtVerfuegbarError(RuntimeError):
    """Wird geworfen, wenn kein OpenGL-Kontext erzeugt werden kann (keine GPU/Treiber)."""


def create_gl_context() -> "moderngl.Context":
    """Erzeugt einen standalone OpenGL-Kontext mit verstaendlicher Fehlermeldung.

    Kapselt moderngl.create_standalone_context(), damit Nutzer ohne
    kompatible GPU/Treiber keine kryptische glcontext-Exception sehen.
    """
    try:
        return moderngl.create_standalone_context()
    except Exception as e:
        raise GPUNichtVerfuegbarError(
            "Keine kompatible GPU gefunden. Audio Visualizer Pro benoetigt "
            "OpenGL 3.3 oder neuer.\n"
            "Moegliche Ursachen:\n"
            "  - Grafiktreiber veraltet oder nicht installiert\n"
            "  - Remote-Desktop/VM ohne GPU-Beschleunigung\n"
            f"Technische Details: {e}"
        ) from e


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
        self.ctx = create_gl_context()

        # === HDR-Pipeline ===
        # Szene wird in Float16-FBOs gerendert (Werte > 1.0 bleiben erhalten,
        # kein Banding durch 8-Bit-Zwischenschritte). Erst der finale
        # Tonemap-Pass quantisiert nach RGBA8 (post_fbo).

        # Haupt-Szenen-FBO (RGBA16F fuer HDR + Alpha-Kanal-Erhalt)
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4, dtype='f2')]
        )

        # Temporaerer FBO fuer Hintergrundbild
        self.bg_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 3, dtype='f2')]
        )

        # Temporaerer FBO fuer Visualizer (wird ueber Hintergrundbild composited)
        self.viz_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4, dtype='f2')]
        )

        # Optionales 4x-MSAA-Target fuer geometriebasierte Visualizer
        # (weiche Kanten bei Linien/Balken). Faellt bei Treiber-Problemen
        # transparent auf das Nicht-MSAA-Target zurueck.
        self.viz_ms_fbo = None
        try:
            self.viz_ms_fbo = self.ctx.framebuffer(
                color_attachments=[
                    self.ctx.renderbuffer((width, height), 4, samples=4, dtype='f2')
                ]
            )
        except Exception as e:
            logger.info(f"[GPU] MSAA nicht verfuegbar, rendere ohne Kantenglaettung: {e}")
            self.viz_ms_fbo = None

        # Dummy schwarze Textur fuer Composite ohne Hintergrundbild (verhindert Memory-Leak)
        self._dummy_black_texture = self.ctx.texture((1, 1), 3, b'\x00\x00\x00')

        # Ausgabe-FBO: finaler Tonemap/Grading-Pass schreibt hierhin (RGBA8)
        self.post_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )

        # HDR-Bloom-Kette (Threshold -> Downsample -> Tent-Upsample)
        try:
            self._bloom = BloomPass(self.ctx, width, height)
        except Exception as e:
            logger.warning(f"[GPU] Bloom nicht verfuegbar: {e}")
            self._bloom = None

        # LUT-Zustand (3D-Textur wird bei Bedarf geladen und gecached).
        # Platzhalter-LUT, damit der sampler3D immer gueltig gebunden ist.
        self._lut_texture = None
        self._lut_path = None
        self._lut_size = 0
        try:
            self._lut_placeholder = self.ctx.texture3d(
                (1, 1, 1), 3, b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype='f4'
            )
        except Exception:
            self._lut_placeholder = None

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
        background_color: str = "#0A0A0A",
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
        timeline=None,
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

        # Vorherigen Quote-Renderer zuruecksetzen, damit bei Wiederverwendung
        # des Renderers keine alten Quotes aktiv bleiben.
        self._quote_overlay_renderer = None

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

        logger.info(
            f"[GPU] Rendere {frame_count} Frames @ {self.fps}fps "
            f"({frame_count / self.fps:.1f}s)"
        )
        logger.info(f"[GPU] Visualizer: {visualizer_type}")
        logger.info(f"[GPU] Aufloesung: {self.width}x{self.height}")
        if gpu_encode:
            logger.info("[GPU] GPU-Encoding aktiviert (NVENC/AMF/QSV)")

        # Quotes optional zu Beats synchronisieren
        if sync_quotes_to_beats and quotes and len(features.beat_frames) > 0:
            from .beat_sync import sync_quotes_to_beats as sync_fn
            quotes = sync_fn(quotes, features.beat_frames, self.fps)
            logger.info(f"[GPU] {len(quotes)} Quotes auf Beats synchronisiert")
        
        # Visualizer-Instanz erzeugen
        viz_cls = get_visualizer(visualizer_type)
        viz = viz_cls(self.ctx, self.width, self.height)
        if params:
            viz.set_params(params)

        # === Timeline vorbereiten ===
        # Alle in der Timeline vorkommenden Visualizer EINMAL instanziieren
        # (Shader-Kompilierung ist einmalig; Instanzen bleiben ueber Szenen
        # erhalten, damit Trails/Feedback beim Wiedereintritt bestehen).
        timeline_scenes = None
        viz_instances = {}
        scene_for_frame = None
        applied_scene = {}
        if timeline is not None and getattr(timeline, "scenes", None):
            timeline_scenes = list(timeline.scenes)
            for sc in timeline_scenes:
                name = sc.visualizer
                if name not in viz_instances:
                    try:
                        viz_instances[name] = get_visualizer(name)(self.ctx, self.width, self.height)
                    except Exception as e:
                        logger.warning(f"[GPU] Timeline-Visualizer '{name}' nicht ladbar: {e}")
            if viz_instances:
                self._ensure_timeline_resources()
                # Frame -> Szenen-Index vorberechnen (O(1)-Lookup pro Frame)
                scene_for_frame = [0] * frame_count
                si = 0
                for fi in range(frame_count):
                    t = fi / self.fps
                    while si + 1 < len(timeline_scenes) and t >= timeline_scenes[si + 1].start:
                        si += 1
                    scene_for_frame[fi] = si
            else:
                timeline_scenes = None
        
        # Hintergrundbild vorbereiten
        bg_texture = None
        bg_video_frames = None
        bg_video_temp_dir = None

        if background_image and os.path.exists(background_image):
            if self._is_video_file(background_image):
                bg_video_frames, bg_video_temp_dir = self._extract_video_frames(
                    background_image, self.fps, self.width, self.height
                )
                # Erstes Frame als initiale Textur laden
                bg_texture = self._load_background_texture(
                    bg_video_frames[0], background_blur
                )
            else:
                bg_texture = self._load_background_texture(
                    background_image, background_blur
                )

        # Feature-Dictionary fuer den Visualizer vorbereiten
        # (gemeinsame Logik mit dem Preview-Renderer in render_common.py)
        features_dict = build_features_dict(features, frame_count, self.fps)

        # Temporaere Videodatei fuer den Video-Stream (ohne Audio)
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_video.close()

        # FFmpeg fuer Video-Encoding starten
        ffmpeg_cmd = self._build_ffmpeg_cmd(
            temp_video.name, codec, quality, gpu_encode=gpu_encode
        )

        # FFmpeg-stderr in Log-Datei umleiten, damit Encoder-Fehler diagnosebar sind
        self._ffmpeg_stderr_path = temp_video.name.replace(".mp4", "_ffmpeg_stderr.log")
        self._ffmpeg_stderr_file = open(self._ffmpeg_stderr_path, "wb")
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._ffmpeg_stderr_file,
            bufsize=8 * 1024 * 1024,  # 8MB Buffer fuer schnelleres Schreiben
        )

        try:
            # Quote-Renderer initialisieren falls noetig
            if quotes and quote_config and quote_config.enabled:
                try:
                    self._init_quote_overlay(quotes, quote_config, frame_count, self.fps)
                except Exception as e:
                    logger.warning(f"[GPU] Quote-Initialisierung fehlgeschlagen: {e}")
                    quotes = None  # Quotes fuer diesen Render deaktivieren

            # === PRODUCER-CONSUMER: Render und Encode parallel ===
            # Der Render-Thread rendert Frames in eine Queue.
            # Ein separater Thread schreibt sie zu FFmpeg stdin.
            # Queue MIT maxsize: Producer wartet auf Encoder, RAM bleibt konstant.
            frame_queue = queue.Queue(maxsize=3)
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
                    logger.error(f"[GPU] Encode-Fehler: {e}")
                finally:
                    try:
                        process.stdin.close()
                    except Exception:
                        pass
                    encode_done.set()

            encode_thread = threading.Thread(target=_encode_worker, daemon=True)
            encode_thread.start()

            # === Hintergrund-Video-Prefetch ===
            # Dekodiert kommende BG-Frames in einem eigenen Thread, damit
            # PIL-Decode + Blur nicht auf dem Render-Thread liegen.
            bg_queue = None
            bg_stop = threading.Event()
            bg_thread = None
            if bg_video_frames is not None:
                from PIL import Image, ImageFilter

                bg_queue = queue.Queue(maxsize=4)

                def _bg_decoder():
                    try:
                        for frame_i in range(frame_count):
                            if bg_stop.is_set():
                                break
                            video_frame_idx = frame_i % len(bg_video_frames)
                            img = Image.open(bg_video_frames[video_frame_idx]).convert('RGB')
                            if background_blur > 0.01:
                                img = img.filter(ImageFilter.GaussianBlur(radius=background_blur))
                            data = np.array(img, dtype=np.uint8).tobytes()
                            while not bg_stop.is_set():
                                try:
                                    bg_queue.put(data, timeout=0.1)
                                    break
                                except queue.Full:
                                    continue
                    except Exception as e:
                        logger.warning(f"[GPU] BG-Video-Decoder-Fehler: {e}")
                        bg_stop.set()

                bg_thread = threading.Thread(target=_bg_decoder, daemon=True)
                bg_thread.start()

            def _emit_frame(pixels, frame_time, frame_i) -> bool:
                """Quote-Overlay anwenden und Frame in die Encoder-Queue geben.

                Returns:
                    False, wenn der Render abgebrochen werden soll.
                """
                # PIL-basierte Quote-Overlays auf das Frame-Array anwenden
                if quotes and quote_config and quote_config.enabled:
                    try:
                        arr = np.frombuffer(pixels, dtype=np.uint8).copy().reshape(
                            (self.height, self.width, 3)
                        )
                        arr = self._quote_overlay_renderer.apply(
                            arr, frame_time, frame_idx=frame_i
                        )
                        pixels = arr.tobytes()
                    except Exception as e:
                        # Quote-Renderer darf NIEMALS den gesamten Render killen
                        logger.warning(
                            f"[GPU] Quote-Render-Fehler bei Frame {frame_i} "
                            f"({frame_time:.2f}s): {e}"
                        )

                # FFmpeg-Health-Check VOR dem put
                if process.poll() is not None:
                    enc = ffmpeg_cmd[ffmpeg_cmd.index("-c:v") + 1] if "-c:v" in ffmpeg_cmd else "unknown"
                    raise RuntimeError(
                        f"FFmpeg ist unerwartet beendet (Code {process.returncode}). "
                        f"Pruefe ob der Encoder '{enc}' verfuegbar ist (ffmpeg -encoders)."
                    )

                if encode_error[0] is not None:
                    raise RuntimeError(f"Encode-Thread-Fehler: {encode_error[0]}")

                # Bei voller Queue blockieren, aber Abbruch UND Encoder-Tod
                # regelmaessig pruefen — sonst deadlockt der Producer, wenn der
                # Encode-Thread stirbt (z.B. Broken Pipe) und die Queue nicht
                # mehr geleert wird.
                while True:
                    try:
                        frame_queue.put(pixels, timeout=0.1)
                        break
                    except queue.Full:
                        if cancel_event is not None and cancel_event.is_set():
                            logger.info("[GPU] Render abgebrochen durch User (Queue voll).")
                            return False
                        if encode_error[0] is not None:
                            raise RuntimeError(f"Encode-Thread-Fehler: {encode_error[0]}")
                        if encode_done.is_set():
                            raise RuntimeError(
                                "Encode-Thread unerwartet beendet (Queue laeuft voll)."
                            )
                return not (cancel_event is not None and cancel_event.is_set())

            # === PBO-Doppelpufferung fuer den Readback ===
            # Der blockierende glReadPixels-Download wird in zwei Pixel-Buffer
            # ueberlappt: Frame N wird angestossen, waehrend Frame N-1
            # ausgelesen und encodiert wird (1 Frame Latenz, kein GPU-Stall).
            use_pbo = True
            pbo_pair = None
            pending = None  # (pbo, time, frame_index) des noch nicht gelesenen Frames
            try:
                buf_size = self.width * self.height * 3
                pbo_pair = [self.ctx.buffer(reserve=buf_size) for _ in range(2)]
            except Exception as e:
                logger.info(f"[GPU] PBO nicht verfuegbar, nutze direkten Readback: {e}")
                use_pbo = False

            try:
                # Haupt-Render-Loop
                for i in range(frame_count):
                    if cancel_event is not None and cancel_event.is_set():
                        logger.info("[GPU] Render abgebrochen durch User.")
                        break

                    time = i / self.fps
                    
                    self.fbo.use()
                    if bg_texture is None:
                        bg_rgb = _hex_to_rgb(background_color)
                        self.ctx.clear(bg_rgb[0], bg_rgb[1], bg_rgb[2])
                    else:
                        self.ctx.clear(0.0, 0.0, 0.0)

                    if _DEBUG and i == 0:
                        self._save_debug(self.fbo, "debug_step1_after_clear.png")

                    if bg_texture is not None:
                        if bg_queue is not None and not bg_stop.is_set():
                            # Vorab dekodiertes BG-Frame aus dem Prefetch-Thread holen
                            try:
                                data = bg_queue.get(timeout=5.0)
                                bg_texture.write(data)
                            except queue.Empty:
                                logger.warning("[GPU] BG-Video-Prefetch zu langsam, Frame wiederverwendet.")
                        self._render_background(bg_texture, background_opacity, background_vignette)
                        if _DEBUG and i == 0:
                            self._save_debug(self.fbo, "debug_step2_after_bg.png")
                    
                    # Visualizer rendern: mit MSAA (falls verfuegbar) fuer
                    # weiche Kanten bei geometriebasierten Visualizern.
                    if timeline_scenes is not None:
                        active_viz_tex = self._render_timeline_frame(
                            timeline_scenes, scene_for_frame, viz_instances,
                            applied_scene, features_dict, i, time,
                        )
                    else:
                        self._render_viz_into(viz, self.viz_fbo, features_dict, time)
                        active_viz_tex = self.viz_fbo.color_attachments[0]
                    if _DEBUG and i == 0:
                        self._save_debug(self.viz_fbo, "debug_step3_after_viz.png")

                    self.fbo.use()
                    self._blit_viz_to_fbo(
                        active_viz_tex,
                        offset_x=viz_offset_x,
                        offset_y=viz_offset_y,
                        scale=viz_scale,
                    )
                    if _DEBUG and i == 0:
                        self._save_debug(self.fbo, "debug_step3b_after_viz_blit.png")

                    pp = postprocess or {}

                    # HDR-Bloom: helle Bereiche (>threshold) leuchten weich aus,
                    # additiv auf die Szene VOR dem Tonemapping
                    bloom_intensity = pp.get("bloom_intensity", 0.6)
                    if self._bloom is not None and bloom_intensity > 0.0:
                        self._apply_bloom(
                            intensity=bloom_intensity,
                            threshold=pp.get("bloom_threshold", 1.0),
                            radius=pp.get("bloom_radius", 1.0),
                        )

                    # Finaler Pass laeuft IMMER: Exposure -> ACES-Tonemap ->
                    # Grading -> LUT -> Vignette -> Grain -> Dither -> RGBA8
                    self._apply_postprocess(
                        self.fbo.color_attachments[0],
                        contrast=pp.get("contrast", 1.0),
                        saturation=pp.get("saturation", 1.0),
                        brightness=pp.get("brightness", 0.0),
                        warmth=pp.get("warmth", 0.0),
                        film_grain=pp.get("film_grain", 0.0),
                        time=time,
                        exposure=pp.get("exposure", 1.0),
                        vignette=pp.get("vignette", 0.0),
                        chromatic_aberration=pp.get("chromatic_aberration", 0.0),
                        lut_path=pp.get("lut"),
                        lut_strength=pp.get("lut_strength", 1.0),
                    )
                    if _DEBUG and i == 0:
                        self._save_debug(self.post_fbo, "debug_step4_after_postprocess.png")
                    target_fbo = self.post_fbo
                    if _DEBUG and i == 0:
                        self._save_debug(target_fbo, "debug_step6_final.png")

                    if use_pbo:
                        # Download von Frame i anstossen (asynchron in den PBO),
                        # danach Frame i-1 auslesen und weiterreichen
                        pbo = pbo_pair[i % 2]
                        try:
                            target_fbo.read_into(pbo, components=3)
                        except Exception as e:
                            logger.warning(f"[GPU] PBO-Readback fehlgeschlagen, Fallback: {e}")
                            use_pbo = False
                            if pending is not None:
                                prev_pbo, prev_time, prev_i = pending
                                if not _emit_frame(prev_pbo.read(), prev_time, prev_i):
                                    break
                                pending = None
                            if not _emit_frame(target_fbo.read(components=3), time, i):
                                break
                        else:
                            if pending is not None:
                                prev_pbo, prev_time, prev_i = pending
                                if not _emit_frame(prev_pbo.read(), prev_time, prev_i):
                                    pending = None
                                    break
                            pending = (pbo, time, i)
                    else:
                        pixels = target_fbo.read(components=3)
                        if not _emit_frame(pixels, time, i):
                            break

                    if i % 30 == 0 or i == frame_count - 1:
                        if progress_callback:
                            progress_callback(i + 1, frame_count)
                        if i % 120 == 0 or i == frame_count - 1:
                            progress_pct = (i + 1) / frame_count * 100
                            logger.info(
                                f"[GPU] {progress_pct:.1f}% ({i + 1}/{frame_count})"
                            )

                # Letzten anhaengigen PBO-Frame nachliefern (1-Frame-Latenz)
                if pending is not None and not (cancel_event is not None and cancel_event.is_set()):
                    prev_pbo, prev_time, prev_i = pending
                    _emit_frame(prev_pbo.read(), prev_time, prev_i)
                    pending = None
            finally:
                # BG-Prefetch-Thread stoppen (Queue leeren, damit put() nicht blockiert)
                bg_stop.set()
                if bg_queue is not None:
                    try:
                        while True:
                            bg_queue.get_nowait()
                    except queue.Empty:
                        pass
                if bg_thread is not None:
                    bg_thread.join(timeout=5)
                if pbo_pair is not None:
                    for pbo in pbo_pair:
                        try:
                            pbo.release()
                        except Exception:
                            pass
                # Sentinel mit Timeout setzen: Bei voller Queue und bereits
                # beendetem Encode-Thread wuerde ein blockierendes put(None)
                # deadlocken (Fehlerpfad, z.B. Broken Pipe im Encoder).
                while True:
                    try:
                        frame_queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        if encode_done.is_set():
                            break
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

            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.error("[GPU] FFmpeg reagiert nicht mehr, Prozess wird beendet.")
                process.kill()
                process.wait()

            # stderr-File schliessen, damit es gelesen werden kann
            try:
                self._ffmpeg_stderr_file.close()
            except Exception:
                pass

            if process.returncode != 0:
                stderr_msg = ""
                try:
                    with open(self._ffmpeg_stderr_path, "r", encoding="utf-8", errors="replace") as f:
                        stderr_msg = f.read().strip()
                except Exception:
                    pass
                if stderr_msg:
                    raise RuntimeError(
                        f"FFmpeg Video-Encoding fehlgeschlagen (Code {process.returncode}):\n{stderr_msg}"
                    )
                else:
                    raise RuntimeError(
                        f"FFmpeg Video-Encoding fehlgeschlagen (Code {process.returncode}). "
                        f"Stderr-Log: {self._ffmpeg_stderr_path}"
                    )

            # Audio mit dem Video muxen
            self._mux_audio(temp_video.name, audio_path, output_path)
            logger.info(f"[GPU] Fertig: {output_path}")

        finally:
            # FFmpeg-Prozess sauber beenden falls noch aktiv
            if "process" in locals() and process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

            # stderr-File schliessen falls noch offen
            if hasattr(self, "_ffmpeg_stderr_file") and self._ffmpeg_stderr_file:
                try:
                    self._ffmpeg_stderr_file.close()
                except Exception:
                    pass
                self._ffmpeg_stderr_file = None

            # Temporaere Datei aufraeumen
            if os.path.exists(temp_video.name):
                os.unlink(temp_video.name)
            # Stderr-Log aufraeumen NUR bei Erfolg (beim Fehler behalten fuer Diagnose)
            returncode_ok = "process" in locals() and process is not None and process.returncode == 0
            if (
                hasattr(self, "_ffmpeg_stderr_path")
                and self._ffmpeg_stderr_path
                and os.path.exists(self._ffmpeg_stderr_path)
                and returncode_ok
            ):
                try:
                    os.unlink(self._ffmpeg_stderr_path)
                except Exception:
                    pass

            # Background-Video Frames aufräumen
            if bg_video_temp_dir and os.path.exists(bg_video_temp_dir):
                import shutil
                shutil.rmtree(bg_video_temp_dir, ignore_errors=True)
            self._last_bg_frame_idx = -1

    def _save_debug(self, fbo_obj, filename: str):
        """Speichert den aktuellen FBO-Inhalt als PNG fuer Debugging."""
        try:
            from PIL import Image
            raw = fbo_obj.read(components=3)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            # ModernGL fbo.read() gibt Daten top-to-bottom (PIL-kompatibel)
            Image.fromarray(arr).save(filename)
            logger.debug(f"[GPU] DEBUG: {filename} gespeichert ({self.width}x{self.height})")
        except Exception as e:
            logger.warning(f"[GPU] DEBUG: Konnte {filename} nicht speichern: {e}")

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

    def _is_video_file(self, path: str) -> bool:
        """Prüft ob eine Datei ein Video ist (basierend auf Endung)."""
        return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS

    def _extract_video_frames(self, video_path: str, fps: int, width: int, height: int):
        """Extrahiert alle Frames aus einem Video mit FFmpeg.

        Args:
            video_path: Pfad zum Video.
            fps: Ziel-Framerate für die Extraktion.
            width: Ziel-Breite.
            height: Ziel-Höhe.

        Returns:
            Tuple(List[str], str): Liste der Frame-Pfade und Temp-Verzeichnis.
        """
        temp_dir = tempfile.mkdtemp(prefix="avp_bg_frames_")
        pattern = os.path.join(temp_dir, "frame_%05d.png")

        cmd = [
            get_ffmpeg_path(), "-y", "-i", video_path,
            "-vf", f"fps={fps},scale={width}:{height}",
            "-pix_fmt", "rgb24",
            pattern
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(
                f"FFmpeg konnte Video-Frames nicht extrahieren:\n{result.stderr[:800]}"
            )

        frames = sorted([
            os.path.join(temp_dir, f)
            for f in os.listdir(temp_dir)
            if f.endswith('.png')
        ])

        if not frames:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError("Keine Frames aus dem Video extrahiert.")

        logger.info(f"[GPU] {len(frames)} Background-Frames aus Video extrahiert")
        return frames, temp_dir

    def _extract_video_frame_at_time(self, video_path: str, time_sec: float, width: int, height: int) -> str:
        """Extrahiert ein einzelnes Frame aus einem Video zur gegebenen Zeit.

        Args:
            video_path: Pfad zum Video.
            time_sec: Zeitpunkt in Sekunden.
            width: Ziel-Breite.
            height: Ziel-Höhe.

        Returns:
            Pfad zur extrahierten PNG-Datei.
        """
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()

        # Video-Dauer ermitteln für Loop
        try:
            result = subprocess.run(
                [get_ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, timeout=10
            )
            duration = float(result.stdout.strip())
            if duration > 0:
                time_sec = time_sec % duration
        except Exception:
            pass

        cmd = [
            get_ffmpeg_path(), "-y", "-ss", str(time_sec), "-i", video_path,
            "-vframes", "1",
            "-vf", f"scale={width}:{height}",
            "-pix_fmt", "rgb24",
            temp_file.name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            os.unlink(temp_file.name)
            raise RuntimeError(
                f"FFmpeg konnte Frame nicht extrahieren:\n{result.stderr[:500]}"
            )
        return temp_file.name

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
            get_ffmpeg_path(),
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
                logger.info(f"[GPU] GPU-Encoder verifiziert: {enc}")
                break
        
        self._gpu_encoder_cache[cache_key] = detected
        return detected
    
    def _ffmpeg_has_encoder(self, encoder_name: str) -> bool:
        """Prueft ob FFmpeg einen bestimmten Encoder unterstuetzt."""
        try:
            result = subprocess.run(
                [get_ffmpeg_path(), "-encoders"],
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
                    get_ffmpeg_path(), "-y",
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
        """Initialisiert den finalen Tonemap/Color-Grading-Pass.

        Dieser Pass laeuft IMMER am Ende der Pipeline:
        HDR-Szene (f16) -> Exposure -> ACES-Tonemap -> Grading -> Grain
        -> Triangular-Dither -> RGBA8 (post_fbo).
        """
        from .gpu_visualizers.base import (
            SHADER_COMMON_GLSL, TEXTURED_VERTEX_SHADER, LYGIA_MATH_GLSL,
            compose_fragment, create_textured_quad,
        )

        fragment = compose_fragment(
            """
            uniform sampler2D u_texture;
            uniform sampler3D u_lut;
            uniform float u_exposure;
            uniform float u_contrast;
            uniform float u_saturation;
            uniform float u_brightness;
            uniform float u_warmth;
            uniform float u_film_grain;
            uniform float u_vignette;
            uniform float u_chromatic_aberration;
            uniform float u_lut_strength;
            uniform float u_lut_size;
            uniform float u_time;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 uv = v_uv;
                vec4 tex = texture(u_texture, uv);
                vec3 col = max(tex.rgb, 0.0);
                float alpha = tex.a;

                // Echte chromatische Aberration: radialer RGB-Sample-Versatz
                // (staerker zum Bildrand hin, wie bei realen Linsen)
                if (u_chromatic_aberration > 0.0) {
                    vec2 dir = uv - 0.5;
                    vec2 offset = dir * u_chromatic_aberration * 0.004;
                    col.r = max(texture(u_texture, uv + offset).r, 0.0);
                    col.b = max(texture(u_texture, uv - offset).b, 0.0);
                }

                // Exposure (im HDR-Raum, vor dem Tonemapping)
                col *= u_exposure;

                // Tonemapping: weiche Highlight-Kompression statt hartem Clipping.
                // Mischung aus per-Kanal-ACES (filmisch, entsaettigt Highlights)
                // und luminanzbasiertem ACES (erhaelt Neon-Saettigung) —
                // so bleiben die kraeftigen Visualizer-Farben erhalten.
                vec3 tmPerChannel = tonemapACES(col);
                float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
                float tmLum = tonemapACES(vec3(lum)).x;
                vec3 tmSaturated = col * (tmLum / max(lum, 1e-5));
                col = clamp(mix(tmSaturated, tmPerChannel, 0.35), 0.0, 1.0);

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

                // 3D-LUT Color-Grading (nach dem Tonemapping, display-referred)
                if (u_lut_strength > 0.0 && u_lut_size > 1.0) {
                    vec3 lut_uv = clamp(col, 0.0, 1.0)
                        * ((u_lut_size - 1.0) / u_lut_size) + 0.5 / u_lut_size;
                    vec3 lutted = texture(u_lut, lut_uv).rgb;
                    col = mix(col, lutted, u_lut_strength);
                }

                // Vignette auf dem Gesamtbild
                if (u_vignette > 0.0) {
                    float dist = length(uv - 0.5) * 1.4142;
                    col *= 1.0 - u_vignette * smoothstep(0.3, 1.0, dist);
                }

                // Film Grain: luminanzabhaengig (Schatten staerker als Lichter),
                // dreieckig verteilt und pro Frame animiert
                if (u_film_grain > 0.0) {
                    float t = fract(u_time * 100.0);
                    float g1 = hash12(gl_FragCoord.xy * 1.31 + t * 271.0);
                    float g2 = hash12(gl_FragCoord.xy * 0.73 + t * 137.0 + 43.0);
                    float grain = g1 + g2 - 1.0;
                    float lumaG = dot(clamp(col, 0.0, 1.0), vec3(0.299, 0.587, 0.114));
                    float weight = 0.3 + 0.7 * (1.0 - smoothstep(0.0, 1.0, lumaG));
                    col += grain * u_film_grain * 0.08 * weight;
                }

                // Triangular-Dither gegen Banding bei 8-Bit-Quantisierung
                col += ditherTriangular(gl_FragCoord.xy, fract(u_time));

                col = clamp(col, 0.0, 1.0);

                // Alpha-Kanal der Original-Textur erhalten!
                f_color = vec4(col, alpha);
            }
            """,
            includes=(LYGIA_MATH_GLSL, SHADER_COMMON_GLSL),
        )

        self._pp_prog = self.ctx.program(
            vertex_shader=TEXTURED_VERTEX_SHADER,
            fragment_shader=fragment,
        )
        self._pp_vao, self._pp_vbo = create_textured_quad(self.ctx, self._pp_prog)

    def _apply_bloom(self, intensity=0.6, threshold=1.0, radius=1.0):
        """Berechnet HDR-Bloom aus der Szene und addiert ihn auf self.fbo."""
        if self._bloom is None:
            return
        self._bloom.apply(
            self.fbo,
            self.fbo.color_attachments[0],
            intensity=intensity,
            threshold=threshold,
            radius=radius,
        )

    def _ensure_lut(self, lut_path):
        """Laedt die 3D-LUT-Textur bei Bedarf (gecached pro Pfad).

        Returns:
            Groesse der geladenen LUT (0 wenn keine LUT aktiv).
        """
        if not lut_path:
            return 0
        if lut_path == self._lut_path and self._lut_texture is not None:
            return self._lut_size
        try:
            arr = load_cube_lut(lut_path)
            size = arr.shape[0]
            if self._lut_texture is not None:
                try:
                    self._lut_texture.release()
                except Exception:
                    pass
            self._lut_texture = self.ctx.texture3d(
                (size, size, size), 3, arr.tobytes(), dtype='f4'
            )
            self._lut_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._lut_path = lut_path
            self._lut_size = size
            logger.info(f"[GPU] LUT geladen: {lut_path} ({size}x{size}x{size})")
            return size
        except Exception as e:
            logger.warning(f"[GPU] LUT konnte nicht geladen werden ({lut_path}): {e}")
            self._lut_path = lut_path  # Nicht bei jedem Frame erneut versuchen
            self._lut_size = 0
            return 0

    def _apply_postprocess(self, texture, contrast=1.0, saturation=1.0, brightness=0.0,
                           warmth=0.0, film_grain=0.0, time=0.0, exposure=1.0,
                           vignette=0.0, chromatic_aberration=0.0,
                           lut_path=None, lut_strength=1.0):
        """Wendet den finalen Tonemap/Color-Grading-Pass auf die Textur an.

        Rendert das Ergebnis in self.post_fbo (RGBA8, bereit fuer den Readback).
        """
        lut_size = self._ensure_lut(lut_path)

        self.post_fbo.use()
        self._pp_prog["u_texture"].value = 0
        self._pp_prog["u_lut"].value = 1
        self._pp_prog["u_exposure"].value = exposure
        self._pp_prog["u_contrast"].value = contrast
        self._pp_prog["u_saturation"].value = saturation
        self._pp_prog["u_brightness"].value = brightness
        self._pp_prog["u_warmth"].value = warmth
        self._pp_prog["u_film_grain"].value = film_grain
        self._pp_prog["u_vignette"].value = vignette
        self._pp_prog["u_chromatic_aberration"].value = chromatic_aberration
        self._pp_prog["u_lut_strength"].value = lut_strength if lut_size > 0 else 0.0
        self._pp_prog["u_lut_size"].value = float(lut_size)
        self._pp_prog["u_time"].value = time

        texture.use(location=0)
        if self._lut_texture is not None and lut_size > 0:
            self._lut_texture.use(location=1)
        elif self._lut_placeholder is not None:
            self._lut_placeholder.use(location=1)
        self._pp_vao.render(mode=moderngl.TRIANGLE_STRIP)
    
    def _init_composite_shader(self):
        """Initialisiert einen Shader, der Visualizer (mit Alpha) ueber Hintergrundbild mischt."""
        from .gpu_visualizers.base import TEXTURED_VERTEX_SHADER, create_textured_quad

        self._composite_prog = self.ctx.program(
            vertex_shader=TEXTURED_VERTEX_SHADER,
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
        self._composite_vao, self._composite_vbo = create_textured_quad(
            self.ctx, self._composite_prog
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
        from .gpu_visualizers.base import TEXTURED_VERTEX_SHADER, create_textured_quad

        self._blit_prog = self.ctx.program(
            vertex_shader=TEXTURED_VERTEX_SHADER,
            fragment_shader="""
            #version 330
            uniform sampler2D u_texture;
            uniform sampler2D u_subject_mask;
            uniform vec2 u_resolution;
            uniform float u_opacity;
            uniform float u_viz_alpha_cap;        // Default 1.0 = kein Cap
            uniform float u_viz_alpha_from_luma;  // 0.0 = Bestand, 1.0 = Studio (C14)
            uniform float u_luma_knee_lo;
            uniform float u_luma_knee_hi;
            uniform float u_subject_strength;     // Default 0.0 = keine Maskierung
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                vec4 tex = texture(u_texture, v_uv);
                float a_viz = tex.a;
                // Studio-Pfad (C14): Helligkeit IST die Deckung fuer Emitter auf
                // Schwarz. Gilt UNABHAENGIG von tex.a — auch alpha=1.0-Stacks
                // (composite.py) zeichnen grossflaechig Schwarz. Laeuft VOR dem
                // Cap (Reihenfolge bindend, Spec §3.2.2).
                if (u_viz_alpha_from_luma > 0.5) {
                    float luma = dot(tex.rgb, vec3(0.2126, 0.7152, 0.0722));
                    a_viz = smoothstep(u_luma_knee_lo, u_luma_knee_hi, luma);
                }
                // Subjekt-Maske liegt im Bildschirmraum, nicht im Quad-UV-Raum
                // (der Blit-Quad hat Offset/Scale — v_uv waere falsch).
                vec2 screen_uv = gl_FragCoord.xy / u_resolution;
                float subject_mask = texture(u_subject_mask, screen_uv).r;
                float a_eff = min(a_viz, u_viz_alpha_cap) * u_opacity
                            * (1.0 - u_subject_strength * subject_mask);
                f_color = vec4(tex.rgb, a_eff);
            }
            """
        )
        self._blit_vao, self._blit_vbo = create_textured_quad(self.ctx, self._blit_prog)
    
    def _render_viz_into(self, viz, dest_fbo, features_dict, time):
        """Rendert einen Visualizer (mit MSAA falls verfuegbar) in dest_fbo."""
        if self.viz_ms_fbo is not None:
            self.viz_ms_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            viz.render(features_dict, time)
            self.ctx.copy_framebuffer(dest_fbo, self.viz_ms_fbo)
        else:
            dest_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            viz.render(features_dict, time)

    def _ensure_timeline_resources(self):
        """Legt die zusaetzlichen FBOs und den Blend-Shader fuer Crossfades an."""
        if getattr(self, "_timeline_ready", False):
            return
        from .gpu_visualizers.base import create_textured_quad
        # Zweites Resolve-Target (ausgehende Szene) + Blend-Ziel (RGBA16F HDR)
        self.viz_fbo_b = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.width, self.height), 4, dtype='f2')]
        )
        self.viz_fbo_blend = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.width, self.height), 4, dtype='f2')]
        )
        self._xfade_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos; in vec2 in_uv; out vec2 v_uv;
            void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D u_from;   // ausgehende Szene
            uniform sampler2D u_to;     // eingehende Szene
            uniform float u_alpha;      // 0 = ausgehend, 1 = eingehend
            in vec2 v_uv; out vec4 f_color;
            void main() {
                vec4 a = texture(u_from, v_uv);
                vec4 b = texture(u_to, v_uv);
                f_color = mix(a, b, clamp(u_alpha, 0.0, 1.0));
            }
            """,
        )
        self._xfade_vao, self._xfade_vbo = create_textured_quad(self.ctx, self._xfade_prog)
        self._timeline_ready = True

    def _apply_scene_params(self, viz, viz_name, scene, scene_idx, applied_scene):
        """Setzt die Params einer Szene auf ihre Visualizer-Instanz (nur bei Wechsel)."""
        if applied_scene.get(viz_name) != scene_idx:
            try:
                if scene.params:
                    viz.set_params(scene.params)
            except Exception as e:
                logger.debug(f"[GPU] Szenen-Params fuer '{viz_name}' fehlgeschlagen: {e}")
            applied_scene[viz_name] = scene_idx

    def _render_timeline_frame(self, scenes, scene_for_frame, viz_instances,
                                applied_scene, features_dict, frame_i, time):
        """Rendert einen Timeline-Frame und liefert die anzuzeigende Viz-Textur.

        Ausserhalb einer Transition entspricht das exakt dem Einzel-Viz-Pfad.
        Im Crossfade-Fenster werden aus- und eingehende Szene getrennt gerendert
        und ueber mix() ineinander geblendet.
        """
        si = scene_for_frame[frame_i]
        scene = scenes[si]
        viz = viz_instances.get(scene.visualizer)
        if viz is None:
            # Fallback: leeres FBO
            self.viz_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            return self.viz_fbo.color_attachments[0]

        self._apply_scene_params(viz, scene.visualizer, scene, si, applied_scene)

        # Crossfade nur am Szenen-Anfang, wenn Vorgaenger existiert, Transition
        # 'crossfade' ist und die Visualizer sich unterscheiden (eine Instanz
        # kann nicht gleichzeitig zwei Param-Saetze rendern).
        in_xfade = False
        alpha = 1.0
        prev = scenes[si - 1] if si > 0 else None
        if (prev is not None and scene.transition == "crossfade"
                and scene.transition_duration > 0.0
                and prev.visualizer != scene.visualizer
                and time < scene.start + scene.transition_duration):
            prev_viz = viz_instances.get(prev.visualizer)
            if prev_viz is not None:
                in_xfade = True
                alpha = (time - scene.start) / scene.transition_duration

        if not in_xfade:
            self._render_viz_into(viz, self.viz_fbo, features_dict, time)
            return self.viz_fbo.color_attachments[0]

        # Eingehende Szene -> viz_fbo, ausgehende -> viz_fbo_b
        self._render_viz_into(viz, self.viz_fbo, features_dict, time)
        self._apply_scene_params(prev_viz, prev.visualizer, prev, si - 1, applied_scene)
        self._render_viz_into(prev_viz, self.viz_fbo_b, features_dict, time)

        # Blenden -> viz_fbo_blend
        self.viz_fbo_blend.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.viz_fbo_b.color_attachments[0].use(location=0)
        self.viz_fbo.color_attachments[0].use(location=1)
        self._xfade_prog["u_from"].value = 0
        self._xfade_prog["u_to"].value = 1
        self._xfade_prog["u_alpha"].value = float(alpha)
        self._xfade_vao.render(mode=moderngl.TRIANGLE_STRIP)
        return self.viz_fbo_blend.color_attachments[0]

    def _blit_viz_to_fbo(
        self, source_texture, offset_x=0.0, offset_y=0.0, scale=1.0,
        opacity=1.0, alpha_cap=1.0, alpha_from_luma=False,
        luma_knee_lo=0.02, luma_knee_hi=0.25,
        subject_strength=0.0, subject_mask=None,
    ):
        """Blittet die Visualizer-Textur auf den aktuellen FBO.

        Defaults sind bit-identisch zum bisherigen Verhalten. Die Studio-
        Parameter (C14) aktivieren Luma-Alpha, Cap und Subjekt-Maskierung.
        """
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
        source_texture.use(location=0)
        # Subjekt-Maske: Default schwarz (= kein Subjekt), Dummy wiederverwenden
        if subject_mask is not None:
            subject_mask.use(location=1)
        else:
            self._dummy_black_texture.use(location=1)

        prog = self._blit_prog
        if "u_subject_mask" in prog:
            prog["u_subject_mask"].value = 1
        if "u_resolution" in prog:
            prog["u_resolution"].value = (float(self.width), float(self.height))
        if "u_viz_alpha_cap" in prog:
            prog["u_viz_alpha_cap"].value = float(alpha_cap)
        if "u_viz_alpha_from_luma" in prog:
            prog["u_viz_alpha_from_luma"].value = 1.0 if alpha_from_luma else 0.0
        if "u_luma_knee_lo" in prog:
            prog["u_luma_knee_lo"].value = float(luma_knee_lo)
        if "u_luma_knee_hi" in prog:
            prog["u_luma_knee_hi"].value = float(luma_knee_hi)
        if "u_subject_strength" in prog:
            prog["u_subject_strength"].value = float(subject_strength)
        prog["u_opacity"].value = opacity

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
            from .gpu_visualizers.base import TEXTURED_VERTEX_SHADER, create_textured_quad

            self._bg_prog = self.ctx.program(
                vertex_shader=TEXTURED_VERTEX_SHADER,
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
            self._bg_vao, self._bg_vbo = create_textured_quad(self.ctx, self._bg_prog)
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        texture.use(location=0)
        self._bg_prog['u_texture'].value = 0
        self._bg_prog['u_opacity'].value = opacity
        self._bg_prog['u_vignette'].value = vignette
        self._bg_vao.render(mode=moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
    
    def _init_quote_overlay(self, quotes, quote_config, frame_count, fps):
        """Initialisiert den PIL-basierten Quote-Overlay-Renderer.

        Wird bei jedem render()-Aufruf mit neuen Quotes neu aufgebaut,
        um Stale-State bei Renderer-Wiederverwendung zu vermeiden.
        """
        self._quote_overlay_renderer = QuoteOverlayRenderer(quotes=quotes, config=quote_config)
        self._quote_overlay_renderer.build_frame_index(frame_count, fps)

    def _mux_audio(self, video_path: str, audio_path: str, output_path: str):
        """Kombiniert Video-Stream mit Original-Audio.

        Args:
            video_path: Pfad zur temporaren Videodatei (ohne Ton).
            audio_path: Pfad zur Original-Audiodatei.
            output_path: Pfad fuer die finale Ausgabedatei.
        """
        cmd = [
            get_ffmpeg_path(),
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
            if hasattr(self, "viz_fbo") and self.viz_fbo:
                self.viz_fbo.release()
                self.viz_fbo = None
            if hasattr(self, "bg_fbo") and self.bg_fbo:
                self.bg_fbo.release()
                self.bg_fbo = None
            if hasattr(self, "viz_ms_fbo") and self.viz_ms_fbo:
                self.viz_ms_fbo.release()
                self.viz_ms_fbo = None
            for tl_attr in ("viz_fbo_b", "viz_fbo_blend"):
                obj = getattr(self, tl_attr, None)
                if obj:
                    try:
                        obj.release()
                    except Exception:
                        pass
                    setattr(self, tl_attr, None)
            if hasattr(self, "_bloom") and self._bloom:
                self._bloom.release()
                self._bloom = None
            for lut_attr in ("_lut_texture", "_lut_placeholder"):
                obj = getattr(self, lut_attr, None)
                if obj:
                    try:
                        obj.release()
                    except Exception:
                        pass
                    setattr(self, lut_attr, None)
            if hasattr(self, "_dummy_black_texture") and self._dummy_black_texture:
                self._dummy_black_texture.release()
                self._dummy_black_texture = None
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
            if hasattr(self, "_quote_overlay_renderer") and self._quote_overlay_renderer:
                self._quote_overlay_renderer = None
            # Zusaetzliche Shader/VAO/VBO aus dem Render-Pipeline freigeben
            for name in ("_pp_prog", "_pp_vao", "_pp_vbo", "_composite_prog",
                         "_composite_vao", "_composite_vbo",
                         "_blit_prog", "_blit_vao", "_blit_vbo", "_bg_prog", "_bg_vao",
                         "_bg_vbo", "_box_prog", "_box_vao", "_box_vbo"):
                obj = getattr(self, name, None)
                if obj:
                    try:
                        obj.release()
                    except Exception:
                        pass
                    setattr(self, name, None)
            if hasattr(self, "ctx") and self.ctx:
                self.ctx.release()
                self.ctx = None
        except Exception:
            pass

    def __del__(self):
        """Gibt GPU-Ressourcen beim Zerstoeren der Instanz frei."""
        try:
            self.release()
        except Exception:
            pass


class GPUPreviewRenderer(GPUBatchRenderer):
    """Schneller Vorschau-Renderer mit reduzierter Aufloesung."""

    def __init__(self, width: int = 854, height: int = 480, fps: int = 30):
        super().__init__(width, height, fps)
