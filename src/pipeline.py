"""
Pipeline - Der Haupt-Orchestrator

Steuert den kompletten Render-Flow von Audio-Analyse bis zum finalen Video.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import tempfile
import subprocess
from PIL import Image, ImageFilter, ImageDraw

from .analyzer import AudioAnalyzer
from .visuals.registry import VisualizerRegistry
from .types import ProjectConfig, AudioFeatures
from .renderers.pil_renderer import PILRenderer
from .postprocess import PostProcessor
from .quote_overlay import QuoteOverlayRenderer, QuoteOverlayConfig


class RenderPipeline:
    """
    Haupt-Controller. Der KI-Agent startet hier.
    """
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.analyzer = AudioAnalyzer()
        self.post_processor = PostProcessor(config.postprocess)
        self.quote_overlay = None
        if config.quotes:
            overlay_cfg = QuoteOverlayConfig(
                enabled=True,
                font_size=config.postprocess.get('quote_font_size', 36),
                font_color=tuple(config.postprocess.get('quote_font_color', [255, 255, 255])),
                box_color=tuple(config.postprocess.get('quote_box_color', [0, 0, 0, 160])),
                display_duration=config.postprocess.get('quote_display_duration', 8.0),
                position=config.postprocess.get('quote_position', 'bottom'),
                font_path=config.postprocess.get('quote_font_path', None),
                text_align=config.postprocess.get('quote_text_align', 'center'),
                latency_offset=config.postprocess.get('quote_latency_offset', 0.0),
                buffer_lookahead=config.postprocess.get('quote_buffer_lookahead', 2.0),
            )
            self.quote_overlay = QuoteOverlayRenderer(config.quotes, overlay_cfg)
        
    def run(self, preview_mode: bool = False, preview_duration: float = 5.0,
            cancel_event=None):
        """
        Führt die komplette Pipeline aus.

        Args:
            preview_mode: Rendert nur erste N Sekunden für schnelles Testen
            preview_duration: Dauer der Vorschau in Sekunden
            cancel_event: Optional threading.Event fuer User-Abbruch.
        """
        audio_path = Path(self.config.audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio nicht gefunden: {audio_path}")

        # Schritt 1: Analyse (oder Cache laden)
        features = self.analyzer.analyze(
            str(audio_path),
            fps=self.config.visual.fps
        )

        print(f"[Pipeline] Audio: {features.duration:.1f}s @ {features.tempo:.0f} BPM")
        print(f"[Pipeline] Mode: {features.mode}, Key: {features.key}")

        # Schritt 2: Visualizer initialisieren
        VisualizerRegistry.autoload()  # Lade alle Plugins
        visualizer_class = VisualizerRegistry.get(self.config.visual.type)
        visualizer = visualizer_class(self.config.visual, features)
        visualizer.setup()

        # Schritt 3: Rendering
        if preview_mode:
            print(f"[Pipeline] PREVIEW MODE: Nur erste {preview_duration} Sekunden")
            features.frame_count = int(preview_duration * features.fps)

        # Frame-Index fuer Quote-Buffer bauen (O(1) Lookups im Render-Loop)
        if self.quote_overlay is not None:
            self.quote_overlay.build_frame_index(features.frame_count, features.fps)

        self._render_video(visualizer, features, audio_path, cancel_event=cancel_event)

        print(f"[Pipeline] Fertig! Output: {self.config.output_file}")
    
    def _render_video(self, visualizer, features: AudioFeatures, audio_path: Path,
                      cancel_event=None):
        """Intern: Frame-Generierung + FFmpeg-Encoding.

        Args:
            visualizer: Der zu verwendende Visualizer.
            features: AudioFeatures fuer die Frame-Generierung.
            audio_path: Pfad zur Audio-Datei fuer das Muxing.
            cancel_event: Optional threading.Event fuer User-Abbruch.
        """
        import threading

        # Temporäre Datei für Video (ohne Audio)
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video.close()

        # Temporäre Datei für FFmpeg stderr (verhindert Blockieren bei langen Videos)
        stderr_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        stderr_file.close()
        stderr_handle = None
        process = None
        cancelled = False

        try:
            # FFmpeg-Writer über subprocess
            fps = features.fps
            width, height = self.config.visual.resolution

            # FFmpeg-Befehl für Video-Encoding
            preset = 'ultrafast' if self.config.turbo_mode else 'medium'
            crf = '28' if self.config.turbo_mode else '23'

            # Frame-Skip: Wir rendern nur jeden N-ten Frame
            frame_skip = max(1, self.config.frame_skip)
            input_fps = fps / frame_skip

            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(input_fps),  # Eingabe-FPS reduziert
                '-i', '-',  # Input von stdin
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', preset,
                '-crf', crf,
                '-r', str(fps),  # Ausgabe-FPS bleibt gleich (FFmpeg dupliziert)
                '-movflags', '+faststart',
                temp_video.name
            ]

            print(f"[Pipeline] Starte Rendering ({features.frame_count} Frames @ {fps}fps)...")

            # Starte FFmpeg-Prozess (stderr in Datei umleiten)
            stderr_handle = open(stderr_file.name, 'w')
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_handle
            )

            # Hintergrundbild vorbereiten (wenn konfiguriert)
            bg_array = None
            if self.config.background_image and Path(self.config.background_image).exists():
                bg_array = self._prepare_background(width, height)

            # Frame-Loop mit Fortschrittsanzeige
            # Progress-Log-Intervall: Bei langen Videos seltener loggen
            total_render_frames = features.frame_count // frame_skip
            log_interval = max(1, total_render_frames // 100)

            for render_idx in range(total_render_frames):
                i = render_idx * frame_skip

                # Cancel-Check: Wenn User abgebrochen hat, sofort raus
                if cancel_event is not None and cancel_event.is_set():
                    print("[Pipeline] Render abgebrochen durch User.")
                    cancelled = True
                    break

                # === RENDER-LOOP PERFORMANCE-GARANTIE ===
                # Dieser Loop darf NIEMALS auf Netzwerk-IO oder blockierende
                # KI-Aufrufe warten. Alle Gemini/Transkription-Operationen muessen
                # VOR dem Loop (via ThreadPoolExecutor/Async) abgeschlossen sein.
                # Verletzung fuehrt zu Frame-Drops und AV-Desynchronisation.

                frame = visualizer.render_frame(i)

                # Post-Processing anwenden (Bloom, Grain, Vignette, Chromatic Aberration, LUT)
                frame = self.post_processor.apply(frame)

                # Hintergrundbild kompositieren (NumPy, sehr schnell)
                if bg_array is not None:
                    frame = self._composite_background_fast(frame, bg_array)

                # Quote Overlays anwenden (zeitbasiert + frame-synchroner Buffer)
                # STRIKT NACH allen Post-Processing-Effekten, damit Typografie nicht
                # von Bloom, Grain oder Chromatic-Aberration verzerrt wird.
                if self.quote_overlay is not None:
                    frame_time = i / features.fps
                    frame = self.quote_overlay.apply(frame, frame_time, frame_idx=i)

                # Direkt zu FFmpeg schreiben (kein Batching - war langsamer)
                process.stdin.write(frame.tobytes())

                # Fortschritt loggen
                if render_idx % log_interval == 0 or render_idx == total_render_frames - 1:
                    progress = (render_idx + 1) / total_render_frames * 100
                    print(f"[Pipeline] Rendering: {progress:.1f}% ({render_idx+1}/{total_render_frames} frames, skip={frame_skip})", flush=True)

            # Schließe stdin und warte auf FFmpeg (nur wenn nicht abgebrochen)
            if not cancelled:
                process.stdin.close()
                process.wait()

                if process.returncode != 0:
                    with open(stderr_file.name, 'r') as f:
                        stderr = f.read().strip() or "Unbekannter Fehler"
                    raise RuntimeError(f"FFmpeg Fehler beim Video-Encoding: {stderr}")

                # Audio hinzufügen (Muxing)
                self._mux_audio(temp_video.name, audio_path, self.config.output_file)

        finally:
            # FFmpeg-Prozess sauber beenden (verhindert Zombies)
            if process is not None and process.poll() is None:
                try:
                    process.stdin.close()
                except Exception:
                    pass
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

            # stderr File-Handle schliessen (verhindert File-Descriptor-Leak)
            if stderr_handle is not None:
                try:
                    stderr_handle.close()
                except Exception:
                    pass

            # Temporaere Dateien aufraeumen
            if Path(temp_video.name).exists():
                Path(temp_video.name).unlink()
            if Path(stderr_file.name).exists():
                Path(stderr_file.name).unlink()
    
    def _prepare_background(self, width: int, height: int) -> np.ndarray:
        """Laedt und bereitet das Hintergrundbild vor (Blur + Vignette).
        
        Gibt ein NumPy-Array zurueck fuer schnelles Compositing.
        """
        bg = Image.open(self.config.background_image).convert('RGB')
        bg = bg.resize((width, height), Image.LANCZOS)
        
        # Blur anwenden
        if self.config.background_blur > 0:
            bg = bg.filter(ImageFilter.GaussianBlur(radius=self.config.background_blur))
        
        bg_array = np.array(bg, dtype=np.float32)
        
        # Vignette anwenden (NumPy, sehr schnell)
        if self.config.background_vignette > 0:
            bg_array = self._apply_vignette_numpy(bg_array, self.config.background_vignette)
        
        return bg_array
    
    def _apply_vignette_numpy(self, img: np.ndarray, strength: float) -> np.ndarray:
        """Wendet eine Vignette auf ein Bild an (NumPy-Version, ~100x schneller als PIL)."""
        h, w = img.shape[:2]
        center_y, center_x = h / 2.0, w / 2.0
        
        # Erstelle Distanz-Maske
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Vignette-Faktor: 1.0 in der Mitte, abnehmend zum Rand
        vignette = 1.0 - (dist / max_dist) * strength
        vignette = np.clip(vignette, 0.3, 1.0)
        
        return (img * vignette[:, :, np.newaxis]).astype(np.float32)
    
    def _composite_background_fast(self, frame: np.ndarray, bg_array: np.ndarray) -> np.ndarray:
        """Mischt einen gerenderten Frame (RGB oder RGBA) mit dem Hintergrundbild.

        Wenn der Frame 4 Kanäle hat, wird der Alpha-Kanal für pixelgenaues
        Compositing verwendet. Der Rückgabewert ist immer RGB (3 Kanäle).
        """
        opacity = self.config.background_opacity
        opacity = max(0.0, min(1.0, opacity))

        if opacity <= 0.01:
            if frame.shape[2] == 4:
                # Alpha voranwenden, da kein Hintergrund sichtbar ist
                return (
                    frame[..., :3].astype(np.float32) * (frame[..., 3:4] / 255.0)
                ).clip(0, 255).astype(np.uint8)
            return frame.astype(np.uint8)

        # Konvertiere zu float32 für präzise Berechnung
        frame_f = frame.astype(np.float32)
        bg_f = bg_array[..., :3].astype(np.float32)

        # 1. Alpha-Kanal extrahieren
        if frame_f.shape[2] == 4:
            alpha = frame_f[..., 3:4] / 255.0
            fg = frame_f[..., :3]
        else:
            alpha = 1.0
            fg = frame_f

        # 3. Pixelgenaues Alpha-Compositing
        # 4. Globale background_opacity verrechnet (skaliert den Hintergrund-Anteil)
        result = fg * alpha + bg_f * opacity * (1.0 - alpha)

        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _mux_audio(self, video_path: str, audio_path: Path, output_path: str):
        """Kombiniert Video mit Original-Audio."""
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', str(audio_path),
            '-c:v', 'copy',      # Video unverändert
            '-c:a', 'aac',       # AAC Audio Codec
            '-b:a', '320k',      # Hohe Audio-Qualität
            '-shortest',         # Kürzeste Länge bestimmt
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg Fehler beim Audio-Muxing: {result.stderr}")


class PreviewPipeline(RenderPipeline):
    """Schnelle Vorschau mit niedrigerer Auflösung."""
    
    def run(self, preview_mode: bool = True, preview_duration: float = 5.0):
        # Überschreibe Config mit 480p für Speed
        original_resolution = self.config.visual.resolution
        original_fps = self.config.visual.fps
        
        self.config.visual.resolution = (854, 480)
        self.config.visual.fps = 30
        
        try:
            super().run(preview_mode=True, preview_duration=preview_duration)
        finally:
            # Stelle Original-Werte wieder her
            self.config.visual.resolution = original_resolution
            self.config.visual.fps = original_fps
