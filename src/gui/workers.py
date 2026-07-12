"""QThread-Worker fuer blockierende GUI-Operationen."""

import threading
import traceback

from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image

from src.analyzer import AudioAnalyzer


class AnalyzeWorker(QThread):
    analysis_ready = pyqtSignal(object)
    # Alle *_error-Signale liefern (Meldung, Traceback) — Traceback nur fuers Log
    analysis_error = pyqtSignal(str, str)

    def __init__(self, audio_path: str, fps: int = 30, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.fps = fps

    def run(self):
        try:
            analyzer = AudioAnalyzer()
            features = analyzer.analyze(self.audio_path, fps=self.fps)
            self.analysis_ready.emit(features)
        except Exception as e:
            self.analysis_error.emit(str(e), traceback.format_exc())


class PreviewWorker(QThread):
    preview_ready = pyqtSignal(Image.Image)
    preview_error = pyqtSignal(str, str)

    def __init__(
        self,
        audio_path: str,
        visualizer_type: str,
        width: int,
        height: int,
        fps: int,
        preview_time_percent: float,
        params: dict | None = None,
        background_image: str | None = None,
        background_blur: float = 0.0,
        background_vignette: float = 0.0,
        background_opacity: float = 0.3,
        background_color: str = "#0A0A0A",
        postprocess: dict | None = None,
        viz_offset_x: float = 0.0,
        viz_offset_y: float = 0.0,
        viz_scale: float = 1.0,
        features=None,
        quotes=None,
        quote_config=None,
        timeline=None,
        parent=None,
    ):
        super().__init__(parent)
        self.audio_path = audio_path
        self.visualizer_type = visualizer_type
        self.params = params
        self.timeline = timeline
        self.width = width
        self.height = height
        self.fps = fps
        self.preview_time_percent = preview_time_percent
        self.background_image = background_image
        self.background_blur = background_blur
        self.background_vignette = background_vignette
        self.background_opacity = background_opacity
        self.background_color = background_color
        self.postprocess = postprocess
        self.viz_offset_x = viz_offset_x
        self.viz_offset_y = viz_offset_y
        self.viz_scale = viz_scale
        self.features = features
        self.quotes = quotes
        self.quote_config = quote_config

    def run(self):
        try:
            from src.gpu_preview import render_gpu_preview

            img = render_gpu_preview(
                audio_path=self.audio_path,
                visualizer_type=self.visualizer_type,
                params=self.params,
                width=self.width,
                height=self.height,
                fps=self.fps,
                preview_time_percent=self.preview_time_percent,
                background_image=self.background_image,
                background_blur=self.background_blur,
                background_vignette=self.background_vignette,
                background_opacity=self.background_opacity,
                background_color=self.background_color,
                postprocess=self.postprocess,
                viz_offset_x=self.viz_offset_x,
                viz_offset_y=self.viz_offset_y,
                viz_scale=self.viz_scale,
                features=self.features,
                quotes=self.quotes,
                quote_config=self.quote_config,
                cancel_check=self.isInterruptionRequested,
                timeline=self.timeline,
            )
            if img is not None:
                self.preview_ready.emit(img)
            elif self.isInterruptionRequested():
                # Abbruch durch neuere Vorschau — kein Fehler
                pass
            else:
                self.preview_error.emit("Vorschau lieferte kein Bild", "")
        except Exception as e:
            self.preview_error.emit(str(e), traceback.format_exc())


class RenderWorker(QThread):
    render_progress = pyqtSignal(float)
    render_finished = pyqtSignal(str)
    render_error = pyqtSignal(str, str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._cancel_event = threading.Event()

    def cancel(self):
        """Bricht das Rendering asynchron ab."""
        self._cancel_event.set()

    def run(self):
        """Fuehrt den GPU-Render-Job aus."""
        renderer = None
        try:
            from src.gpu_renderer import GPUBatchRenderer

            width = self.config.get("width", 1920)
            height = self.config.get("height", 1080)
            fps = self.config.get("fps", 30)

            renderer = GPUBatchRenderer(width=width, height=height, fps=fps)

            def _progress(frame: int, total: int):
                if total > 0:
                    self.render_progress.emit(frame / total)

            renderer.render(
                audio_path=self.config["audio_path"],
                visualizer_type=self.config["visualizer_type"],
                output_path=self.config["output_path"],
                features=self.config.get("features"),
                params=self.config.get("params", {}),
                background_image=self.config.get("background_image"),
                background_blur=self.config.get("background_blur", 0.0),
                background_vignette=self.config.get("background_vignette", 0.0),
                background_opacity=self.config.get("background_opacity", 0.3),
                background_color=self.config.get("background_color", "#0A0A0A"),
                postprocess=self.config.get("postprocess", {}),
                quotes=self.config.get("quotes"),
                quote_config=self.config.get("quote_config"),
                codec=self.config.get("codec", "h264"),
                quality=self.config.get("quality", "high"),
                gpu_encode=self.config.get("gpu_encode", False),
                viz_offset_x=self.config.get("viz_offset_x", 0.0),
                viz_offset_y=self.config.get("viz_offset_y", 0.0),
                viz_scale=self.config.get("viz_scale", 1.0),
                progress_callback=_progress,
                cancel_event=self._cancel_event,
                timeline=self.config.get("timeline"),
            )

            if self._cancel_event.is_set():
                self.render_error.emit("Rendering abgebrochen.", "")
            else:
                self.render_finished.emit(self.config["output_path"])
        except Exception as e:
            self.render_error.emit(str(e), traceback.format_exc())
        finally:
            if renderer is not None:
                try:
                    renderer.release()
                except Exception:
                    pass


class IntroWorker(QThread):
    intro_progress = pyqtSignal(float)
    intro_finished = pyqtSignal(str)
    intro_error = pyqtSignal(str, str)

    def __init__(
        self,
        intro_path: str,
        main_video_path: str,
        output_path: str,
        fade_duration: float = 1.0,
        parent=None,
    ):
        super().__init__(parent)
        self.intro_path = intro_path
        self.main_video_path = main_video_path
        self.output_path = output_path
        self.fade_duration = fade_duration
        self._cancel_event = threading.Event()

    def cancel(self):
        """Bricht das Intro-Rendering asynchron ab."""
        self._cancel_event.set()

    def run(self):
        """Fuegt das Intro vor das Haupt-Video."""
        try:
            from src.intro_renderer import render_with_intro

            def _progress(p: float):
                self.intro_progress.emit(p)

            render_with_intro(
                intro_path=self.intro_path,
                main_video_path=self.main_video_path,
                output_path=self.output_path,
                fade_duration=self.fade_duration,
                progress_callback=_progress,
                cancel_event=self._cancel_event,
            )

            if self._cancel_event.is_set():
                self.intro_error.emit("Intro-Rendering abgebrochen.", "")
            else:
                self.intro_finished.emit(self.output_path)
        except Exception as e:
            self.intro_error.emit(str(e), traceback.format_exc())


class AIOptimizeWorker(QThread):
    optimize_ready = pyqtSignal(dict)
    optimize_error = pyqtSignal(str, str)

    def __init__(
        self,
        gemini,
        visualizer_type: str,
        current_params: dict,
        audio_features: dict,
        colors: dict,
        param_specs: dict | None = None,
        user_prompt: str | None = None,
        recommendation: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.gemini = gemini
        self.visualizer_type = visualizer_type
        self.current_params = current_params
        self.audio_features = audio_features
        self.colors = colors
        self.param_specs = param_specs or {}
        self.user_prompt = user_prompt
        self.recommendation = recommendation

    def run(self):
        try:
            future = self.gemini.optimize_all_settings_async(
                visualizer_type=self.visualizer_type,
                current_params=self.current_params,
                audio_features=self.audio_features,
                colors=self.colors,
                param_specs=self.param_specs,
                user_prompt=self.user_prompt,
                recommendation=self.recommendation,
            )
            result = future.result(timeout=60)
            self.optimize_ready.emit(result)
        except Exception as e:
            self.optimize_error.emit(str(e), traceback.format_exc())


class QuoteExtractWorker(QThread):
    quotes_ready = pyqtSignal(list)
    quotes_error = pyqtSignal(str, str)

    def __init__(
        self,
        gemini,
        audio_path: str,
        audio_duration: float | None = None,
        max_quotes: int | None = None,
        use_cache: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.gemini = gemini
        self.audio_path = audio_path
        self.audio_duration = audio_duration
        self.max_quotes = max_quotes
        self.use_cache = use_cache

    def run(self):
        try:
            future = self.gemini.extract_quotes_async(
                audio_path=self.audio_path,
                audio_duration=self.audio_duration,
                max_quotes=self.max_quotes,
                use_cache=self.use_cache,
            )
            quotes = future.result(timeout=120)
            self.quotes_ready.emit(quotes)
        except Exception as e:
            self.quotes_error.emit(str(e), traceback.format_exc())


class FullAIWorker(QThread):
    """Voll-KI-Modus: Segmentierung -> Timeline (offline + optional Gemini)."""

    progress = pyqtSignal(str)
    timeline_ready = pyqtSignal(object)   # src.types.Timeline
    ai_error = pyqtSignal(str, str)

    def __init__(self, features, gemini=None, use_gemini=True, parent=None):
        super().__init__(parent)
        self.features = features
        self.gemini = gemini
        self.use_gemini = use_gemini

    def run(self):
        try:
            from src.segmentation import segment_audio
            from src.ai_matcher import SmartMatcher

            self.progress.emit("Analysiere Songstruktur...")
            segments = segment_audio(self.features)

            self.progress.emit("Erstelle Szenen-Timeline...")
            matcher = SmartMatcher()
            timeline = matcher.suggest_timeline(self.features, segments)

            # Optionale Gemini-Verfeinerung (Labels + Visualizer je Segment)
            if self.use_gemini and self.gemini is not None and segments:
                try:
                    self.progress.emit("KI verfeinert die Szenen...")
                    from src.gpu_visualizers import list_visualizers
                    stats = [
                        {"start": round(s.start, 2), "end": round(s.end, 2), **s.stats}
                        for s in segments
                    ]
                    refined = self.gemini.generate_scene_timeline(
                        segments_stats=stats,
                        available_visualizers=list_visualizers(),
                        mode=getattr(self.features, "mode", "music"),
                    )
                    for item in refined:
                        idx = item.get("index")
                        if isinstance(idx, int) and 0 <= idx < len(timeline.scenes):
                            timeline.scenes[idx].visualizer = item["visualizer"]
                            if item.get("label"):
                                timeline.scenes[idx].label = item["label"]
                except Exception as e:
                    # Gemini optional — Offline-Timeline bleibt gueltig
                    import logging
                    logging.getLogger("avp.gui.workers").warning(
                        f"Gemini-Timeline uebersprungen: {e}"
                    )

            self.progress.emit(f"{len(timeline.scenes)} Szenen erstellt.")
            self.timeline_ready.emit(timeline)
        except Exception as e:
            self.ai_error.emit(str(e), traceback.format_exc())


class TranscribeWorker(QThread):
    transcribe_ready = pyqtSignal(str)
    transcribe_error = pyqtSignal(str, str)

    def __init__(self, gemini, audio_path: str, parent=None):
        super().__init__(parent)
        self.gemini = gemini
        self.audio_path = audio_path

    def run(self):
        try:
            future = self.gemini.transcribe_audio_async(audio_path=self.audio_path)
            transcript = future.result(timeout=300)
            self.transcribe_ready.emit(transcript)
        except Exception as e:
            self.transcribe_error.emit(str(e), traceback.format_exc())
