"""QThread-Worker fuer blockierende GUI-Operationen."""

from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image

from src.analyzer import AudioAnalyzer
from src.gpu_preview import render_gpu_preview


class AnalyzeWorker(QThread):
    analysis_ready = pyqtSignal(object)
    analysis_error = pyqtSignal(str)

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
            self.analysis_error.emit(str(e))


class PreviewWorker(QThread):
    preview_ready = pyqtSignal(Image.Image)
    preview_error = pyqtSignal(str)

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
        postprocess: dict | None = None,
        viz_offset_x: float = 0.0,
        viz_offset_y: float = 0.0,
        viz_scale: float = 1.0,
        features=None,
        quotes=None,
        quote_config=None,
        parent=None,
    ):
        super().__init__(parent)
        self.audio_path = audio_path
        self.visualizer_type = visualizer_type
        self.params = params
        self.width = width
        self.height = height
        self.fps = fps
        self.preview_time_percent = preview_time_percent
        self.background_image = background_image
        self.background_blur = background_blur
        self.background_vignette = background_vignette
        self.background_opacity = background_opacity
        self.postprocess = postprocess
        self.viz_offset_x = viz_offset_x
        self.viz_offset_y = viz_offset_y
        self.viz_scale = viz_scale
        self.features = features
        self.quotes = quotes
        self.quote_config = quote_config

    def run(self):
        try:
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
                postprocess=self.postprocess,
                viz_offset_x=self.viz_offset_x,
                viz_offset_y=self.viz_offset_y,
                viz_scale=self.viz_scale,
                features=self.features,
                quotes=self.quotes,
                quote_config=self.quote_config,
            )
            if img is not None:
                self.preview_ready.emit(img)
            else:
                self.preview_error.emit("Preview returned None")
        except Exception as e:
            self.preview_error.emit(str(e))


class RenderWorker(QThread):
    render_progress = pyqtSignal(float)
    render_finished = pyqtSignal(str)
    render_error = pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        # Wird spaeter implementiert, zunaechst nur Stub
        try:
            self.render_finished.emit("output.mp4")
        except Exception as e:
            self.render_error.emit(str(e))
