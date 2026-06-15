"""Zentraler Zustand fuer die Audio Visualizer Pro GUI."""

from PyQt6.QtCore import QObject, pyqtSignal
from src.quote_overlay import QuoteOverlayConfig


class AppState(QObject):
    changed = pyqtSignal(str)

    _STATE_KEYS = frozenset({
        "audio_path", "features", "audio_duration",
        "visualizer_type", "viz_params", "viz_offset_x", "viz_offset_y", "viz_scale",
        "color_mode", "base_hue", "color_saturation",
        "background_path", "bg_blur", "bg_vignette", "bg_opacity",
        "pp_contrast", "pp_saturation", "pp_brightness", "pp_warmth", "pp_grain",
        "preview_time_percent", "preview_fps", "preview_width", "preview_height",
        "resolution", "render_fps", "codec", "quality", "gpu_encode", "output_dir",
        "quotes", "quotes_enabled", "quote_config",
        "status_message", "status_kind",
    })

    def __init__(self, parent=None):
        super().__init__(parent)
        object.__setattr__(self, "_initialized", False)

        self.audio_path: str | None = None
        self.features = None
        self.audio_duration: float = 0.0

        self.visualizer_type: str = "lumina_core"
        self.viz_params: dict = {}
        self.viz_offset_x: float = 0.0
        self.viz_offset_y: float = 0.0
        self.viz_scale: float = 1.0

        self.color_mode: str = "chroma"
        self.base_hue: float = 0.55
        self.color_saturation: float = 0.7

        self.background_path: str | None = None
        self.bg_blur: float = 0.0
        self.bg_vignette: float = 0.0
        self.bg_opacity: float = 0.3

        self.pp_contrast: float = 1.0
        self.pp_saturation: float = 1.0
        self.pp_brightness: float = 0.0
        self.pp_warmth: float = 0.0
        self.pp_grain: float = 0.0

        self.preview_time_percent: float = 0.3
        self.preview_fps: int = 30
        self.preview_width: int = 854
        self.preview_height: int = 480

        self.resolution: tuple[int, int] = (1920, 1080)
        self.render_fps: int = 30
        self.codec: str = "h264"
        self.quality: str = "high"
        self.gpu_encode: bool = False
        self.output_dir: str = "output"

        self.quotes: list = []
        self.quotes_enabled: bool = False
        self.quote_config: QuoteOverlayConfig = QuoteOverlayConfig(enabled=True)

        self.status_message: str = "Bereit."
        self.status_kind: str = "info"  # info | ok | warn | error

        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if getattr(self, "_initialized", False) and key in self._STATE_KEYS:
            self._notify(key)

    def _notify(self, key: str):
        self.changed.emit(key)

    def set(self, key: str, value):
        if hasattr(self, key):
            setattr(self, key, value)

    def get_postprocess(self) -> dict:
        return {
            "contrast": self.pp_contrast,
            "saturation": self.pp_saturation,
            "brightness": self.pp_brightness,
            "warmth": self.pp_warmth,
            "film_grain": self.pp_grain,
        }

    def get_params(self) -> dict:
        base = {
            "offset_x": self.viz_offset_x,
            "offset_y": self.viz_offset_y,
            "scale": self.viz_scale,
            "color_mode": self.color_mode,
            "base_hue": self.base_hue,
            "color_saturation": self.color_saturation,
        }
        base.update(self.viz_params)
        return base

    def to_dict(self) -> dict:
        qc = self.quote_config
        return {
            "version": 1,
            "audio_path": self.audio_path,
            "background_path": self.background_path,
            "visualizer_type": self.visualizer_type,
            "viz_params": self.viz_params,
            "viz_offset_x": self.viz_offset_x,
            "viz_offset_y": self.viz_offset_y,
            "viz_scale": self.viz_scale,
            "color_mode": self.color_mode,
            "base_hue": self.base_hue,
            "color_saturation": self.color_saturation,
            "bg_blur": self.bg_blur,
            "bg_vignette": self.bg_vignette,
            "bg_opacity": self.bg_opacity,
            "pp_contrast": self.pp_contrast,
            "pp_saturation": self.pp_saturation,
            "pp_brightness": self.pp_brightness,
            "pp_warmth": self.pp_warmth,
            "pp_grain": self.pp_grain,
            "preview_time_percent": self.preview_time_percent,
            "preview_fps": self.preview_fps,
            "resolution": list(self.resolution),
            "render_fps": self.render_fps,
            "codec": self.codec,
            "quality": self.quality,
            "gpu_encode": self.gpu_encode,
            "output_dir": self.output_dir,
            "quotes_enabled": self.quotes_enabled,
            "quote_config": {
                "enabled": qc.enabled,
                "position": qc.position,
                "font_size": qc.font_size,
                "font_color": list(qc.font_color),
                "box_color": list(qc.box_color),
                "display_duration": qc.display_duration,
                "fade_duration": qc.fade_duration,
                "max_chars_per_line": qc.max_chars_per_line,
                "line_spacing": qc.line_spacing,
                "text_align": qc.text_align,
            },
        }

    @classmethod
    def from_dict(cls, data: dict):
        s = cls()
        for key, value in data.items():
            if key == "quote_config" and isinstance(value, dict):
                s.quote_config = QuoteOverlayConfig(**value)
            elif key == "resolution" and isinstance(value, list):
                s.resolution = tuple(value)
            elif hasattr(s, key):
                setattr(s, key, value)
        return s
