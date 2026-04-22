"""
GPU-Live-Preview fuer schnelles Einzel-Frame-Rendering.

Rendert ein einzelnes Frame mit dem GPU-Renderer und gibt es als
PIL-Image zurueck fuer die Streamlit-Vorschau.

NEU: Renderer und Visualizer werden gecacht, um Shader-Neukompilierung
bei jedem Slider-Zug zu vermeiden.
"""

import numpy as np
from PIL import Image

from .analyzer import AudioAnalyzer
from .gpu_renderer import GPUPreviewRenderer
from .gpu_visualizers import get_visualizer
from .quote_overlay import QuoteOverlayConfig


# === Modul-Level Cache fuer Preview-Renderer ===
# Das verhindert, dass bei jedem Auto-Preview-Slider-Zug der komplette
# GPU-Context + Shader neu aufgebaut werden.
_PREVIEW_CACHE = {
    "key": None,           # (visualizer_type, width, height, fps)
    "renderer": None,      # GPUPreviewRenderer Instanz
    "viz": None,           # Visualizer Instanz
}


def _get_cached_renderer(visualizer_type: str, width: int, height: int, fps: int):
    """Holt oder erstellt einen gecachten Renderer + Visualizer."""
    global _PREVIEW_CACHE

    cache_key = (visualizer_type, width, height, fps)

    # Cache invalidieren wenn sich Visualizer oder Aufloesung aendert
    if _PREVIEW_CACHE["key"] != cache_key:
        _release_preview_cache()
        _PREVIEW_CACHE["key"] = cache_key

    # Renderer erstellen falls noetig
    if _PREVIEW_CACHE["renderer"] is None:
        _PREVIEW_CACHE["renderer"] = GPUPreviewRenderer(width=width, height=height, fps=fps)

    # Visualizer erstellen falls noetig
    if _PREVIEW_CACHE["viz"] is None or _PREVIEW_CACHE["viz_type"] != visualizer_type:
        viz_cls = get_visualizer(visualizer_type)
        _PREVIEW_CACHE["viz"] = viz_cls(_PREVIEW_CACHE["renderer"].ctx, width, height)
        _PREVIEW_CACHE["viz_type"] = visualizer_type

    return _PREVIEW_CACHE["renderer"], _PREVIEW_CACHE["viz"]


def _release_preview_cache():
    """Gibt den gecachten Renderer ordentlich frei."""
    global _PREVIEW_CACHE
    if _PREVIEW_CACHE["renderer"] is not None:
        try:
            _PREVIEW_CACHE["renderer"].release()
        except Exception:
            pass
        _PREVIEW_CACHE["renderer"] = None
    _PREVIEW_CACHE["viz"] = None
    _PREVIEW_CACHE["viz_type"] = None
    _PREVIEW_CACHE["key"] = None


def render_gpu_preview(
    audio_path: str,
    visualizer_type: str,
    params: dict = None,
    width: int = 480,
    height: int = 270,
    fps: int = 30,
    preview_time_percent: float = 0.3,
    background_image: str = None,
    background_blur: float = 0.0,
    background_vignette: float = 0.0,
    background_opacity: float = 0.3,
    postprocess: dict = None,
    quotes: list = None,
    quote_config: QuoteOverlayConfig = None,
    viz_offset_x: float = 0.0,
    viz_offset_y: float = 0.0,
    viz_scale: float = 1.0,
):
    """
    Rendert ein einzelnes Frame fuer die Live-Vorschau.

    Args:
        audio_path: Pfad zur Audio-Datei
        visualizer_type: Name des GPU-Visualizers
        params: Visualizer-Parameter (optional)
        width: Breite des Preview-Frames
        height: Hoehe des Preview-Frames
        fps: FPS fuer Feature-Extraktion
        preview_time_percent: Zeitpunkt im Audio (0.0-1.0)
        background_image: Pfad zum Hintergrundbild (optional)
        background_blur: Blur fuer Hintergrund
        background_vignette: Vignette fuer Hintergrund
        background_opacity: Opacity fuer Hintergrund
        viz_offset_x: Horizontaler Offset in normalisierten Koordinaten (-1.0 bis 1.0).
        viz_offset_y: Vertikaler Offset in normalisierten Koordinaten (-1.0 bis 1.0).
        viz_scale: Skalierungsfaktor des Visualizers (0.5 bis 2.0).

    Returns:
        PIL.Image oder None bei Fehler
    """
    try:
        # Audio analysieren (gecached)
        analyzer = AudioAnalyzer()
        features = analyzer.analyze(audio_path, fps=fps)

        # Renderer und Visualizer aus Cache holen (NICHT jedes Mal neu erstellen)
        renderer, viz = _get_cached_renderer(visualizer_type, width, height, fps)

        # Parameter aktualisieren (billig, kein Neuerstellen noetig)
        if params:
            viz.set_params(params)

        # Hintergrundbild laden
        bg_texture = None
        if background_image:
            bg_texture = renderer._load_background_texture(
                background_image, background_blur
            )

        # Feature-Dict vorbereiten
        features_dict = {
            "rms": features.rms,
            "onset": features.onset,
            "chroma": features.chroma,
            "spectral_centroid": features.spectral_centroid,
            "fps": fps,
            "frame_count": features.frame_count,
        }

        # Zeitpunkt fuer Preview
        preview_time = features.duration * preview_time_percent

        # Frame rendern
        renderer.fbo.use()
        renderer.ctx.clear(0.05, 0.05, 0.05)

        if bg_texture is not None:
            renderer._render_background(bg_texture, background_opacity)

        # Visualizer in temporären viz_fbo rendern
        renderer.viz_fbo.use()
        renderer.ctx.clear(0.0, 0.0, 0.0, 0.0)
        viz.render(features_dict, preview_time)

        # Visualizer von viz_fbo auf main fbo blitten (mit Offset/Scale)
        renderer.fbo.use()
        renderer._blit_viz_to_fbo(
            renderer.viz_fbo.color_attachments[0],
            offset_x=viz_offset_x,
            offset_y=viz_offset_y,
            scale=viz_scale,
        )

        # Quote-Overlays auf GPU rendern
        if quotes and quote_config and quote_config.enabled:
            renderer._init_text_renderer()
            renderer._render_quotes_gpu(preview_time, quotes, quote_config)

        # Post-Process (Color-Grading) anwenden falls konfiguriert
        if postprocess:
            renderer._apply_postprocess(
                renderer.fbo.color_attachments[0],
                contrast=postprocess.get("contrast", 1.0),
                saturation=postprocess.get("saturation", 1.0),
                brightness=postprocess.get("brightness", 0.0),
                warmth=postprocess.get("warmth", 0.0),
                film_grain=postprocess.get("film_grain", 0.0),
                time=preview_time,
            )
            pixels = renderer.post_fbo.read(components=3)
        else:
            pixels = renderer.fbo.read(components=3)

        # Zu PIL Image konvertieren
        img_array = np.frombuffer(pixels, dtype=np.uint8)
        img_array = img_array.reshape((height, width, 3))
        # Flip vertically (OpenGL hat Ursprung unten links)
        img_array = np.flipud(img_array)
        img = Image.fromarray(img_array, mode='RGB')

        return img

    except Exception as e:
        print(f"[GPU Preview] Fehler: {e}")
        return None
