"""
GPU-Live-Preview fuer schnelles Einzel-Frame-Rendering.

Rendert ein einzelnes Frame mit dem GPU-Renderer und gibt es als
PIL-Image zurueck fuer die Streamlit-Vorschau.
"""

import numpy as np
from PIL import Image

from .analyzer import AudioAnalyzer
from .gpu_renderer import GPUPreviewRenderer
from .gpu_visualizers import get_visualizer


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

    Returns:
        PIL.Image oder None bei Fehler
    """
    try:
        # Audio analysieren
        analyzer = AudioAnalyzer()
        features = analyzer.analyze(audio_path, fps=fps)

        # Renderer erstellen
        renderer = GPUPreviewRenderer(width=width, height=height, fps=fps)

        # Visualizer erstellen
        viz_cls = get_visualizer(visualizer_type)
        viz = viz_cls(renderer.ctx, width, height)
        if params:
            viz.set_params(params)

        # Hintergrundbild laden
        bg_texture = None
        if background_image:
            bg_texture = renderer._load_background_texture(
                background_image, background_blur, background_vignette
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

        viz.render(features_dict, preview_time)

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

        # Ressourcen freigeben
        renderer.__del__()

        return img

    except Exception as e:
        print(f"[GPU Preview] Fehler: {e}")
        return None
