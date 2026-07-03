"""
GPU-Live-Preview fuer schnelles Einzel-Frame-Rendering.

Rendert ein einzelnes Frame mit dem GPU-Renderer und gibt es als
PIL-Image zurueck fuer die GUI-Vorschau.
"""

import os
import traceback

import numpy as np
from PIL import Image

from .analyzer import AudioAnalyzer
from .app_logging import get_logger

logger = get_logger(__name__)


def _hex_to_rgb(hex_color: str) -> tuple:
    """Wandelt einen 6-stelligen Hex-String in RGB (0.0-1.0) um."""
    hex_color = hex_color.lstrip('#')
    return (
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
    )
from .gpu_renderer import GPUPreviewRenderer
from .gpu_visualizers import get_visualizer
from .quote_overlay import QuoteOverlayConfig, QuoteOverlayRenderer
from .types import AudioFeatures


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
    background_color: str = "#0A0A0A",
    postprocess: dict = None,
    quotes: list = None,
    quote_config: QuoteOverlayConfig = None,
    viz_offset_x: float = 0.0,
    viz_offset_y: float = 0.0,
    viz_scale: float = 1.0,
    features: AudioFeatures = None,
    cancel_check=None,
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
        cancel_check: Optionales Callable; gibt es True zurueck, wird die
            Vorschau abgebrochen (Rueckgabe None). Erlaubt kooperativen
            Abbruch verworfener Previews aus dem Worker-Thread.

    Returns:
        PIL.Image oder None bei Fehler/Abbruch
    """
    def _cancelled():
        return cancel_check is not None and cancel_check()

    temp_bg_frame = None
    bg_texture = None
    renderer = None
    try:
        # Audio analysieren (gecached) nur wenn nicht schon vorhanden
        if features is None:
            analyzer = AudioAnalyzer()
            features = analyzer.analyze(audio_path, fps=fps)

        if _cancelled():
            return None

        # Fuer jeden Preview-Frame einen frischen Renderer erstellen.
        # Das verhindert Cross-Thread-Probleme mit dem ModernGL-Context,
        # wenn der Preview-Worker in einem QThread laeuft.
        renderer = GPUPreviewRenderer(width=width, height=height, fps=fps)
        viz_cls = get_visualizer(visualizer_type)
        viz = viz_cls(renderer.ctx, width, height)

        # Parameter aktualisieren
        if params:
            viz.set_params(params)

        # Hintergrundbild laden
        if background_image:
            try:
                if renderer._is_video_file(background_image):
                    preview_time = features.duration * preview_time_percent
                    temp_bg_frame = renderer._extract_video_frame_at_time(
                        background_image, preview_time, width, height
                    )
                    bg_texture = renderer._load_background_texture(
                        temp_bg_frame, background_blur
                    )
                else:
                    bg_texture = renderer._load_background_texture(
                        background_image, background_blur
                    )
            except Exception as e:
                logger.warning(f'[GPU Preview] Konnte Hintergrundbild nicht laden: {e}')
                bg_texture = None

        # Beat-Intensity vektorisiert berechnen (fuer Visualizer die es brauchen)
        frame_count = features.frame_count
        beat_intensity = np.zeros(frame_count, dtype=np.float32)
        if len(features.beat_frames) > 0:
            decay_frames = max(3, int(fps * 0.1))
            for bf in features.beat_frames:
                if bf >= frame_count:
                    continue
                end = min(bf + decay_frames + 1, frame_count)
                if end > bf:
                    dists = np.arange(end - bf, dtype=np.float32)
                    vals = 1.0 - dists / decay_frames
                    vals = np.clip(vals, 0.0, 1.0)
                    beat_intensity[bf:end] = np.maximum(beat_intensity[bf:end], vals)

        # Feature-Dict vorbereiten (vollstaendig wie im Haupt-Renderer)
        def _slice_or_zeros(arr, fc):
            if arr is None or len(arr) == 0:
                return np.zeros(fc, dtype=np.float32)
            return arr[:fc]

        features_dict = {
            "rms": _slice_or_zeros(features.rms, frame_count),
            "onset": _slice_or_zeros(features.onset, frame_count),
            "chroma": features.chroma[:, :frame_count] if features.chroma.ndim > 1 and features.chroma.shape[1] >= frame_count else features.chroma,
            "spectral_centroid": _slice_or_zeros(features.spectral_centroid, frame_count),
            "spectral_rolloff": _slice_or_zeros(features.spectral_rolloff, frame_count),
            "zero_crossing_rate": _slice_or_zeros(features.zero_crossing_rate, frame_count),
            "transient": _slice_or_zeros(features.transient, frame_count),
            "voice_clarity": _slice_or_zeros(features.voice_clarity, frame_count),
            "voice_band": _slice_or_zeros(features.voice_band, frame_count),
            "mfcc": features.mfcc[:, :frame_count] if features.mfcc.ndim > 1 and features.mfcc.shape[1] >= frame_count else features.mfcc,
            "tempogram": features.tempogram[:, :frame_count] if features.tempogram.ndim > 1 and features.tempogram.shape[1] >= frame_count else features.tempogram,
            "beat_frames": features.beat_frames,
            "beat_intensity": beat_intensity,
            "tempo": float(features.tempo),
            "mode": features.mode,
            "duration": float(features.duration),
            "fps": fps,
            "frame_count": frame_count,
        }

        # Zeitpunkt fuer Preview
        preview_time = features.duration * preview_time_percent

        if _cancelled():
            return None

        # Frame rendern
        renderer.fbo.use()
        if bg_texture is None:
            bg_rgb = _hex_to_rgb(background_color)
            renderer.ctx.clear(bg_rgb[0], bg_rgb[1], bg_rgb[2])
        else:
            renderer.ctx.clear(0.0, 0.0, 0.0)

        if bg_texture is not None:
            renderer._render_background(bg_texture, background_opacity, background_vignette)

        # Visualizer rendern: mit MSAA (falls verfuegbar), wie im Haupt-Renderer
        if getattr(renderer, "viz_ms_fbo", None) is not None:
            renderer.viz_ms_fbo.use()
            renderer.ctx.clear(0.0, 0.0, 0.0, 0.0)
            viz.render(features_dict, preview_time)
            renderer.ctx.copy_framebuffer(renderer.viz_fbo, renderer.viz_ms_fbo)
        else:
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

        pp = postprocess or {}

        # HDR-Bloom wie im Haupt-Renderer (vor dem Tonemapping)
        bloom_intensity = pp.get("bloom_intensity", 0.6)
        if getattr(renderer, "_bloom", None) is not None and bloom_intensity > 0.0:
            renderer._apply_bloom(
                intensity=bloom_intensity,
                threshold=pp.get("bloom_threshold", 1.0),
                radius=pp.get("bloom_radius", 1.0),
            )

        # Finaler Pass laeuft IMMER (Tonemap + Dither), damit die Vorschau
        # exakt dem gerenderten Video entspricht.
        renderer._apply_postprocess(
            renderer.fbo.color_attachments[0],
            contrast=pp.get("contrast", 1.0),
            saturation=pp.get("saturation", 1.0),
            brightness=pp.get("brightness", 0.0),
            warmth=pp.get("warmth", 0.0),
            film_grain=pp.get("film_grain", 0.0),
            time=preview_time,
            exposure=pp.get("exposure", 1.0),
            vignette=pp.get("vignette", 0.0),
            chromatic_aberration=pp.get("chromatic_aberration", 0.0),
            lut_path=pp.get("lut"),
            lut_strength=pp.get("lut_strength", 1.0),
        )
        if _cancelled():
            return None

        pixels = renderer.post_fbo.read(components=3)

        # Zu PIL Image konvertieren
        img_array = np.frombuffer(pixels, dtype=np.uint8)
        img_array = img_array.reshape((height, width, 3))

        # Quote-Overlays mit PIL anwenden (nach GPU-Rendering)
        if quotes and quote_config and quote_config.enabled:
            quote_renderer = QuoteOverlayRenderer(quotes=quotes, config=quote_config)
            quote_renderer.build_frame_index(frame_count, fps)
            preview_frame_idx = int(preview_time * fps)
            img_array = quote_renderer.apply(img_array, preview_time, frame_idx=preview_frame_idx)

        # ModernGL fbo.read() gibt bereits top-down (PIL-kompatibel)
        img = Image.fromarray(img_array)

        return img

    except Exception as e:
        logger.error(f"[GPU Preview] Fehler: {e}")
        logger.debug(traceback.format_exc())
        return None
    finally:
        if bg_texture is not None:
            try:
                bg_texture.release()
            except Exception:
                pass
        if temp_bg_frame and os.path.exists(temp_bg_frame):
            os.unlink(temp_bg_frame)
        if renderer is not None:
            try:
                renderer.release()
            except Exception:
                pass
