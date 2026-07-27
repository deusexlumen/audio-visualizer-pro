"""Messraster und Metriken M1–M6 (Spec §3.1, §3.3).

Alle Metriken rechnen auf dem normalisierten Messraster: lange Kante
854 px, Linear-Light, float32 in [0, 1].
"""

import numpy as np
from PIL import Image

MEASURE_LONG_EDGE = 854


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """sRGB (float, [0,1]) -> Linear-Light (float, [0,1])."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def to_measure_raster(frame: np.ndarray, long_edge: int = MEASURE_LONG_EDGE) -> np.ndarray:
    """uint8-RGB-Frame -> normalisiertes Messraster (float32, linear).

    Downscale per BOX (Area-Mittelung), Seitenverhältnis bleibt erhalten,
    kein Upscale.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, long_edge / max(h, w))
    if scale < 1.0:
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.BOX)
        frame = np.asarray(img)
    return srgb_to_linear(frame.astype(np.float32) / 255.0)
