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


# --- Differenz-Render und Metriken M1/M2/M3/M5/M6 (Spec §3.2, §3.3) ---


def contribution(a_linear: np.ndarray, b_linear: np.ndarray) -> np.ndarray:
    """Post-FX-wirksamer Visualizer-Einfluss pro Pixel (H, W, float32)."""
    diff = np.abs(a_linear.astype(np.float32) - b_linear.astype(np.float32))
    return np.clip(diff.mean(axis=-1), 0.0, 1.0)


def overlay_energy(contrib: np.ndarray) -> float:
    """M1: mittlere Overlay-Energie (kontinuierlich, hart)."""
    return float(np.mean(contrib))


def overlay_coverage(contrib: np.ndarray, threshold: float = 0.5) -> float:
    """M2: Flächenanteil oberhalb der Schwelle (weich/warn)."""
    return float(np.mean(contrib > threshold))


def subject_disturbance(contrib: np.ndarray, mask: np.ndarray) -> float:
    """M3: maskengewichtete Störung; 0.0 wenn keine Subjektfläche."""
    denom = float(mask.sum())
    if denom <= 0.0:
        return 0.0
    return float((contrib * mask).sum() / denom)


def vitality(contrib_t: np.ndarray, contrib_t_delta: np.ndarray) -> float:
    """M5: mittlere zeitliche Änderung zwischen zwei contrib-Maps."""
    return float(np.mean(np.abs(contrib_t_delta - contrib_t)))


def integrity_violations(frame_linear: np.ndarray) -> list[str]:
    """M6: binäre Integritätsprüfung (NaN/Inf, Blackframe, Clipping)."""
    violations: list[str] = []
    if not np.isfinite(frame_linear).all():
        violations.append("nan_inf")
    luma = frame_linear.mean(axis=-1)
    if float(np.percentile(luma, 99)) < 0.02:
        violations.append("blackframe")
    clipped = frame_linear >= 1.0 - 1e-6
    if float(np.mean(clipped)) > 0.15:
        violations.append("clipping")
    return violations


# --- M4: Text-Kontrast (Spec §3.3, Glyphenmaske + Ring, WCAG 2.x) ---


def _relative_luminance(rgb_linear: np.ndarray) -> np.ndarray:
    """WCAG-relative Luminanz auf bereits linearem RGB."""
    return (
        0.2126 * rgb_linear[..., 0]
        + 0.7152 * rgb_linear[..., 1]
        + 0.0722 * rgb_linear[..., 2]
    )


def text_contrast_wcag(
    frame_linear: np.ndarray,
    glyph_mask: np.ndarray,
    ring_dilate_px: int = 3,
) -> float:
    """Kontrast-Ratio Glyphen vs. dilatierter Hintergrund-Ring (WCAG 2.x).

    Vordergrund = p5 der Glyphen-Luminanz (Worst-Case-nah), Hintergrund =
    Median des Rings um die Glyphen. 0.0 bei leerer Glyphenmaske.
    """
    from PIL import ImageFilter

    glyph = np.asarray(glyph_mask, dtype=bool)
    if not glyph.any():
        return 0.0
    kernel = 2 * ring_dilate_px + 1
    dilated = np.asarray(
        Image.fromarray(glyph).filter(ImageFilter.MaxFilter(kernel)), dtype=bool
    )
    ring = dilated & ~glyph
    if not ring.any():
        return 0.0
    luma = _relative_luminance(frame_linear)
    fg = float(np.percentile(luma[glyph], 5))
    bg = float(np.median(luma[ring]))
    hi, lo = max(fg, bg), min(fg, bg)
    return (hi + 0.05) / (lo + 0.05)


def aggregate_text_contrast(per_frame_ratios: list[float]) -> float:
    """M4-Aggregation: Minimum über alle Sample-Frames (Spec §3.3)."""
    valid = [r for r in per_frame_ratios if r > 0.0]
    return min(valid) if valid else 0.0
