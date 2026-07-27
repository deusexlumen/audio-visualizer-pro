"""Subjekt-Masken-Service (Spec §6.2).

Erzeugt pro Hintergrundbild eine Salienz-Maske im Quellbildraum
(float32, HxW, [0,1]; 1 = Subjekt) und cached sie als NPZ.
Fallback-Kette: rembg/u2net -> OpenCV GrabCut -> Zentrums-Gauß.
Cache-Key = sha256(Bilddatei) + provider_id + model_hash + service_version.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

SERVICE_VERSION = "mask-service/1"
DEFAULT_CACHE_DIR = ".cache/subject_masks"


@dataclass
class MaskResult:
    """Ergebnis der Maskenerzeugung inkl. Provenance."""

    mask: np.ndarray
    provider: str
    cache_hit: bool
    warnings: list[str] = field(default_factory=list)


def _center_gauss(w: int, h: int, sigma: float = 0.3) -> np.ndarray:
    """Notfallback: Gaußsche Zentrums-Gewichtung (immer verfügbar)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dist2 = ((x - cx) / (sigma * w)) ** 2 + ((y - cy) / (sigma * h)) ** 2
    return np.exp(-0.5 * dist2).astype(np.float32)


def _try_rembg(img: Image.Image) -> np.ndarray | None:
    """ML-Segmentierung via rembg (optionale Dependency)."""
    try:
        from rembg import remove
    except ImportError:
        return None
    out = remove(img)  # RGBA, Alpha = Vordergrund
    alpha = np.asarray(out)[..., 3].astype(np.float32) / 255.0
    return alpha


def _try_opencv(img: Image.Image) -> np.ndarray | None:
    """GrabCut-Segmentierung via OpenCV (optionale Dependency)."""
    try:
        import cv2
    except ImportError:
        return None
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    grab_mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(arr, grab_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.isin(grab_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    return fg.astype(np.float32)


def _model_hash() -> str:
    """Hash des rembg-Modells, falls vorhanden (Provenance, Spec §6.2)."""
    model = Path.home() / ".u2net" / "u2net.onnx"
    if model.exists():
        return hashlib.sha256(model.read_bytes()).hexdigest()[:12]
    return "none"


def _cache_key(image_bytes: bytes, provider: str) -> str:
    raw = hashlib.sha256(image_bytes).hexdigest()[:16]
    # SERVICE_VERSION enthält "/" — für Dateinamen sanitizen
    version = SERVICE_VERSION.replace("/", "-")
    return f"{raw}_{provider}_{_model_hash()}_{version}.npz"


def get_subject_mask(
    image_path: str, cache_dir: str = DEFAULT_CACHE_DIR
) -> MaskResult:
    """Liefert die Subjekt-Maske eines Hintergrundbilds (gecached).

    Provider-Kette: rembg -> OpenCV -> Zentrums-Gauß. Jeder Fallback
    erzeugt eine Warnung; der genutzte Provider steht im Ergebnis.
    """
    p = Path(image_path)
    image_bytes = p.read_bytes()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    img = Image.open(p).convert("RGB")
    warnings: list[str] = []

    mask, provider = None, None
    for name, fn in (("rembg:u2net", _try_rembg), ("opencv:grabcut", _try_opencv)):
        candidate = fn(img)
        if candidate is not None:
            mask, provider = candidate, name
            break
        warnings.append(f"Provider {name} nicht verfügbar — Fallback")
    if mask is None:
        mask = _center_gauss(img.width, img.height)
        provider = "center_gauss"

    cache_file = cache / _cache_key(image_bytes, provider)
    if cache_file.exists():
        loaded = np.load(cache_file)["mask"]
        return MaskResult(loaded, provider, cache_hit=True, warnings=warnings)

    np.savez(cache_file, mask=mask)
    return MaskResult(mask.astype(np.float32), provider, cache_hit=False,
                      warnings=warnings)


def resize_mask(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Skaliert die Maske geometrisch identisch zum Hintergrund-Pfad.

    Nicht-negativer Kernel (AREA bei Downscale, BILINEAR bei Upscale),
    danach Clamp auf [0, 1] — keine LANCZOS-Überschwinger an Maskenkanten
    (Spec §6.2).
    """
    img = Image.fromarray(mask.astype(np.float32), mode="F")
    downscale = target_w < mask.shape[1] or target_h < mask.shape[0]
    kernel = Image.Resampling.BOX if downscale else Image.Resampling.BILINEAR
    out = np.asarray(img.resize((target_w, target_h), kernel), dtype=np.float32)
    return np.clip(out, 0.0, 1.0)
