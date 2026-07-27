"""Tests für den Subjekt-Masken-Service (Spec §6.2)."""

import numpy as np
import pytest
from PIL import Image

from src.studio import mask_service
from src.studio.mask_service import get_subject_mask, resize_mask


def _test_image(path, size=(64, 48)):
    Image.new("RGB", size, (120, 80, 40)).save(path)


def test_center_gauss_eigenschaften():
    mask = mask_service._center_gauss(64, 48)
    assert mask.shape == (48, 64)
    assert mask.dtype == np.float32
    # Zentrum deutlich höher als Rand, Werte in [0, 1]
    assert mask[24, 32] > 0.9
    assert mask[0, 0] < 0.1
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_fallback_auf_center_gauss_ohne_provider(tmp_path, monkeypatch):
    # rembg und cv2 als nicht installiert simulieren
    monkeypatch.setattr(mask_service, "_try_rembg", lambda img: None)
    monkeypatch.setattr(mask_service, "_try_opencv", lambda img: None)
    img_path = tmp_path / "bg.png"
    _test_image(img_path)
    result = get_subject_mask(str(img_path), cache_dir=str(tmp_path / "cache"))
    assert result.provider == "center_gauss"
    assert result.mask.shape == (48, 64)  # Quellbildraum
    assert any("rembg" in w or "Fallback" in w for w in result.warnings)


def test_cache_hit_bei_zweitem_aufruf(tmp_path, monkeypatch):
    monkeypatch.setattr(mask_service, "_try_rembg", lambda img: None)
    monkeypatch.setattr(mask_service, "_try_opencv", lambda img: None)
    img_path = tmp_path / "bg.png"
    _test_image(img_path)
    cache = tmp_path / "cache"
    first = get_subject_mask(str(img_path), cache_dir=str(cache))
    assert first.cache_hit is False
    second = get_subject_mask(str(img_path), cache_dir=str(cache))
    assert second.cache_hit is True
    np.testing.assert_array_equal(first.mask, second.mask)


def test_cache_schluessel_reagiert_auf_bild(tmp_path, monkeypatch):
    monkeypatch.setattr(mask_service, "_try_rembg", lambda img: None)
    monkeypatch.setattr(mask_service, "_try_opencv", lambda img: None)
    cache = tmp_path / "cache"
    img_a = tmp_path / "a.png"
    _test_image(img_a)
    get_subject_mask(str(img_a), cache_dir=str(cache))
    img_b = tmp_path / "b.png"
    _test_image(img_b, size=(32, 32))  # anderes Bild => anderer Key
    result_b = get_subject_mask(str(img_b), cache_dir=str(cache))
    assert result_b.cache_hit is False


def test_resize_mask_bleibt_in_wertebereich():
    # Extremes Schachbrett: nicht-negative Kernels bleiben in [0, 1]
    mask = np.indices((40, 40)).sum(axis=0) % 2
    mask = mask.astype(np.float32)
    out = resize_mask(mask, 100, 100)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resize_geometrische_paritaet_zum_hintergrund():
    # Der Hintergrund-Pfad resized hart auf Zielgröße (LANCZOS). Die Maske
    # muss auf dieselbe Zielgröße/Aspect kommen — Geometrie identisch,
    # Filterwahl egal (Spec §6.2).
    from src.studio.mask_service import resize_mask
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[12:36, 16:48] = 1.0  # zentrales Rechteck (25%-75% je Achse)
    out = resize_mask(mask, 160, 90)
    assert out.shape == (90, 160)
    # Rechteck liegt an denselben relativen Koordinaten
    assert out[45, 80] > 0.9           # Zentrum bleibt Subjekt
    assert out[5, 5] < 0.1             # Ecke bleibt frei
    # Flächenanteil grob erhalten (25 % ± 3 %)
    assert (out > 0.5).mean() == pytest.approx(0.25, abs=0.03)


def test_resize_keine_ueberschwinger_an_kanten():
    # LANCZOS würde an harten Kanten über 1.0/unter 0.0 schießen;
    # der nicht-negative Kernel darf das nicht (Kanten-Härtung, Spec §6.2).
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[:, 32:] = 1.0  # harte vertikale Kante
    out = resize_mask(mask, 160, 90)
    assert out.max() <= 1.0
    assert out.min() >= 0.0
