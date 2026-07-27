"""Tests für Messraster und Metriken M1–M6 (Spec §3.1, §3.3)."""

import numpy as np
import pytest

from src.studio.metrics import srgb_to_linear, to_measure_raster


def test_srgb_to_linear_known_values():
    assert srgb_to_linear(np.array(0.0)) == pytest.approx(0.0)
    assert srgb_to_linear(np.array(1.0)) == pytest.approx(1.0)
    # sRGB-Mittengrau 0.5 -> linear ~0.2140
    assert srgb_to_linear(np.array(0.5)) == pytest.approx(0.2140, abs=1e-3)


def test_measure_raster_long_edge_and_aspect():
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    out = to_measure_raster(frame, long_edge=854)
    assert out.shape == (427, 854, 3)
    assert out.dtype == np.float32


def test_measure_raster_no_upscale():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = to_measure_raster(frame, long_edge=854)
    assert out.shape == (100, 200, 3)


def test_measure_raster_is_linear_light():
    frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    out = to_measure_raster(frame, long_edge=854)
    assert float(out.mean()) == pytest.approx(0.2158, abs=1e-2)


from src.studio.metrics import (
    contribution,
    integrity_violations,
    overlay_coverage,
    overlay_energy,
    subject_disturbance,
    vitality,
)


def test_contribution_and_m1_m2():
    a = np.full((10, 10, 3), 0.8, dtype=np.float32)
    b = np.full((10, 10, 3), 0.2, dtype=np.float32)
    contrib = contribution(a, b)
    assert overlay_energy(contrib) == pytest.approx(0.6)
    assert overlay_coverage(contrib) == pytest.approx(1.0)
    assert overlay_coverage(contrib, threshold=0.7) == pytest.approx(0.0)


def test_m1_zero_for_identical_frames():
    frame = np.random.rand(8, 8, 3).astype(np.float32)
    assert overlay_energy(contribution(frame, frame)) == pytest.approx(0.0)


def test_m3_subject_disturbance():
    contrib = np.zeros((4, 4), dtype=np.float32)
    contrib[:2, :] = 0.4  # obere Hälfte gestört
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:2, :] = 1.0     # Subjekt oben
    assert subject_disturbance(contrib, mask) == pytest.approx(0.4)
    mask_zero = np.zeros((4, 4), dtype=np.float32)
    assert subject_disturbance(contrib, mask_zero) == 0.0  # kein Subjekt -> 0


def test_m5_vitality():
    t0 = np.zeros((4, 4), dtype=np.float32)
    t1 = np.full((4, 4), 0.3, dtype=np.float32)
    assert vitality(t0, t1) == pytest.approx(0.3)
    assert vitality(t1, t1) == pytest.approx(0.0)


def test_m6_integrity():
    # Plan-Abweichung: fill_value 0.5 ergänzt (Plan-Code wirft TypeError);
    # 0.5 triggert weder Blackframe- noch Clipping-Schwelle.
    ok = np.full((10, 10, 3), 0.5, dtype=np.float32)
    assert integrity_violations(ok) == []
    nan_frame = ok.copy(); nan_frame[0, 0, 0] = np.nan
    assert "nan_inf" in integrity_violations(nan_frame)
    black = np.zeros((10, 10, 3), dtype=np.float32)
    assert "blackframe" in integrity_violations(black)
    clipped = np.ones((10, 10, 3), dtype=np.float32)
    assert "clipping" in integrity_violations(clipped)
