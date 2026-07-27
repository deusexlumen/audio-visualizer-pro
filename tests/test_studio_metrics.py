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
