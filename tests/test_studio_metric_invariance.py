"""Metrik-Invarianz über Auflösungen (Spec §3.1, §15).

Ohne diesen Test ist 'Preview = Batch' eine Behauptung.
"""

import numpy as np
import pytest
from PIL import Image

from src.studio.metrics import contribution, overlay_energy, to_measure_raster


def test_metric_invariance_across_resolutions():
    rng = np.random.default_rng(42)
    base = (rng.random((90, 160, 3)) * 255).astype(np.uint8)
    zeros = np.zeros_like(base)

    ref = overlay_energy(
        contribution(to_measure_raster(base), to_measure_raster(zeros))
    )
    for factor in (2, 4):  # simuliert 1080p/4K-Varianten desselben Inhalts
        big = np.asarray(
            Image.fromarray(base).resize(
                (160 * factor, 90 * factor), Image.Resampling.BOX
            )
        )
        big_zeros = np.zeros_like(big)
        m = overlay_energy(
            contribution(to_measure_raster(big), to_measure_raster(big_zeros))
        )
        assert abs(m - ref) <= 0.01
