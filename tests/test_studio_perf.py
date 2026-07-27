"""Perf-Budgets als Akzeptanzkriterien (Spec §13).

Überschreitung = P2-Defekt, nicht „halt langsam". CI-Faktor ×4 gegen
Schwankungen auf geteilten Runnern.
"""

import time

import numpy as np
import pytest

CI_FACTOR = 4.0


def test_metrik_pro_sample_budget():
    """≤ 15 ms CPU pro Sample @854 px (Spec §13)."""
    from src.studio.metrics import (contribution, overlay_energy,
                                    subject_disturbance, to_measure_raster)
    rng = np.random.default_rng(1)
    a = (rng.random((480, 854, 3)) * 255).astype(np.uint8)
    b = (rng.random((480, 854, 3)) * 255).astype(np.uint8)
    # Abweichung vom Plan: Maske muss zum Contrib-Raster (480, 854) passen
    # (480x854 liegt bereits auf dem Messraster, kein Downscale).
    mask = (rng.random((480, 854)) > 0.5).astype(np.float32)

    start = time.perf_counter()
    ra, rb = to_measure_raster(a), to_measure_raster(b)
    c = contribution(ra, rb)
    overlay_energy(c)
    subject_disturbance(c, mask)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms <= 15 * CI_FACTOR


def test_feasibility_budget():
    """≤ 200 ms rein analytisch (Spec §7/§13)."""
    from src.studio.feasibility import check_feasibility
    rng = np.random.default_rng(2)
    mask = (rng.random((1080, 1920)) > 0.5).astype(np.float32)

    start = time.perf_counter()
    for _ in range(10):
        check_feasibility(mask, requires_text_zone=True)
    elapsed_ms = (time.perf_counter() - start) * 1000 / 10
    assert elapsed_ms <= 200 * CI_FACTOR


def test_solver_synthetisch_budget():
    """Vollständiger Solve (synthetisch) deutlich unter 20 s (Spec §13)."""
    from src.studio.solver import solve
    from src.studio.thresholds import load_thresholds
    ts = load_thresholds()
    fn = lambda p: {"M1": 0.8 * p.get("alpha_cap", 0.0),
                    "M5": 0.1 * p.get("intensity", 0.0)
                          + 0.5 * p.get("chroma_modulation", 0.0)}

    start = time.perf_counter()
    result = solve(fn, {"alpha_cap": 1.0}, ts)
    elapsed_s = time.perf_counter() - start
    assert result.status == "solved"
    assert elapsed_s <= 20 * CI_FACTOR  # synthetisch: ms-Bereich
