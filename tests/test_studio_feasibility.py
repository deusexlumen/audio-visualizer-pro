"""Tests für den Feasibility-Precheck (Spec §7, §15)."""

import numpy as np
import pytest

from src.studio.feasibility import (
    PERIPHERAL_VISUALS,
    FeasibilityResult,
    check_feasibility,
)


def test_ohne_maske_ok():
    result = check_feasibility(None)
    assert result.status == "ok"
    assert result.should_render is True
    assert result.visualizer_whitelist is None


def test_kleine_subjektflaeche_ok():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[:16, :16] = 1.0  # 6 % Subjekt
    result = check_feasibility(mask)
    assert result.status == "ok"


def test_grosse_subjektflaeche_layout_fallback():
    mask = np.ones((64, 64), dtype=np.float32)
    mask[:8, :8] = 0.0  # ~98 % Subjekt (> 0.75)
    result = check_feasibility(mask)
    assert result.status == "layout_fallback"
    assert result.should_render is True
    assert result.visualizer_whitelist == list(PERIPHERAL_VISUALS)


def test_zielkonflikt_infeasible_ohne_render():
    """Spec §15: unlösbarer Fall bricht mit 0 Render-Aufrufen ab."""
    mask = np.ones((64, 64), dtype=np.float32)  # 100 % Subjekt
    render_calls = []

    result = check_feasibility(mask, requires_text_zone=True)
    assert result.status == "infeasible"
    assert result.should_render is False

    # Treiber-Logik: nur rendern wenn should_render — Zähler bleibt 0
    if result.should_render:
        render_calls.append("render")
    assert render_calls == []


def test_keine_textzone_erzwingt_scrim():
    mask = np.ones((64, 64), dtype=np.float32)
    mask[0:6, 0:6] = 0.0  # winzige freie Ecke < Mindestfläche
    result = check_feasibility(mask, requires_text_zone=True, m3_active=False)
    assert "scrim" in " ".join(result.actions).lower()
    assert any("0.05" in a for a in result.actions)


def test_periphere_whitelist_ist_valide():
    from src.gpu_visualizers import VISUALIZER_MAP
    for key in PERIPHERAL_VISUALS:
        assert key in VISUALIZER_MAP, f"Unbekannter Visualizer: {key}"
