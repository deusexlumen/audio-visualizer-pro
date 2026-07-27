"""Tests für die Studio-GUI-Elemente (Spec §11.2)."""

import pytest

pytestmark = pytest.mark.gui


def test_modus_badge_zeigt_modus_und_konfidenz(qtbot):
    from src.gui.state import AppState
    from src.gui.studio_panel import StudioPanel
    from src.studio.mode_gate import ModeResult

    panel = StudioPanel(AppState())
    qtbot.addWidget(panel)
    panel.update_studio_badges(
        ModeResult(value="PODCAST", resolved="podcast", confidence=0.87,
                   speech_score=0.63, hysteresis_applied=False)
    )
    assert "PODCAST" in panel.mode_badge.text()
    assert "0.87" in panel.mode_badge.text()


def test_quality_badge_zeigt_metriken(qtbot):
    from src.gui.state import AppState
    from src.gui.studio_panel import StudioPanel
    from src.studio.mode_gate import ModeResult

    panel = StudioPanel(AppState())
    qtbot.addWidget(panel)
    panel.update_studio_badges(
        ModeResult(value="MUSIC", resolved="music", confidence=0.9,
                   speech_score=0.1, hysteresis_applied=False),
        verify_metrics={"M1": 0.19, "M3": 0.06, "M5": 0.05},
    )
    assert "M1" in panel.quality_badge.text()
    assert "0.19" in panel.quality_badge.text()


def test_solver_trace_listet_schritte(qtbot):
    from src.gui.state import AppState
    from src.gui.studio_panel import StudioPanel
    from src.studio.mode_gate import ModeResult
    from src.studio.solver import SolveResult

    panel = StudioPanel(AppState())
    qtbot.addWidget(panel)
    result = SolveResult(
        params={"alpha_cap": 0.84},
        j_trace=[0.41, 0.18, 0.0],
        steps=[{"lever": "alpha_cap", "op": "-0.08",
                "j_before": 0.41, "j_after": 0.18}],
        iterations=2, status="solved",
    )
    panel.update_studio_badges(
        ModeResult(value="MUSIC", resolved="music", confidence=0.9,
                   speech_score=0.1, hysteresis_applied=False),
        solver_result=result,
    )
    assert panel.solver_trace_list.count() == 1
    assert "alpha_cap" in panel.solver_trace_list.item(0).text()


def test_preset_button_ruft_callback(qtbot):
    from src.gui.state import AppState
    from src.gui.ki_panel import KIPanel

    panel = KIPanel(AppState())
    qtbot.addWidget(panel)
    received = []
    panel.set_studio_preset_callback(received.append)
    panel.studio_preset_button.click()
    assert len(received) == 1
