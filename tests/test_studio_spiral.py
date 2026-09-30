"""Studio-Modus: Spiral-Zoom wird abgeschaltet (Messungen setzen ein unverbogenes Bild voraus)."""

from unittest.mock import MagicMock, patch

import pytest

from src.studio.engine import disable_spiral_for_studio, run_studio


def test_helper_switches_spiral_off_with_warning():
    pp, warnings = disable_spiral_for_studio({"spiral_enabled": True, "bloom_intensity": 0.5})
    assert pp["spiral_enabled"] is False
    assert pp["bloom_intensity"] == 0.5
    assert len(warnings) == 1
    assert "Spiral-Zoom" in warnings[0]


def test_helper_leaves_other_configs_alone():
    original = {"contrast": 1.2}
    pp, warnings = disable_spiral_for_studio(original)
    assert pp == original
    assert pp is not original  # Kopie, Aufrufer-Dict bleibt unberuehrt
    assert warnings == []
    assert disable_spiral_for_studio(None) == ({}, [])


def test_helper_reads_flag_like_renderer():
    """Text "false" ist aus (wie im Renderer) — keine unnoetige Warnung."""
    _, warnings = disable_spiral_for_studio({"spiral_enabled": "false"})
    assert warnings == []
    _, warnings = disable_spiral_for_studio({"spiral_enabled": True, "spiral_mix": 0.0})
    assert warnings == []  # Staerke 0 = wirkungslos, nichts abzuschalten


class _Stop(Exception):
    pass


def test_run_studio_measures_without_spiral():
    """Schon der Probe-Solve bekommt spiral_enabled = False."""
    seen = {}

    def fake_solve(probe, viz_factory, features_dict, plan, postprocess, *args, **kwargs):
        seen.update(postprocess)
        raise _Stop

    with patch("src.studio.engine.check_feasibility", return_value=MagicMock(should_render=True)), \
         patch("src.studio.engine.build_sample_plan", return_value=MagicMock()), \
         patch("src.studio.engine.solve_constraints", side_effect=fake_solve), \
         patch("src.studio.probe.ProbeRenderer"):
        with pytest.raises(_Stop):
            run_studio(
                audio_path="song.mp3", visualizer="lumina_core", features=None,
                features_dict={}, output_path="out.mp4",
                postprocess={"spiral_enabled": True, "spiral_arms": 1},
            )
    assert seen["spiral_enabled"] is False
