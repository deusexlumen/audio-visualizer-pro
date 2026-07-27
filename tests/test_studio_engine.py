"""Tests für die Engine-Probe-Loop (Spec §8, §9)."""

import numpy as np
import pytest

from src.studio.constraints import ConstraintSet
from src.studio.engine import evaluate_params, solve_constraints
from src.studio.probe import ProbeRenderer
from src.studio.sampling import build_sample_plan
from src.studio.thresholds import load_thresholds

pytestmark = pytest.mark.gpu


@pytest.fixture
def probe():
    p = ProbeRenderer(width=160, height=90, fps=30)
    yield p
    p.release()


@pytest.fixture
def features_dict(dummy_audio_features):
    from src.render_common import build_features_dict
    return build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )


def _viz(probe):
    from src.gpu_visualizers import get_visualizer
    return get_visualizer("spectrum_bars")(probe.ctx, 160, 90)


def test_evaluate_params_liefert_metriken(probe, features_dict):
    cs = ConstraintSet(max_overlay_alpha=1.0)
    metrics = evaluate_params(
        probe, _viz(probe), features_dict, [0.2, 0.5, 0.8], {},
        cs.to_measure_constraints(),
    )
    assert set(metrics) >= {"M1", "M5", "M6_violations"}
    assert metrics["M1"] > 0.0  # spectrum_bars zeichnet sichtbar
    assert metrics["M6_violations"] == 0


def test_solve_senkt_alpha_cap_bei_engem_m1(probe, features_dict):
    ts = load_thresholds()
    # Künstlich enge M1-Schwelle: Solver muss alpha_cap senken
    ts = ts.model_copy(update={"m1_overlay_energy_max": 0.01})
    plan = build_sample_plan(features_dict)
    cs = ConstraintSet(max_overlay_alpha=1.0)
    params, result, final_metrics = solve_constraints(
        probe, lambda: _viz(probe), features_dict, plan, {}, cs, ts, "music"
    )
    assert result.j_trace[0] > result.j_trace[-1]  # J gesunken
    assert all(b < a for a, b in zip(result.j_trace, result.j_trace[1:]))
    if result.status == "solved":
        assert params["alpha_cap"] < 1.0


def test_run_studio_auto_end_to_end(tmp_path, dummy_audio_features):
    from unittest.mock import MagicMock, patch
    from src.gpu_renderer import GPUBatchRenderer
    from src.render_common import build_features_dict
    from src.studio.engine import run_studio_auto

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    with patch.object(GPUBatchRenderer, "render",
                      MagicMock(side_effect=RuntimeError("mock"))), \
         patch("src.gpu_renderer.subprocess.run") as mock_run, \
         patch("src.studio.mode_gate.classify_mode") as mock_mode:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        mock_mode.return_value.resolved = "music"
        mock_mode.return_value.value = "MUSIC"
        mock_mode.return_value.confidence = 0.9
        mock_mode.return_value.speech_score = 0.1
        mock_mode.return_value.hysteresis_applied = False
        sidecar = run_studio_auto(
            str(audio), dummy_audio_features, features_dict, str(out),
            profile_name="music_default",
        )

    assert sidecar["mode"]["value"] == "MUSIC"
    assert sidecar["profile"]["name"] == "music_default"
    assert sidecar["verify"]["status"] in ("pass", "drift_abort")
