"""Integrationstest: genau ein Commit-Render, Sidecar, Verify grün
(Spec §9, §15, §16 P3)."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_run_studio_genau_ein_commit_render(tmp_path, dummy_audio_features):
    from src.render_common import build_features_dict
    from src.studio.constraints import ConstraintSet
    from src.studio.engine import run_studio
    from src.gpu_renderer import GPUBatchRenderer

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    render_spy = MagicMock(side_effect=RuntimeError("stop-after-first"))
    with patch.object(GPUBatchRenderer, "render", render_spy), \
         patch("src.gpu_renderer.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        sidecar = run_studio(
            str(audio), "spectrum_bars", dummy_audio_features, features_dict,
            str(out), constraints=ConstraintSet(max_overlay_alpha=1.0),
        )

    # Genau ein Commit-Render-Versuch (kein automatischer Re-Render)
    assert render_spy.call_count == 1
    # Sidecar wurde geschrieben, Verify hat gemessen
    assert (tmp_path / "out.studio.json").exists()
    data = json.loads((tmp_path / "out.studio.json").read_text())
    assert data["schema_version"] == "studio-decision/2.1"
    assert data["verify"]["status"] in ("pass", "drift_abort")
    assert data["verify"]["drift_within_budget"] is True
