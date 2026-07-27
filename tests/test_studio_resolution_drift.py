"""Drift-Messung Probe vs. Ziel (C16, Spec §3.4, §15).

Der Test schlägt nicht bei Drift fehl, sondern wenn kein Driftwert
erfasst wurde — unbekannte Drift ist der Defekt, nicht Drift selbst.
"""

import json
import pytest

from src.studio.probe import probe_resolution

pytestmark = pytest.mark.gpu


def test_drift_is_measured_and_recorded(tmp_path, dummy_audio_features):
    from src.render_common import build_features_dict
    from tools.measure_drift import measure_visualizer_drift, write_drift_file

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    target = (320, 180)
    probe = probe_resolution(*target)
    entry = measure_visualizer_drift(
        "spectrum_bars", features_dict, probe, target, times=[0.2, 0.5, 0.8]
    )
    assert "M1" in entry and "M5" in entry
    assert all(v >= 0.0 for v in entry.values())

    out = tmp_path / "studio_drift.v1.json"
    write_drift_file({"spectrum_bars": entry}, str(out))
    data = json.loads(out.read_text())
    assert data["version"] == "studio-drift/1"
    assert data["per_visualizer"]["spectrum_bars"]["M1"] == entry["M1"]
