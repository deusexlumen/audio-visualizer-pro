"""Tests für das Provenance-Sidecar (Spec §12)."""

import json

import pytest

from src.studio.provenance import SCHEMA_VERSION, build_sidecar, write_sidecar


def _sections():
    return {
        "input": {"audio_sha256": "ab" * 32, "duration_s": 1.0},
        "mode": {"value": "MUSIC", "confidence": 1.0},
        "profile": {"name": "manual", "version": 0},
        "thresholds": {"set": "config/studio_thresholds.v1.json", "calibrated": False},
        "mask": {"provider": "none", "cache_hit": False},
        "sampling": {"n": 18, "seed": "cd", "timestamps_s": [0.1]},
        "solver": {"iterations": 0, "j_trace": [0.0], "steps": []},
        "verify": {"metrics": {"M1": 0.1}, "status": "pass"},
        "renderer": {"app_version": "dev"},
    }


def test_schema_version_und_created_utc():
    sc = build_sidecar(_sections())
    assert sc["schema_version"] == SCHEMA_VERSION
    assert "created_utc" in sc


def test_pflichtblock_fehlt_wirft():
    sections = _sections()
    del sections["solver"]
    with pytest.raises(ValueError, match="solver"):
        build_sidecar(sections)


def test_write_sidecar_datei(tmp_path):
    out = tmp_path / "video.mp4"
    out.write_bytes(b"fake")
    path = write_sidecar(str(out), build_sidecar(_sections()))
    assert path.endswith(".studio.json")
    data = json.loads((tmp_path / "video.studio.json").read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["sampling"]["n"] == 18
