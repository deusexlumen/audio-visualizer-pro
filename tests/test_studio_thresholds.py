"""Tests für das versionierte Schwellwert-Set des Visualizer Studio."""

from src.studio.thresholds import ThresholdSet, load_thresholds


def test_load_thresholds_defaults():
    ts = load_thresholds()
    assert ts.version == "studio-thresholds/1"
    assert ts.m1_overlay_energy_max == 0.04
    assert ts.m2_coverage_warn == 0.60
    assert ts.m3_subject_max == 0.01
    assert ts.m4_contrast_min == 4.5
    assert ts.m5_music_min == 0.02
    assert ts.m5_podcast_max == 0.09
    assert ts.epsilon == 0.01
    assert ts.luma_knee_lo < ts.luma_knee_hi


def test_thresholds_file_sha256():
    ts = load_thresholds()
    assert len(ts.file_sha256) == 64
    int(ts.file_sha256, 16)  # gültiges Hex


def test_thresholds_provenance_present():
    ts = load_thresholds()
    for key in ("m1_overlay_energy_max", "m4_contrast_min"):
        value = ts.provenance[key]
        assert value == "assumed" or value.startswith("calibrated@")


def test_measure_constraints_defaults():
    from src.studio.types import MeasureConstraints
    mc = MeasureConstraints()
    assert mc.alpha_cap == 1.0
    assert mc.alpha_from_luma is False
    assert mc.grain_free is False
