"""Tests für das Laufzeit-ConstraintSet (Spec §6.1)."""

import pytest

from src.studio.constraints import ConstraintSet


def test_defaults_aus_spec():
    cs = ConstraintSet()
    assert cs.max_overlay_alpha == 0.6
    assert cs.subject_strength == 0.8
    assert cs.text_zone_alpha == 0.15
    assert cs.alpha_from_luma is True
    assert cs.luma_knee_lo < cs.luma_knee_hi


def test_to_measure_constraints_mapping():
    cs = ConstraintSet(max_overlay_alpha=0.4, subject_strength=0.5)
    mc = cs.to_measure_constraints()
    assert mc.alpha_cap == 0.4
    assert mc.alpha_from_luma is True
    assert mc.subject_strength == 0.5
    assert mc.luma_knee_lo == cs.luma_knee_lo


def test_clamp_postprocess_mit_warnung():
    cs = ConstraintSet(max_bloom_intensity=1.0, max_film_grain=0.5)
    pp = {"bloom_intensity": 1.8, "film_grain": 0.9, "contrast": 1.1}
    clamped, warnings = cs.clamp_postprocess(pp)
    assert clamped["bloom_intensity"] == 1.0
    assert clamped["film_grain"] == 0.5
    assert clamped["contrast"] == 1.1  # unberührt
    assert len(warnings) == 2
    assert any("bloom_intensity" in w for w in warnings)


def test_clamp_postprocess_unveraendert_keine_warnung():
    cs = ConstraintSet()
    clamped, warnings = cs.clamp_postprocess({"bloom_intensity": 0.6})
    assert clamped["bloom_intensity"] == 0.6
    assert warnings == []
