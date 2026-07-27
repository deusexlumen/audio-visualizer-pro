"""Tests für die PresetFactory (Spec §10)."""

import numpy as np
import pytest

from src.studio.preset_factory import PRESET_SCHEMA_VERSION, build_preset
from src.studio.profiles import load_profile


def test_visualizer_aus_whitelist(dummy_audio_features):
    for profil in ("music_default", "podcast_default"):
        preset = build_preset(dummy_audio_features, load_profile(profil))
        assert preset.visualizer in load_profile(profil).visualizer_whitelist


def test_params_innerhalb_der_korridore(dummy_audio_features):
    preset = build_preset(dummy_audio_features, load_profile("podcast_default"))
    corridors = load_profile("podcast_default").param_corridors
    for key, (lo, hi) in corridors.items():
        if key in preset.params:
            assert lo <= preset.params[key] <= hi


def test_postprocess_innerhalb_budget(dummy_audio_features):
    preset = build_preset(dummy_audio_features, load_profile("podcast_default"))
    assert preset.postprocess.get("bloom_intensity", 0) <= 0.4
    assert preset.postprocess.get("film_grain", 0) <= 0.1


def test_schema_version_und_threshold_referenz(dummy_audio_features):
    preset = build_preset(dummy_audio_features, load_profile("music_default"))
    assert preset.schema_version == PRESET_SCHEMA_VERSION
    assert preset.threshold_set.endswith("studio_thresholds.v1.json")


def test_podcast_farben_entsaettigt(dummy_audio_features):
    preset = build_preset(dummy_audio_features, load_profile("podcast_default"))
    # Entsättigung: RGB-Kanäle liegen näher beieinander als Saturated-Rot
    from src.gpu_visualizers.base import hex_to_rgb
    rgb = hex_to_rgb(preset.colors["primary"])
    spread = max(rgb) - min(rgb)
    assert spread < 0.7  # reines Rot hätte 1.0
