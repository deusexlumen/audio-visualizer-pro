"""Spiral-Zoom in Config-Schema und Presets."""

import json
from dataclasses import fields

import pytest

from config.schemas import PostProcessConfig, load_and_validate_config
from src.spiral_zoom import SpiralSettings


def test_schema_defaults_match_renderer_defaults():
    cfg = PostProcessConfig()
    d = SpiralSettings()
    for f in fields(SpiralSettings):
        assert getattr(cfg, f"spiral_{f.name}") == getattr(d, f.name), f.name


def test_bounds_and_integer_arms():
    with pytest.raises(Exception):
        PostProcessConfig(spiral_arms=4)
    with pytest.raises(Exception):
        PostProcessConfig(spiral_arms=1.5)
    with pytest.raises(Exception):
        PostProcessConfig(spiral_ratio=1.0)
    with pytest.raises(Exception):
        PostProcessConfig(spiral_speed=3.0)
    assert PostProcessConfig(spiral_arms=-3).spiral_arms == -3


def test_config_file_reaches_render_postprocess(tmp_path):
    """Spiral-Schluessel ueberleben load_and_validate_config + model_dump (CLI-Pfad)."""
    cfg_path = tmp_path / "spiral.json"
    cfg_path.write_text(json.dumps({
        "audio_file": "input.mp3",
        "output_file": "out.mp4",
        "visual": {"type": "spectrum_bars", "resolution": [1280, 720], "fps": 30},
        "postprocess": {"spiral_enabled": True, "spiral_arms": 2, "spiral_speed": -0.5},
    }), encoding="utf-8")
    pp = load_and_validate_config(str(cfg_path)).postprocess.model_dump()
    s = SpiralSettings.from_postprocess(pp)
    assert s.is_active
    assert s.arms == 2
    assert s.speed == -0.5


def test_old_config_without_spiral_keys_stays_off():
    pp = load_and_validate_config("config/music_aggressive.json").postprocess.model_dump()
    assert SpiralSettings.from_postprocess(pp).is_active is False


def test_example_preset_is_valid_and_active():
    pp = load_and_validate_config("config/music_spiral_zoom.json").postprocess.model_dump()
    assert SpiralSettings.from_postprocess(pp).is_active
