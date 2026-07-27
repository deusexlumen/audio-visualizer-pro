"""Tests für die Modus-Profile (Spec §5)."""

import pytest

from src.studio.profiles import load_profile


def test_music_default_laedt():
    p = load_profile("music_default")
    assert p.mode == "music"
    assert "spectrum_bars" in p.visualizer_whitelist
    assert "voice_flow" not in p.visualizer_whitelist
    assert p.desaturate_colors is False


def test_podcast_default_laedt():
    p = load_profile("podcast_default")
    assert p.mode == "podcast"
    assert "voice_flow" in p.visualizer_whitelist
    assert "speech_focus" in p.visualizer_whitelist
    assert "bass_temple" not in p.visualizer_whitelist
    assert p.desaturate_colors is True
    # Podcast: enger Vitalitätskorridor (Spec §3.3 M5)
    assert p.vitality_corridor[1] <= 0.09


def test_whitelist_keys_existieren_in_registry():
    from src.gpu_visualizers import VISUALIZER_MAP
    for name in ("music_default", "podcast_default"):
        for key in load_profile(name).visualizer_whitelist:
            assert key in VISUALIZER_MAP


def test_unbekanntes_profil_wirft():
    with pytest.raises(KeyError, match="unbekannt"):
        load_profile("unbekannt")


def test_fail_fast_bei_unbekanntem_visualizer():
    from src.studio.profiles import StudioProfile
    with pytest.raises(ValueError, match="gibt_es_nicht"):
        StudioProfile(
            name="kaputt", version=1, mode="music",
            visualizer_whitelist=["gibt_es_nicht"],
            param_corridors={}, postfx_budget={},
            vitality_corridor=(0.0, 1.0),
        )
