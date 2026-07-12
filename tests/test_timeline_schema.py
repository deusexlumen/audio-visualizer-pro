"""Tests fuer Timeline-/Szenen-Schema und Laufzeit-Modell."""

import pytest
from pydantic import ValidationError

from config.schemas import SceneSchema, TimelineSchema
from src.types import Scene, Timeline


def test_scene_gueltig():
    s = SceneSchema(start=0.0, end=10.0, visualizer="lumina_core")
    assert s.transition == "crossfade"
    assert s.end > s.start


def test_scene_ende_vor_start_abgelehnt():
    with pytest.raises(ValidationError):
        SceneSchema(start=5.0, end=5.0, visualizer="lumina_core")


def test_scene_unbekannter_visualizer_abgelehnt():
    with pytest.raises(ValidationError):
        SceneSchema(start=0.0, end=5.0, visualizer="gibtsnicht")


def test_timeline_ueberlappung_abgelehnt():
    with pytest.raises(ValidationError):
        TimelineSchema(scenes=[
            SceneSchema(start=0.0, end=10.0, visualizer="lumina_core"),
            SceneSchema(start=8.0, end=15.0, visualizer="nebula_drift"),
        ])


def test_timeline_gueltige_folge():
    tl = TimelineSchema(scenes=[
        SceneSchema(start=0.0, end=10.0, visualizer="lumina_core"),
        SceneSchema(start=10.0, end=20.0, visualizer="nebula_drift"),
    ])
    assert len(tl.scenes) == 2


def test_runtime_timeline_scene_at():
    tl = Timeline(scenes=[
        Scene(start=0.0, end=10.0, visualizer="a"),
        Scene(start=10.0, end=20.0, visualizer="b"),
    ])
    assert tl.scene_at(5.0).visualizer == "a"
    assert tl.scene_at(15.0).visualizer == "b"
    # Ueber das Ende hinaus -> letzte Szene
    assert tl.scene_at(25.0).visualizer == "b"
