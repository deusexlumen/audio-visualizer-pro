"""Tests fuer das Datenmodell des Folgen-Looks (Kapitel, Buehnen-Saetze)."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from config.schemas import load_and_validate_config, validate_config
from src.types import Chapter, ProjectConfig, Quote


def _story_dict():
    with open("config/podcast_story.json", encoding="utf-8") as f:
        return json.load(f)


def test_old_quote_json_loads():
    q = Quote(text="Hallo Welt heute", start_time=1.0, end_time=2.0)
    assert q.kind == "quote" and q.accent_word is None


def test_stage_quote_roundtrip():
    q = Quote(text="Der Mensch stört", start_time=1, end_time=2,
              kind="stage", accent_word="stört")
    assert Quote(**q.model_dump()) == q


def test_project_config_defaults_classic():
    cfg = ProjectConfig(audio_file="a.mp3", output_file="a.mp4",
                        visual={"type": "voice_line"})
    assert cfg.look == "classic" and cfg.chapters == []
    assert cfg.episode_title is None and cfg.episode_number is None


def test_project_defaults_classic():
    cfg = load_and_validate_config("config/podcast_story.json")
    assert cfg.look == "classic" and cfg.chapters == [] and cfg.episode_title is None


def test_schema_accepts_episode_fields():
    data = _story_dict()
    data.update({
        "look": "episode",
        "chapters": [{"start": 0, "title": "Einstieg"}],
        "episode_title": "Folge X",
        "episode_number": "12",
        "quotes": [{"text": "Der Mensch stört", "start_time": 1, "end_time": 2,
                    "kind": "stage", "accent_word": "stört"}],
    })
    cfg = validate_config(data)
    assert cfg.look == "episode" and cfg.chapters[0].title == "Einstieg"
    assert cfg.episode_number == "12"
    assert cfg.quotes[0].kind == "stage" and cfg.quotes[0].accent_word == "stört"


def test_schema_rejects_negative_chapter_start():
    data = _story_dict()
    data["chapters"] = [{"start": -1, "title": "x"}]
    with pytest.raises(ValidationError):
        validate_config(data)


def test_chapter_model():
    assert Chapter(start=1.5, title="A").start == 1.5


def _stage():
    return Quote(text="Der echte Mensch stört hier", start_time=2.0, end_time=4.0,
                 kind="stage", accent_word="stört")


def test_stage_survives_beat_sync():
    from src.beat_sync import sync_quotes_to_beats
    out = sync_quotes_to_beats([_stage()], np.array([62]), 30)
    assert out[0].start_time == pytest.approx(62 / 30)
    assert out[0].kind == "stage" and out[0].accent_word == "stört"


def test_stage_survives_refiner():
    from src.quote_refiner import refine_quote_timestamps
    features = {"fps": 30, "onset": np.zeros(300), "beat_frames": np.array([]),
                "duration": 10.0}
    out = refine_quote_timestamps([_stage()], features)
    assert out[0].kind == "stage" and out[0].accent_word == "stört"


def test_stage_survives_editor(qtbot):
    from src.gui.quote_editor import QuoteEditorDialog
    dlg = QuoteEditorDialog(_stage())
    qtbot.addWidget(dlg)
    dlg.spin_start.setValue(2.5)
    r = dlg.result_quote()
    assert r.start_time == pytest.approx(2.5)
    assert r.kind == "stage" and r.accent_word == "stört"
