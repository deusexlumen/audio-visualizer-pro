"""Tests fuer den YouTube-Kapiteltext, den Intro-Versatz und den Folgentitel."""

import pytest

from src.episode.chapters import (
    default_episode_title, format_timestamp, write_chapter_file, youtube_chapter_text,
)
from src.intro_renderer import _build_filter_complex, main_start_offset
from src.types import Chapter

CH = [Chapter(start=0, title="Einstieg"), Chapter(start=95.4, title="Das System"),
      Chapter(start=3725, title="Ausweg")]


def test_format_timestamp():
    assert [format_timestamp(s) for s in (0, 95.4, 3725)] == ["0:00", "1:35", "1:02:05"]


def test_chapter_text_basic():
    text, warn = youtube_chapter_text(CH, total_duration=4000)
    assert text.splitlines() == ["0:00 Einstieg", "1:35 Das System", "1:02:05 Ausweg"]
    assert warn == []


def test_chapter_text_first_line_always_zero():
    text, _ = youtube_chapter_text([Chapter(start=3, title="A")] + CH[1:], total_duration=4000)
    assert text.splitlines()[0] == "0:00 A"


def test_chapter_text_with_intro_offset():
    text, _ = youtube_chapter_text(CH, total_duration=4000, intro_offset=4.0)
    assert text.splitlines()[:2] == ["0:00 Einstieg", "1:39 Das System"]


def test_chapter_text_warns_too_few():
    _, warn = youtube_chapter_text(CH[:2], total_duration=4000)
    assert any("3" in w for w in warn)


def test_chapter_text_warns_short_chapter():
    _, warn = youtube_chapter_text([CH[0], Chapter(start=5, title="kurz"), CH[1]],
                                   total_duration=4000)
    assert any("10 s" in w for w in warn)


def test_chapter_text_warns_short_last_chapter():
    _, warn = youtube_chapter_text(CH, total_duration=3730)
    assert any("Ausweg" in w and "10 s" in w for w in warn)


def test_main_start_offset_matches_filter():
    assert main_start_offset(5.0, 1.0) == pytest.approx(4.0)
    assert main_start_offset(0.5, 1.0) == pytest.approx(0.1)


def test_filter_complex_uses_same_offset():
    info = {"width": 64, "height": 36, "fps": 30, "audio_sample_rate": 44100,
            "duration": 5.0, "has_audio": False}
    fc, _ = _build_filter_complex(info, dict(info, duration=60.0), 1.0)
    assert "offset=4.0" in fc


def test_write_chapter_file(tmp_path):
    video = tmp_path / "folge.mp4"
    path, warn = write_chapter_file(str(video), CH, 4000)
    assert path == tmp_path / "folge.kapitel.txt"
    assert path.read_text(encoding="utf-8").splitlines()[1] == "1:35 Das System"
    assert warn == []


def test_default_episode_title():
    assert (default_episode_title("x/Digitale_Gorilla-Taktiken_gegen_den_KI-Müll.m4a")
            == "Digitale Gorilla-Taktiken gegen den KI-Müll")
