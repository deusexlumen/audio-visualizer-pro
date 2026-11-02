"""Tests fuer die Folgen-Zeitachse: Kamerafahrt, Buehne, Kapitel-Karte, Text."""

import numpy as np
import pytest

from src.episode.config import EpisodeLookConfig
from src.episode.timeline import EpisodeTimeline
from src.episode.timing import display_window
from src.types import Chapter, Quote

CFG = EpisodeLookConfig()
DUR = 600.0
CHAPTERS = [Chapter(start=0, title="Einstieg"), Chapter(start=95.4, title="Das System"),
            Chapter(start=300, title="Ende")]
STAGE = Quote(text="Der echte Mensch stört in diesem System.", start_time=30.0,
              end_time=33.0, kind="stage", accent_word="stört")
CONFLICT = Quote(text="Das fällt in die Karte hinein.", start_time=96.0, end_time=98.0)
DENSE = [Quote(text="Erster dichter Satz hier.", start_time=60.0, end_time=61.0),
         Quote(text="Zweiter dichter Satz folgt.", start_time=61.5, end_time=62.5),
         Quote(text="Dritter Satz direkt dahinter.", start_time=61.6, end_time=63.0)]
QUOTES = [CONFLICT, STAGE] + DENSE


@pytest.fixture
def tl():
    return EpisodeTimeline(CHAPTERS, QUOTES, DUR, CFG)


def test_items_sorted(tl):
    assert [q.start_time for q in tl.items] == sorted(q.start_time for q in QUOTES)


def test_camera_stays_inside_image(tl):
    for t in np.linspace(0, DUR, 2000):
        c = tl.state_at(float(t)).camera
        assert 1.0 <= c.scale <= 1.08 + 1e-9
        assert abs(c.offset_x) <= (c.scale - 1) / 2 + 1e-9
        assert abs(c.offset_y) <= (c.scale - 1) / 2 + 1e-9


def test_camera_moves(tl):
    assert tl.state_at(10).camera != tl.state_at(80).camera


def test_camera_continuous_at_chapter_boundary(tl):
    a, b = tl.state_at(95.4 - 1 / 30).camera, tl.state_at(95.4).camera
    assert abs(a.scale - b.scale) < 1e-3 and abs(a.offset_x - b.offset_x) < 1e-3


def test_card_full_at_chapter_start(tl):
    s = tl.state_at(95.4)
    assert s.card_amount == pytest.approx(1.0) and s.chapter_index == 1


def test_first_card_full_at_zero(tl):
    assert tl.state_at(0.0).card_amount == pytest.approx(1.0)
    assert tl.state_at(0.0).chapter_index == 0


def test_card_ends(tl):
    assert tl.state_at(95.4 + 4.0 + 0.01).card_amount == 0.0


def test_card_hides_text_and_reports_conflict(tl):
    idx = tl.items.index(CONFLICT)
    assert tl.state_at(97.0).text_index == -1
    assert tl.card_conflicts() == [idx]


def test_stage_amount_ramps(tl):
    i = tl.items.index(STAGE)
    end = tl.windows[i][1]
    assert tl.state_at(29.2).stage_amount == pytest.approx(0.0, abs=1e-6)
    assert tl.state_at(29.6).stage_amount == pytest.approx(0.5)
    assert tl.state_at(30.0).stage_amount == pytest.approx(1.0)
    assert tl.state_at(end + 0.8).stage_amount == pytest.approx(0.0, abs=1e-6)


def test_text_index_and_fade(tl):
    i = tl.items.index(STAGE)
    end = tl.windows[i][1]
    assert tl.state_at(31.0).text_index == i and tl.state_at(31.0).text_alpha == 1.0
    assert tl.state_at(end - 0.35 / 2).text_alpha == pytest.approx(0.5)
    assert tl.state_at(end + 0.01).text_index == -1


def test_windows_respect_display_window(tl):
    i = tl.items.index(STAGE)
    assert tl.windows[i] == pytest.approx(display_window(STAGE, tl.items[i + 1].start_time, CFG))


def test_state_never_two_texts(tl):
    assert all(w0[1] <= w1[0] + 1e-9 for w0, w1 in zip(tl.windows, tl.windows[1:]))


def test_quote_window_stops_before_card():
    q = Quote(text="Kurz vor dem Kapitel gesagt.", start_time=93.0, end_time=94.0)
    tl = EpisodeTimeline(CHAPTERS, [q], DUR, CFG)
    assert tl.windows[0][1] <= 95.4 - 0.4 - 0.3 + 1e-9


def test_timeline_without_chapters():
    tl = EpisodeTimeline([], QUOTES, DUR, CFG)
    s = tl.state_at(10)
    assert s.card_amount == 0 and s.chapter_index == -1
    assert tl.state_at(0).camera != tl.state_at(DUR).camera
