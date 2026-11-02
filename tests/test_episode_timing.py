"""Tests fuer die Lesezeit-Regel und die Wortzeiten des Folgen-Looks."""

import pytest

from src.episode.config import EpisodeLookConfig, EpisodeRenderSpec
from src.episode.timing import display_window, visible_word_count, word_reveal_times
from src.types import Quote

CFG = EpisodeLookConfig()


def q(text, s, e):
    return Quote(text=text, start_time=s, end_time=e)


def test_config_defaults_match_spec():
    assert (CFG.chars_per_second, CFG.min_hold, CFG.gap_before_next) == (15.0, 1.5, 0.3)
    assert (CFG.card_duration, CFG.stage_fade, CFG.max_zoom) == (4.0, 0.8, 1.08)
    assert CFG.accent_color == "#F2B866" and CFG.brand_text == "DEUS EX LUMEN"


def test_render_spec_default_config():
    spec = EpisodeRenderSpec(image_path="x.png", chapters=[], quotes=[], title="T")
    assert spec.config == EpisodeLookConfig() and spec.number is None


def test_display_window_short_uses_hold():
    assert display_window(q("Kurz und klar gesagt.", 10, 11.5), None, CFG) == pytest.approx((10, 13.0))


def test_display_window_long_uses_reading_time():
    assert display_window(q("x" * 150, 20, 28), None, CFG) == pytest.approx((20, 30.0))


def test_display_window_dense_next_caps():
    assert display_window(q("Kurz und klar gesagt.", 10, 11.5), 12.0, CFG) == pytest.approx((10, 11.7))


def test_display_window_overlapped_next():
    assert display_window(q("Kurz und klar gesagt.", 10, 11.5), 10.2, CFG) == pytest.approx((10, 10.2))


def test_word_reveal_times():
    assert word_reveal_times(q("eins zwei drei", 0, 1.4)) == pytest.approx([0.0, 0.5, 1.0])


def test_visible_word_count():
    assert [visible_word_count([0, 0.5, 1.0], t) for t in (-0.1, 0, 0.7, 2)] == [0, 1, 2, 3]
