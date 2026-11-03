"""Tests fuer den Text-Renderer des Folgen-Looks (Zitat, Buehne, Kapitel-Karte)."""

import numpy as np
import pytest
from fontTools.ttLib import TTFont

from src.episode.config import EpisodeLookConfig
from src.episode.text import SERIF_FONT_PATH, EpisodeTextRenderer, wrap_balanced
from src.episode.timeline import EpisodeTimeline
from src.episode.timing import word_reveal_times
from src.types import Chapter, Quote

W, H = 640, 360
QUOTE = Quote(text="Wir sind mitten in der Ära des Dead Internet",
              start_time=1.0, end_time=4.0)
STAGE = Quote(text="Der Mensch stört das System", start_time=1.0, end_time=3.0,
              kind="stage", accent_word="stört")


def _renderer(quotes=(), chapters=(), number=None):
    timeline = EpisodeTimeline(list(chapters), list(quotes), 20.0, EpisodeLookConfig())
    return EpisodeTextRenderer(timeline, W, H, "Folge X", number, EpisodeLookConfig())


def _gray(value=128):
    return np.full((H, W, 3), value, dtype=np.uint8)


def _change(out, frame):
    return np.abs(out.astype(int) - frame.astype(int)).max(axis=2)


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def test_font_has_german_glyphs():
    cmap = TTFont(SERIF_FONT_PATH).getBestCmap()
    assert all(ord(c) in cmap for c in "äöüÄÖÜß„“")


def test_wrap_balanced_no_orphan():
    lines = wrap_balanced("Wir sind mitten in der Ära des Dead Internet".split(), len, 30)
    assert len(lines[-1].split()) >= 2
    assert all(len(line) <= 30 for line in lines)


def test_wrap_balanced_short_text_one_line():
    assert wrap_balanced(["Hallo", "Welt"], len, 30) == ["Hallo Welt"]


def test_quote_has_no_box():
    frame = _gray()
    out = _renderer([QUOTE]).apply(frame, 5.0)
    changed = _change(out, frame) > 40
    # Markenzeile oben links ausblenden, nur das Zitat zaehlt
    changed[: int(0.15 * H), :] = False
    y0, y1, x0, x1 = _bbox(changed)
    assert changed[y0:y1, x0:x1].mean() < 0.5
    assert x0 >= int(0.05 * W) and y0 > H // 2


def test_words_appear_over_time():
    frame = _gray()
    r = _renderer([QUOTE])
    times = word_reveal_times(QUOTE)

    def diff_at(t):
        return _change(r.apply(frame, t), frame)[int(0.15 * H):].sum()

    assert diff_at(times[1] + 0.3) < diff_at(times[2] + 0.3) < diff_at(5.0)


def test_stage_accent_word_coloured():
    frame = _gray(30)
    out = _renderer([STAGE]).apply(frame, 4.0)
    accent = np.array([0xF2, 0xB8, 0x66])
    near = np.abs(out.astype(int) - accent).max(axis=2) < 30
    assert near.sum() > 20
    ys, xs = np.nonzero(near)
    assert abs(ys.mean() - H / 2) < 0.2 * H


def test_card_left_column():
    frame = _gray()
    out = _renderer(chapters=[Chapter(start=0.0, title="Einstieg")], number="12").apply(frame, 2.0)
    change = _change(out, frame)
    assert change[:, : int(0.35 * W)].mean() > 5 * change[:, int(0.6 * W):].mean()


def test_no_text_outside_windows_except_brand():
    frame = _gray()
    out = _renderer([QUOTE]).apply(frame, 10.0)
    change = _change(out, frame)
    brand = change[: int(0.12 * H), : int(0.4 * W)]
    rest = change.copy()
    rest[: int(0.12 * H), : int(0.4 * W)] = 0
    assert brand.max() > 40
    assert rest.max() == 0


def test_card_long_chapter_name_keeps_gap_below_title():
    def chapter_top(name):
        chapters = [Chapter(start=0.0, title=name)]
        r = EpisodeTextRenderer(EpisodeTimeline(chapters, [], 20.0, EpisodeLookConfig()),
                                960, 540, "Die Macy Stiftung", "7", EpisodeLookConfig())
        chap = r._card_layers(0)[3]
        ys = np.nonzero(chap.alpha[:, :, 0].max(axis=1) > 0.5)[0]
        return chap.y + ys.min()

    # Ein umbrechender Kapitelname waechst nach unten, nicht in den Titel
    short = chapter_top("Einstieg")
    long = chapter_top("Die Macy Konferenzen als Startpunkt der Kontrolle")
    assert abs(long - short) <= 3
