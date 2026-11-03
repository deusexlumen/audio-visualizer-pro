"""Tests fuer den KI-Plan des Folgen-Looks (ohne Netzwerk)."""

import pytest

from src.episode.config import EpisodeLookConfig
from src.episode.plan import EpisodePlan, build_episode_plan, episode_plan_prompt

SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "Willkommen zu einer neuen Folge."},
    {"start": 5.0, "end": 12.0, "text": "Heute geht es um das Internet."},
    {"start": 30.0, "end": 36.0, "text": "Der echte Mensch stört in diesem geschlossenen System."},
    {"start": 40.0, "end": 45.0, "text": "Die Gesellschaft verliert ihr Gedächtnis."},
    {"start": 60.0, "end": 66.0, "text": "Wir leben mitten in der Ära des Dead Internet."},
    {"start": 120.0, "end": 126.0, "text": "Was können wir also konkret dagegen tun?"},
    {"start": 200.0, "end": 206.0, "text": "Am Ende entscheidet jeder selbst."},
]


def _raw(**over):
    raw = {
        "chapters": [{"segment_index": 1, "title": "Einstieg"},
                     {"segment_index": 5, "title": "Gegenwehr"},
                     {"segment_index": 6, "title": "Schluss"}],
        "stage": [{"text": "Der echte Mensch stört in diesem geschlossenen System.",
                   "accent_word": "stört"}],
        "quotes": [{"text": "Wir leben mitten in der Ära des Dead Internet.", "confidence": 0.9}],
    }
    raw.update(over)
    return raw


def test_plan_first_chapter_at_zero():
    plan = build_episode_plan(_raw(), SEGMENTS, 210.0)
    assert isinstance(plan, EpisodePlan)
    assert plan.chapters[0].start == 0.0
    assert [c.title for c in plan.chapters] == ["Einstieg", "Gegenwehr", "Schluss"]
    assert plan.chapters[1].start == 120.0


def test_plan_drops_chapter_closer_than_10s():
    raw = _raw(chapters=[{"segment_index": 0, "title": "A"},
                         {"segment_index": 1, "title": "B"},
                         {"segment_index": 5, "title": "C"}])
    plan = build_episode_plan(raw, SEGMENTS, 210.0)
    assert [c.title for c in plan.chapters] == ["A", "C"]


def test_plan_caps_chapters():
    raw = _raw(chapters=[{"segment_index": i, "title": f"K{i}"} for i in range(7)])
    plan = build_episode_plan(raw, SEGMENTS, 210.0, EpisodeLookConfig(max_chapters=3))
    assert len(plan.chapters) <= 3


def test_plan_ignores_bad_segment_index():
    raw = _raw(chapters=[{"segment_index": 99, "title": "X"}, {"segment_index": 1, "title": "A"}])
    assert [c.title for c in build_episode_plan(raw, SEGMENTS, 210.0).chapters] == ["A"]


def test_plan_drops_unlocatable_text():
    raw = _raw(quotes=[{"text": "Dieser Satz wurde nie gesagt worden sein.", "confidence": 1}])
    plan = build_episode_plan(raw, SEGMENTS, 210.0)
    assert all(q.kind == "stage" for q in plan.quotes)


def test_plan_stage_times_from_segments():
    stage = [q for q in build_episode_plan(_raw(), SEGMENTS, 210.0).quotes if q.kind == "stage"][0]
    assert stage.start_time == pytest.approx(30.0, abs=0.01)
    assert stage.accent_word == "stört"


def test_plan_caps_stage_at_three():
    texts = [s["text"] for s in SEGMENTS[1:6]]
    raw = _raw(stage=[{"text": t, "accent_word": ""} for t in texts], quotes=[])
    plan = build_episode_plan(raw, SEGMENTS, 210.0)
    assert sum(q.kind == "stage" for q in plan.quotes) == 3


def test_plan_accent_word_fallback_longest():
    raw = _raw(stage=[{"text": "Die Gesellschaft verliert ihr Gedächtnis.",
                       "accent_word": "Zukunft"}], quotes=[])
    stage = build_episode_plan(raw, SEGMENTS, 210.0).quotes[0]
    assert stage.accent_word == "Gesellschaft"


def test_plan_accent_word_case_insensitive():
    raw = _raw(stage=[{"text": "Der echte Mensch stört in diesem geschlossenen System.",
                       "accent_word": "SYSTEM"}], quotes=[])
    assert build_episode_plan(raw, SEGMENTS, 210.0).quotes[0].accent_word == "System"


def test_plan_drops_quote_overlapping_stage():
    raw = _raw(quotes=[{"text": "Der echte Mensch stört", "confidence": 0.8}])
    plan = build_episode_plan(raw, SEGMENTS, 210.0)
    assert [q.kind for q in plan.quotes] == ["stage"]


def test_plan_quotes_sorted():
    plan = build_episode_plan(_raw(), SEGMENTS, 210.0)
    assert [q.start_time for q in plan.quotes] == sorted(q.start_time for q in plan.quotes)


def test_plan_prompt_names_boldest_claims():
    prompt = episode_plan_prompt(SEGMENTS, 3, 8, 2)
    assert "kühnsten Behauptungen" in prompt
    assert "Nicht die beste Zusammenfassung" in prompt
    assert "[2] Der echte Mensch" in prompt
    assert "3–8 Kapitel" in prompt and "etwa 2 weitere Zitate" in prompt


def test_plan_episode_uses_cache(monkeypatch):
    from src import gemini_integration as gi
    store = {}
    monkeypatch.setattr(gi, "load_json_result", lambda a, k, s: store.get((a, k, s)))
    monkeypatch.setattr(gi, "save_json_result", lambda a, k, s, d: store.__setitem__((a, k, s), d))

    g = gi.GeminiIntegration.__new__(gi.GeminiIntegration)
    g._model = "test-model"
    calls = []
    monkeypatch.setattr(gi.GeminiIntegration, "model", property(lambda self: "test-model"))
    monkeypatch.setattr(g, "transcribe_segments", lambda *a, **k: SEGMENTS, raising=False)

    def fake_select(segments, duration):
        calls.append(1)
        return _raw()

    monkeypatch.setattr(g, "_select_episode_plan", fake_select, raising=False)
    p1 = g.plan_episode("folge.m4a", 210.0)
    p2 = g.plan_episode("folge.m4a", 210.0)
    assert len(calls) == 1 and p1 == p2 and p1.chapters[0].title == "Einstieg"


LONG = ("Wir schauen uns heute an wie eine strategische Architektur aus enormem privatem Geld "
        "und militaerischen Interessen den Grundstein fuer eine voellig neue Form der Kontrolle legte.")


def test_plan_drops_stage_fragment():
    # Echter Lauf: "Denn bevor grosse Ideen ... werden," war kein ganzer Satz
    raw = _raw(stage=[{"text": "Der echte Mensch stört in diesem geschlossenen",
                       "accent_word": "stört"}], quotes=[])
    assert build_episode_plan(raw, SEGMENTS, 210.0).quotes == []


def test_plan_drops_long_stage():
    segments = SEGMENTS[:6] + [{"start": 150.0, "end": 165.0, "text": LONG}] + SEGMENTS[6:]
    assert len(LONG.split()) > 20
    raw = _raw(stage=[{"text": LONG, "accent_word": "Kontrolle"}], quotes=[])
    assert build_episode_plan(raw, segments, 210.0).quotes == []


def test_plan_prompt_limits_stage_sentences():
    prompt = episode_plan_prompt(SEGMENTS, 3, 8, 2)
    assert "höchstens 20 Wörter" in prompt
    assert "vollständiger Satz" in prompt
