"""Pruefliste des Folgen-Looks (Reiter "Folge")."""

import pytest
from PIL import Image
from PyQt6.QtWidgets import QDialog

from src.gui.episode_panel import EpisodePanel
from src.gui.quote_editor import QuoteEditorDialog
from src.gui.state import AppState
from src.types import Chapter, Quote


@pytest.fixture
def state():
    s = AppState()
    s.audio_duration = 300.0
    return s


def _panel(qtbot, state):
    panel = EpisodePanel(state)
    qtbot.addWidget(panel)
    return panel


def _times(panel):
    return [panel.table.item(r, 1).text() for r in range(panel.table.rowCount())]


def _row_of(panel, text):
    for r in range(panel.table.rowCount()):
        if panel.table.item(r, 2).text() == text:
            return r
    raise AssertionError(text)


def test_panel_rows_sorted_by_time(qtbot, state):
    state.chapters = [Chapter(start=0.0, title="A"), Chapter(start=60.0, title="B")]
    state.quotes = [Quote(text="spaet", start_time=90, end_time=92),
                    Quote(text="frueh", start_time=30, end_time=32)]
    panel = _panel(qtbot, state)
    assert _times(panel) == ["0:00", "0:30", "1:00", "1:30"]


def test_panel_delete_removes_from_state(qtbot, state):
    state.quotes = [Quote(text="eins", start_time=10, end_time=12),
                    Quote(text="zwei", start_time=20, end_time=22)]
    panel = _panel(qtbot, state)
    panel.table.cellWidget(_row_of(panel, "eins"), 5).click()
    assert [q.text for q in state.quotes] == ["zwei"]
    assert panel.table.rowCount() == 1


def test_panel_toggle_kind_to_stage(qtbot, state):
    state.quotes = [Quote(text="Der Mensch stört das System", start_time=10, end_time=12)]
    panel = _panel(qtbot, state)
    panel.table.cellWidget(0, 0).setCurrentText("Bühne")
    assert state.quotes[0].kind == "stage"
    assert state.quotes[0].accent_word in "Der Mensch stört das System".split()


def test_panel_marks_card_conflict(qtbot, state):
    state.chapters = [Chapter(start=0.0, title="A"), Chapter(start=60.0, title="B")]
    state.quotes = [Quote(text="in der Karte", start_time=60.5, end_time=62),
                    Quote(text="frei", start_time=120, end_time=122)]
    panel = _panel(qtbot, state)
    assert "Kapitel-Karte" in panel.table.item(_row_of(panel, "in der Karte"), 2).toolTip()
    assert "Kapitel-Karte" not in panel.table.item(_row_of(panel, "frei"), 2).toolTip()


def test_panel_title_defaults_from_audio(qtbot, state):
    panel = _panel(qtbot, state)
    state.audio_path = "C:/podcast/Das_tote_Internet.mp3"
    assert panel.edit_title.placeholderText() == "Das tote Internet"
    # Review: kein fester Titel im Zustand, sonst bleibt er bei der naechsten Folge stehen
    assert state.episode_title == ""
    state.audio_path = "C:/podcast/Folge_zwei.mp3"
    assert panel.edit_title.placeholderText() == "Folge zwei"
    assert state.episode_title == ""


def test_panel_warns_small_image(qtbot, state, tmp_path):
    img = tmp_path / "quadrat.png"
    Image.new("RGB", (1024, 1024)).save(img)
    panel = _panel(qtbot, state)
    state.background_path = str(img)
    assert "Querformat" in panel.lbl_hint.text()


def test_panel_shows_error_keeps_title(qtbot, state):
    state.episode_title = "Meine Folge"
    panel = _panel(qtbot, state)
    panel.set_plan_error("Zeitlimit erreicht")
    assert "Zeitlimit erreicht" in panel.lbl_hint.text()
    assert panel.edit_title.text() == "Meine Folge"
    assert panel.btn_add_chapter.isEnabled() and panel.btn_add_quote.isEnabled()


def test_panel_add_rows_by_hand(qtbot, state):
    state.preview_time_percent = 0.5
    panel = _panel(qtbot, state)
    panel.btn_add_chapter.click()
    panel.btn_add_quote.click()
    assert len(state.chapters) == 1 and state.chapters[0].start == pytest.approx(150.0)
    assert len(state.quotes) == 1 and state.quotes[0].start_time == pytest.approx(150.0)
    assert panel.table.rowCount() == 2


def test_panel_edit_text_writes_state(qtbot, state):
    state.quotes = [Quote(text="alt", start_time=10, end_time=12, kind="stage", accent_word="alt")]
    panel = _panel(qtbot, state)
    panel.table.item(0, 2).setText("neu und besser")
    assert state.quotes[0].text == "neu und besser" and state.quotes[0].kind == "stage"


def test_panel_edit_time_keeps_stage(qtbot, state, monkeypatch):
    state.quotes = [Quote(text="Der Mensch stört", start_time=10, end_time=12,
                          kind="stage", accent_word="stört")]
    panel = _panel(qtbot, state)

    def fake_exec(dialog):
        dialog.spin_start.setValue(10.5)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QuoteEditorDialog, "exec", fake_exec)
    panel.table.cellDoubleClicked.emit(0, 1)
    q = state.quotes[0]
    assert q.start_time == pytest.approx(10.5)
    assert q.kind == "stage" and q.accent_word == "stört"


def test_panel_play_requests_preview(qtbot, state):
    state.quotes = [Quote(text="eins", start_time=42, end_time=44)]
    panel = _panel(qtbot, state)
    with qtbot.waitSignal(panel.preview_requested) as blocker:
        panel.table.cellWidget(0, 4).click()
    # Vorschau dort, wo der ganze Text steht (Sprechende), nicht beim ersten Wort
    assert blocker.args == [pytest.approx(44.0)]


def test_panel_play_chapter_previews_full_card(qtbot, state):
    state.chapters = [Chapter(start=60.0, title="Mitte")]
    panel = _panel(qtbot, state)
    with qtbot.waitSignal(panel.preview_requested) as blocker:
        panel.table.cellWidget(0, 4).click()
    assert blocker.args == [pytest.approx(61.0)]


def test_panel_toggle_look(qtbot, state):
    panel = _panel(qtbot, state)
    panel.chk_enabled.setChecked(True)
    assert state.look == "episode"
    state.look = "classic"
    assert not panel.chk_enabled.isChecked()


def test_panel_accepts_chatgpt_wide_image(qtbot, state, tmp_path):
    # Review: 1536x1024 ist das breiteste ChatGPT-Format; der Tipp empfahl genau das
    img = tmp_path / "breit.png"
    Image.new("RGB", (1536, 1024)).save(img)
    panel = _panel(qtbot, state)
    state.background_path = str(img)
    assert "Querformat" not in panel.lbl_hint.text()
