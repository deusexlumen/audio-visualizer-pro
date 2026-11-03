"""Reiter "Folge": Pruefliste fuer Kapitel, Buehnen-Saetze und Zitate.

Eine Liste nach Zeit; jede Aenderung landet sofort in AppState
(chapters, quotes, episode_title, episode_number, look).
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.app_logging import get_logger
from src.episode.chapters import default_episode_title, format_timestamp
from src.episode.compositor import image_too_small
from src.episode.config import EpisodeLookConfig
from src.episode.timeline import EpisodeTimeline
from src.gui.quote_editor import QuoteEditorDialog
from src.gui.styles import Theme
from src.types import Chapter, Quote

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
except ImportError:  # QtMultimedia ist optional
    QAudioOutput = None
    QMediaPlayer = None

logger = get_logger(__name__)

KIND_LABELS = {"quote": "Zitat", "stage": "Bühne"}
CONFLICT_TIP = "Fällt in eine Kapitel-Karte; die Karte gewinnt, das Zitat ist nicht zu sehen."
SMALL_IMAGE_TIP = ("Bild zu klein für die Ausgabe. In ChatGPT ein Querformat anfordern, "
                   "z. B. 1536×1024; Wichtiges nicht an den oberen und unteren Rand.")
# Anteil der Zielbreite, ab dem ein Bild ohne Hinweis hochskaliert wird
MIN_SCALE = 0.8
_STATE_KEYS = {"chapters", "quotes", "look", "episode_title", "episode_number",
               "audio_path", "background_path", "resolution", "audio_duration"}

COL_KIND, COL_TIME, COL_TEXT, COL_ACCENT, COL_PLAY, COL_DELETE = range(6)


def _words(text: str) -> list[str]:
    return [w for w in (re.sub(r"^\W+|\W+$", "", t) for t in text.split()) if w]


class EpisodePanel(QWidget):
    plan_requested = pyqtSignal()
    preview_requested = pyqtSignal(float)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._rows: list[tuple[str, int]] = []   # (Art, Index in state-Liste)
        self._error = ""
        self._filling = False
        self._player = None
        self._build_ui()
        self.state.changed.connect(self._on_state_changed)
        self.refresh()

    # --- Aufbau ---

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.chk_enabled = QCheckBox("Folgen-Look verwenden")
        self.chk_enabled.toggled.connect(self._on_look_toggled)
        layout.addWidget(self.chk_enabled)

        form = QFormLayout()
        self.edit_title = QLineEdit()
        self.edit_title.setPlaceholderText("Folgentitel")
        self.edit_title.textEdited.connect(lambda t: self._write("episode_title", t))
        form.addRow("Titel", self.edit_title)
        self.edit_number = QLineEdit()
        self.edit_number.setPlaceholderText("leer = ohne Nummer")
        self.edit_number.textEdited.connect(lambda t: self._write("episode_number", t.strip()))
        form.addRow("Folge Nr.", self.edit_number)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.btn_plan = QPushButton("KI vorbereiten")
        self.btn_plan.setToolTip("Kapitel, 2–3 Bühnen-Sätze und Zitate aus dem Transkript.")
        self.btn_plan.clicked.connect(self._on_plan_clicked)
        buttons.addWidget(self.btn_plan)
        self.btn_add_chapter = QPushButton("+ Kapitel")
        self.btn_add_chapter.clicked.connect(self._add_chapter)
        buttons.addWidget(self.btn_add_chapter)
        self.btn_add_quote = QPushButton("+ Zitat")
        self.btn_add_quote.clicked.connect(self._add_quote)
        buttons.addWidget(self.btn_add_quote)
        layout.addLayout(buttons)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet(f"color: rgb{Theme.WARNING};")
        layout.addWidget(self.lbl_hint)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Art", "Zeit", "Text", "Schlüsselwort", "", ""])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_TEXT, QHeaderView.ResizeMode.Stretch)
        for col in (COL_KIND, COL_TIME, COL_ACCENT, COL_PLAY, COL_DELETE):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(True)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        layout.addWidget(self.table, stretch=1)

    # --- Zustand -> Anzeige ---

    def _on_state_changed(self, key: str):
        if self._filling or key not in _STATE_KEYS:
            return
        self.refresh()

    def refresh(self):
        self._filling = True
        try:
            self.chk_enabled.setChecked(self.state.look == "episode")
            if self.edit_title.text() != self.state.episode_title:
                self.edit_title.setText(self.state.episode_title)
            # Leerer Titel = Dateiname (so rendert es auch _episode_spec)
            audio = self.state.audio_path
            self.edit_title.setPlaceholderText(
                default_episode_title(audio) if audio else "Folgentitel")
            if self.edit_number.text() != self.state.episode_number:
                self.edit_number.setText(self.state.episode_number)
            self._fill_table()
            self._update_hint()
        finally:
            self._filling = False

    def _duration(self) -> float:
        d = float(self.state.audio_duration or 0.0)
        if d <= 0 and self.state.features is not None:
            d = float(self.state.features.duration)
        if d <= 0:
            ends = [q.end_time for q in self.state.quotes] + [c.start for c in self.state.chapters]
            d = max(ends, default=0.0) + 10.0
        return d

    def _conflict_ids(self) -> set[int]:
        timeline = EpisodeTimeline(self.state.chapters, self.state.quotes,
                                   self._duration(), EpisodeLookConfig())
        return {id(timeline.items[i]) for i in timeline.card_conflicts()}

    def _fill_table(self):
        entries = [(c.start, "chapter", i) for i, c in enumerate(self.state.chapters)]
        entries += [(q.start_time, "quote", i) for i, q in enumerate(self.state.quotes)]
        entries.sort(key=lambda e: (e[0], e[1] != "chapter"))
        self._rows = [(kind, idx) for _, kind, idx in entries]
        conflicts = self._conflict_ids()

        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.setRowCount(len(self._rows))
        for row, (kind, idx) in enumerate(self._rows):
            if kind == "chapter":
                c = self.state.chapters[idx]
                self.table.setItem(row, COL_KIND, self._fixed_item("Kapitel"))
                self.table.setItem(row, COL_TIME, self._fixed_item(format_timestamp(c.start)))
                self.table.setItem(row, COL_TEXT, QTableWidgetItem(c.title))
                self.table.setItem(row, COL_ACCENT, self._fixed_item(""))
                t = c.start
            else:
                q = self.state.quotes[idx]
                combo = QComboBox()
                combo.addItems(list(KIND_LABELS.values()))
                combo.setCurrentText(KIND_LABELS.get(q.kind, "Zitat"))
                combo.currentTextChanged.connect(lambda text, i=idx: self._set_kind(i, text))
                self.table.setCellWidget(row, COL_KIND, combo)
                self.table.setItem(row, COL_TIME, self._fixed_item(format_timestamp(q.start_time)))
                text_item = QTableWidgetItem(q.text)
                text_item.setToolTip(q.text)
                self.table.setItem(row, COL_TEXT, text_item)
                if q.kind == "stage":
                    accent = QComboBox()
                    accent.addItems(_words(q.text))
                    if q.accent_word:
                        accent.setCurrentText(q.accent_word)
                    accent.currentTextChanged.connect(lambda w, i=idx: self._set_accent(i, w))
                    self.table.setCellWidget(row, COL_ACCENT, accent)
                else:
                    self.table.setItem(row, COL_ACCENT, self._fixed_item(""))
                if id(q) in conflicts:
                    text_item.setToolTip(f"{CONFLICT_TIP}\n\n{q.text}")
                    text_item.setBackground(QBrush(QColor(*Theme.WARNING, 60)))
                t = q.start_time
            play = QPushButton("▶")
            play.setToolTip("Ab hier anhören und Vorschau zeigen")
            play.clicked.connect(lambda _, s=t: self._play(s))
            self.table.setCellWidget(row, COL_PLAY, play)
            delete = QPushButton("✕")
            delete.setToolTip("Zeile löschen")
            delete.clicked.connect(lambda _, k=kind, i=idx: self._delete(k, i))
            self.table.setCellWidget(row, COL_DELETE, delete)
        self.table.blockSignals(False)

    @staticmethod
    def _fixed_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def _update_hint(self):
        parts = []
        if self._error:
            parts.append(f"KI-Fehler: {self._error}. Kapitel und Zitate gehen auch von Hand.")
        bg = self.state.background_path
        if bg and Path(bg).exists():
            try:
                with Image.open(bg) as img:
                    size = img.size
                # Leichtes Hochskalieren (ChatGPT 1536x1024 -> 1080p) ist normal;
                # nur warnen, wenn der Ausschnitt deutlich zu schmal ist
                w, h = self.state.resolution
                if image_too_small(size, int(w * MIN_SCALE), int(h * MIN_SCALE)):
                    parts.append(SMALL_IMAGE_TIP)
            except Exception:
                pass  # Videos oder unlesbare Dateien: kein Hinweis
        elif self.state.look == "episode":
            parts.append("Kein Bild gewählt. Der Folgen-Look braucht ein Bild (links bei Assets).")
        self.lbl_hint.setText("\n".join(parts))
        self.lbl_hint.setVisible(bool(parts))

    def set_plan_error(self, msg: str):
        self._error = msg
        self.set_plan_busy(False)
        self._update_hint()

    def set_plan_busy(self, busy: bool):
        self.btn_plan.setEnabled(not busy)
        self.btn_plan.setText("⏳ KI arbeitet..." if busy else "KI vorbereiten")
        # Das Ergebnis ersetzt die Liste: waehrenddessen keine Aenderungen
        for widget in (self.table, self.btn_add_chapter, self.btn_add_quote):
            widget.setEnabled(not busy)
        if busy:
            self._error = ""
            self._update_hint()

    # --- Anzeige -> Zustand ---

    def _write(self, key: str, value):
        self._filling = True
        try:
            setattr(self.state, key, value)
        finally:
            self._filling = False

    def _on_look_toggled(self, checked: bool):
        if self._filling:
            return
        self._write("look", "episode" if checked else "classic")
        self._update_hint()

    def _on_plan_clicked(self):
        self.set_plan_busy(True)
        self.plan_requested.emit()

    def _now(self) -> float:
        return float(self.state.preview_time_percent) * self._duration()

    def _add_chapter(self):
        chapter = Chapter(start=round(self._now(), 2), title="Neues Kapitel")
        self.state.chapters = list(self.state.chapters) + [chapter]

    def _add_quote(self):
        t = round(self._now(), 2)
        self.state.quotes = list(self.state.quotes) + [
            Quote(text="Neues Zitat", start_time=t, end_time=t + 2.0)]

    def _replace_quote(self, idx: int, quote: Quote, rebuild: bool = True):
        quotes = list(self.state.quotes)
        quotes[idx] = quote
        if rebuild:
            self.state.quotes = quotes
        else:
            self._write("quotes", quotes)

    def _set_kind(self, idx: int, label: str):
        kind = "stage" if label == KIND_LABELS["stage"] else "quote"
        q = self.state.quotes[idx]
        accent = q.accent_word
        if kind == "stage" and not accent:
            words = _words(q.text)
            accent = max(words, key=len) if words else None
        self._replace_quote(idx, q.model_copy(update={"kind": kind, "accent_word": accent}))

    def _set_accent(self, idx: int, word: str):
        q = self.state.quotes[idx]
        self._replace_quote(idx, q.model_copy(update={"accent_word": word or None}), rebuild=False)

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._filling or item.column() != COL_TEXT:
            return
        kind, idx = self._rows[item.row()]
        text = item.text().strip()
        if not text:
            return
        if kind == "chapter":
            chapters = list(self.state.chapters)
            chapters[idx] = chapters[idx].model_copy(update={"title": text})
            self._write("chapters", chapters)
        else:
            q = self.state.quotes[idx]
            update = {"text": text}
            if q.kind == "stage" and q.accent_word not in _words(text):
                words = _words(text)
                update["accent_word"] = max(words, key=len) if words else None
            self._replace_quote(idx, q.model_copy(update=update), rebuild=False)

    def _on_cell_double_clicked(self, row: int, col: int):
        if col != COL_TIME or not (0 <= row < len(self._rows)):
            return
        kind, idx = self._rows[row]
        if kind == "chapter":
            c = self.state.chapters[idx]
            value, ok = QInputDialog.getDouble(self, "Kapitel-Start", "Start in Sekunden",
                                               c.start, 0.0, self._duration(), 1)
            if ok:
                chapters = list(self.state.chapters)
                chapters[idx] = c.model_copy(update={"start": float(value)})
                self.state.chapters = chapters
            return
        dialog = QuoteEditorDialog(self.state.quotes[idx], audio_path=self.state.audio_path,
                                   features=self.state.features, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # result_quote behaelt kind und accent_word (model_copy)
            self._replace_quote(idx, dialog.result_quote())

    def _delete(self, kind: str, idx: int):
        if kind == "chapter":
            self.state.chapters = [c for i, c in enumerate(self.state.chapters) if i != idx]
        else:
            self.state.quotes = [q for i, q in enumerate(self.state.quotes) if i != idx]

    # --- Abspielen ---

    def _play(self, t: float):
        self._play_audio(t)
        self.preview_requested.emit(float(t))

    def _play_audio(self, t: float):
        path = self.state.audio_path
        if QMediaPlayer is None or not path or not Path(path).exists():
            return
        try:
            if self._player is None:
                self._player = QMediaPlayer(self)
                self._audio_out = QAudioOutput(self)
                self._player.setAudioOutput(self._audio_out)
            if self._player.source() != QUrl.fromLocalFile(path):
                self._player.setSource(QUrl.fromLocalFile(path))
            self._player.setPosition(int(t * 1000))
            self._player.play()
        except Exception as e:
            logger.warning(f"[Folge] Wiedergabe nicht moeglich: {e}")
