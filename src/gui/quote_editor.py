"""Dialog zum Bearbeiten eines Zitats — Text UND Zeitpunkt.

Der automatisch gesetzte Zeitstempel trifft den richtigen Satz, liegt an
den Raendern aber gern ein paar Zehntel daneben. Ohne Gehoer laesst sich
das nicht beurteilen, darum: Wellenform mit Sprech-Kontur, ziehbare
Grenzen, Feinschritte und Abhoeren des Ausschnitts.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QUrl, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from src.app_logging import get_logger
from src.gui.styles import Theme
from src.quote_timing import speech_segments, snap_to_speech
from src.types import Quote

logger = get_logger(__name__)

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
except ImportError:  # QtMultimedia ist optional
    QAudioOutput = None
    QMediaPlayer = None

# Kontext links und rechts des Zitats in der Wellenform
CONTEXT_SECONDS = 4.0
# Fangbereich der Griffe in Pixeln
HANDLE_GRAB_PX = 7


class QuoteWaveform(QWidget):
    """Wellenform-Ausschnitt mit ziehbaren Zitat-Grenzen."""

    rangeChanged = pyqtSignal(float, float)
    seeked = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(96)
        self.setMouseTracking(True)
        self._rms = None
        self._fps = 30.0
        self._segments = []
        self._view = (0.0, 1.0)
        self._start = 0.0
        self._end = 1.0
        self._playhead = None
        self._drag = None

    def set_audio(self, rms, fps, segments=None):
        self._rms = np.asarray(rms, dtype=np.float32).ravel() if rms is not None else None
        self._fps = float(fps) if fps else 30.0
        self._segments = list(segments or [])
        self.update()

    def set_range(self, start, end):
        self._start, self._end = float(start), float(end)
        self.update()

    def set_view(self, view_start, view_end):
        if view_end <= view_start:
            view_end = view_start + 1.0
        self._view = (max(0.0, view_start), view_end)
        self.update()

    def set_playhead(self, t):
        self._playhead = None if t is None else float(t)
        self.update()

    # --- Koordinaten ---

    def _t_to_x(self, t):
        v0, v1 = self._view
        return (t - v0) / (v1 - v0) * self.width()

    def _x_to_t(self, x):
        v0, v1 = self._view
        return v0 + max(0.0, min(1.0, x / max(self.width(), 1))) * (v1 - v0)

    # --- Maus ---

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        if abs(x - self._t_to_x(self._start)) <= HANDLE_GRAB_PX:
            self._drag = "start"
        elif abs(x - self._t_to_x(self._end)) <= HANDLE_GRAB_PX:
            self._drag = "end"
        else:
            self._drag = None
            self.seeked.emit(self._x_to_t(x))

    def mouseMoveEvent(self, event):
        x = event.position().x()
        if self._drag is None:
            near = (abs(x - self._t_to_x(self._start)) <= HANDLE_GRAB_PX
                    or abs(x - self._t_to_x(self._end)) <= HANDLE_GRAB_PX)
            self.setCursor(Qt.CursorShape.SplitHCursor if near
                           else Qt.CursorShape.PointingHandCursor)
            return
        t = self._x_to_t(x)
        if self._drag == "start":
            self._start = min(t, self._end - 0.1)
        else:
            self._end = max(t, self._start + 0.1)
        self.update()
        self.rangeChanged.emit(self._start, self._end)

    def mouseReleaseEvent(self, event):
        self._drag = None

    # --- Zeichnen ---

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        center_y = h / 2.0
        v0, v1 = self._view

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*Theme.INPUT))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)

        # Sprech-Abschnitte als Band (zeigt, wo ueberhaupt geredet wird)
        painter.setBrush(QColor(*Theme.ACCENT, 40))
        for s, e in self._segments:
            if e < v0 or s > v1:
                continue
            x0, x1 = self._t_to_x(s), self._t_to_x(e)
            painter.drawRect(QRectF(x0, 4, max(1.0, x1 - x0), h - 8))

        # Ausgewaehlter Bereich
        sx0, sx1 = self._t_to_x(self._start), self._t_to_x(self._end)
        painter.setBrush(QColor(*Theme.ACCENT, 55))
        painter.drawRect(QRectF(sx0, 0, max(1.0, sx1 - sx0), h))

        # Wellenform
        if self._rms is not None and self._rms.size > 0:
            i0 = max(0, int(v0 * self._fps))
            i1 = min(self._rms.size, int(v1 * self._fps))
            if i1 > i0:
                view = self._rms[i0:i1]
                bins = max(1, min(w // 2, view.size))
                chunks = np.array_split(view, bins)
                peaks = np.array([float(c.max()) if c.size else 0.0 for c in chunks])
                peak_max = float(peaks.max()) or 1.0
                bar_w = w / bins
                for i, amp in enumerate(peaks):
                    bar_h = max(2.0, (amp / peak_max) * (h - 22))
                    x = i * bar_w
                    inside = self._start <= self._x_to_t(x + bar_w / 2) <= self._end
                    painter.setBrush(QColor(*(Theme.ACCENT if inside else Theme.BORDER)))
                    painter.drawRoundedRect(
                        QRectF(x + bar_w * 0.15, center_y - bar_h / 2,
                               max(1.0, bar_w * 0.7), bar_h), 1, 1)

        # Griffe
        for x, label in ((sx0, "Start"), (sx1, "Ende")):
            painter.setPen(QPen(QColor(*Theme.TEXT_PRIMARY), 2))
            painter.drawLine(int(x), 2, int(x), h - 12)
            painter.setPen(QColor(*Theme.TEXT_SECONDARY))
            painter.drawText(QRectF(x - 30, h - 14, 60, 12),
                             Qt.AlignmentFlag.AlignCenter, label)

        # Abspielkopf
        if self._playhead is not None and v0 <= self._playhead <= v1:
            painter.setPen(QPen(QColor(*Theme.WARNING), 2))
            px = self._t_to_x(self._playhead)
            painter.drawLine(int(px), 0, int(px), h)

        painter.end()


class QuoteEditorDialog(QDialog):
    """Bearbeitet Text und Zeitfenster eines Zitats."""

    def __init__(self, quote, audio_path=None, features=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zitat bearbeiten")
        self.setMinimumWidth(620)
        self._quote = quote
        self._audio_path = audio_path
        self._duration = float(getattr(features, "duration", 0.0) or 0.0)
        self._segments = []
        if features is not None:
            try:
                self._segments = speech_segments(features.rms, float(features.fps))
            except Exception as e:
                logger.debug(f"[Zitat-Editor] Sprech-Erkennung fehlgeschlagen: {e}")

        self._player = None
        self._audio_out = None
        self._stop_at = None
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui(features)
        self._setup_player()
        self._sync_view()

    # --- Aufbau ---

    def _build_ui(self, features):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Text"))
        self.txt = QTextEdit()
        self.txt.setPlainText(self._quote.text)
        self.txt.setMaximumHeight(80)
        layout.addWidget(self.txt)

        self.wave = QuoteWaveform()
        if features is not None:
            self.wave.set_audio(getattr(features, "rms", None),
                                float(getattr(features, "fps", 30) or 30),
                                self._segments)
        self.wave.set_range(self._quote.start_time, self._quote.end_time)
        self.wave.rangeChanged.connect(self._on_range_dragged)
        self.wave.seeked.connect(self._play_from)
        layout.addWidget(self.wave)

        self.spin_start = self._make_spin(self._quote.start_time)
        self.spin_end = self._make_spin(self._quote.end_time)
        self.spin_start.valueChanged.connect(self._on_spin_changed)
        self.spin_end.valueChanged.connect(self._on_spin_changed)

        layout.addLayout(self._time_row("Start", self.spin_start))
        layout.addLayout(self._time_row("Ende", self.spin_end))

        self.lbl_info = QLabel("")
        layout.addWidget(self.lbl_info)

        actions = QHBoxLayout()
        self.btn_play = QPushButton("Ausschnitt abspielen")
        self.btn_play.clicked.connect(self._play_selection)
        actions.addWidget(self.btn_play)
        self.btn_stop = QPushButton("Stopp")
        self.btn_stop.clicked.connect(self._stop)
        actions.addWidget(self.btn_stop)
        self.btn_snap = QPushButton("Auf Sprechgrenzen einrasten")
        self.btn_snap.setToolTip(
            "Verschiebt Start und Ende auf die naechste erkannte Sprech-Kante."
        )
        self.btn_snap.clicked.connect(self._snap)
        self.btn_snap.setEnabled(bool(self._segments))
        actions.addWidget(self.btn_snap)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_info()

    def _make_spin(self, value):
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setSuffix(" s")
        spin.setRange(0.0, self._duration if self._duration > 0 else 99999.0)
        spin.setValue(float(value))
        return spin

    def _time_row(self, label, spin):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(spin)
        for delta in (-0.5, -0.1, 0.1, 0.5):
            btn = QPushButton(f"{delta:+.1f}")
            btn.setMaximumWidth(52)
            btn.clicked.connect(lambda _, s=spin, d=delta: s.setValue(s.value() + d))
            row.addWidget(btn)
        row.addStretch()
        return row

    def _setup_player(self):
        if QMediaPlayer is None or not self._audio_path:
            for b in (self.btn_play, self.btn_stop):
                b.setEnabled(False)
                b.setToolTip("Audio-Wiedergabe nicht verfuegbar (QtMultimedia fehlt).")
            return
        try:
            self._player = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_out)
            self._player.setSource(QUrl.fromLocalFile(self._audio_path))
        except Exception as e:
            logger.warning(f"[Zitat-Editor] Wiedergabe nicht moeglich: {e}")
            self._player = None
            for b in (self.btn_play, self.btn_stop):
                b.setEnabled(False)

    # --- Interaktion ---

    def _on_range_dragged(self, start, end):
        for spin, value in ((self.spin_start, start), (self.spin_end, end)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._update_info()

    def _on_spin_changed(self):
        if self.spin_end.value() <= self.spin_start.value():
            self.spin_end.blockSignals(True)
            self.spin_end.setValue(self.spin_start.value() + 0.1)
            self.spin_end.blockSignals(False)
        self.wave.set_range(self.spin_start.value(), self.spin_end.value())
        self._sync_view()
        self._update_info()

    def _sync_view(self):
        start, end = self.spin_start.value(), self.spin_end.value()
        self.wave.set_view(max(0.0, start - CONTEXT_SECONDS), end + CONTEXT_SECONDS)

    def _update_info(self):
        dur = self.spin_end.value() - self.spin_start.value()
        self.lbl_info.setText(f"Dauer: {dur:.2f} s")

    def _snap(self):
        start, end = snap_to_speech(self.spin_start.value(), self.spin_end.value(),
                                    self._segments)
        self.spin_start.setValue(start)
        self.spin_end.setValue(end)

    # --- Wiedergabe ---

    def _play_from(self, t, stop_at=None):
        if self._player is None:
            return
        self._stop_at = stop_at
        self._player.setPosition(int(max(0.0, t) * 1000))
        self._player.play()
        self._timer.start()

    def _play_selection(self):
        # Kleiner Vorlauf, damit hoerbar wird, ob das erste Wort vollstaendig ist
        self._play_from(max(0.0, self.spin_start.value() - 0.4),
                        self.spin_end.value() + 0.4)

    def _stop(self):
        self._timer.stop()
        if self._player is not None:
            self._player.pause()
        self.wave.set_playhead(None)

    def _on_tick(self):
        if self._player is None:
            return
        pos = self._player.position() / 1000.0
        self.wave.set_playhead(pos)
        if self._stop_at is not None and pos >= self._stop_at:
            self._stop()

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)

    # --- Ergebnis ---

    def result_quote(self):
        return Quote(
            text=self.txt.toPlainText().strip() or self._quote.text,
            start_time=self.spin_start.value(),
            end_time=self.spin_end.value(),
            confidence=self._quote.confidence,
        )
