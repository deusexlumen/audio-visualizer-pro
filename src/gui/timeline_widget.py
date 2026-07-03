"""Timeline-Widget mit RMS-Wellenform, Beat-Markern und Klick-Seek."""

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
)

from src.gui.styles import Theme


class WaveformBar(QWidget):
    """Zeichnet die RMS-Wellenform mit Beat-Markern; Klick/Ziehen = Seek."""

    seeked = pyqtSignal(float)  # 0.0-1.0

    # Anzahl der Wellenform-Saeulen (auf Widget-Breite verteilt)
    _NUM_BINS = 400

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rms: np.ndarray | None = None
        self._beats: list[float] = []  # Positionen 0.0-1.0
        self._percent: float = 0.3

    def set_features(self, features):
        """Uebernimmt Analyzer-Features (RMS-Wellenform + Beat-Frames)."""
        try:
            rms = np.asarray(features.rms, dtype=np.float32)
            frame_count = max(int(features.frame_count), 1)
            if rms.size > 0:
                # Auf feste Saeulenzahl herunterbrechen (Maximum pro Bin,
                # damit Transienten sichtbar bleiben)
                bins = np.array_split(rms, min(self._NUM_BINS, rms.size))
                peaks = np.array([float(b.max()) for b in bins], dtype=np.float32)
                peak_max = float(peaks.max())
                self._rms = peaks / peak_max if peak_max > 0 else peaks
            else:
                self._rms = None
            beat_frames = np.asarray(features.beat_frames)
            self._beats = [
                float(bf) / frame_count for bf in beat_frames if 0 <= bf < frame_count
            ]
        except Exception:
            self._rms = None
            self._beats = []
        self.update()

    def clear_features(self):
        self._rms = None
        self._beats = []
        self.update()

    def set_percent(self, percent: float):
        self._percent = max(0.0, min(1.0, percent))
        self.update()

    def _seek_to(self, x: float):
        width = max(self.width(), 1)
        percent = max(0.0, min(1.0, x / width))
        self._percent = percent
        self.update()
        self.seeked.emit(percent)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._seek_to(event.position().x())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to(event.position().x())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        center_y = h / 2.0

        # Hintergrund
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*Theme.INPUT))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)

        played = QColor(*Theme.ACCENT)
        unplayed = QColor(*Theme.BORDER)
        playhead_x = self._percent * w

        if self._rms is not None and self._rms.size > 0:
            n = self._rms.size
            bar_w = w / n
            for i in range(n):
                amp = float(self._rms[i])
                bar_h = max(2.0, amp * (h - 10))
                x = i * bar_w
                color = played if (x + bar_w / 2) <= playhead_x else unplayed
                painter.setBrush(color)
                painter.drawRoundedRect(
                    QRectF(x + bar_w * 0.15, center_y - bar_h / 2,
                           bar_w * 0.7, bar_h), 1, 1
                )
        else:
            # Ohne Analyse: dezente Mittellinie
            painter.setPen(QPen(unplayed, 2))
            painter.drawLine(int(8), int(center_y), int(w - 8), int(center_y))

        # Beat-Marker (kleine Punkte am unteren Rand)
        if self._beats:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*Theme.WARNING))
            for pos in self._beats:
                painter.drawEllipse(QRectF(pos * w - 1.5, h - 5, 3, 3))

        # Playhead
        painter.setPen(QPen(QColor(*Theme.TEXT_PRIMARY), 2))
        painter.drawLine(int(playhead_x), 2, int(playhead_x), h - 2)

        painter.end()


class TimelineWidget(QWidget):
    time_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration = 0.0
        self._percent = 0.3

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(4)

        self.waveform = WaveformBar()
        self.waveform.set_percent(self._percent)
        self.waveform.setToolTip(
            "Klicken oder ziehen, um den Vorschau-Zeitpunkt zu waehlen. "
            "Gelbe Punkte markieren erkannte Beats."
        )
        self.waveform.seeked.connect(self._on_seeked)
        layout.addWidget(self.waveform)

        bottom = QHBoxLayout()
        self.time_label = QLabel("0.0s / 0.0s")
        bottom.addWidget(self.time_label)
        bottom.addStretch()

        for pct in [0, 25, 50, 75, 100]:
            btn = QPushButton(f"{pct}%")
            btn.setToolTip(f"Vorschau-Zeitpunkt auf {pct}% setzen")
            # Kompaktes Padding, damit der Text nicht abgeschnitten wird
            btn.setStyleSheet("padding: 4px 6px;")
            btn.setMinimumWidth(44)
            btn.clicked.connect(lambda checked, p=pct: self.set_percent(p / 100.0))
            bottom.addWidget(btn)

        layout.addLayout(bottom)

    def set_duration(self, duration: float):
        self._duration = max(0.0, duration)
        self._update_label()

    def set_features(self, features):
        """Zeigt Wellenform und Beat-Marker der analysierten Audio-Datei."""
        self.waveform.set_features(features)

    def set_percent(self, percent: float):
        percent = max(0.0, min(1.0, percent))
        self._percent = percent
        self.waveform.set_percent(percent)
        self._update_label()
        self.time_changed.emit(percent)

    def get_percent(self) -> float:
        return self._percent

    def _on_seeked(self, percent: float):
        self._percent = percent
        self._update_label()
        self.time_changed.emit(percent)

    def _update_label(self):
        pos = self._percent * self._duration
        self.time_label.setText(f"{pos:.1f}s / {self._duration:.1f}s")
