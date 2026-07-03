"""Timeline-Widget mit Slider und Zeit-Sprung-Buttons."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSlider, QLabel, QPushButton, QHBoxLayout,
)


class TimelineWidget(QWidget):
    time_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(4)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(300)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

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

    def set_percent(self, percent: float):
        percent = max(0.0, min(1.0, percent))
        self.slider.blockSignals(True)
        self.slider.setValue(int(percent * 1000))
        self.slider.blockSignals(False)
        self._update_label()
        self.time_changed.emit(percent)

    def get_percent(self) -> float:
        return self.slider.value() / 1000.0

    def _on_slider_changed(self, value: int):
        percent = value / 1000.0
        self._update_label()
        self.time_changed.emit(percent)

    def _update_label(self):
        pos = self.get_percent() * self._duration
        self.time_label.setText(f"{pos:.1f}s / {self._duration:.1f}s")
