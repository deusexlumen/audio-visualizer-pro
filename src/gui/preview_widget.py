"""Preview-Widget zur Anzeige gerenderter Frames."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from PIL import Image

from src.gui.styles import Theme


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setStyleSheet(
            f"background-color: {Theme.hex(Theme.BACKGROUND)};"
            f"border: 1px solid {Theme.rgb(Theme.BORDER)};"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("Vorschau")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(
            f"color: {Theme.rgb(Theme.TEXT_DISABLED)}; border: none;"
        )
        layout.addWidget(self.label)

        # Halbtransparentes Busy-Overlay waehrend der Berechnung
        self._busy_overlay = QLabel("Wird berechnet…", self)
        self._busy_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._busy_overlay.setStyleSheet(
            f"background-color: {Theme.rgba(Theme.BACKGROUND, 0.55)};"
            f"color: {Theme.rgb(Theme.TEXT_PRIMARY)};"
            f"border: none; font-size: 14px;"
        )
        self._busy_overlay.hide()

    def set_busy(self, busy: bool):
        """Zeigt oder versteckt das Busy-Overlay."""
        if busy:
            self._busy_overlay.setGeometry(self.rect())
            self._busy_overlay.raise_()
            self._busy_overlay.show()
        else:
            self._busy_overlay.hide()

    def set_image(self, img: Image.Image):
        """Aktualisiert das Preview-Bild."""
        if img is None:
            return
        img_rgb = img.convert("RGB")
        data = img_rgb.tobytes()
        width, height = img_rgb.size
        image = QImage(data, width, height, width * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        scaled = pixmap.scaled(
            self.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._busy_overlay.isVisible():
            self._busy_overlay.setGeometry(self.rect())
        pixmap = self.label.pixmap()
        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                self.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.label.setPixmap(scaled)
