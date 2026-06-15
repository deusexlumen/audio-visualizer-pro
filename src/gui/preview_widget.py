"""Preview-Widget zur Anzeige gerenderter Frames."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from PIL import Image


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setStyleSheet("background-color: #050505; border: 1px solid #2a2d3a;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("Preview")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #5a5e6b; border: none;")
        layout.addWidget(self.label)

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
        if self.label.pixmap():
            pixmap = self.label.pixmap()
            scaled = pixmap.scaled(
                self.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.label.setPixmap(scaled)
