"""Einfarbige SVG-Icons fuer die GUI, zur Laufzeit in Theme-Farben eingefaerbt.

Die SVGs in assets/icons/ nutzen 'currentColor' als Platzhalter — get_icon()
ersetzt ihn durch die gewuenschte Farbe und rendert das SVG in mehreren
Aufloesungen in ein QIcon.
"""

from pathlib import Path

from PyQt6.QtCore import QByteArray, Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap, QPainter
from PyQt6.QtSvg import QSvgRenderer

from src.gui.styles import Theme

_ICON_DIR = Path(__file__).resolve().parents[2] / "assets" / "icons"
_cache: dict = {}


def get_icon(name: str, color: tuple[int, int, int] = None) -> QIcon:
    """Laedt ein SVG-Icon und faerbt es in der angegebenen Theme-Farbe ein.

    Args:
        name: Dateiname ohne .svg (z.B. "play", "folder-open").
        color: RGB-Tupel; Default ist Theme.TEXT_PRIMARY.

    Returns:
        QIcon (leer, falls die Datei fehlt — Buttons zeigen dann nur Text).
    """
    if color is None:
        color = Theme.TEXT_PRIMARY
    key = (name, color)
    if key in _cache:
        return _cache[key]

    path = _ICON_DIR / f"{name}.svg"
    if not path.exists():
        icon = QIcon()
        _cache[key] = icon
        return icon

    svg = path.read_text(encoding="utf-8").replace("currentColor", Theme.hex(color))
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))

    icon = QIcon()
    for size in (16, 24, 32, 48):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        icon.addPixmap(pixmap)
    _cache[key] = icon
    return icon


def get_app_icon() -> QIcon:
    """App-/Fenster-Icon (Waveform-Kreis in Akzentfarbe)."""
    return get_icon("logo", Theme.ACCENT)
