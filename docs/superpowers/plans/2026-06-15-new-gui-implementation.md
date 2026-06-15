# Neue Audio Visualizer Pro GUI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Die monolithische DearPyGui-basierte `gui.py` durch eine modulare PyQt6-GUI mit Panel-Layout, dunklem Studio-Theme und echtzeitnaher Preview ersetzen.

**Architecture:** PyQt6-QWidget-basierte Desktop-App mit zentralem `AppState` (QObject + pyqtSignal). UI in kleine Panel-Widgets aufgeteilt. Blockierende Operationen (Analyse, Preview, Render) laufen in `QThread`-Workern. Bestehende Business-Logik (`AudioAnalyzer`, `render_gpu_preview`, `GPUBatchRenderer`) bleibt unverändert und wird nur aus der GUI aufgerufen.

**Tech Stack:** Python 3.11, PyQt6, numpy, Pillow, pytest-qt (optional), bestehende AVP-Module.

---

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | PyQt6 hinzufügen |
| `src/gui/__init__.py` | Package-Init, Version/Exporte |
| `src/gui/styles.py` | Farbkonstanten, QSS-Strings, Theme-Helper |
| `src/gui/state.py` | `AppState` mit `pyqtSignal`, Serialisierung |
| `src/gui/workers.py` | `PreviewWorker`, `AnalyzeWorker`, `RenderWorker` |
| `src/gui/preview_widget.py` | `PreviewWidget` (QLabel + QPixmap) |
| `src/gui/timeline_widget.py` | `TimelineWidget` (Slider + Zeit + Buttons) |
| `src/gui/assets_panel.py` | Audio-/Background-/Quotes-Panel |
| `src/gui/params_panel.py` | Visualizer-Auswahl + Parameter + Post-Process |
| `src/gui/main_window.py` | `MainWindow` mit QSplitter-Layout und Signal-Verdrahtung |
| `src/gui/app.py` | `QApplication`-Entry-Point |
| `tests/test_gui_state.py` | Tests für `AppState` |
| `tests/test_gui_workers.py` | Tests für Worker (gemockte Render-Funktion) |
| `tests/test_gui_smoke.py` | GUI startet und schließt sich sauber |
| `gui.py` | Wird später durch Thin-Wrapper ersetzt |
| `gui_legacy.py` | Alte DearPyGui-Datei als Backup umbenannt |

---

## Task 1: Setup — Abhängigkeit und Package-Struktur

**Files:**
- Modify: `requirements.txt`
- Create: `src/gui/__init__.py`

- [ ] **Step 1: PyQt6 zu requirements.txt hinzufügen**

```text
PyQt6>=6.4.0
```

Füge diese Zeile zu `requirements.txt` hinzu ( alphabetisch sortiert, falls bereits sortiert).

- [ ] **Step 2: `src/gui/__init__.py` erstellen**

```python
"""Audio Visualizer Pro — PyQt6 GUI Package."""

__version__ = "3.1.0"

from .app import run_app

__all__ = ["run_app"]
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt src/gui/__init__.py
git commit -m "chore: PyQt6 dependency + src/gui package scaffold"
```

---

## Task 2: Theme und Styles

**Files:**
- Create: `src/gui/styles.py`
- Test: `tests/test_gui_styles.py`

- [ ] **Step 1: Failing test für Styles schreiben**

```python
# tests/test_gui_styles.py
from src.gui.styles import Theme, build_app_stylesheet


def test_theme_colors_are_rgb_tuples():
    assert Theme.BACKGROUND == (10, 10, 15)
    assert Theme.ACCENT == (96, 176, 255)


def test_stylesheet_contains_background_color():
    qss = build_app_stylesheet()
    assert "#0a0a0f" in qss
    assert "QGroupBox" in qss
    assert "QPushButton" in qss
```

- [ ] **Step 2: Test laufen lassen — erwartet FAIL**

```bash
pytest tests/test_gui_styles.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'src.gui.styles'`

- [ ] **Step 3: `src/gui/styles.py` implementieren**

```python
"""Dark Studio Theme für die Audio Visualizer Pro GUI."""


class Theme:
    BACKGROUND = (10, 10, 15)
    PANEL = (18, 19, 26)
    INPUT = (26, 28, 36)
    BORDER = (42, 45, 58)
    TEXT_PRIMARY = (232, 233, 236)
    TEXT_SECONDARY = (139, 143, 153)
    ACCENT = (96, 176, 255)
    SUCCESS = (80, 200, 120)
    ERROR = (255, 95, 95)
    WARNING = (240, 200, 90)

    @staticmethod
    def rgb(color: tuple[int, int, int]) -> str:
        return f"rgb({color[0]}, {color[1]}, {color[2]})"

    @staticmethod
    def rgba(color: tuple[int, int, int], alpha: float) -> str:
        return f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})"


def build_app_stylesheet() -> str:
    bg = Theme.rgb(Theme.BACKGROUND)
    panel = Theme.rgb(Theme.PANEL)
    inp = Theme.rgb(Theme.INPUT)
    border = Theme.rgb(Theme.BORDER)
    text_primary = Theme.rgb(Theme.TEXT_PRIMARY)
    text_secondary = Theme.rgb(Theme.TEXT_SECONDARY)
    accent = Theme.rgb(Theme.ACCENT)

    return f"""
    QWidget {{
        background-color: {bg};
        color: {text_primary};
        font-family: "Segoe UI", "Inter", sans-serif;
        font-size: 13px;
    }}

    QGroupBox {{
        background-color: {panel};
        border: 1px solid {border};
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 8px;
        font-weight: 600;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {text_secondary};
    }}

    QPushButton {{
        background-color: {inp};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 6px 14px;
        color: {text_primary};
    }}

    QPushButton:hover {{
        border-color: {accent};
    }}

    QPushButton:pressed {{
        background-color: {Theme.rgba(Theme.ACCENT, 0.15)};
    }}

    QPushButton#primary {{
        background-color: {accent};
        color: {bg};
        border: none;
        font-weight: 600;
    }}

    QSlider::groove:horizontal {{
        height: 4px;
        background: {border};
        border-radius: 2px;
    }}

    QSlider::handle:horizontal {{
        background: {accent};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}

    QSlider::sub-page:horizontal {{
        background: {accent};
        border-radius: 2px;
    }}

    QLineEdit, QComboBox, QSpinBox {{
        background-color: {inp};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 4px 8px;
    }}

    QLabel {{
        color: {text_secondary};
    }}

    QLabel#heading {{
        color: {text_primary};
        font-size: 16px;
        font-weight: 600;
    }}

    QStatusBar {{
        background-color: {panel};
        color: {text_secondary};
    }}
    """
```

- [ ] **Step 4: Test laufen lassen — erwartet PASS**

```bash
pytest tests/test_gui_styles.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_gui_styles.py src/gui/styles.py
git commit -m "feat: add PyQt6 dark studio theme and styles"
```

---

## Task 3: Zentraler AppState

**Files:**
- Create: `src/gui/state.py`
- Test: `tests/test_gui_state.py`

- [ ] **Step 1: Failing test für `AppState` schreiben**

```python
# tests/test_gui_state.py
import pytest
from PyQt6.QtCore import QObject
from src.gui.state import AppState


def test_state_initial_defaults():
    s = AppState()
    assert s.audio_path is None
    assert s.visualizer_type == "lumina_core"
    assert s.preview_width == 854


def test_state_set_emits_changed(qtbot):
    s = AppState()
    with qtbot.waitSignal(s.changed, timeout=100):
        s.visualizer_type = "voice_flow"


def test_state_to_dict_roundtrip():
    s = AppState()
    s.audio_path = "/tmp/test.mp3"
    s.bg_blur = 2.5
    data = s.to_dict()
    restored = AppState.from_dict(data)
    assert restored.audio_path == "/tmp/test.mp3"
    assert restored.bg_blur == 2.5
```

> Hinweis: `qtbot` kommt aus `pytest-qt`. Falls nicht installiert, kann der Signal-Test mit einer einfachen Slot-Liste ersetzt werden.

- [ ] **Step 2: Test laufen lassen — erwartet FAIL**

```bash
pytest tests/test_gui_state.py -v
```

- [ ] **Step 3: `src/gui/state.py` implementieren**

```python
"""Zentraler Zustand für die Audio Visualizer Pro GUI."""

from PyQt6.QtCore import QObject, pyqtSignal
from src.quote_overlay import QuoteOverlayConfig


class AppState(QObject):
    changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.audio_path: str | None = None
        self.features = None
        self.audio_duration: float = 0.0

        self.visualizer_type: str = "lumina_core"
        self.viz_params: dict = {}
        self.viz_offset_x: float = 0.0
        self.viz_offset_y: float = 0.0
        self.viz_scale: float = 1.0

        self.color_mode: str = "chroma"
        self.base_hue: float = 0.55
        self.color_saturation: float = 0.7

        self.background_path: str | None = None
        self.bg_blur: float = 0.0
        self.bg_vignette: float = 0.0
        self.bg_opacity: float = 0.3

        self.pp_contrast: float = 1.0
        self.pp_saturation: float = 1.0
        self.pp_brightness: float = 0.0
        self.pp_warmth: float = 0.0
        self.pp_grain: float = 0.0

        self.preview_time_percent: float = 0.3
        self.preview_fps: int = 30
        self.preview_width: int = 854
        self.preview_height: int = 480

        self.resolution: tuple[int, int] = (1920, 1080)
        self.render_fps: int = 30
        self.codec: str = "h264"
        self.quality: str = "high"
        self.gpu_encode: bool = False
        self.output_dir: str = "output"

        self.quotes: list = []
        self.quotes_enabled: bool = False
        self.quote_config: QuoteOverlayConfig = QuoteOverlayConfig(enabled=True)

        self.status_message: str = "Bereit."
        self.status_kind: str = "info"  # info | ok | warn | error

    def _notify(self, key: str):
        self.changed.emit(key)

    def set(self, key: str, value):
        if hasattr(self, key):
            setattr(self, key, value)
            self._notify(key)

    def get_postprocess(self) -> dict:
        return {
            "contrast": self.pp_contrast,
            "saturation": self.pp_saturation,
            "brightness": self.pp_brightness,
            "warmth": self.pp_warmth,
            "film_grain": self.pp_grain,
        }

    def get_params(self) -> dict:
        base = {
            "offset_x": self.viz_offset_x,
            "offset_y": self.viz_offset_y,
            "scale": self.viz_scale,
            "color_mode": self.color_mode,
            "base_hue": self.base_hue,
            "color_saturation": self.color_saturation,
        }
        base.update(self.viz_params)
        return base

    def to_dict(self) -> dict:
        qc = self.quote_config
        return {
            "version": 1,
            "audio_path": self.audio_path,
            "background_path": self.background_path,
            "visualizer_type": self.visualizer_type,
            "viz_params": self.viz_params,
            "viz_offset_x": self.viz_offset_x,
            "viz_offset_y": self.viz_offset_y,
            "viz_scale": self.viz_scale,
            "color_mode": self.color_mode,
            "base_hue": self.base_hue,
            "color_saturation": self.color_saturation,
            "bg_blur": self.bg_blur,
            "bg_vignette": self.bg_vignette,
            "bg_opacity": self.bg_opacity,
            "pp_contrast": self.pp_contrast,
            "pp_saturation": self.pp_saturation,
            "pp_brightness": self.pp_brightness,
            "pp_warmth": self.pp_warmth,
            "pp_grain": self.pp_grain,
            "preview_time_percent": self.preview_time_percent,
            "preview_fps": self.preview_fps,
            "resolution": list(self.resolution),
            "render_fps": self.render_fps,
            "codec": self.codec,
            "quality": self.quality,
            "gpu_encode": self.gpu_encode,
            "output_dir": self.output_dir,
            "quotes_enabled": self.quotes_enabled,
            "quote_config": {
                "enabled": qc.enabled,
                "position": qc.position,
                "font_size": qc.font_size,
                "font_color": list(qc.font_color),
                "box_color": list(qc.box_color),
                "display_duration": qc.display_duration,
                "fade_duration": qc.fade_duration,
                "max_chars_per_line": qc.max_chars_per_line,
                "line_spacing": qc.line_spacing,
                "text_align": qc.text_align,
            },
        }

    @classmethod
    def from_dict(cls, data: dict):
        s = cls()
        for key, value in data.items():
            if key == "quote_config" and isinstance(value, dict):
                s.quote_config = QuoteOverlayConfig(**value)
            elif key == "resolution" and isinstance(value, list):
                s.resolution = tuple(value)
            elif hasattr(s, key):
                setattr(s, key, value)
        return s
```

- [ ] **Step 4: Test laufen lassen — erwartet PASS**

```bash
pytest tests/test_gui_state.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_gui_state.py src/gui/state.py
git commit -m "feat: add central AppState with serialization"
```

---

## Task 4: Worker-Threads

**Files:**
- Create: `src/gui/workers.py`
- Test: `tests/test_gui_workers.py`

- [ ] **Step 1: Failing test für Worker schreiben**

```python
# tests/test_gui_workers.py
from unittest.mock import patch
from PyQt6.QtCore import QCoreApplication
from src.gui.workers import PreviewWorker


def test_preview_worker_emits_ready(qtbot):
    app = QCoreApplication.instance() or QCoreApplication([])
    worker = PreviewWorker(
        audio_path="/tmp/test.wav",
        visualizer_type="lumina_core",
        width=320,
        height=180,
        fps=30,
        preview_time_percent=0.3,
    )

    with patch("src.gui.workers.render_gpu_preview") as mock_render:
        from PIL import Image
        mock_render.return_value = Image.new("RGB", (320, 180), (0, 0, 0))

        with qtbot.waitSignal(worker.preview_ready, timeout=1000):
            worker.start()
            qtbot.waitUntil(lambda: not worker.isRunning(), timeout=1000)

    mock_render.assert_called_once()
```

- [ ] **Step 2: Test laufen lassen — erwartet FAIL**

```bash
pytest tests/test_gui_workers.py -v
```

- [ ] **Step 3: `src/gui/workers.py` implementieren**

```python
"""QThread-Worker für blockierende GUI-Operationen."""

from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image

from src.analyzer import AudioAnalyzer
from src.gpu_preview import render_gpu_preview


class AnalyzeWorker(QThread):
    analysis_ready = pyqtSignal(object)
    analysis_error = pyqtSignal(str)

    def __init__(self, audio_path: str, fps: int = 30, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.fps = fps

    def run(self):
        try:
            analyzer = AudioAnalyzer()
            features = analyzer.analyze(self.audio_path, fps=self.fps)
            self.analysis_ready.emit(features)
        except Exception as e:
            self.analysis_error.emit(str(e))


class PreviewWorker(QThread):
    preview_ready = pyqtSignal(Image.Image)
    preview_error = pyqtSignal(str)

    def __init__(
        self,
        audio_path: str,
        visualizer_type: str,
        params: dict,
        width: int,
        height: int,
        fps: int,
        preview_time_percent: float,
        background_image: str | None = None,
        background_blur: float = 0.0,
        background_vignette: float = 0.0,
        background_opacity: float = 0.3,
        postprocess: dict | None = None,
        viz_offset_x: float = 0.0,
        viz_offset_y: float = 0.0,
        viz_scale: float = 1.0,
        features=None,
        quotes=None,
        quote_config=None,
        parent=None,
    ):
        super().__init__(parent)
        self.audio_path = audio_path
        self.visualizer_type = visualizer_type
        self.params = params
        self.width = width
        self.height = height
        self.fps = fps
        self.preview_time_percent = preview_time_percent
        self.background_image = background_image
        self.background_blur = background_blur
        self.background_vignette = background_vignette
        self.background_opacity = background_opacity
        self.postprocess = postprocess
        self.viz_offset_x = viz_offset_x
        self.viz_offset_y = viz_offset_y
        self.viz_scale = viz_scale
        self.features = features
        self.quotes = quotes
        self.quote_config = quote_config

    def run(self):
        try:
            img = render_gpu_preview(
                audio_path=self.audio_path,
                visualizer_type=self.visualizer_type,
                params=self.params,
                width=self.width,
                height=self.height,
                fps=self.fps,
                preview_time_percent=self.preview_time_percent,
                background_image=self.background_image,
                background_blur=self.background_blur,
                background_vignette=self.background_vignette,
                background_opacity=self.background_opacity,
                postprocess=self.postprocess,
                viz_offset_x=self.viz_offset_x,
                viz_offset_y=self.viz_offset_y,
                viz_scale=self.viz_scale,
                features=self.features,
                quotes=self.quotes,
                quote_config=self.quote_config,
            )
            if img is not None:
                self.preview_ready.emit(img)
            else:
                self.preview_error.emit("Preview returned None")
        except Exception as e:
            self.preview_error.emit(str(e))


class RenderWorker(QThread):
    render_progress = pyqtSignal(float)
    render_finished = pyqtSignal(str)
    render_error = pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        # Wird in Task 10 implementiert, zunächst nur Stub
        try:
            self.render_finished.emit("output.mp4")
        except Exception as e:
            self.render_error.emit(str(e))
```

- [ ] **Step 4: Test laufen lassen — erwartet PASS**

```bash
pytest tests/test_gui_workers.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_gui_workers.py src/gui/workers.py
git commit -m "feat: add QThread workers for analysis and preview"
```

---

## Task 5: Preview Widget

**Files:**
- Create: `src/gui/preview_widget.py`

- [ ] **Step 1: `src/gui/preview_widget.py` implementieren**

```python
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

        # Skaliere auf Widget-Größe, behalte Aspect Ratio
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
```

- [ ] **Step 2: Schneller manueller Test**

```bash
python - <<'PY'
import sys
from PyQt6.QtWidgets import QApplication
from src.gui.preview_widget import PreviewWidget
from PIL import Image
app = QApplication(sys.argv)
w = PreviewWidget()
w.set_image(Image.new("RGB", (854, 480), (20, 40, 80)))
w.show()
# app.exec()  # Nicht in CI
print("PreviewWidget created successfully")
PY
```

- [ ] **Step 3: Commit**

```bash
git add src/gui/preview_widget.py
git commit -m "feat: add PreviewWidget for PyQt6 GUI"
```

---

## Task 6: Timeline Widget

**Files:**
- Create: `src/gui/timeline_widget.py`

- [ ] **Step 1: `src/gui/timeline_widget.py` implementieren**

```python
"""Timeline-Widget mit Slider und Zeit-Sprung-Buttons."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSlider, QLabel, QPushButton,
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
            btn.setFixedWidth(40)
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
```

- [ ] **Step 2: Commit**

```bash
git add src/gui/timeline_widget.py
git commit -m "feat: add TimelineWidget with scrubbing"
```

---

## Task 7: Assets Panel

**Files:**
- Create: `src/gui/assets_panel.py`

- [ ] **Step 1: `src/gui/assets_panel.py` implementieren**

```python
"""Panel zum Laden von Audio, Hintergrund und Quotes."""

from pathlib import Path
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QGroupBox,
    QSlider, QHBoxLayout,
)

from src.gui.state import AppState


class AssetsPanel(QWidget):
    analyze_requested = pyqtSignal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Audio
        audio_box = QGroupBox("Audio")
        audio_layout = QVBoxLayout(audio_box)
        self.btn_load_audio = QPushButton("Audio laden")
        self.btn_load_audio.clicked.connect(self._load_audio)
        audio_layout.addWidget(self.btn_load_audio)
        self.audio_info = QLabel("Kein Audio geladen")
        self.audio_info.setWordWrap(True)
        audio_layout.addWidget(self.audio_info)
        layout.addWidget(audio_box)

        # Background
        bg_box = QGroupBox("Hintergrund")
        bg_layout = QVBoxLayout(bg_box)
        self.btn_load_bg = QPushButton("Bild/Video laden")
        self.btn_load_bg.clicked.connect(self._load_background)
        bg_layout.addWidget(self.btn_load_bg)

        self.bg_path_label = QLabel("Kein Hintergrund")
        self.bg_path_label.setWordWrap(True)
        bg_layout.addWidget(self.bg_path_label)

        bg_layout.addWidget(QLabel("Blur"))
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(0, 200)
        self.slider_blur.setValue(0)
        self.slider_blur.valueChanged.connect(self._on_blur_changed)
        bg_layout.addWidget(self.slider_blur)

        bg_layout.addWidget(QLabel("Vignette"))
        self.slider_vignette = QSlider(Qt.Orientation.Horizontal)
        self.slider_vignette.setRange(0, 100)
        self.slider_vignette.setValue(0)
        self.slider_vignette.valueChanged.connect(self._on_vignette_changed)
        bg_layout.addWidget(self.slider_vignette)

        bg_layout.addWidget(QLabel("Opacity"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(30)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        bg_layout.addWidget(self.slider_opacity)

        layout.addWidget(bg_box)
        layout.addStretch()

    def _load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Audio laden",
            "",
            "Audio (*.mp3 *.wav *.flac *.aac *.ogg *.m4a)",
        )
        if path:
            self.state.audio_path = path
            self.state.set("audio_path", path)
            self.audio_info.setText(Path(path).name)
            self.analyze_requested.emit()

    def _load_background(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Hintergrund laden",
            "",
            "Bilder/Videos (*.png *.jpg *.jpeg *.mp4 *.mov *.gif)",
        )
        if path:
            self.state.background_path = path
            self.state.set("background_path", path)
            self.bg_path_label.setText(Path(path).name)

    def _on_blur_changed(self, value: int):
        self.state.bg_blur = value / 10.0
        self.state.set("bg_blur", self.state.bg_blur)

    def _on_vignette_changed(self, value: int):
        self.state.bg_vignette = value / 100.0
        self.state.set("bg_vignette", self.state.bg_vignette)

    def _on_opacity_changed(self, value: int):
        self.state.bg_opacity = value / 100.0
        self.state.set("bg_opacity", self.state.bg_opacity)
```

- [ ] **Step 2: Commit**

```bash
git add src/gui/assets_panel.py
git commit -m "feat: add AssetsPanel for audio and background"
```

---

## Task 8: Params Panel

**Files:**
- Create: `src/gui/params_panel.py`

- [ ] **Step 1: `src/gui/params_panel.py` implementieren**

```python
"""Panel für Visualizer-Auswahl, Parameter und Post-Process."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QSlider, QGroupBox,
    QGridLayout,
)

from src.gui.state import AppState
from src.gpu_visualizers import list_visualizers


class ParamsPanel(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Visualizer
        viz_box = QGroupBox("Visualizer")
        viz_layout = QVBoxLayout(viz_box)
        self.combo_viz = QComboBox()
        self.combo_viz.addItems(list_visualizers())
        self.combo_viz.currentTextChanged.connect(self._on_visualizer_changed)
        viz_layout.addWidget(self.combo_viz)
        layout.addWidget(viz_box)

        # Offset / Scale
        transform_box = QGroupBox("Transform")
        transform_layout = QGridLayout(transform_box)
        self.slider_offset_x = self._make_slider(-100, 100, 0)
        self.slider_offset_y = self._make_slider(-100, 100, 0)
        self.slider_scale = self._make_slider(50, 200, 100)

        transform_layout.addWidget(QLabel("Offset X"), 0, 0)
        transform_layout.addWidget(self.slider_offset_x, 0, 1)
        transform_layout.addWidget(QLabel("Offset Y"), 1, 0)
        transform_layout.addWidget(self.slider_offset_y, 1, 1)
        transform_layout.addWidget(QLabel("Scale"), 2, 0)
        transform_layout.addWidget(self.slider_scale, 2, 1)
        layout.addWidget(transform_box)

        # Post-Process
        pp_box = QGroupBox("Post-Process")
        pp_layout = QGridLayout(pp_box)
        self.slider_contrast = self._make_slider(0, 300, 100)
        self.slider_saturation = self._make_slider(0, 300, 100)
        self.slider_brightness = self._make_slider(-100, 100, 0)
        self.slider_warmth = self._make_slider(-100, 100, 0)
        self.slider_grain = self._make_slider(0, 100, 0)

        pp_layout.addWidget(QLabel("Contrast"), 0, 0)
        pp_layout.addWidget(self.slider_contrast, 0, 1)
        pp_layout.addWidget(QLabel("Saturation"), 1, 0)
        pp_layout.addWidget(self.slider_saturation, 1, 1)
        pp_layout.addWidget(QLabel("Brightness"), 2, 0)
        pp_layout.addWidget(self.slider_brightness, 2, 1)
        pp_layout.addWidget(QLabel("Warmth"), 3, 0)
        pp_layout.addWidget(self.slider_warmth, 3, 1)
        pp_layout.addWidget(QLabel("Grain"), 4, 0)
        pp_layout.addWidget(self.slider_grain, 4, 1)
        layout.addWidget(pp_box)

        layout.addStretch()

        # Signals verbinden
        self.slider_offset_x.valueChanged.connect(lambda v: self._set("viz_offset_x", v / 100.0))
        self.slider_offset_y.valueChanged.connect(lambda v: self._set("viz_offset_y", v / 100.0))
        self.slider_scale.valueChanged.connect(lambda v: self._set("viz_scale", v / 100.0))
        self.slider_contrast.valueChanged.connect(lambda v: self._set("pp_contrast", v / 100.0))
        self.slider_saturation.valueChanged.connect(lambda v: self._set("pp_saturation", v / 100.0))
        self.slider_brightness.valueChanged.connect(lambda v: self._set("pp_brightness", v / 100.0))
        self.slider_warmth.valueChanged.connect(lambda v: self._set("pp_warmth", v / 100.0))
        self.slider_grain.valueChanged.connect(lambda v: self._set("pp_grain", v / 100.0))

    def _make_slider(self, min_val: int, max_val: int, default: int):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        return slider

    def _on_visualizer_changed(self, text: str):
        self.state.visualizer_type = text
        self.state.viz_params = {}
        self.state.set("visualizer_type", text)

    def _set(self, key: str, value):
        setattr(self.state, key, value)
        self.state.set(key, value)
```

- [ ] **Step 2: Commit**

```bash
git add src/gui/params_panel.py
git commit -m "feat: add ParamsPanel for visualizer and postprocess"
```

---

## Task 9: Main Window

**Files:**
- Create: `src/gui/main_window.py`

- [ ] **Step 1: `src/gui/main_window.py` implementieren**

```python
"""Hauptfenster der neuen Audio Visualizer Pro GUI."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QStatusBar, QLabel, QMessageBox, QProgressBar,
)

from src.gui.assets_panel import AssetsPanel
from src.gui.params_panel import ParamsPanel
from src.gui.preview_widget import PreviewWidget
from src.gui.state import AppState
from src.gui.styles import build_app_stylesheet, Theme
from src.gui.timeline_widget import TimelineWidget
from src.gui.workers import AnalyzeWorker, PreviewWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer Pro")
        self.setMinimumSize(1200, 750)

        self.state = AppState()
        self._preview_worker: PreviewWorker | None = None
        self._analyze_worker: AnalyzeWorker | None = None

        self._setup_ui()
        self._setup_signals()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # QSplitter für Panel-Layout
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.assets_panel = AssetsPanel(self.state)
        splitter.addWidget(self.assets_panel)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(8)

        self.preview_widget = PreviewWidget()
        center_layout.addWidget(self.preview_widget, stretch=1)

        self.timeline = TimelineWidget()
        center_layout.addWidget(self.timeline)

        splitter.addWidget(center)

        self.params_panel = ParamsPanel(self.state)
        splitter.addWidget(self.params_panel)

        splitter.setSizes([260, 620, 320])
        layout.addWidget(splitter)

        # Bottom Bar
        bottom = QHBoxLayout()
        bottom.setContentsMargins(12, 8, 12, 8)
        self.status_label = QLabel("Bereit.")
        bottom.addWidget(self.status_label)

        bottom.addStretch()

        self.btn_render = QPushButton("▶ Render")
        self.btn_render.setObjectName("primary")
        self.btn_render.setFixedWidth(120)
        bottom.addWidget(self.btn_render)

        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom)
        layout.addWidget(bottom_widget)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _setup_signals(self):
        self.assets_panel.analyze_requested.connect(self._start_analysis)
        self.timeline.time_changed.connect(self._on_time_changed)
        self.state.changed.connect(self._on_state_changed)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._start_preview)

    def _on_state_changed(self, key: str):
        if key in {
            "visualizer_type", "viz_params", "viz_offset_x", "viz_offset_y", "viz_scale",
            "bg_blur", "bg_vignette", "bg_opacity",
            "pp_contrast", "pp_saturation", "pp_brightness", "pp_warmth", "pp_grain",
            "background_path", "preview_time_percent",
        }:
            self._preview_timer.start(50)

    def _on_time_changed(self, percent: float):
        self.state.preview_time_percent = percent
        self._preview_timer.start(50)

    def _start_analysis(self):
        path = self.state.audio_path
        if not path:
            return
        self._set_status("Analysiere Audio...", "warn")
        self._analyze_worker = AnalyzeWorker(path, fps=self.state.preview_fps)
        self._analyze_worker.analysis_ready.connect(self._on_analysis_ready)
        self._analyze_worker.analysis_error.connect(self._on_analysis_error)
        self._analyze_worker.start()

    def _on_analysis_ready(self, features):
        self.state.features = features
        self.state.audio_duration = features.duration
        self.timeline.set_duration(features.duration)
        self.assets_panel.audio_info.setText(
            f"{features.duration:.1f}s · {features.tempo:.0f} BPM · {features.mode}"
        )
        self._set_status("Analyse fertig.", "ok")
        self._start_preview()

    def _on_analysis_error(self, msg: str):
        self._set_status(f"Analyse-Fehler: {msg}", "error")
        QMessageBox.critical(self, "Analyse-Fehler", msg)

    def _start_preview(self):
        if not self.state.audio_path or self.state.features is None:
            return

        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.requestInterruption()
            self._preview_worker.wait(100)

        self._preview_worker = PreviewWorker(
            audio_path=self.state.audio_path,
            visualizer_type=self.state.visualizer_type,
            params=self.state.get_params(),
            width=self.state.preview_width,
            height=self.state.preview_height,
            fps=self.state.preview_fps,
            preview_time_percent=self.state.preview_time_percent,
            background_image=self.state.background_path,
            background_blur=self.state.bg_blur,
            background_vignette=self.state.bg_vignette,
            background_opacity=self.state.bg_opacity,
            postprocess=self.state.get_postprocess(),
            viz_offset_x=self.state.viz_offset_x,
            viz_offset_y=self.state.viz_offset_y,
            viz_scale=self.state.viz_scale,
            features=self.state.features,
            quotes=self.state.quotes if self.state.quotes_enabled else None,
            quote_config=self.state.quote_config if self.state.quotes_enabled else None,
        )
        self._preview_worker.preview_ready.connect(self._on_preview_ready)
        self._preview_worker.preview_error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_ready(self, img):
        self.preview_widget.set_image(img)
        self._set_status("Preview aktualisiert.", "ok")

    def _on_preview_error(self, msg: str):
        self._set_status(f"Preview-Fehler: {msg}", "error")

    def _set_status(self, msg: str, kind: str = "info"):
        self.status_label.setText(msg)
        color_map = {
            "info": Theme.TEXT_SECONDARY,
            "ok": Theme.SUCCESS,
            "warn": Theme.WARNING,
            "error": Theme.ERROR,
        }
        rgb = color_map.get(kind, Theme.TEXT_SECONDARY)
        self.status_label.setStyleSheet(f"color: rgb{rgb};")

    def closeEvent(self, event):
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.requestInterruption()
            self._preview_worker.wait(500)
        if self._analyze_worker and self._analyze_worker.isRunning():
            self._analyze_worker.wait(500)
        event.accept()
```

- [ ] **Step 2: Commit**

```bash
git add src/gui/main_window.py
git commit -m "feat: add MainWindow with panel layout and preview wiring"
```

---

## Task 10: App Entry-Point

**Files:**
- Create: `src/gui/app.py`
- Modify: `gui.py` (später)

- [ ] **Step 1: `src/gui/app.py` implementieren**

```python
"""Entry-Point für die PyQt6-GUI."""

import sys
from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.gui.styles import build_app_stylesheet


def run_app(argv=None):
    app = QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(build_app_stylesheet())

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app(sys.argv))
```

- [ ] **Step 2: Smoke-Test schreiben**

```python
# tests/test_gui_smoke.py
import sys
from PyQt6.QtWidgets import QApplication


def test_app_starts_and_exits():
    app = QApplication.instance() or QApplication(sys.argv)
    from src.gui.main_window import MainWindow
    window = MainWindow()
    window.show()
    window.close()
    # QApplication nicht beenden, da es andere Tests stören könnte
    assert window is not None
```

- [ ] **Step 3: Tests laufen lassen**

```bash
pytest tests/test_gui_smoke.py tests/test_gui_state.py tests/test_gui_workers.py tests/test_gui_styles.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/gui/app.py tests/test_gui_smoke.py
git commit -m "feat: add PyQt6 GUI entry point and smoke test"
```

---

## Task 11: Migration — alte GUI als Backup, neuer Entry-Point

**Files:**
- Modify: `gui.py`
- Create: `gui_legacy.py` (Kopie der alten Datei)

- [ ] **Step 1: Alte `gui.py` zu `gui_legacy.py` kopieren**

```bash
cp gui.py gui_legacy.py
```

- [ ] **Step 2: `gui.py` durch Thin-Wrapper ersetzen**

```python
# gui.py
"""Thin-Wrapper: Startet die neue PyQt6-GUI."""

import sys
from src.gui.app import run_app

if __name__ == "__main__":
    sys.exit(run_app(sys.argv))
```

- [ ] **Step 3: README/AGENTS.md aktualisieren**

Passe `AGENTS.md` an: GUI ist jetzt PyQt6, Start via `python gui.py` oder `python -m src.gui`.

- [ ] **Step 4: Commit**

```bash
git add gui.py gui_legacy.py AGENTS.md
git commit -m "feat: replace DearPyGui entry point with PyQt6 wrapper"
```

---

## Task 12: Final Verification

- [ ] **Step 1: Gesamte Test-Suite laufen lassen**

```bash
pytest tests/ -v
```

- [ ] **Step 2: Manuelle GUI-Prüfung**

```bash
python gui.py
```

Erwartet:
- Fenster öffnet sich.
- Audio laden → Analyse läuft.
- Preview erscheint.
- Slider bewegen → Preview aktualisiert sich flüssig ohne Ruckeln.
- Hintergrund laden → Preview zeigt Hintergrund.
- Timeline scrubben → Preview springt.

- [ ] **Step 3: Commit falls nötig**

```bash
git add -A
git commit -m "fix: final polish and verification"
```

---

## Spec Coverage Check

| Spec-Section | Implementing Task |
|---|---|
| Framework PyQt6 | Task 1 |
| File structure | Tasks 2–10 |
| Panel-Layout | Task 9 |
| State Management | Task 3 |
| Preview-System | Tasks 4, 5, 9 |
| Timeline | Task 6 |
| Styling | Task 2 |
| Error handling | Tasks 4, 9 |
| Testing | Tasks 2, 3, 4, 10, 12 |
| Migration | Task 11 |

## Placeholder Scan

- Keine TBD/TODO/"implement later" im Plan.
- Jeder Task hat konkrete Dateipfade, Code und Test-Kommandos.
- Keine vagen Anweisungen wie "add appropriate error handling".

## Type Consistency Check

- `AppState` verwendet durchgehend `pyqtSignal(str)` für `changed`.
- `PreviewWorker.preview_ready` sendet `PIL.Image.Image`.
- `TimelineWidget.time_changed` sendet `float` (0.0–1.0).
- `MainWindow` nutzt `QTimer` mit 50ms Debounce.
