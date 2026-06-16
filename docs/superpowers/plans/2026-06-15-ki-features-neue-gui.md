# KI-Features in neue PyQt6-GUI – Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** KI-gestützte Parameter-Optimierung, Auto-Visualizer-Empfehlung und Key-Zitat-Extraktion/-Verwaltung in die neue PyQt6-GUI integrieren.

**Architecture:** Rechtes GUI-Panel wird zu einem `QTabWidget` mit Tabs Params/KI/Quotes. `KIPanel` und `QuotesPanel` sind eigenständige Widgets, die über `AppState` kommunizieren. Langlaufende KI-Aufrufe laufen in `AIOptimizeWorker` und `QuoteExtractWorker` (QThread).

**Tech Stack:** Python 3.11, PyQt6, ModernGL, google-genai, pytest-qt

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/gui/state.py` | Erweiterung um KI-Zustand + Serialisierung |
| `src/gui/workers.py` | Neue `AIOptimizeWorker`, `QuoteExtractWorker` |
| `src/gui/ki_panel.py` | KI-Tab UI (SmartMatcher + Gemini-Optimierung) |
| `src/gui/quotes_panel.py` | Quotes-Tab UI (Extraktion, Liste, Erscheinungsbild) |
| `src/gui/main_window.py` | Rechtes Panel zu Tabs umbauen, Worker-Lifecycle |
| `src/gui/params_panel.py` | Ggf. Anpassung für Tab-Einbettung |
| `tests/test_app_state.py` | Tests für erweiterten AppState |
| `tests/test_gui_workers.py` | Tests für neue Worker |
| `tests/test_ki_panel.py` | GUI-Tests für KIPanel |
| `tests/test_quotes_panel.py` | GUI-Tests für QuotesPanel |

---

## Task 1: AppState um KI-Felder erweitern

**Files:**
- Modify: `src/gui/state.py`
- Test: `tests/test_app_state.py`

**Context:** `AppState` verwendet `__setattr__` mit `_STATE_KEYS`, um `changed`-Signale zu senden.

- [ ] **Step 1: Write failing test for KI fields**

```python
# tests/test_app_state.py
import pytest
from src.gui.state import AppState


def test_app_state_has_ki_fields():
    s = AppState()
    assert s.ki_prompt == ""
    assert s.ki_suggested_colors == {}
    assert s.ki_status == ""
    assert s.ki_error is False
    assert s.ki_optimizing is False
    assert s.quotes_extracting is False


def test_ki_fields_emit_changed_signal(qtbot):
    s = AppState()
    with qtbot.waitSignal(s.changed, timeout=100):
        s.ki_prompt = "dunkler Kontrast"


def test_ki_serialization():
    s = AppState()
    s.ki_prompt = "test prompt"
    s.ki_suggested_colors = {"primary": "#FF0000"}
    data = s.to_dict()
    assert data["ki_prompt"] == "test prompt"
    assert data["ki_suggested_colors"] == {"primary": "#FF0000"}

    restored = AppState.from_dict(data)
    assert restored.ki_prompt == "test prompt"
    assert restored.ki_suggested_colors == {"primary": "#FF0000"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app_state.py -v`
Expected: FAIL – `AttributeError: 'AppState' object has no attribute 'ki_prompt'`

- [ ] **Step 3: Implement KI fields in AppState**

Edit `src/gui/state.py`:

```python
class AppState(QObject):
    changed = pyqtSignal(str)

    _STATE_KEYS = frozenset({
        "audio_path", "features", "audio_duration",
        "visualizer_type", "viz_params", "viz_offset_x", "viz_offset_y", "viz_scale",
        "color_mode", "base_hue", "color_saturation",
        "background_path", "bg_blur", "bg_vignette", "bg_opacity",
        "pp_contrast", "pp_saturation", "pp_brightness", "pp_warmth", "pp_grain",
        "preview_time_percent", "preview_fps", "preview_width", "preview_height",
        "resolution", "render_fps", "codec", "quality", "gpu_encode", "output_dir",
        "quotes", "quotes_enabled", "quote_config",
        # KI
        "ki_prompt", "ki_suggested_colors", "ki_status", "ki_error",
        "ki_optimizing", "quotes_extracting",
        "status_message", "status_kind",
    })

    def __init__(self, parent=None):
        super().__init__(parent)
        object.__setattr__(self, "_initialized", False)

        # ... existing fields ...

        # KI
        self.ki_prompt: str = ""
        self.ki_suggested_colors: dict = {}
        self.ki_status: str = ""
        self.ki_error: bool = False
        self.ki_optimizing: bool = False
        self.quotes_extracting: bool = False

        # ... rest ...
```

- [ ] **Step 4: Extend serialization**

In `to_dict()`:

```python
"quotes_enabled": self.quotes_enabled,
"quote_config": { ... },
"ki_prompt": self.ki_prompt,
"ki_suggested_colors": self.ki_suggested_colors,
```

In `from_dict()`:

```python
s.ki_prompt = data.get("ki_prompt", "")
s.ki_suggested_colors = data.get("ki_suggested_colors", {})
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_app_state.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/gui/state.py tests/test_app_state.py
git commit -m "feat(gui): erweitere AppState um KI-Felder und Serialisierung"
```

---

## Task 2: Worker für KI-Optimierung und Quote-Extraktion

**Files:**
- Modify: `src/gui/workers.py`
- Test: `tests/test_gui_workers.py`

**Context:** `GeminiIntegration` hat `optimize_all_settings_async` und `extract_quotes_async`, die `concurrent.futures.Future` zurückgeben.

- [ ] **Step 1: Write failing tests for workers**

```python
# tests/test_gui_workers.py
from unittest.mock import MagicMock
import pytest
from src.gui.workers import AIOptimizeWorker, QuoteExtractWorker


def test_ai_optimize_worker_emits_result(qtbot):
    mock_gemini = MagicMock()
    future = MagicMock()
    future.result.return_value = {"params": {"scale": 1.2}}
    mock_gemini.optimize_all_settings_async.return_value = future

    worker = AIOptimizeWorker(
        gemini=mock_gemini,
        visualizer_type="voice_flow",
        current_params={},
        audio_features={},
        colors={},
        param_specs={},
        user_prompt=None,
    )

    with qtbot.waitSignal(worker.optimize_ready, timeout=100):
        worker.run()

    result = worker.optimize_ready.current_args[0]
    assert result["params"]["scale"] == 1.2


def test_quote_extract_worker_emits_quotes(qtbot):
    mock_gemini = MagicMock()
    future = MagicMock()
    future.result.return_value = [
        {"text": "Hello", "start_time": 1.0, "end_time": 3.0, "confidence": 0.9}
    ]
    mock_gemini.extract_quotes_async.return_value = future

    worker = QuoteExtractWorker(
        gemini=mock_gemini,
        audio_path="test.mp3",
        audio_duration=10.0,
    )

    with qtbot.waitSignal(worker.quotes_ready, timeout=100):
        worker.run()

    quotes = worker.quotes_ready.current_args[0]
    assert len(quotes) == 1
    assert quotes[0]["text"] == "Hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gui_workers.py -v`
Expected: FAIL – `ImportError: cannot import name 'AIOptimizeWorker'`

- [ ] **Step 3: Implement workers**

Append to `src/gui/workers.py`:

```python
class AIOptimizeWorker(QThread):
    optimize_ready = pyqtSignal(dict)
    optimize_error = pyqtSignal(str)

    def __init__(
        self,
        gemini,
        visualizer_type: str,
        current_params: dict,
        audio_features: dict,
        colors: dict,
        param_specs: dict | None = None,
        user_prompt: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.gemini = gemini
        self.visualizer_type = visualizer_type
        self.current_params = current_params
        self.audio_features = audio_features
        self.colors = colors
        self.param_specs = param_specs or {}
        self.user_prompt = user_prompt

    def run(self):
        try:
            future = self.gemini.optimize_all_settings_async(
                visualizer_type=self.visualizer_type,
                current_params=self.current_params,
                audio_features=self.audio_features,
                colors=self.colors,
                param_specs=self.param_specs,
                user_prompt=self.user_prompt,
            )
            result = future.result(timeout=60)
            self.optimize_ready.emit(result)
        except Exception as e:
            self.optimize_error.emit(str(e))


class QuoteExtractWorker(QThread):
    quotes_ready = pyqtSignal(list)
    quotes_error = pyqtSignal(str)

    def __init__(
        self,
        gemini,
        audio_path: str,
        audio_duration: float | None = None,
        max_quotes: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.gemini = gemini
        self.audio_path = audio_path
        self.audio_duration = audio_duration
        self.max_quotes = max_quotes

    def run(self):
        try:
            future = self.gemini.extract_quotes_async(
                audio_path=self.audio_path,
                audio_duration=self.audio_duration,
                max_quotes=self.max_quotes,
            )
            quotes = future.result(timeout=120)
            self.quotes_ready.emit(quotes)
        except Exception as e:
            self.quotes_error.emit(str(e))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_gui_workers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gui/workers.py tests/test_gui_workers.py
git commit -m "feat(gui): füge AIOptimizeWorker und QuoteExtractWorker hinzu"
```

---

## Task 3: KIPanel erstellen

**Files:**
- Create: `src/gui/ki_panel.py`
- Test: `tests/test_ki_panel.py`

**Context:** Panel nutzt `AppState` und optional `GeminiIntegration`. SmartMatcher ist synchron, Gemini asynchron.

- [ ] **Step 1: Write failing GUI test**

```python
# tests/test_ki_panel.py
import pytest
from src.gui.state import AppState
from src.gui.ki_panel import KIPanel


@pytest.fixture
def state():
    return AppState()


def test_ki_panel_creates_widgets(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)
    assert panel.btn_auto_viz is not None
    assert panel.btn_optimize is not None
    assert panel.prompt_input is not None


def test_auto_viz_disabled_without_features(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)
    assert not panel.btn_auto_viz.isEnabled()


def test_auto_viz_enabled_when_features_available(qtbot, state):
    panel = KIPanel(state, gemini=None)
    qtbot.addWidget(panel)
    state.features = {"duration": 10.0}
    assert panel.btn_auto_viz.isEnabled()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ki_panel.py -v`
Expected: FAIL – `ModuleNotFoundError: No module named 'src.gui.ki_panel'`

- [ ] **Step 3: Implement KIPanel**

Create `src/gui/ki_panel.py`:

```python
"""KI-Panel für die PyQt6-GUI."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QTextEdit, QProgressBar,
)

from src.ai_matcher import SmartMatcher
from src.gpu_visualizers import get_visualizer


class KIPanel(QWidget):
    def __init__(self, state, gemini=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.gemini = gemini
        self._matcher = SmartMatcher()

        self._setup_ui()
        self._connect_signals()
        self._update_button_states()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # --- SmartMatcher ---
        auto_box = QGroupBox("Auto-Empfehlung")
        auto_layout = QVBoxLayout(auto_box)
        self.btn_auto_viz = QPushButton("Auto-Visualizer empfehlen")
        self.btn_auto_viz.clicked.connect(self._on_auto_viz)
        auto_layout.addWidget(self.btn_auto_viz)

        self.lbl_recommendation = QLabel("Noch keine Empfehlung")
        self.lbl_recommendation.setWordWrap(True)
        auto_layout.addWidget(self.lbl_recommendation)

        self.btn_apply_recommendation = QPushButton("Empfehlung übernehmen")
        self.btn_apply_recommendation.setEnabled(False)
        self.btn_apply_recommendation.clicked.connect(self._on_apply_recommendation)
        auto_layout.addWidget(self.btn_apply_recommendation)

        layout.addWidget(auto_box)

        # --- Gemini Optimierung ---
        opt_box = QGroupBox("KI-Parameter-Optimierung")
        opt_layout = QVBoxLayout(opt_box)

        opt_layout.addWidget(QLabel("Dein Wunsch (optional):"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("z.B. dunkler, mehr Kontrast, cyberpunk-Stil")
        opt_layout.addWidget(self.prompt_input)

        self.btn_optimize = QPushButton("Parameter optimieren")
        self.btn_optimize.clicked.connect(self._on_optimize)
        opt_layout.addWidget(self.btn_optimize)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        opt_layout.addWidget(self.lbl_status)

        self.lbl_colors = QLabel("")
        self.lbl_colors.setWordWrap(True)
        opt_layout.addWidget(self.lbl_colors)

        layout.addWidget(opt_box)
        layout.addStretch()

    def _connect_signals(self):
        self.state.changed.connect(self._on_state_changed)

    def _on_state_changed(self, key: str):
        if key == "features":
            self._update_button_states()

    def _update_button_states(self):
        has_features = self.state.features is not None
        has_gemini = self.gemini is not None
        self.btn_auto_viz.setEnabled(has_features)
        self.btn_optimize.setEnabled(has_features and has_gemini)
        if not has_gemini:
            self.lbl_status.setText("KI nicht verfügbar. Prüfe API-Key.")

    def _on_auto_viz(self):
        if self.state.features is None:
            return
        try:
            rec = self._matcher.match(self.state.features)
            self._last_recommendation = rec
            self.lbl_recommendation.setText(
                f"{rec.visualizer} (Confidence: {rec.confidence:.0%})\n{rec.reason}"
            )
            self.btn_apply_recommendation.setEnabled(True)
        except Exception as e:
            self.lbl_recommendation.setText(f"Fehler: {e}")

    def _on_apply_recommendation(self):
        rec = getattr(self, "_last_recommendation", None)
        if rec is None:
            return
        self.state.visualizer_type = rec.visualizer
        self.state.viz_params.update(rec.params)
        self.state.ki_suggested_colors = rec.colors
        self.lbl_status.setText("Empfehlung übernommen.")

    def _on_optimize(self):
        if self.state.features is None or self.gemini is None:
            return
        self.state.ki_optimizing = True
        self.btn_optimize.setEnabled(False)
        self.btn_optimize.setText("⏳ KI denkt nach...")
        self.lbl_status.setText("Sende Anfrage an Gemini...")
        # Worker wird von MainWindow gestartet

    def on_optimize_finished(self, result: dict):
        self.state.ki_optimizing = False
        self.btn_optimize.setEnabled(True)
        self.btn_optimize.setText("Parameter optimieren")
        self._apply_optimize_result(result)
        self.lbl_status.setText("Parameter optimiert!")

    def on_optimize_error(self, msg: str):
        self.state.ki_optimizing = False
        self.btn_optimize.setEnabled(True)
        self.btn_optimize.setText("Parameter optimieren")
        self.lbl_status.setText(f"KI-Fehler: {msg}")

    def _apply_optimize_result(self, result: dict):
        if not isinstance(result, dict):
            return

        params = result.get("params", {})
        self.state.viz_params.update(params)

        pp = result.get("postprocess", {})
        if "contrast" in pp:
            self.state.pp_contrast = float(pp["contrast"])
        if "saturation" in pp:
            self.state.pp_saturation = float(pp["saturation"])
        if "brightness" in pp:
            self.state.pp_brightness = float(pp["brightness"])
        if "warmth" in pp:
            self.state.pp_warmth = float(pp["warmth"])
        if "film_grain" in pp:
            self.state.pp_grain = float(pp["film_grain"])

        bg = result.get("background", {})
        if "blur" in bg:
            self.state.bg_blur = float(bg["blur"])
        if "vignette" in bg:
            self.state.bg_vignette = float(bg["vignette"])
        if "opacity" in bg:
            self.state.bg_opacity = float(bg["opacity"])

        colors = result.get("colors", {})
        if colors:
            self.state.ki_suggested_colors = colors
            self.lbl_colors.setText(
                f"Primary: {colors.get('primary', '-')}  "
                f"Secondary: {colors.get('secondary', '-')}  "
                f"BG: {colors.get('background', '-')}"
            )

    def get_optimize_request(self) -> dict:
        """Liefert die Daten, die der AIOptimizeWorker braucht."""
        viz_class = get_visualizer(self.state.visualizer_type)
        param_specs = {}
        if hasattr(viz_class, "EFFECTS"):
            param_specs.update(viz_class.EFFECTS)
        if hasattr(viz_class, "PARAMS"):
            param_specs.update(viz_class.PARAMS)

        return {
            "gemini": self.gemini,
            "visualizer_type": self.state.visualizer_type,
            "current_params": self.state.get_params(),
            "audio_features": self._features_to_dict(self.state.features),
            "colors": self.state.ki_suggested_colors or {},
            "param_specs": param_specs,
            "user_prompt": self.prompt_input.text().strip() or None,
        }

    @staticmethod
    def _features_to_dict(features) -> dict:
        import numpy as np

        def _mean(arr):
            arr = np.asarray(arr)
            return float(arr.mean()) if arr.size else 0.0

        def _std(arr):
            arr = np.asarray(arr)
            return float(arr.std()) if arr.size else 0.0

        return {
            "duration": float(getattr(features, "duration", 0)),
            "tempo": float(getattr(features, "tempo", 120)),
            "mode": str(getattr(features, "mode", "music")),
            "rms_mean": _mean(getattr(features, "rms", [])),
            "rms_std": _std(getattr(features, "rms", [])),
            "onset_mean": _mean(getattr(features, "onset", [])),
            "onset_std": _std(getattr(features, "onset", [])),
            "spectral_mean": _mean(getattr(features, "spectral_centroid", [])),
            "transient_mean": _mean(getattr(features, "transient", [])),
            "voice_clarity_mean": _mean(getattr(features, "voice_clarity", [])),
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ki_panel.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gui/ki_panel.py tests/test_ki_panel.py
git commit -m "feat(gui): füge KIPanel mit SmartMatcher und Gemini-Optimierung hinzu"
```

---

## Task 4: QuotesPanel erstellen

**Files:**
- Create: `src/gui/quotes_panel.py`
- Test: `tests/test_quotes_panel.py`

**Context:** `QuoteOverlayConfig` hat viele Felder. Wir nutzen bestehende Felder, die in `AppState.quote_config` bereits serialisiert werden.

- [ ] **Step 1: Write failing GUI test**

```python
# tests/test_quotes_panel.py
import pytest
from src.gui.state import AppState
from src.gui.quotes_panel import QuotesPanel


@pytest.fixture
def state():
    return AppState()


def test_quotes_panel_creates_widgets(qtbot, state):
    panel = QuotesPanel(state, gemini=None)
    qtbot.addWidget(panel)
    assert panel.chk_enabled is not None
    assert panel.btn_extract is not None
    assert panel.list_quotes is not None


def test_quotes_panel_adds_demo_quote(qtbot, state):
    panel = QuotesPanel(state, gemini=None)
    qtbot.addWidget(panel)
    qtbot.mouseClick(panel.btn_demo, Qt.MouseButton.LeftButton)
    assert len(state.quotes) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quotes_panel.py -v`
Expected: FAIL – `ModuleNotFoundError: No module named 'src.gui.quotes_panel'`

- [ ] **Step 3: Implement QuotesPanel**

Create `src/gui/quotes_panel.py`:

```python
"""Quotes-Panel für die PyQt6-GUI."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QComboBox, QSlider,
    QColorDialog, QGroupBox, QGridLayout, QInputDialog,
)

from src.types import Quote


class QuotesPanel(QWidget):
    def __init__(self, state, gemini=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.gemini = gemini

        self._setup_ui()
        self._connect_signals()
        self._refresh_list()
        self._update_button_states()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # --- Extraktion ---
        extract_box = QGroupBox("Zitate extrahieren")
        extract_layout = QVBoxLayout(extract_box)

        self.chk_enabled = QCheckBox("Zitate aktivieren")
        self.chk_enabled.setChecked(self.state.quotes_enabled)
        self.chk_enabled.stateChanged.connect(self._on_enabled_changed)
        extract_layout.addWidget(self.chk_enabled)

        btn_row = QHBoxLayout()
        self.btn_extract = QPushButton("Key-Zitate extrahieren")
        self.btn_extract.clicked.connect(self._on_extract)
        btn_row.addWidget(self.btn_extract)

        self.btn_demo = QPushButton("Demo-Zitate")
        self.btn_demo.clicked.connect(self._on_demo)
        btn_row.addWidget(self.btn_demo)
        extract_layout.addLayout(btn_row)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        extract_layout.addWidget(self.lbl_status)

        layout.addWidget(extract_box)

        # --- Liste ---
        list_box = QGroupBox("Zitat-Liste")
        list_layout = QVBoxLayout(list_box)
        self.list_quotes = QListWidget()
        self.list_quotes.setMaximumHeight(180)
        list_layout.addWidget(self.list_quotes)

        list_btn_row = QHBoxLayout()
        self.btn_add = QPushButton("➕ Hinzufügen")
        self.btn_add.clicked.connect(self._on_add)
        list_btn_row.addWidget(self.btn_add)

        self.btn_remove = QPushButton("🗑 Entfernen")
        self.btn_remove.clicked.connect(self._on_remove)
        list_btn_row.addWidget(self.btn_remove)

        self.btn_edit = QPushButton("✏️ Bearbeiten")
        self.btn_edit.clicked.connect(self._on_edit)
        list_btn_row.addWidget(self.btn_edit)
        list_layout.addLayout(list_btn_row)

        layout.addWidget(list_box)

        # --- Erscheinungsbild ---
        style_box = QGroupBox("Erscheinungsbild")
        style_layout = QGridLayout(style_box)

        style_layout.addWidget(QLabel("Position"), 0, 0)
        self.combo_position = QComboBox()
        self.combo_position.addItems(["bottom", "center", "top"])
        self.combo_position.setCurrentText(self.state.quote_config.position)
        self.combo_position.currentTextChanged.connect(self._on_style_changed)
        style_layout.addWidget(self.combo_position, 0, 1)

        style_layout.addWidget(QLabel("Schriftgröße"), 1, 0)
        self.slider_font_size = self._make_slider(16, 96, self.state.quote_config.font_size)
        self.slider_font_size.valueChanged.connect(self._on_style_changed)
        style_layout.addWidget(self.slider_font_size, 1, 1)

        style_layout.addWidget(QLabel("Fade-Dauer"), 2, 0)
        self.slider_fade = self._make_slider(1, 20, int(self.state.quote_config.fade_duration * 10))
        self.slider_fade.valueChanged.connect(self._on_style_changed)
        style_layout.addWidget(self.slider_fade, 2, 1)

        style_layout.addWidget(QLabel("Anzeigedauer"), 3, 0)
        self.slider_display = self._make_slider(20, 200, int(self.state.quote_config.display_duration * 10))
        self.slider_display.valueChanged.connect(self._on_style_changed)
        style_layout.addWidget(self.slider_display, 3, 1)

        self.btn_font_color = QPushButton("Textfarbe wählen")
        self.btn_font_color.clicked.connect(self._on_font_color)
        style_layout.addWidget(self.btn_font_color, 4, 0)

        self.btn_box_color = QPushButton("Box-Farbe wählen")
        self.btn_box_color.clicked.connect(self._on_box_color)
        style_layout.addWidget(self.btn_box_color, 4, 1)

        layout.addWidget(style_box)
        layout.addStretch()

    def _make_slider(self, min_val: int, max_val: int, default: int):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        return slider

    def _connect_signals(self):
        self.state.changed.connect(self._on_state_changed)

    def _on_state_changed(self, key: str):
        if key in {"features", "audio_path"}:
            self._update_button_states()
        if key == "quotes":
            self._refresh_list()

    def _update_button_states(self):
        has_audio = bool(self.state.audio_path)
        has_features = self.state.features is not None
        has_gemini = self.gemini is not None
        self.btn_extract.setEnabled(has_audio and has_features and has_gemini)
        if not has_gemini:
            self.lbl_status.setText("KI nicht verfügbar. Prüfe API-Key.")

    def _refresh_list(self):
        self.list_quotes.clear()
        for q in self.state.quotes:
            text = f"{q.text[:40]}{'...' if len(q.text) > 40 else ''} ({q.start_time:.1f}s - {q.end_time:.1f}s)"
            self.list_quotes.addItem(text)

    def _on_enabled_changed(self, state):
        self.state.quotes_enabled = bool(state)

    def _on_extract(self):
        if self.state.audio_path is None or self.state.features is None or self.gemini is None:
            return
        self.state.quotes_extracting = True
        self.btn_extract.setEnabled(False)
        self.btn_extract.setText("⏳ Extrahiere...")
        self.lbl_status.setText("Sende Anfrage an Gemini...")
        # Worker wird von MainWindow gestartet

    def on_extract_finished(self, quotes: list):
        self.state.quotes_extracting = False
        self.btn_extract.setEnabled(True)
        self.btn_extract.setText("Key-Zitate extrahieren")
        self.state.quotes = quotes
        self.lbl_status.setText(f"{len(quotes)} Zitate extrahiert.")

    def on_extract_error(self, msg: str):
        self.state.quotes_extracting = False
        self.btn_extract.setEnabled(True)
        self.btn_extract.setText("Key-Zitate extrahieren")
        self.lbl_status.setText(f"Fehler: {msg}")

    def _on_demo(self):
        duration = getattr(self.state.features, "duration", 10.0) or 10.0
        demo = [
            Quote(text="Das ist ein Beispielzitat.", start_time=1.0, end_time=4.0, confidence=0.9),
            Quote(text="Und hier noch ein zweites Highlight.", start_time=duration * 0.4, end_time=duration * 0.4 + 3.0, confidence=0.85),
        ]
        self.state.quotes = demo
        self.lbl_status.setText("Demo-Zitate hinzugefügt.")

    def _on_add(self):
        duration = getattr(self.state.features, "duration", 10.0) or 10.0
        text, ok = QInputDialog.getText(self, "Zitat hinzufügen", "Text:")
        if ok and text:
            new_quote = Quote(text=text, start_time=duration * 0.3, end_time=duration * 0.3 + 3.0, confidence=1.0)
            self.state.quotes = self.state.quotes + [new_quote]

    def _on_remove(self):
        row = self.list_quotes.currentRow()
        if row >= 0:
            quotes = list(self.state.quotes)
            quotes.pop(row)
            self.state.quotes = quotes

    def _on_edit(self):
        row = self.list_quotes.currentRow()
        if row < 0 or row >= len(self.state.quotes):
            return
        q = self.state.quotes[row]
        text, ok = QInputDialog.getText(self, "Zitat bearbeiten", "Text:", text=q.text)
        if ok and text:
            quotes = list(self.state.quotes)
            quotes[row] = Quote(text=text, start_time=q.start_time, end_time=q.end_time, confidence=q.confidence)
            self.state.quotes = quotes

    def _on_style_changed(self):
        qc = self.state.quote_config
        qc.position = self.combo_position.currentText()
        qc.font_size = self.slider_font_size.value()
        qc.fade_duration = self.slider_fade.value() / 10.0
        qc.display_duration = self.slider_display.value() / 10.0
        # Trigger update via state.set to emit signal for dependent widgets
        self.state.set("quote_config", qc)

    def _on_font_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.state.quote_config.font_color = (color.red(), color.green(), color.blue())
            self.state.set("quote_config", self.state.quote_config)

    def _on_box_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
            self.state.quote_config.box_color = (r, g, b, a)
            self.state.set("quote_config", self.state.quote_config)

    def get_extract_request(self) -> dict:
        """Liefert die Daten, die der QuoteExtractWorker braucht."""
        return {
            "gemini": self.gemini,
            "audio_path": self.state.audio_path,
            "audio_duration": getattr(self.state.features, "duration", None),
            "max_quotes": None,
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_quotes_panel.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gui/quotes_panel.py tests/test_quotes_panel.py
git commit -m "feat(gui): füge QuotesPanel für Extraktion und Verwaltung hinzu"
```

---

## Task 5: MainWindow – rechtes Panel zu Tabs umbauen

**Files:**
- Modify: `src/gui/main_window.py`
- Test: `tests/test_gui_main_window.py` (neu)

**Context:** Aktuell ist `self.params_panel` direkt im Splitter. Wir wandeln es in ein `QTabWidget` um.

- [ ] **Step 1: Write failing smoke test**

```python
# tests/test_gui_main_window.py
import pytest
from src.gui.main_window import MainWindow


def test_main_window_has_three_tabs(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.right_tabs.count() == 3
    assert window.right_tabs.tabText(0) == "Params"
    assert window.right_tabs.tabText(1) == "KI"
    assert window.right_tabs.tabText(2) == "Quotes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gui_main_window.py -v`
Expected: FAIL – `AttributeError: 'MainWindow' object has no attribute 'right_tabs'`

- [ ] **Step 3: Modify MainWindow**

Edit `src/gui/main_window.py`:

```python
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QStatusBar, QLabel, QMessageBox, QTabWidget,
)

from src.gui.assets_panel import AssetsPanel
from src.gui.params_panel import ParamsPanel
from src.gui.ki_panel import KIPanel
from src.gui.quotes_panel import QuotesPanel
from src.gui.preview_widget import PreviewWidget
from src.gui.state import AppState
from src.gui.styles import build_app_stylesheet, Theme
from src.gui.timeline_widget import TimelineWidget
from src.gui.workers import AnalyzeWorker, PreviewWorker, AIOptimizeWorker, QuoteExtractWorker

# ...

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer Pro")
        self.setMinimumSize(1200, 750)

        self.state = AppState()
        self._preview_worker: PreviewWorker | None = None
        self._analyze_worker: AnalyzeWorker | None = None
        self._ai_optimize_worker: AIOptimizeWorker | None = None
        self._quote_extract_worker: QuoteExtractWorker | None = None

        # Gemini-Integration einmalig initialisieren
        self.gemini = None
        try:
            from src.gemini_integration import GeminiIntegration
            self.gemini = GeminiIntegration()
        except Exception as e:
            print(f"[GUI] Gemini nicht verfügbar: {e}")

        self._setup_ui()
        self._setup_signals()
```

In `_setup_ui`, ersetze den rechten Panel-Block:

```python
        # Rechtes Panel als Tabs
        self.right_tabs = QTabWidget()
        self.params_panel = ParamsPanel(self.state)
        self.ki_panel = KIPanel(self.state, gemini=self.gemini)
        self.quotes_panel = QuotesPanel(self.state, gemini=self.gemini)

        self.right_tabs.addTab(self.params_panel, "Params")
        self.right_tabs.addTab(self.ki_panel, "KI")
        self.right_tabs.addTab(self.quotes_panel, "Quotes")
        splitter.addWidget(self.right_tabs)
```

- [ ] **Step 4: Verbinde KI-Worker-Lifecycle**

In `_setup_signals`:

```python
        self.ki_panel.btn_optimize.clicked.connect(self._start_ai_optimize)
        self.quotes_panel.btn_extract.clicked.connect(self._start_quote_extract)
```

Neue Methoden in `MainWindow`:

```python
    def _start_ai_optimize(self):
        if self._ai_optimize_worker and self._ai_optimize_worker.isRunning():
            return
        req = self.ki_panel.get_optimize_request()
        self._ai_optimize_worker = AIOptimizeWorker(
            gemini=req["gemini"],
            visualizer_type=req["visualizer_type"],
            current_params=req["current_params"],
            audio_features=req["audio_features"],
            colors=req["colors"],
            param_specs=req["param_specs"],
            user_prompt=req["user_prompt"],
            parent=self,
        )
        self._ai_optimize_worker.optimize_ready.connect(self.ki_panel.on_optimize_finished)
        self._ai_optimize_worker.optimize_error.connect(self.ki_panel.on_optimize_error)
        self._ai_optimize_worker.start()

    def _start_quote_extract(self):
        if self._quote_extract_worker and self._quote_extract_worker.isRunning():
            return
        req = self.quotes_panel.get_extract_request()
        self._quote_extract_worker = QuoteExtractWorker(
            gemini=req["gemini"],
            audio_path=req["audio_path"],
            audio_duration=req["audio_duration"],
            parent=self,
        )
        self._quote_extract_worker.quotes_ready.connect(self.quotes_panel.on_extract_finished)
        self._quote_extract_worker.quotes_error.connect(self.quotes_panel.on_extract_error)
        self._quote_extract_worker.start()
```

In `closeEvent` Worker sauber beenden:

```python
    def closeEvent(self, event):
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.requestInterruption()
            self._preview_worker.wait(500)
        if self._analyze_worker and self._analyze_worker.isRunning():
            self._analyze_worker.wait(500)
        if self._ai_optimize_worker and self._ai_optimize_worker.isRunning():
            self._ai_optimize_worker.wait(500)
        if self._quote_extract_worker and self._quote_extract_worker.isRunning():
            self._quote_extract_worker.wait(500)
        event.accept()
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_gui_main_window.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/gui/main_window.py tests/test_gui_main_window.py
git commit -m "feat(gui): rechtes Panel zu Tabs mit KI und Quotes umbauen"
```

---

## Task 6: GUI-Smoke-Test und Integration

**Files:**
- Test: `tests/test_gui_smoke.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/test_gui_smoke.py
import pytest
from src.gui.main_window import MainWindow


def test_main_window_opens(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.isVisible() or not window.isVisible()
    assert window.preview_widget is not None
    assert window.right_tabs is not None
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_gui_smoke.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_gui_smoke.py
git commit -m "test(gui): füge GUI-Smoke-Test hinzu"
```

---

## Task 7: Full Test Suite und Linting

- [ ] **Step 1: Run full GUI tests**

Run: `pytest tests/test_app_state.py tests/test_gui_workers.py tests/test_ki_panel.py tests/test_quotes_panel.py tests/test_gui_main_window.py tests/test_gui_smoke.py -v`
Expected: ALL PASS

- [ ] **Step 2: Run existing tests for regressions**

Run: `pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: ALL PASS (oder bekannte GPU-bedingte Fehler)

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "test(gui): verifiziere KI-Feature-Integration"
```

---

## Spec Coverage Check

| Spec Requirement | Implementing Task |
|------------------|-------------------|
| Rechtes Panel als Tabs (Params/KI/Quotes) | Task 5 |
| AppState KI-Felder + Serialisierung | Task 1 |
| SmartMatcher Auto-Visualizer | Task 3 |
| Gemini Parameter-Optimierung | Task 2 + Task 3 + Task 5 |
| Quote-Extraktion | Task 2 + Task 4 + Task 5 |
| Quote-Verwaltung + Erscheinungsbild | Task 4 |
| Worker für asynchrone KI-Aufrufe | Task 2 |
| Fehlerbehandlung (kein API-Key, etc.) | Task 3 + Task 4 |
| Tests | Alle Tasks |

## Placeholder Scan

Keine TBD/TODO. Alle Steps enthalten konkreten Code, Dateipfade oder Befehle.

## Type Consistency Check

- `AIOptimizeWorker` und `QuoteExtractWorker` Signaturen stimmen in Task 2 und Task 5 überein.
- `get_optimize_request()` / `get_extract_request()` liefern Dicts mit den erwarteten Keys.
- `AppState`-Felder `ki_prompt`, `ki_suggested_colors`, etc. werden in Task 1 definiert und in Task 3/4 genutzt.
