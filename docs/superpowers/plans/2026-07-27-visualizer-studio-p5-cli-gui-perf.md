# Visualizer Studio — P5 CLI, GUI, Video-Degradation & Perf Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Die Studio-Pipeline wird nutzbar: CLI-Flags (`--studio`, `--studio-dry`, `--studio-strict`), Video-Hintergrund-Degradation statt Abbruch, GUI-Anbindung (Modus-Badge, Quality-Badge, Solver-Trace, Studio-Preset-Button) und Perf-Budget-Tests als harte Akzeptanzkriterien.

**Architecture:** Vier kleine, getrennte Integrationen. CLI: drei Flags am bestehenden `render`-Kommando (`main.py:73-93`), die in `run_studio_auto` (P4) münden; `--studio-dry` stoppt nach dem Solve, `--studio-strict` macht Masken-Fallbacks zum Fehler. Video-Degradation: `run_studio` erkennt Video-Hintergründe und deaktiviert M3 (statt Abbruch, Spec §14/C17). GUI: Badges im `StudioPanel` + Preset-Button im `KIPanel` (PyQt6, pytest-qt vorhanden). Perf: Budgets aus Spec §13 als Assertions mit CI-Faktor. Spec: `docs/superpowers/specs/2026-07-27-visualizer-studio-design.md` (studio-spec/2.1, §11, §13, §14, §16 P5).

**Tech Stack:** Python 3.11, click, PyQt6, pytest + pytest-qt (Bestand, `plugins: qt-4.5.0`). Keine neuen Dependencies.

**Voraussetzung (P0–P4, abgeschlossen):** `src/studio/` komplett inkl. `run_studio`, `run_studio_auto`, `classify_mode`, `build_preset`, `get_subject_mask`. GUI: `src/gui/studio_panel.py` (`StudioPanel`), `src/gui/ki_panel.py` (`KIPanel`), `src/gui/state.py` (`AppState`, QObject + pyqtSignal).

## Global Constraints

Werte wörtlich aus der Spec — jeder Task erbt diese Anforderungen implizit:

- CLI: `--studio` (Engine-Flow), `--studio-dry` (Analyse + Solve, kein Commit-Render), `--studio-strict` (Fallback-Maskenprovider = Fehler statt Warnung); Sidecar `<output>.studio.json`; Exit-Code ≠ 0 bei Gate-Abbruch (Spec §11.3).
- Video-Hintergrund: **Degradation statt Abbruch** — M3/Maskenregeln aus, Rest aktiv, `mask.provider = "none:video_background"`, Warnung im Report (Spec §14, C17).
- Perf-Budgets (Spec §13): Metrik/Sample @854 px ≤ 15 ms CPU · Feasibility ≤ 200 ms · Solver (synthetisch) ≤ 20 s Worst Case. Überschreitung = Defekt. Tests nutzen **CI-Faktor ×4**.
- GUI: Maskenerzeugung **niemals** im UI-Thread (Spec §11.1); kein neues Hauptfenster (Spec §11.2).
- Code-Kommentare und Commit-Messages auf Deutsch (AGENTS.md).

## Dateistruktur

| Datei | Verantwortung |
|-------|---------------|
| `main.py` (Modify) | `--studio`, `--studio-dry`, `--studio-strict` am render-Kommando |
| `src/studio/engine.py` (Modify) | Video-Erkennung + Degradation in `run_studio`; `dry_run`-Parameter |
| `src/studio/mask_service.py` (Modify) | `strict`-Parameter: Fallback-Kette bricht statt Gauß ab |
| `src/gui/studio_panel.py` (Modify) | Modus-Badge, Quality-Badge, Solver-Trace |
| `src/gui/ki_panel.py` (Modify) | Button „Studio-Preset anwenden" |
| `tests/test_studio_cli.py` (neu) | CLI-Flags (CliRunner, gemockte Engine) |
| `tests/test_studio_video_background.py` (neu) | Video-Degradation (Spec §15) |
| `tests/test_studio_gui.py` (neu) | Badges + Preset-Button (pytest-qt) |
| `tests/test_studio_perf.py` (neu) | Budget-Assertions (Spec §13) |

---

### Task 1: CLI-Flags + Dry-Run

**Files:**
- Modify: `main.py` (render-Kommando, `main.py:73-93` + Rumpf)
- Modify: `src/studio/engine.py` (`run_studio`/`run_studio_auto`: `dry_run`-Parameter)
- Test: `tests/test_studio_cli.py`

**Interfaces:**
- Consumes: `run_studio_auto` (P4), `AudioAnalyzer` (Bestand).
- Produces:
  - CLI-Optionen `--studio`, `--studio-dry`, `--studio-strict` am `render`-Kommando
  - `run_studio_auto(..., dry_run: bool = False, strict: bool = False)` — bei `dry_run=True`: kein Commit-Render, Sidecar mit `verify.status = "dry_run"`
  - Konsumiert von Nutzern + Task 2 (strict in Maskenpfad).

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_cli.py`:

```python
"""Tests für die Studio-CLI-Flags (Spec §11.3)."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from main import cli


def _fake_sidecar(out):
    return {
        "schema_version": "studio-decision/2.1",
        "mode": {"value": "MUSIC"},
        "verify": {"status": "pass"},
    }


def test_studio_flag_ruft_engine(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    runner = CliRunner()
    with patch("main._run_studio_pipeline") as mock_pipe:
        mock_pipe.return_value = _fake_sidecar(str(tmp_path / "out.mp4"))
        result = runner.invoke(cli, ["render", str(audio), "--studio",
                                     "-o", str(tmp_path / "out.mp4")])
    assert result.exit_code == 0, result.output
    mock_pipe.assert_called_once()
    assert mock_pipe.call_args.kwargs.get("dry_run") is False


def test_studio_dry_ohne_commit(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    runner = CliRunner()
    with patch("main._run_studio_pipeline") as mock_pipe:
        mock_pipe.return_value = _fake_sidecar(str(tmp_path / "out.mp4"))
        result = runner.invoke(cli, ["render", str(audio), "--studio",
                                     "--studio-dry",
                                     "-o", str(tmp_path / "out.mp4")])
    assert result.exit_code == 0, result.output
    assert mock_pipe.call_args.kwargs.get("dry_run") is True


def test_studio_strict_wird_durchgereicht(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    runner = CliRunner()
    with patch("main._run_studio_pipeline") as mock_pipe:
        mock_pipe.return_value = _fake_sidecar(str(tmp_path / "out.mp4"))
        result = runner.invoke(cli, ["render", str(audio), "--studio",
                                     "--studio-strict",
                                     "-o", str(tmp_path / "out.mp4")])
    assert result.exit_code == 0, result.output
    assert mock_pipe.call_args.kwargs.get("strict") is True


def test_ohne_studio_flag_bleibt_klassisch(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    runner = CliRunner()
    with patch("main._run_studio_pipeline") as mock_pipe, \
         patch("main.GPUBatchRenderer") as mock_renderer:
        mock_renderer.return_value.render = MagicMock()
        result = runner.invoke(cli, ["render", str(audio),
                                     "-o", str(tmp_path / "out.mp4")])
    mock_pipe.assert_not_called()
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_cli.py -v`
Expected: FAIL — 3 Tests mit `AssertionError` (mock_pipe nicht aufgerufen, weil Flags fehlen: `Error: No such option: --studio`)

- [ ] **Step 3: Implementierung**

**3a) `src/studio/engine.py`** — `run_studio_auto`-Signatur erweitern und am Ende vor dem Commit verzweigen:

```python
def run_studio_auto(audio_path, features, features_dict, output_path,
                    profile_name=None, params_override=None,
                    postprocess_override=None, background_image=None,
                    subject_mask=None, dry_run=False, strict=False) -> dict:
```

und nach dem Solve-Abschnitt in `run_studio` einen Dry-Run-Ausstieg einbauen (die sauberste Stelle: `run_studio` bekommt ebenfalls `dry_run=False`; bei `dry_run=True` werden Schritte 3 (Commit) und 4 (Verify) übersprungen und das Sidecar mit `verify: {"status": "dry_run", "metrics": probe_metrics}` geschrieben). `strict` wird an `get_subject_mask` weitergereicht, sobald `background_image` genutzt wird (Task 2 verdrahtet den Maskenpfad; hier nur Parameter durchreichen).

**3b) `main.py`** — am render-Kommando drei Optionen ergänzen (nach `--intro-fade`):

```python
@click.option('--studio', is_flag=True, help='Studio-Engine: Probe→Solve→1×Render→Verify + Sidecar')
@click.option('--studio-dry', is_flag=True, help='Nur Analyse + Solve, kein Commit-Render')
@click.option('--studio-strict', is_flag=True, help='Masken-Fallback = Fehler statt Warnung')
```

Funktionssignatur um `studio, studio_dry, studio_strict` erweitern und am Anfang des Rumpfs:

```python
    if studio or studio_dry or studio_strict:
        sidecar = _run_studio_pipeline(
            audio_file, output, visual=visual, resolution=resolution,
            fps=fps, background_image=background_image,
            dry_run=studio_dry, strict=studio_strict,
        )
        status = sidecar.get("verify", {}).get("status")
        click.echo(f"[Studio] Modus: {sidecar['mode']['value']} | Verify: {status} | Sidecar: {output.replace('.mp4', '.studio.json')}")
        if status not in ("pass", "dry_run"):
            raise SystemExit(1)
        return
```

Sowie die Hilfsfunktion in `main.py` (z.B. direkt über dem render-Kommando):

```python
def _run_studio_pipeline(audio_file, output, visual=None, resolution="1920x1080",
                         fps=60, background_image=None, dry_run=False,
                         strict=False):
    """Studio-Flow für die CLI: Analyse → run_studio_auto (Spec §11.3)."""
    from src.analyzer import AudioAnalyzer
    from src.render_common import build_features_dict
    from src.studio.engine import run_studio_auto
    from src.studio.mask_service import get_subject_mask

    analyzer = AudioAnalyzer()
    features = analyzer.analyze(audio_file, fps=fps)
    features_dict = build_features_dict(features, features.frame_count, fps)

    mask = None
    if background_image:
        from src.studio.engine import is_video_background
        if not is_video_background(background_image):
            mask = get_subject_mask(background_image, strict=strict).mask

    return run_studio_auto(
        audio_file, features, features_dict, output,
        background_image=background_image, subject_mask=mask,
        dry_run=dry_run, strict=strict,
    )
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_cli.py tests/test_cli.py -v`
Expected: alle PASS (Bestands-CLI-Tests unberührt)

- [ ] **Step 5: Commit**

```bash
git add main.py src/studio/engine.py tests/test_studio_cli.py
git commit -m "feat(studio): P5 CLI — --studio/--studio-dry/--studio-strict"
```

---

### Task 2: Video-Degradation + Maskenpfad

**Files:**
- Modify: `src/studio/engine.py` (`is_video_background`, Degradation in `run_studio`)
- Modify: `src/studio/mask_service.py` (`strict`-Parameter)
- Test: `tests/test_studio_video_background.py`

**Interfaces:**
- Consumes: `get_subject_mask` (P1), `run_studio` (P3).
- Produces:
  - `is_video_background(path: str) -> bool` (in `engine.py`; Endungen mp4/mov/avi/mkv/webm/gif)
  - `get_subject_mask(image_path, cache_dir=..., strict: bool = False)` — bei `strict=True` und weder rembg noch cv2 verfügbar: `RuntimeError` statt Gauß-Fallback (Spec §14)
  - `run_studio` mit Video-Hintergrund: M3 aus, `mask.provider = "none:video_background"`, Warnung im Sidecar (Spec §14)
  - Konsumiert von Task 1 (CLI) und Spec §15-Test.

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_video_background.py`:

```python
"""Video-Hintergrund: Degradation statt Abbruch (Spec §14, §15, C17)."""

from unittest.mock import MagicMock, patch

import pytest

from src.studio.engine import is_video_background

pytestmark = pytest.mark.gpu


def test_is_video_background():
    assert is_video_background("clip.mp4") is True
    assert is_video_background("clip.MKV") is True
    assert is_video_background("bild.png") is False


def test_video_hintergrund_degradiert_statt_abbruch(tmp_path, dummy_audio_features):
    from src.gpu_renderer import GPUBatchRenderer
    from src.render_common import build_features_dict
    from src.studio.constraints import ConstraintSet
    from src.studio.engine import run_studio

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    with patch.object(GPUBatchRenderer, "render",
                      MagicMock(side_effect=RuntimeError("mock"))), \
         patch("src.gpu_renderer.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        sidecar = run_studio(
            str(audio), "spectrum_bars", dummy_audio_features, features_dict,
            str(out), constraints=ConstraintSet(max_overlay_alpha=1.0),
            background_image=str(tmp_path / "hintergrund.mp4"),
        )

    # Explizit KEIN Abbruch: Lauf erfolgreich, M3 deaktiviert (Spec §15)
    assert sidecar["mask"]["provider"] == "none:video_background"
    assert sidecar["verify"]["metrics"]["M3"] is None
    assert any("video" in w.lower()
               for w in sidecar["mask"].get("warnings", []))


def test_strict_ohne_provider_wirft(tmp_path, monkeypatch):
    from PIL import Image
    from src.studio import mask_service

    monkeypatch.setattr(mask_service, "_try_rembg", lambda img: None)
    monkeypatch.setattr(mask_service, "_try_opencv", lambda img: None)
    img = tmp_path / "bg.png"
    Image.new("RGB", (32, 32)).save(img)
    with pytest.raises(RuntimeError, match="strict"):
        mask_service.get_subject_mask(str(img),
                                      cache_dir=str(tmp_path / "c"),
                                      strict=True)
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_video_background.py -v`
Expected: FAIL mit `ImportError: cannot import name 'is_video_background'`

- [ ] **Step 3: Implementierung**

**3a) `src/studio/mask_service.py`** — `get_subject_mask` um `strict` erweitern:

```python
def get_subject_mask(image_path: str, cache_dir: str = DEFAULT_CACHE_DIR,
                     strict: bool = False) -> MaskResult:
```

und im Fallback-Zweig (vor dem Gauß):

```python
    if mask is None and strict:
        raise RuntimeError(
            "strict: weder rembg noch OpenCV verfügbar — "
            "kein Masken-Fallback erlaubt (Spec §14)."
        )
```

**3b) `src/studio/engine.py`** — anfügen bzw. `run_studio` anpassen:

```python
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif")


def is_video_background(path: str) -> bool:
    """Erkennt Video-Hintergründe an der Endung (Spec §14)."""
    return str(path).lower().endswith(VIDEO_EXTENSIONS)
```

In `run_studio`, am Anfang (vor Feasibility):

```python
    mask_warnings: list[str] = []
    if background_image and is_video_background(background_image):
        # C17: Degradation statt Abbruch — M3 aus, Rest aktiv (Spec §14)
        subject_mask = None
        mask_warnings.append(
            "Video-Hintergrund: Subjekt-Maskierung (M3) deaktiviert, "
            "übrige Metriken aktiv (Spec §14)."
        )
```

und im Sidecar-Block `"mask"`:

```python
        "mask": {
            "provider": ("none:video_background"
                         if background_image and is_video_background(background_image)
                         else ("provided" if subject_mask is not None else "none")),
            "cache_hit": False,
            "warnings": mask_warnings,
        },
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_video_background.py -v`
Expected: 3 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/engine.py src/studio/mask_service.py tests/test_studio_video_background.py
git commit -m "feat(studio): P5 Video-Degradation (C17) + strict-Maskenpfad"
```

---

### Task 3: GUI — Badges + Studio-Preset-Button

**Files:**
- Modify: `src/gui/studio_panel.py` (Modus-Badge, Quality-Badge, Solver-Trace)
- Modify: `src/gui/ki_panel.py` (Button „Studio-Preset anwenden")
- Test: `tests/test_studio_gui.py`

**Interfaces:**
- Consumes: `StudioPanel`/`KIPanel` (`src/gui/`), `classify_mode` (P4), `build_preset` (P4), `AppState` (`src/gui/state.py`).
- Produces:
  - `StudioPanel.update_studio_badges(mode_result: ModeResult, verify_metrics: dict | None = None, solver_result: SolveResult | None = None)` — setzt Modus-Badge (+Konfidenz), Quality-Badge (M1/M3/M5), Solver-Trace (aufklappbare Liste der Hebel-Schritte + J-Verlauf)
  - `KIPanel.set_studio_preset_callback(fn)` + Button; der Callback ruft `build_preset` auf und übergibt das Preset an `AppState` (kein direkter Engine-Aufruf aus dem Panel — Qt-trennbar testbar)
  - Konsumiert vom Nutzer; `main_window`-Verdrahtung minimal (bestehende Signale).

Hinweis für den Implementierer: Die genaue interne Struktur von `StudioPanel`/`KIPanel`/`AppState` vorher lesen und die neuen Elemente **anhängen** (eigene Sektion unten im Panel), nichts Bestehendes umbauen. Badge = `QLabel`, Solver-Trace = `QToolButton` (checkable) + `QListWidget`.

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_gui.py`:

```python
"""Tests für die Studio-GUI-Elemente (Spec §11.2)."""

import pytest

pytestmark = pytest.mark.gui


def test_modus_badge_zeigt_modus_und_konfidenz(qtbot):
    from src.gui.studio_panel import StudioPanel
    from src.studio.mode_gate import ModeResult

    panel = StudioPanel()
    qtbot.addWidget(panel)
    panel.update_studio_badges(
        ModeResult(value="PODCAST", resolved="podcast", confidence=0.87,
                   speech_score=0.63, hysteresis_applied=False)
    )
    assert "PODCAST" in panel.mode_badge.text()
    assert "0.87" in panel.mode_badge.text()


def test_quality_badge_zeigt_metriken(qtbot):
    from src.gui.studio_panel import StudioPanel
    from src.studio.mode_gate import ModeResult

    panel = StudioPanel()
    qtbot.addWidget(panel)
    panel.update_studio_badges(
        ModeResult(value="MUSIC", resolved="music", confidence=0.9,
                   speech_score=0.1, hysteresis_applied=False),
        verify_metrics={"M1": 0.19, "M3": 0.06, "M5": 0.05},
    )
    assert "M1" in panel.quality_badge.text()
    assert "0.19" in panel.quality_badge.text()


def test_solver_trace_listet_schritte(qtbot):
    from src.gui.studio_panel import StudioPanel
    from src.studio.mode_gate import ModeResult
    from src.studio.solver import SolveResult

    panel = StudioPanel()
    qtbot.addWidget(panel)
    result = SolveResult(
        params={"alpha_cap": 0.84},
        j_trace=[0.41, 0.18, 0.0],
        steps=[{"lever": "alpha_cap", "op": "-0.08",
                "j_before": 0.41, "j_after": 0.18}],
        iterations=2, status="solved",
    )
    panel.update_studio_badges(
        ModeResult(value="MUSIC", resolved="music", confidence=0.9,
                   speech_score=0.1, hysteresis_applied=False),
        solver_result=result,
    )
    assert panel.solver_trace_list.count() == 1
    assert "alpha_cap" in panel.solver_trace_list.item(0).text()


def test_preset_button_ruft_callback(qtbot):
    from src.gui.ki_panel import KIPanel

    panel = KIPanel()
    qtbot.addWidget(panel)
    received = []
    panel.set_studio_preset_callback(received.append)
    panel.studio_preset_button.click()
    assert len(received) == 1
```

Hinweis: Konstruktor-Signaturen der Panels vorher prüfen (`src/gui/studio_panel.py`, `src/gui/ki_panel.py`) und die Tests an die reale Signatur anpassen (z.B. falls sie `AppState` verlangen).

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_gui.py -v`
Expected: FAIL mit `AttributeError: ... 'update_studio_badges'` / `'set_studio_preset_callback'`

- [ ] **Step 3: Implementierung**

**3a) `src/gui/studio_panel.py`** — am Ende von `__init__` (Layout anhängen):

```python
        # --- Studio-Badges (Spec §11.2) ---
        from PyQt6.QtWidgets import QLabel, QListWidget, QToolButton
        self.mode_badge = QLabel("Modus: —")
        self.quality_badge = QLabel("Qualität: —")
        self.solver_trace_toggle = QToolButton()
        self.solver_trace_toggle.setText("Solver-Trace")
        self.solver_trace_toggle.setCheckable(True)
        self.solver_trace_list = QListWidget()
        self.solver_trace_list.setVisible(False)
        self.solver_trace_toggle.toggled.connect(
            self.solver_trace_list.setVisible)
        layout = self.layout()
        layout.addWidget(self.mode_badge)
        layout.addWidget(self.quality_badge)
        layout.addWidget(self.solver_trace_toggle)
        layout.addWidget(self.solver_trace_list)
```

Sowie die Methode:

```python
    def update_studio_badges(self, mode_result, verify_metrics=None,
                             solver_result=None):
        """Aktualisiert Modus-/Quality-Badge und Solver-Trace (Spec §11.2)."""
        self.mode_badge.setText(
            f"Modus: {mode_result.value} "
            f"(Konfidenz {mode_result.confidence:.2f})"
        )
        if verify_metrics:
            parts = [f"{k}={v:.2f}" for k, v in verify_metrics.items()
                     if isinstance(v, (int, float))]
            self.quality_badge.setText("Qualität: " + ", ".join(parts))
        self.solver_trace_list.clear()
        if solver_result:
            for step in solver_result.steps:
                self.solver_trace_list.addItem(
                    f"{step['lever']} {step['op']}: "
                    f"J {step['j_before']:.2f} → {step['j_after']:.2f}"
                )
```

**3b) `src/gui/ki_panel.py`** — Button + Callback:

```python
        # --- Studio-Preset (Spec §11.2) ---
        from PyQt6.QtWidgets import QPushButton
        self.studio_preset_button = QPushButton("Studio-Preset anwenden")
        self._studio_preset_cb = None
        self.studio_preset_button.clicked.connect(self._on_studio_preset)
        self.layout().addWidget(self.studio_preset_button)
```

```python
    def set_studio_preset_callback(self, fn):
        """Registriert den Callback für „Studio-Preset anwenden"."""
        self._studio_preset_cb = fn

    def _on_studio_preset(self):
        if self._studio_preset_cb is not None:
            self._studio_preset_cb(True)
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_gui.py -v`
Expected: 4 × PASS

Regression: `pytest tests/test_gui_main_window.py tests/test_ki_panel.py tests/test_studio_panel.py -v`
Expected: alle PASS

- [ ] **Step 5: Commit**

```bash
git add src/gui/studio_panel.py src/gui/ki_panel.py tests/test_studio_gui.py
git commit -m "feat(studio): P5 GUI — Modus-/Quality-Badges, Solver-Trace, Preset-Button"
```

---

### Task 4: Perf-Budgets

**Files:**
- Test: `tests/test_studio_perf.py`

**Interfaces:**
- Consumes: Metriken (P0), `check_feasibility` (P2), `solve` (P3).
- Produces: nichts — Verifikations-Task (Spec §13). Budgets als Assertions mit CI-Faktor ×4.

- [ ] **Step 1: Tests schreiben**

`tests/test_studio_perf.py`:

```python
"""Perf-Budgets als Akzeptanzkriterien (Spec §13).

Überschreitung = P2-Defekt, nicht „halt langsam". CI-Faktor ×4 gegen
Schwankungen auf geteilten Runnern.
"""

import time

import numpy as np
import pytest

CI_FACTOR = 4.0


def test_metrik_pro_sample_budget():
    """≤ 15 ms CPU pro Sample @854 px (Spec §13)."""
    from src.studio.metrics import (contribution, overlay_energy,
                                    subject_disturbance, to_measure_raster)
    rng = np.random.default_rng(1)
    a = (rng.random((480, 854, 3)) * 255).astype(np.uint8)
    b = (rng.random((480, 854, 3)) * 255).astype(np.uint8)
    mask = (rng.random((427, 854)) > 0.5).astype(np.float32)

    start = time.perf_counter()
    ra, rb = to_measure_raster(a), to_measure_raster(b)
    c = contribution(ra, rb)
    overlay_energy(c)
    subject_disturbance(c, mask)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms <= 15 * CI_FACTOR


def test_feasibility_budget():
    """≤ 200 ms rein analytisch (Spec §7/§13)."""
    from src.studio.feasibility import check_feasibility
    rng = np.random.default_rng(2)
    mask = (rng.random((1080, 1920)) > 0.5).astype(np.float32)

    start = time.perf_counter()
    for _ in range(10):
        check_feasibility(mask, requires_text_zone=True)
    elapsed_ms = (time.perf_counter() - start) * 1000 / 10
    assert elapsed_ms <= 200 * CI_FACTOR


def test_solver_synthetisch_budget():
    """Vollständiger Solve (synthetisch) deutlich unter 20 s (Spec §13)."""
    from src.studio.solver import solve
    from src.studio.thresholds import load_thresholds
    ts = load_thresholds()
    fn = lambda p: {"M1": 0.8 * p.get("alpha_cap", 0.0),
                    "M5": 0.1 * p.get("intensity", 0.0)
                          + 0.5 * p.get("chroma_modulation", 0.0)}

    start = time.perf_counter()
    result = solve(fn, {"alpha_cap": 1.0}, ts)
    elapsed_s = time.perf_counter() - start
    assert result.status == "solved"
    assert elapsed_s <= 20 * CI_FACTOR  # synthetisch: ms-Bereich
```

- [ ] **Step 2: Erfolg verifizieren**

Run: `pytest tests/test_studio_perf.py -v`
Expected: 3 × PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_studio_perf.py
git commit -m "test(studio): P5 Perf-Budgets als Akzeptanzkriterien (Spec §13)"
```

---

## Abschluss P5 (Definition of Done, Spec §16)

- [ ] `pytest tests/test_studio_cli.py tests/test_studio_video_background.py tests/test_studio_gui.py tests/test_studio_perf.py -v` — alle PASS
- [ ] Video-Hintergrund unter `--studio`: Lauf erfolgreich, M3 deaktiviert, Warnung + `mask.provider = "none:video_background"` im Sidecar — **explizit kein Abbruch**
- [ ] Perf-Budget Preview/CPU eingehalten (Assertions grün)
- [ ] `pytest tests/ -q` — keine Regressionen im Bestand
- [ ] **Studio-Pipeline vollständig (P0–P5)**
