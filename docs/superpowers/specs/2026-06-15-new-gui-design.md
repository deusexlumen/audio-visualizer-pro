# Design-Dokument: Neue Audio Visualizer Pro GUI

**Datum:** 2026-06-15  
**Status:** Abgenickt, wartet auf Spec-Review  
**Autor:** Kimi Code Agent  
**Ziel:** Unhandliche, monolithische `gui.py` durch eine übersichtliche PyQt6-basierte GUI im Video-Editor-Stil ersetzen.

---

## 1. Zusammenfassung

Die aktuelle GUI (`gui.py`, ~3400 Zeilen, DearPyGui) ist unhandlich, schwer wartbar und optisch limitiert. Dieses Design ersetzt sie durch eine modulare PyQt6-Anwendung mit klarem Panel-Layout, dunklem Studio-Theme und echtzeitnaher Preview.

### Kernanforderungen (vom User abgenickt)
- **Framework:** PyQt6 (QWidget-basiert, pragmatisch, gute Live-Preview-Unterstützung).
- **Layout:** Panel-Layout wie ein Video-Editor.
- **Preview:** Echtzeitnahes Update mit Debounce und abbrechenden Worker-Threads.
- **Stil:** Dunkles Studio-Theme.
- **Timeline:** Ja, mit Playhead-Scrubbing.

---

## 2. Architektur

### 2.1 Framework
- **PyQt6** als GUI-Framework.
- **QThread** für alle blockierenden Operationen (Audio-Analyse, Preview-Rendering, Video-Export).
- **QPixmap/QLabel** für die Preview-Anzeige.
- **QSS** für das Styling.

### 2.2 Dateistruktur

```
src/gui/
├── __init__.py          # Package-Init
├── app.py               # QApplication Entry-Point
├── main_window.py       # Hauptfenster + Layout + Menü
├── state.py             # Zentraler AppState mit pyqtSignal
├── styles.py            # Farben, QSS, Theme-Helper
├── preview_widget.py    # Preview-Anzeige
├── timeline_widget.py   # Timeline-Slider + Zeit-Anzeige
├── assets_panel.py      # Audio/Background/Quotes laden
├── params_panel.py      # Visualizer-Parameter + Post-Process
├── render_dialog.py     # Export-Dialog + Fortschritt
└── workers.py           # QThread-Worker für Analyse/Preview/Render
```

### 2.3 Bestehende Logik bleibt unverändert
- `src/analyzer.py` (`AudioAnalyzer`)
- `src/gpu_preview.py` (`render_gpu_preview`)
- `src/gpu_renderer.py` (`GPUBatchRenderer`)
- `src/quote_overlay.py` (`QuoteOverlayConfig`, `QuoteOverlayRenderer`)
- `src/types.py` (`Quote`, `AudioFeatures`, etc.)

---

## 3. Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Menü (Projekt | Bearbeiten | Hilfe)                        │
├──────────────┬──────────────────────────────┬───────────────┤
│              │                              │               │
│  ASSETS      │      PREVIEW (854x480)       │  VISUALIZER   │
│  - Audio     │      + Timeline darunter     │  - Visualizer │
│  - Hinter-   │                              │  - Parameter  │
│    grund     │                              │               │
│  - Quotes    │                              │               │
│              │                              │               │
├──────────────┴──────────────────────────────┴───────────────┤
│  STATUS-LEISTE  |  Render-Button (rechts unten)              │
└─────────────────────────────────────────────────────────────┘
```

- **Linke Spalte (Assets):** ~250px
  - Audio laden, Info (Dauer, BPM, Modus).
  - Hintergrundbild/Video laden, Blur/Vignette/Opacity.
  - Quotes laden/bearbeiten (optional erst später).
- **Mitte (Preview + Timeline):** flexibel
  - Preview-Widget mit schwarzem Hintergrund.
  - Timeline mit Playhead und Zeit-Sprung-Buttons.
- **Rechte Spalte (Parameter):** ~320px
  - Visualizer-Auswahl.
  - Parameter-Controls (Slider, Dropdowns) dynamisch je nach Visualizer.
  - Post-Process (Contrast, Saturation, Brightness, Warmth, Grain).
- **Unten:** Status-Leiste + großer Render-Button.

---

## 4. State Management

### 4.1 `AppState` (zentral)
```python
class AppState(QObject):
    changed = pyqtSignal(str)  # key der geänderten Eigenschaft

    audio_path: str | None
    features: AudioFeatures | None
    audio_duration: float

    visualizer_type: str
    viz_params: dict
    viz_offset_x/y: float
    viz_scale: float

    background_path: str | None
    bg_blur: float
    bg_vignette: float
    bg_opacity: float

    postprocess: dict

    quotes: list[Quote]
    quotes_enabled: bool
    quote_config: QuoteOverlayConfig

    preview_time_percent: float
    preview_fps: int
    preview_width: int = 854
    preview_height: int = 480

    resolution: tuple[int, int]
    render_fps: int
    codec: str
    quality: str
    gpu_encode: bool
    output_dir: str
```

### 4.2 Kommunikation
- Widgets schreiben in `AppState`.
- `AppState.changed` signalisiert Änderungen.
- `MainWindow` leitet Preview-Updates an den Worker weiter.

---

## 5. Preview-System

### 5.1 Ziel
Echtzeitnahes Preview-Update ohne GUI-Ruckeln.

### 5.2 Ablauf
1. Parameter-Änderung (Slider, Dropdown, etc.).
2. `QTimer` mit 50ms Debounce startet.
3. Timer ausgelöst:
   - Laufender Preview-Worker wird abgebrochen (`requestInterruption()`).
   - Neuer `PreviewWorker` startet.
4. Worker ruft `render_gpu_preview(...)` auf.
5. Worker finished → `preview_ready`-Signal sendet `QPixmap`.
6. `PreviewWidget` aktualisiert das Bild.

### 5.3 Worker
```python
class PreviewWorker(QThread):
    preview_ready = pyqtSignal(object)
    preview_error = pyqtSignal(str)

    def run(self):
        try:
            img = render_gpu_preview(...)
            if img is not None:
                self.preview_ready.emit(img)
        except Exception as e:
            self.preview_error.emit(str(e))
```

---

## 6. Timeline

- `QSlider` mit Custom-Style (Playhead).
- Anzeige: `aktuell / gesamt`.
- Buttons für 0%, 25%, 50%, 75%, 100%.
- Scrubbing aktualisiert `preview_time_percent` und triggert Preview-Update.

---

## 7. Styling — Dunkles Studio-Theme

### 7.1 Farbpalette
| Element | Farbe |
|---|---|
| Hintergrund | `#0a0a0f` |
| Panels | `#12131a` |
| Eingaben | `#1a1c24` |
| Border | `#2a2d3a` |
| Text Primary | `#e8e9ec` |
| Text Secondary | `#8b8f99` |
| Akzent | `#60b0ff` |
| Erfolg | `#50c878` |
| Fehler | `#ff5f5f` |

### 7.2 Widgets
- Abgerundete Buttons mit Hover-Effekt.
- `QGroupBox` mit dunklem Header für Gruppierung.
- Slider mit Akzent-Farbe.
- System-Font oder Inter.

---

## 8. Fehlerbehandlung

- Worker fangen Exceptions und emitieren `error_signal`.
- `MainWindow` zeigt `QMessageBox.critical()` oder `warning()` an.
- Lange Operationen (Render) zeigen `QProgressDialog`.
- Netzwerk-/KI-Fehler werden nicht als Crash, sondern als Dialog angezeigt.

---

## 9. Testing

- `tests/test_gui_state.py` — Serialisierung, Signals, Defaults.
- `tests/test_gui_workers.py` — Worker mit gemockten Daten.
- `tests/test_gui_smoke.py` — GUI startet und schließt sich sauber.

---

## 10. Migrationsplan

1. Neue `src/gui/` implementieren.
2. Alte `gui.py` umbenennen zu `gui_legacy.py`.
3. Neue GUI testen.
4. `gui.py` durch Thin-Wrapper ersetzen, der neue GUI startet.
5. `streamlit run gui.py` oder `python -m src.gui` als Entry-Point dokumentieren.

---

## 11. Offene Punkte / Nächste Schritte

- Implementation Plan erstellen (`writing-plans` Skill).
- Abhängigkeit `PyQt6` zu `requirements.txt` hinzufügen.
- Schrittweise Dateien implementieren, beginnend mit `state.py` und `main_window.py`.

---

## 12. Entscheidungen

| Frage | Entscheidung |
|---|---|
| Framework | PyQt6 |
| Layout | Panel-Layout wie Video-Editor |
| Preview | Echtzeitnahes Update via QThread + Debounce |
| Stil | Dunkles Studio-Theme |
| Timeline | Ja, mit Playhead-Scrubbing |
| State-Management | Zentraler `AppState` mit `pyqtSignal` |
| Migration | Alte GUI kurzzeitig als `gui_legacy.py` behalten |
