# Design: KI-Features in die neue PyQt6-GUI übernehmen

## Ziel

Die neue PyQt6-GUI (`gui.py` / `src/gui/`) soll die KI-gestützten Funktionen der alten DearPyGui-Oberfläche (`gui_legacy.py`) erhalten:

1. **KI-Parameter-Optimierung** via Gemini (`optimize_all_settings_async`)
2. **Auto-Visualizer-/Farbempfehlung** via `SmartMatcher`
3. **Key-Zitat-Extraktion** und -Verwaltung via Gemini (`extract_quotes_async`)

## Ausgangslage

- Neue GUI hat aktuell drei Bereiche:
  - Links: `AssetsPanel` (Audio, Hintergrund)
  - Mitte: Preview + Timeline
  - Rechts: `ParamsPanel` (Visualizer, Transform, Post-Process)
- Alte GUI hatte Tabs: Audio, Visualizer, Hintergrund, Post-Process, **KI**, **Zitate**, Export.
- `GeminiIntegration` und `SmartMatcher` existieren bereits.
- `AppState` hat bereits `quotes`, `quotes_enabled`, `quote_config`.

## Architektur-Entscheidung

Das **rechte Panel wird zu einem `QTabWidget`** mit drei Tabs:

1. **Params** – bestehende Visualizer-/Transform-/Post-Process-Steuerung
2. **KI** – SmartMatcher + Gemini-Parameter-Optimierung
3. **Quotes** – Zitat-Extraktion, Liste, Erscheinungsbild

Begründung: Das 3-Spalten-Layout bleibt erhalten, die rechte Spalte wird nur tabbasiert aufgeteilt. Das ist skalierbar und übersichtlicher als eine endlose Scroll-Area.

## Datenmodell

### AppState-Erweiterungen

```python
# KI
self.ki_prompt: str = ""
self.ki_suggested_colors: dict = {}
self.ki_status: str = ""
self.ki_error: bool = False
self.ki_optimizing: bool = False
self.quotes_extracting: bool = False
```

Diese Keys werden in `_STATE_KEYS` aufgenommen, damit `changed`-Signals funktionieren.

### Serialisierung

- `ki_prompt` und `ki_suggested_colors` werden in `to_dict()` / `from_dict()` persistiert.
- Quotes werden bereits serialisiert.

## UI-Komponenten

### 1. KIPanel (`src/gui/ki_panel.py`)

#### SmartMatcher-Sektion

- Button **„Auto-Visualizer empfehlen“**
  - Voraussetzung: `features` ist vorhanden
  - Ruft `SmartMatcher().match(features)`
  - Zeigt: empfohlener Visualizer, Confidence (0-1), Begründungstext
  - Button **„Übernehmen“** wendet Empfehlung an:
    - `state.visualizer_type`
    - `state.viz_params` (aus `recommendation.params`)
    - `state.base_hue`, `state.color_saturation`, `state.color_mode` (falls in params)
    - `state.ki_suggested_colors = recommendation.colors`
    - `state.color_mode` bleibt unverändert; `base_hue` und `color_saturation` werden nur gesetzt, wenn sie explizit in `recommendation.params` enthalten sind.

#### Gemini-Optimierung-Sektion

- `QLineEdit` für optionalen Prompt (z. B. „dunkler, mehr Kontrast, cyberpunk“)
- Button **„Parameter optimieren“**
  - Deaktiviert, wenn `features` fehlt oder API-Key nicht gesetzt
  - Ruft asynchron `GeminiIntegration.optimize_all_settings_async()`
  - Während des Laufens: Button zeigt „⏳ KI denkt nach…“
- Status-Label für Erfolg/Fehler
- Farbanzeige für vorgeschlagene Farben (Primary/Secondary/Background)

#### Anwendung der Gemini-Ergebnisse

`AIOptimizeWorker` liefert ein Dict mit:

```python
{
  "params": {...},        # Visualizer-Parameter
  "postprocess": {...},   # Kontrast, Sättigung, etc.
  "background": {...},    # Blur, Vignette, Opacity
  "colors": {...},        # primary, secondary, background
}
```

Das Panel mapped diese Werte auf `AppState`:

- `params` → `viz_params`, `viz_offset_x`, `viz_offset_y`, `viz_scale`, `base_hue`, `color_saturation`
- `postprocess` → `pp_contrast`, `pp_saturation`, `pp_brightness`, `pp_warmth`, `pp_grain`
- `background` → `bg_blur`, `bg_vignette`, `bg_opacity`
- `colors` → `ki_suggested_colors`

### 2. QuotesPanel (`src/gui/quotes_panel.py`)

#### Extraktion

- Checkbox **„Zitate aktivieren“**
- Button **„Key-Zitate extrahieren“**
  - Ruft asynchron `GeminiIntegration.extract_quotes_async(audio_path, duration)`
  - Füllt `state.quotes`
- Button **„Demo-Zitate“** (optional, für schnelles Testen ohne API)
  - Fügt 1-2 Beispielzitate ein
- Status-Label

#### Zitat-Liste

- `QListWidget` zeigt alle Quotes mit `text (start-end)`
- Buttons: **➕ Hinzufügen**, **🗑 Entfernen**, **✏️ Bearbeiten**
- Bearbeitung via einfachem Dialog (Text, Start, Ende, Confidence)

#### Erscheinungsbild

- Position: `QComboBox` [bottom, center, top]
- Schriftgröße: Slider 16-96
- Textfarbe: Color picker
- Box-Farbe: Color picker mit Alpha
- Fade-Dauer: Slider 0.1-2.0s
- Anzeigedauer: Slider 2.0-20.0s
- Max. Zeichen/Zeile: Slider 20-80
- Zeilenabstand: Slider 0-30px
- Box-Padding, Box-Radius, Abstand unten, Max. Breite, Skalierung

## Worker-Klassen

Erweiterung von `src/gui/workers.py`:

### AIOptimizeWorker

```python
class AIOptimizeWorker(QThread):
    optimize_ready = pyqtSignal(dict)
    optimize_error = pyqtSignal(str)

    def __init__(self, gemini: GeminiIntegration, visualizer_type: str,
                 current_params: dict, audio_features: dict, colors: dict,
                 param_specs: dict, user_prompt: str | None = None):
        ...

    def run(self):
        try:
            future = self.gemini.optimize_all_settings_async(...)
            result = future.result(timeout=60)
            self.optimize_ready.emit(result)
        except Exception as e:
            self.optimize_error.emit(str(e))
```

### QuoteExtractWorker

```python
class QuoteExtractWorker(QThread):
    quotes_ready = pyqtSignal(list)
    quotes_error = pyqtSignal(str)

    def __init__(self, gemini: GeminiIntegration, audio_path: str,
                 audio_duration: float | None = None, max_quotes: int | None = None):
        ...

    def run(self):
        try:
            future = self.gemini.extract_quotes_async(...)
            quotes = future.result(timeout=120)
            self.quotes_ready.emit(quotes)
        except Exception as e:
            self.quotes_error.emit(str(e))
```

## Integration in MainWindow

- Rechtes Panel (`self.params_panel`) wird zu `QTabWidget`.
- Tabs:
  - `Params` enthält die aktuelle `ParamsPanel`-Instanz.
  - `KI` enthält neue `KIPanel`-Instanz.
  - `Quotes` enthält neue `QuotesPanel`-Instanz.
- `MainWindow` hält Referenzen auf `self._ai_optimize_worker` und `self._quote_extract_worker`, um laufende Threads zu verwalten und zu canceln.
- `MainWindow` instantiiert `GeminiIntegration` einmalig (wie in `gui_legacy.py`) und übergibt sie an `KIPanel` und `QuotesPanel`. Falls der API-Key fehlt, wird `None` übergeben; die Panels deaktivieren dann KI-Buttons.
- `KIPanel` hört auf `state.changed` Signal für `features`, um Buttons zu aktivieren/deaktivieren.

## Fehlerbehandlung

- Kein API-Key: Button deaktiviert oder Fehlermeldung „KI nicht verfügbar. Prüfe API-Key."
- Kein Audio / keine Features: Buttons deaktiviert mit Tooltip
- API-Fehler (Timeout, Rate-Limit): Status-Label zeigt Fehler, Worker cancelt laufende Futures
- Ungültiges KI-Ergebnis: Nur gültige Felder werden angewendet, Rest ignoriert

## Tests

- `test_ki_panel.py`: GUI-Unit-Tests für Button-States, Signal-Verknüpfungen
- `test_quotes_panel.py`: Hinzufügen/Entfernen/Bearbeiten von Quotes
- `test_workers.py`: AIOptimizeWorker und QuoteExtractWorker mit gemocktem Gemini
- `test_app_state.py`: Serialisierung der neuen KI-Felder

## Offene Punkte / Annahmen

1. **Gemini API-Key**: Wie bisher über `GEMINI_API_KEY` Umgebungsvariable oder `.env`.
2. **Farb-Parameter**: `base_hue` und `color_saturation` existieren bereits im `AppState`. Wenn Gemini andere Farbschlüssel liefert, werden sie in `ki_suggested_colors` gespeichert, aber nicht automatisch auf den Visualizer angewendet, wenn der Visualizer sie nicht unterstützt.
3. **Demo-Zitate**: Optional, kann bei Bedarf weggelassen werden, um Scope zu reduzieren.
4. **Editieren von Quotes**: Einfacher Dialog; keine Inline-Bearbeitung in der Liste.

## Dateien, die geändert werden

- `src/gui/main_window.py` – rechtes Panel zu Tabs umbauen
- `src/gui/state.py` – KI-Felder erweitern
- `src/gui/workers.py` – neue Worker
- `src/gui/ki_panel.py` – neu
- `src/gui/quotes_panel.py` – neu
- `src/gui/params_panel.py` – ggf. Anpassung, falls nötig
- `tests/test_gui/` oder `tests/test_*.py` – neue Tests
