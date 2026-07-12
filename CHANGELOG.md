# Changelog — Audio Visualizer Pro

Alle nennenswerten Änderungen an diesem Projekt werden in dieser Datei dokumentiert.

Das Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.0.0/),
und dieses Projekt folgt [Semantic Versioning](https://semver.org/lang/de/).

## [2.8.0] — 2026-07-12

Visualizer-Qualität (Phase 4 des Ausbauplans v3.0): schwache Visualizer poliert,
zwei neue Premium-Visualizer, mehr Audio-Features für Shader.

### Added
- **Zwei neue Signature-Visualizer**:
  - `aurora_voice` (Podcast): ruhige, wogende Aurora-Bänder, sprachband-getrieben,
    bewusst ohne Beat-Blitzen — für stundenlange Sprach-Inhalte
  - `nebula_drift` (Musik): treibende fbm-Nebel + Partikelfeld mit Chroma-Hue-Drift
    und beat-getriebenem Pulsieren (Ambient bis EDM)
- **Neue Feature-Uniforms** für alle Visualizer: `spectral_rolloff`,
  `zero_crossing_rate` und `mfcc0` sind jetzt in `_get_feature_at_frame`
  erreichbar; `_map_features_to_uniforms` liefert zusätzlich `u_texture`
  (Rauheit) und `u_warmth` (hell↔dunkel)

### Changed
- **`pulsing_core` runderneuert**: mehrschichtiger HDR-Kern, fbm-Korona,
  beat-getriggerte Schockwellen, Orbit-Partikel
- **`typographic` runderneuert**: kinetisches SDF-Type-Grid mit beat-quantisierter
  Bewegung statt einfacher VU-Balken
- **`spectrum_genesis` aufgewertet**: Peak-Hold-Caps, Reflexions-Tiefe, Chroma-Sweep
- **Schema `visual.type`**: von fester Literal-Liste auf registry-validierten String
  (neue Visualizer und Studio-Rezepte ohne Schema-Änderung nutzbar)
- `SmartMatcher` kennt und bewertet die neuen Visualizer (Podcast/Musik/Hybrid)

## [2.7.0] — 2026-07-12

KI-Härtung (Phase 3 des Ausbauplans v3.0): verlässliche, kostentransparente
Gemini-Integration.

### Added
- **Konfigurierbare Modell-ID** (`config/settings.json`, `src/app_settings.py`):
  Auflösung `GEMINI_MODEL`-env → settings.json → Default. Ungültige/veraltete
  IDs werden beim ersten Aufruf erkannt und per `models.list()` durch ein
  passendes Modell ersetzt (Präferenzliste) — die App crasht nie mehr an einer
  toten Modell-ID
- **Kosten-Tracking** (`src/ai_costs.py`): Session-Ledger über Token-Verbrauch
  und geschätzte USD-Kosten (Preistabelle in settings.json), Anzeige im KI-Panel
- **Result-Caching** für Zitate: identische Anfragen kosten keinen weiteren
  API-Aufruf; „Cache ignorieren"-Checkbox erzwingt Neuanfrage
- **Transkription in der GUI**: „Transkribieren"-Button + Viewer mit Kopieren/
  Speichern im Zitate-Tab (nutzt den bestehenden Transkript-Cache)

### Changed
- **Typisierte Retry-Logik**: Statt String-Matching wird der HTTP-Status des
  google-genai-`APIError` ausgewertet (Retry bei 408/429/5xx, `Retry-After`
  wird beachtet); Auth-/Berechtigungsfehler (401/403) brechen sofort mit klarer
  Meldung ab

## [2.6.2] — 2026-07-12

CI, Test-Lücken und reproduzierbare Installation (Phase 2 des Ausbauplans v3.0).

### Added
- **GitHub-Actions-CI** (`.github/workflows/ci.yml`): Windows + Ubuntu,
  `pytest -m "not gpu"` mit gepinnten Abhängigkeiten
- **`requirements.lock`**: exakt gepinnte Abhängigkeiten (`pip freeze`) für
  reproduzierbare Installationen und den späteren PyInstaller-Build
- **Visueller Smoke-Test**: jeder registrierte Visualizer muss ein sichtbares
  Bild ohne NaN/Inf liefern — Regressionsnetz für Shader-Umbauten
- **Renderer-Fehlerpfad-Tests** (`tests/test_gpu_renderer_failures.py`):
  FFmpeg-Tod mitten im Render, Encode-Thread-Schreibfehler
- Tests für Batch-CLI-Fehlerpfade und Render-UI-Reaktivierung nach Fehler/Erfolg

### Fixed
- **GPU-Probe zerstörte Context-Currency**: Die OpenGL-Verfügbarkeitsprüfung
  läuft jetzt einmalig vor allen Tests (`pytest_sessionstart`), statt mitten in
  der Session den aktiven Test-Kontext zu invalidieren

## [2.6.1] — 2026-07-12

Fundament-Release (Phase 1 des Ausbauplans v3.0): Robustheit und Repo-Hygiene.

### Added
- **Globaler Exception-Handler** (`src/gui/app.py`): Unbehandelte Fehler in
  Qt-Slots und Threads erscheinen jetzt als Fehlerdialog mit Traceback-Details
  und "Log-Ordner öffnen"-Button statt die App still zu beenden
- `.env.example` als Vorlage für den Gemini-API-Schlüssel

### Changed
- **Worker-Fehlersignale** liefern jetzt `(Meldung, Traceback)` — vollständige
  Tracebacks landen in `logs/app.log`, die GUI zeigt weiterhin die kurze Meldung
- README/pyproject: GitHub-URLs auf das echte Repository korrigiert,
  Testzahl aktualisiert (217)

### Removed
- Totes Modul `src/gpu_quote_renderer.py` (357 Zeilen, keine Importer)
- Nutzer-Upload aus dem Git-Index entfernt; `assets/user_uploads/` ignoriert

## [2.6.0] — 2026-07-04

Grosses Qualitaets-Release in 7 Phasen: Stabilität, HDR-Rendering, Premium-Post-FX,
GUI-Politur, Workflow-Komfort, Performance und Code-Bereinigung.

### Added
- **HDR-Rendering-Pipeline**: Szene rendert in RGBA16F (Float16), finaler Pass mit
  Exposure → sättigungserhaltendem ACES-Tonemapping → Triangular-Dithering.
  Behebt Farb-Banding, matschige Glows und hartes Highlight-Clipping global.
- **Echter HDR-Bloom** (`src/gpu_bloom.py`): Soft-Knee-Threshold, progressive
  Downsample-Kette, Tent-Upsample — helle Bereiche leuchten weich aus
- **GPU-LUT-Color-Grading**: `.cube`-Dateien werden als 3D-Textur geladen und
  im Finalpass angewendet (`lut`, `lut_strength`)
- **Neue Post-FX**: echte radiale chromatische Aberration, Vignette auf dem
  Gesamtbild, luminanzabhängiges animiertes Film-Grain, `exposure`-Parameter
- **4x-MSAA** für Visualizer-Geometrie (mit transparentem Fallback)
- **Anti-Aliasing-Infrastruktur**: fwidth-basierte `aastep`/`aafill`-Helfer,
  gemeinsame Shader-Bausteine und Quad-Helper in `base.py`
- **Projekt-Dateien (`.avproj`)**: Speichern/Laden aller Einstellungen,
  Zuletzt-verwendet-Liste, Stern-Marker bei ungespeicherten Änderungen
- **Menüleiste + Shortcuts**: Strg+O/S, F5 Rendern, Esc Abbrechen u.a.
- **Drag & Drop**: Audio, Hintergrund-Medien und Projekte aufs Fenster ziehen
- **Wellenform-Timeline** mit Beat-Markern, Playhead und Klick-Seek
- **Fortschrittsbalken** beim Rendern, Busy-Overlay auf der Vorschau,
  separater Abbrechen-Button, Erfolgs-Dialog mit „Ordner öffnen"
- **Icons & Schrift**: 13 SVG-Icons, App-Icon, gebündelte Inter-Schrift (OFL)
- **Zentrales Logging** (`logs/app.log`), verständliche deutsche Fehlermeldungen
  in CLI und GUI (kein Python-Traceback mehr für Endnutzer)
- **QSettings-Persistenz**: Fenstergeometrie und Splitter-Layout bleiben erhalten

### Changed
- **Performance**: PBO-Doppelpufferung beim Framebuffer-Readback (+33 % Durchsatz
  im Render-Loop), Quote-Overlay wird pro Zitat gecached statt pro Frame neu
  gerendert, `particle_swarm`/`spectrum_bars` komplett vektorisiert,
  Hintergrund-Video-Decode in eigenem Prefetch-Thread
- **GUI durchgängig Deutsch** und vollständig gestylt (Tabs, Scrollbars,
  Checkboxen, Menüs, Dialoge, Hover-/Fokus-/Disabled-Zustände)
- **Vorschau = Endergebnis**: Preview nutzt exakt dieselbe Render-Pipeline
- Verworfene Vorschauen brechen jetzt wirklich ab (kooperativer Abbruch)
- Duplizierter Code konsolidiert: `render_common.py` (Feature-Dict,
  Beat-Intensität), `hex_to_rgb` zentral, GLSL-Helper injiziert statt kopiert

### Fixed
- Rendern mit `--config` brach mit AttributeError ab (`intro_fade`)
- Analyzer-Cache ignorierte `ema_alpha` (lieferte veraltete Features)
- FFmpeg-Hänger nach Encode-Ende (Timeout + Kill-Fallback)
- Doppelt gefeuerte State-Signale im Parameter-Panel
- Stille Fehler sichtbar gemacht: Visualizer-Import-Fehler werden geloggt,
  Preview-Fehler als Dialog, Gemini-Init-Fehler mit Grund im KI-Panel
- `pydantic>=2.0` korrekt gepinnt (vorher `>=1.10` trotz v2-API)

### Removed
- Legacy-GUIs (`gui_legacy.py` DearPyGui, `gui_streamlit_legacy.py` Streamlit)
  samt `streamlit`/`dearpygui`-Abhängigkeiten
- Toter Code: `src/postprocess.py` (durch GPU-Pipeline ersetzt),
  `src/local_transcription.py` (verwaist), Debug-Artefakte aus dem Repo

---

## [2.1.0] — 2026-05-01

### Added
- **Test-Suite massiv erweitert**: 60 → **134 Tests** (+74 neue Tests)
- **GPU-Renderer Mock-Infrastruktur**: Hardware-unabhängige Tests für ModernGL-Context
- **Neue Test-Dateien**:
  - `tests/test_postprocess.py` — 22 Tests für Bloom, Grain, Vignette, LUT, Chromatic Aberration
  - `tests/test_gpu_renderer.py` — 11 Tests für FFmpeg-Cmd-Builder und Render-Flow
  - `tests/test_gpu_renderer_extended.py` — 7 Tests für `_mux_audio`, `_save_debug`, `_load_background_texture`
  - `tests/test_gpu_preview.py` — 9 Tests für Preview-Cache und `render_gpu_preview`
  - `tests/test_gpu_text_renderer.py` — 15 Tests für SDF-Font-Atlas und GPUTextRenderer
  - `tests/conftest.py` — Shared Fixtures mit Pydantic-konformen Dummy-Features
- **Evo-Agent Framework** etabliert:
  - `cognitive_core/agents.md` — State Ledger
  - `cognitive_core/system_prompt.md` — Root Orchestrator
  - `cognitive_core/tool.md` — Skill Dispatcher
  - `skills/skill_*.md` — 5 Skill-Spezifikationen
  - `memory/temp.md` — Working Memory
- **Coverage-Config** in `pyproject.toml` mit `omit` für nicht-testbare Bereiche

### Changed
- `tests/test_quote_overlay.py` erweitert: 15 → **25 Tests** (+10 neue Tests)
- `AGENTS.md` aktualisiert: PIL-Pipeline Referenzen entfernt, GPU-Renderer dokumentiert
- `README.md` aktualisiert: Test-Badge 42 → 134 Passing, Projektstruktur korrigiert

### Fixed
- `src/postprocess.py` — Fehlender `from pathlib import Path` führte zu NameError in `process_video()`
- `src/postprocess.py` — LUT-Parser crashte bei Header-Zeilen wie `TITLE` (ValueError in float())
- `src/pipeline.py` — DeprecationWarning + graceful ImportError-Handling vor Entfernung

### Removed
- **`src/pipeline.py`** — Verwaiste PIL-basierte Pipeline (broken, ImportError bei `PILRenderer`)
- **`src/renderers/`** — Verwaistes Verzeichnis (existierte nicht mehr)

### Coverage-Übersicht

| Modul | Vorher | Nachher |
|---|---|---|
| `postprocess.py` | 0% | **100%** |
| `gpu_preview.py` | 0% | **95%** |
| `gpu_text_renderer.py` | 12% | **78%** |
| `quote_overlay.py` | 77% | **93%** |
| `types.py` | 100% | **100%** |
| **Gesamt** | **63%** | **77%** |

---

## [2.0.0] — 2026-04-21

### Added
- GPU-beschleunigtes Rendering mit ModernGL (OpenGL)
- 16 GPU-Visualizer mit Shader-basiertem Rendering
- Live-Preview mit gecachtem Renderer
- GPU-Text-Rendering mit SDF (Signed Distance Field) Fonts
- Post-Processing: Bloom, Film Grain, Vignette, Chromatic Aberration, LUTs
- KI-Zitat-Extraktion mit Gemini
- Beat-Synchronisation für Quotes
- Multi-Codec Support: H.264, HEVC, ProRes
- Chroma-Subsampling-Fix: `yuv444p` für high/lossless Qualität
