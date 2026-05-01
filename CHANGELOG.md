# Changelog — Audio Visualizer Pro

Alle nennenswerten Änderungen an diesem Projekt werden in dieser Datei dokumentiert.

Das Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.0.0/),
und dieses Projekt folgt [Semantic Versioning](https://semver.org/lang/de/).

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
