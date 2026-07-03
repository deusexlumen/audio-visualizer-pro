# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Note**: This repo already has a detailed `AGENTS.md` (in German) with full architecture docs, the GPU-visualizer authoring template, the feature-key reference table, and per-visualizer parameter lists. Read it for depth — this file is a quick-orientation summary plus things `AGENTS.md` doesn't cover. Prefer `AGENTS.md` for GPU-visualizer/shader conventions and the feature-key table; treat it as authoritative over the README, which is stale in places (e.g. it still describes the GUI as DearPyGui-based).

## Project

Audio Visualizer Pro — a Python CLI + PyQt6 desktop app that turns audio into GPU-rendered music/podcast videos. Pipeline: librosa audio analysis → ModernGL/OpenGL shader visualizers → post-processing → FFmpeg encoding, with optional Gemini-based transcription/quote-extraction and AI parameter matching. It's a solo-developer, AI-assisted project (see README "Entwicklungshintergrund"); code comments, docs, and commit messages are in German.

## Commands

Install: `pip install -r requirements.txt` (FFmpeg must also be installed system-wide and on PATH).

```bash
# CLI
python main.py analyze song.mp3 --fps 30
python main.py list-visuals
python main.py render song.mp3 --visual lumina_core --preview      # 5s/480p preview
python main.py render song.mp3 --visual spectrum_bars -o output.mp4
python main.py render song.mp3 --config config/music_aggressive.json
python main.py create-visualizer mein_visualizer --type shader     # scaffolds new GPU visualizer
python main.py create-config --output meine_config.json
python main.py batch jobs.json

# GUI
python gui.py

# Tests
pytest tests/ -v
pytest tests/test_gpu_renderer.py -v          # single suite
pytest tests/test_visuals.py::TestClassName::test_name -v   # single test

# Validate a config against the Pydantic schema
python -c "from config.schemas import load_and_validate_config; load_and_validate_config('config/mein_preset.json')"

# Dev tooling (per README contributing section)
black src/ tests/
flake8 src/ tests/
```

## Architecture

Four-layer render pipeline, bottom to top:

1. **Audio analysis** (`src/analyzer.py`) — `AudioAnalyzer.analyze(audio_path, fps)` extracts RMS, onset, chroma, spectral centroid/rolloff, zero-crossing rate, transients, voice-clarity/band, beat intensity, tempo, mode (speech/music/hybrid). Results are cached to `.cache/audio_features/` as NPZ, keyed off the audio file — deterministic and thread-safe. **Do not modify `analyze()` itself**; extend by adding a new field to `AudioFeatures` (`src/types.py`), extracting it in `analyzer.py`, and threading it into the `features_dict` in `gpu_renderer.py`/`gpu_preview.py`. Changing the method signature/caching logic invalidates all existing caches.
2. **Visualization** (`src/gpu_visualizers/`) — each visualizer subclasses `BaseGPUVisualizer` (`base.py`) and renders one frame per call into the active OpenGL framebuffer via `render(features, time)`. Auto-discovered on import; `VISUALIZER_MAP` in `src/gpu_visualizers/__init__.py` is kept for backwards compatibility but isn't the source of truth for new visualizers. See `AGENTS.md` for the required class skeleton, the `PARAMS` dict convention (`(default, min, max, step)` tuples merged from `EFFECTS`/`COLOR_PARAMS`/`PARAMS`), and the feature-key reference table.
3. **Rendering** (`src/gpu_renderer.py`) — `GPUBatchRenderer` (full render) and `GPUPreviewRenderer`/`gpu_preview.py` (single-frame/live preview) drive the OpenGL context, apply the post-processing shader pass (contrast/saturation/brightness/warmth/grain/vignette/chromatic aberration), composite quote overlays, and shell out to FFmpeg for encoding + audio muxing.
4. **Quote overlays** (`src/quote_overlay.py`, `src/gpu_text_renderer.py`) — PIL-based `QuoteOverlayRenderer` is what's active in the render loop today; `gpu_text_renderer.py` provides SDF-based GPU text rendering; `gpu_quote_renderer.py` (GPU-based quote renderer) exists but is currently inactive.

**AI integration** (`src/gemini_integration.py`, `src/ai_matcher.py`): `GeminiIntegration` handles transcription and key-quote extraction (with timestamps) via the Gemini API; `local_transcription.py` is a faster-whisper fallback when no API key/network is available. `SmartMatcher` (`ai_matcher.py`) analyzes audio features and recommends a visualizer + color palette + parameters ("Auto-Modus"). Quotes surfaced by Gemini are reviewed/edited/filtered in the GUI before rendering.

**Config**: JSON presets in `config/` (music vs. podcast presets) are validated against Pydantic v2 schemas in `config/schemas.py`. All domain models (`AudioFeatures`, `VisualConfig`, `ProjectConfig`, `Quote`, etc.) live in `src/types.py`. Config JSON uses flat `background_*` fields (not nested), an open `visual.params` dict (each visualizer defines its own params), and colors as hex strings or RGBA lists.

**GUI** (`src/gui/`): PyQt6 desktop app; `gui.py` at the repo root is a thin entry-point wrapper. Key modules: `main_window.py` (main window/layout), `state.py` (app state), `params_panel.py` (parameter controls, largest file), `ki_panel.py` (AI/Gemini panel), `quotes_panel.py`, `assets_panel.py`, `preview_widget.py`/`timeline_widget.py`, `workers.py` (background threads for analysis/render), `styles.py`. `gui_legacy.py` and `gui_streamlit_legacy.py` at the repo root are retained old implementations (DearPyGui and Streamlit respectively) — not part of the active app.

**`cognitive_core/`**: docs (`agents.md`, `system_prompt.md`, `tool.md`) for an "Evo-Agent Framework" (state ledger / orchestrator / skill-dispatcher) referenced in the README roadmap — check these before assuming a plain script structure if working in this area.

## Conventions

- New GPU visualizers, the feature-key table (`rms`, `onset`, `chroma`, `beat_intensity`, etc.), and per-visualizer `PARAMS` are documented in detail in `AGENTS.md` — read that before adding or modifying a visualizer.
- Visualizer `render()` output must be `np.ndarray`, shape `(H, W, 3)`, `dtype=uint8`, values 0-255 (enforced by `tests/test_visuals.py` for every registered visualizer).
- Comments, docstrings, and commit messages are written in German, matching the existing codebase.
- Audio file inputs are validated by extension (`.mp3`, `.wav`, `.flac`, etc.); render outputs must be `.mp4`.
