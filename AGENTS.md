# Audio Visualizer Pro - Agent Guide

Dieses Dokument enthält alle relevanten Informationen für KI-Code-Agents, die an diesem Projekt arbeiten.

## Projekt-Übersicht

**Audio Visualizer Pro** ist ein modulares, KI-optimiertes Audio-Visualisierungs-System für professionelle Musikvideos, Podcast-Visuals und kreative Projekte.

### Kern-Features
- **16 integrierte GPU-Visualizer**: 10 Classic (Pulsing Core, Spectrum Bars, Chroma Field, Particle Swarm, Typographic, Neon Oscilloscope, Sacred Mandala, Liquid Blobs, Neon Wave Circle, Frequency Flower) + 6 Signature Pro (Lumina Core, Voice Flow, Spectrum Genesis, Speech Focus, Bass Temple, Orchestral Swell)
- **GPU-basiertes Rendering**: ModernGL/OpenGL Offscreen-Rendering mit FFmpeg-Encoding
- **Intelligente Audio-Analyse**: Beat-Erkennung, Key-Erkennung, Chroma-Features, Mode-Detection (speech/music/hybrid), Transienten, Voice-Clarity
- **KI-gestützter Auto-Modus**: Smart Matcher analysiert Audio und empfehlt automatisch Visualizer + Farbpalette + Parameter
- **PyQt6-Oberfläche**: Desktop-GUI mit Drag & Drop, Live-Analyse und One-Click-Render
- **Aggressives Caching**: Analysiere einmal, rendere millionenmal
- **Professionelle Codecs**: FFmpeg-basiert mit libx264/libx265/prores und AAC
- **Post-Processing**: Kontrast, Sättigung, Helligkeit, Wärme, Film Grain, Vignette, Chromatic Aberration

## Technologie-Stack

| Komponente | Bibliothek | Zweck |
|------------|------------|--------|
| Audio-Analyse | librosa>=0.10.0 | Feature-Extraktion (RMS, Onset, Chroma, etc.) |
| Bildverarbeitung | Pillow>=9.0.0 | Quote-Overlays, CPU-Fallbacks |
| Datenvalidierung | pydantic>=2.0.0 | Konfiguration-Models (Pydantic v2) |
| CLI | click>=8.0.0 | Kommandozeilen-Interface |
| Numerik | numpy>=1.21.0 | Array-Operationen |
| Testing | pytest>=7.0.0 | Test-Framework |
| GPU-Rendering | moderngl>=5.0 | OpenGL Offscreen-Rendering |
| GUI | PyQt6>=6.0 | Desktop-GUI |

> **Hinweis**: Die GUI basiert auf PyQt6. Die alten DearPyGui/Streamlit-Implementierungen wurden in v2.6 entfernt.
| Video-Encoding | FFmpeg (system) | H.264/H.265/ProRes Encoding |

**System-Voraussetzung**: FFmpeg muss system-seitig installiert sein.
- Ubuntu: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: https://ffmpeg.org/download.html

## Projektstruktur

```
audio_visualizer_pro/
├── main.py                 # CLI Entry Point (Click-basiert)
├── gui.py                  # PyQt6 Desktop-GUI (Thin-Wrapper)
├── requirements.txt        # Python-Abhängigkeiten
├── assets/                 # SVG-Icons + Inter-Schrift (OFL)
├── config/                 # Konfigurations-Presets und Validierung
│   ├── __init__.py
│   ├── schemas.py          # Pydantic v2 Schemas für Config-Validierung
│   ├── default.json        # Standard-Konfiguration
│   ├── music_aggressive.json
│   ├── podcast_minimal.json
│   └── ...
├── src/
│   ├── __init__.py
│   ├── analyzer.py         # AudioAnalyzer mit Caching
│   ├── ai_matcher.py       # SmartMatcher - KI-gestützte Visualizer-Empfehlung
│   ├── app_logging.py      # Zentrales Logging (Konsole + logs/app.log)
│   ├── gpu_renderer.py     # GPUBatchRenderer, GPUPreviewRenderer (HDR, PBO)
│   ├── gpu_bloom.py        # HDR-Bloom-Kette + .cube-LUT-Parser
│   ├── gpu_preview.py      # Einzel-Frame Preview-Renderer
│   ├── gpu_text_renderer.py # SDF Text-Rendering auf der GPU
│   ├── gpu_quote_renderer.py # GPU-basierter Quote-Renderer (aktuell nicht aktiv)
│   ├── render_common.py    # Gemeinsame Renderer-Hilfen (features_dict, Beat-Decay)
│   ├── types.py            # Pydantic Models (AudioFeatures, VisualConfig, etc.)
│   ├── quote_overlay.py    # QuoteOverlayRenderer mit Overlay-Cache (aktiv im Render-Loop)
│   ├── quote_refiner.py    # Zeitstempel-Verfeinerung für Quotes
│   ├── quote_cache.py      # Caching für Gemini Uploads/Transkripte
│   ├── gemini_integration.py # Gemini KI für Transkription und Zitat-Extraktion
│   ├── beat_sync.py        # Beat-Synchronisation
│   ├── intro_renderer.py   # Intro-Video vor Hauptvideo (FFmpeg)
│   ├── visualizer_wizard.py # Generator für eigene Visualizer
│   ├── gui/                # PyQt6-GUI (Panels, AppState, QThread-Worker)
│   └── gpu_visualizers/    # GPU-Visualizer Plugin-System
│       ├── __init__.py     # VISUALIZER_MAP Registry
│       ├── base.py         # BaseGPUVisualizer (abstrakte Basisklasse)
│       ├── pulsing_core.py
│       ├── spectrum_bars.py
│       ├── chroma_field.py
│       ├── particle_swarm.py
│       ├── typographic.py
│       ├── neon_oscilloscope.py
│       ├── sacred_mandala.py
│       ├── liquid_blobs.py
│       ├── neon_wave_circle.py
│       ├── frequency_flower.py
│       ├── lumina_core.py
│       ├── voice_flow.py
│       ├── spectrum_genesis.py
│       ├── speech_focus.py
│       ├── bass_temple.py
│       └── orchestral_swell.py
└── tests/
    ├── __init__.py
    ├── conftest.py         # Shared fixtures
    ├── test_analyzer.py    # Tests für AudioAnalyzer
    ├── test_ai_matcher.py  # Tests für SmartMatcher
    ├── test_quote_overlay.py # Tests für Quote Overlay Renderer
    ├── test_gpu_renderer.py # Tests für GPU-Renderer
    ├── test_gpu_preview.py # Tests für GPU-Preview
    ├── test_gemini_integration.py # Tests für Gemini
    └── ...
```

## Build- und Test-Kommandos

### Installation
```bash
pip install -r requirements.txt
```

### CLI-Befehle
```bash
# Audio analysieren
python main.py analyze song.mp3
python main.py analyze song.mp3 --fps 30

# Verfügbare Visualizer anzeigen
python main.py list-visuals

# 5-Sekunden Vorschau rendern
python main.py render song.mp3 --visual lumina_core --preview

# Volles Video rendern
python main.py render song.mp3 --visual spectrum_bars -o output.mp4

# Mit Config-Datei (wird in config/schemas.py validiert)
python main.py render song.mp3 --config config/music_aggressive.json

# Mit Quote-Config und Hintergrund
python main.py render song.mp3 --config config/podcast_interview.json

# GUI starten
python gui.py

# Neues GPU-Visualizer-Template erstellen (Boilerplate)
python main.py create-template mein_visualizer

# Neues GPU-Visualizer-Template mit reichhaltigem Startpunkt erstellen
python main.py create-visualizer mein_visualizer --type shader
python main.py create-visualizer mein_visualizer --type geometry
python main.py create-visualizer mein_visualizer --type particles

# Beispiel-Config erstellen
python main.py create-config --output meine_config.json

# Batch-Jobs ausführen
python main.py batch jobs.json
```

### Testing
```bash
# Alle Tests ausführen
pytest tests/ -v

# Spezifische Tests
pytest tests/test_visuals.py -v
pytest tests/test_analyzer.py -v
pytest tests/test_gpu_renderer.py -v
```

## Architektur

### Layer-Architektur

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Quote Overlays (Key-Zitat Text-Overlays)         │
│  → QuoteOverlayRenderer.apply(frame, time_seconds)         │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Post-Processing (Kontrast, Sättigung, Grain)     │
│  → GPU-PostProcess Shader in GPUBatchRenderer              │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Visualization (GPU Frame-Generierung)            │
│  → BaseGPUVisualizer.render(features, time)                │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Audio-Analyse (Feature-Extraktion)               │
│  → AudioAnalyzer.analyze(audio_path, fps)                  │
└─────────────────────────────────────────────────────────────┘
```

### Datenfluss

1. **Audio-Analyse** (`analyzer.py`):
   - Extrahiert Features: RMS, Onset, Chroma, Spectral Centroid, Transient, Voice-Clarity, etc.
   - Caching in `.cache/audio_features/` (NPZ-Format)
   - Deterministisch und thread-safe

2. **Visualization** (`gpu_visualizers/`):
   - Jeder Visualizer erbt von `BaseGPUVisualizer`
   - Registrierung via Eintrag in `VISUALIZER_MAP` in `src/gpu_visualizers/__init__.py`
   - `render(features, time)` rendert in den aktiven OpenGL-Framebuffer

3. **Rendering** (`gpu_renderer.py`):
   - `GPUBatchRenderer` steuert den kompletten GPU-beschleunigten Flow
   - FFmpeg-Subprozess für Video-Encoding
   - Quote Overlays werden zeitbasiert auf Frames angewendet (aktuell PIL-basiert)
   - Audio-Muxing zum Schluss

4. **KI-Integration** (`gemini_integration.py`):
   - `GeminiIntegration.transcribe_audio()` - Audio-Transkription
   - `GeminiIntegration.extract_quotes()` - Key-Zitate mit Zeitstempeln
   - `GeminiIntegration.optimize_all_settings()` - Parameter/Farben/Post-Process Optimierung
   - Quotes werden in der GUI reviewt, editiert und gefiltert

## Code-Style Guidelines

### GPU-Visualizer erstellen

**WICHTIG**: Neue GPU-Visualizer MÜSSEN diese Struktur folgen.

Seit v2.6 stellt `base.py` gemeinsame Bausteine bereit — bitte nutzen statt kopieren:
- `create_fullscreen_quad(ctx, prog)` / `create_textured_quad(ctx, prog)` für Quad-Setup
- `compose_fragment(body, includes=(...))` mit `LYGIA_MATH_GLSL`, `LYGIA_NOISE_GLSL`,
  `LYGIA_SDF_GLSL`, `SHADER_COMMON_GLSL` (enthält `aastep`/`aafill` für pixelgenaues
  Anti-Aliasing, `tonemapACES`, `hash12`, Dithering)
- `self._features_at_time(features, time)` statt Frame-Index-Boilerplate
- Shader geben HDR aus (kein `clamp` am Ende) — Tonemapping macht zentral der Renderer.

```python
import numpy as np
import moderngl
from .base import BaseGPUVisualizer

class MeinVisualizer(BaseGPUVisualizer):
    """Dokumentation hier."""

    # Parameter: (default, min, max, step)
    PARAMS = {
        "intensity": (1.0, 0.0, 3.0, 0.1),
        "speed": (1.0, 0.0, 5.0, 0.1),
    }

    def _setup(self):
        """Einmalige Initialisierung: Shader, VAOs, Texturen erstellen."""
        self._build_program()
        self._setup_quad()

    def _build_program(self):
        self.prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
            #version 330
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_rms;
            uniform float u_onset;
            out vec4 f_color;
            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                f_color = vec4(vec3(u_rms), 1.0);
            }
            """,
        )

    def render(self, features: dict, time: float):
        """Rendert EINEN Frame in den aktiven Framebuffer."""
        frame_idx = int(time * features.get("fps", 30))
        f = self._get_feature_at_frame(features, frame_idx)
        self.prog["u_time"].value = time
        self.prog["u_rms"].value = f["rms"]
        self.prog["u_onset"].value = f["onset"]
        self.prog["u_resolution"].value = (self.width, self.height)
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
```

### Feature-Keys

| Key | Bereich | Verwendung |
|-----|---------|------------|
| `rms` | 0.0-1.0 | Lautstärke → Größe/Opazität |
| `onset` | 0.0-1.0 | Beats → Trigger/Explosionen |
| `chroma` | Array[12] | Tonart → Farben (C, C#, D, ...) |
| `spectral_centroid` | 0.0-1.0 | Helligkeit/Detail |
| `spectral_rolloff` | 0.0-1.0 | Bandbreite |
| `zero_crossing_rate` | 0.0-1.0 | Noise vs Tonal |
| `transient` | 0.0-1.0 | Kick/Snare-Transienten |
| `voice_clarity` | 0.0-1.0 | Sprach-Präsenz |
| `voice_band` | 0.0-1.0 | Sprach-Band-Energie |
| `beat_intensity` | 0.0-1.0 | Beat-Decay-Envelope |
| `beat_frames` | Array[int] | Frame-Indizes der erkannten Beats |
| `tempo` | float | BPM |
| `mode` | str | "music", "speech", "hybrid" |
| `progress` | 0.0-1.0 | Zeit-Fortschritt |

### Konfiguration

Pfade und Einstellungen werden in `src/types.py` als Pydantic v2 Models definiert:

```python
# AudioFeatures: Schema für alle Audio-Features
# VisualConfig: Jeder Visualizer hat diese Konfiguration
# ProjectConfig: Gesamtkonfiguration einer Render-Job
```

JSON-Configs werden in `config/schemas.py` validiert. Das Schema erwartet:
- Flache `background_*` Felder (nicht verschachtelt)
- `visual.params` als offenes Dict (jeder Visualizer hat eigene PARAMS)
- `quotes` als Liste von Dicts mit `text`, `start_time`, `end_time`, `confidence`
- Farben in `quote_overlay` als Hex-String oder RGBA-Liste

## Testing Strategie

### Test-Dateien

- **`test_analyzer.py`**: Testet Audio-Feature-Extraktion
  - Feature-Shapes validieren
  - Caching-Verhalten testen
  - Wertebereiche prüfen (0-1)

- **`test_ai_matcher.py`**: Testet den SmartMatcher
  - Empfiehlt korrekte Visualizer basierend auf Audio-Features
  - Validiert KI-Parameter und Farbpaletten

- **`test_visuals.py`**: Testet alle GPU-Visualizer
  - Rückgabe muss `np.ndarray` sein
  - Shape muss `(H, W, 3)` sein
  - `dtype` muss `uint8` sein
  - Werte müssen in 0-255 liegen

- **`test_gpu_renderer.py`**: Testet GPU-Renderer
  - FFmpeg-Cmd-Builder
  - Render-Flow mit gemocktem FFmpeg

- **`test_quote_overlay.py`**: Testet Quote-Overlays
  - Timing, Fade, Text-Wrapping, Thread-Safety

### Test-Hilfsfunktionen

```python
# Dummy-Features für schnelle Tests
dummy_features = AudioFeatures(
    duration=1.0,
    sample_rate=44100,
    fps=30,
    rms=np.random.rand(30),
    onset=np.random.rand(30),
    # ... weitere Features
)
```

## Sicherheitsaspekte

1. **Datei-Validierung**: Audio-Dateien werden auf gültige Endungen geprüft (`.mp3`, `.wav`, `.flac`, etc.)
2. **Output-Validierung**: Output-Dateien müssen `.mp4` Endung haben
3. **Cache-Isolierung**: Cache wird in `.cache/` gespeichert, nicht im Output-Verzeichnis
4. **Temporäre Dateien**: Werden mit `tempfile` erstellt und aufgeräumt

## Performance-Tipps

1. **Vorschau zuerst**: Nutze `--preview` für schnelles Testen (5 Sekunden, 480p)
2. **Caching**: Audio-Analyse wird automatisch gecached (`.cache/audio_features/`)
3. **Niedrigere FPS**: 30fps statt 60fps für schnelleres Rendering
4. **Niedrigere Auflösung**: Preview nutzt automatisch 854x480

## Wichtige Dateien für KI-Agents

| Datei | Beschreibung |
|-------|--------------|
| `src/gpu_visualizers/base.py` | Muss gelesen werden für neue GPU-Visualizer |
| `src/gpu_visualizers/__init__.py` | VISUALIZER_MAP Registry |
| `src/types.py` | Alle Pydantic Models |
| `config/schemas.py` | Config-Validierung (Pydantic v2) |
| `src/analyzer.py` | Audio-Feature-Extraktion (NICHT ÄNDERN, nur erweitern) |
| `src/ai_matcher.py` | KI-Empfehlungslogik (Smart Matcher) |
| `src/quote_overlay.py` | Text-Overlay Rendering für Quotes |
| `src/gemini_integration.py` | Gemini KI Integration |
| `src/gpu_renderer.py` | Render-Flow verstehen |

## Sprache und Kommentare

- **Code-Kommentare**: Deutsch
- **Dokumentation**: Deutsch
- **README**: Deutsch
- **Commit-Messages**: Deutsch (empfohlen)

## Häufige Aufgaben

### Neuen GPU-Visualizer hinzufügen

1. `python main.py create-visualizer mein_visualizer --type shader` ausführen
   (Alternativ: `create-template` für ein minimales Boilerplate)
2. `src/gpu_visualizers/mein_visualizer.py` implementieren
3. Auto-Discovery übernimmt die Registrierung automatisch beim nächsten Import.
   Manuelle Einträge in `src/gpu_visualizers/__init__.py` bleiben für
   Rückwärts-Kompatibilität erhalten.
4. In `test_visuals.py` automatisch getestet (sofern in Registry)

### Neue Config-Preset erstellen

1. `python main.py create-config --output config/mein_preset.json`
2. Werte anpassen
3. Schema in `config/schemas.py` bei Bedarf erweitern
4. Config mit `python -c "from config.schemas import load_and_validate_config; load_and_validate_config('config/mein_preset.json')"` testen

### KI-Parameter für Visualizer nutzen

Jeder GPU-Visualizer liest Parameter aus `self.params` (merge von `EFFECTS`, `COLOR_PARAMS`, `PARAMS`):

```python
def _setup(self):
    p = self.params
    self.num_particles = p.get('particle_count', 150)
    self.explosion_threshold = p.get('explosion_threshold', 0.4)
```

Wichtige KI-Parameter pro Visualizer (siehe `PARAMS` in jeder Datei):

| Visualizer | Parameter |
|-----------|-----------|
| pulsing_core | pulse_intensity, ring_count, glow_radius, bg_brightness |
| spectrum_bars | bar_count, height_scale, spacing, color_shift |
| chroma_field | field_resolution, connection_dist, particle_size |
| particle_swarm | particle_count, explosion_threshold, glow_size, trail_length |
| typographic | bar_width, bar_spacing, animation_speed |
| neon_oscilloscope | line_thickness, trail_length, num_points, glow_radius |
| sacred_mandala | rotation_speed |
| liquid_blobs | blob_count, fluidity |
| neon_wave_circle | circle_count, wave_amplitude |
| frequency_flower | num_petals, layer_count |
| lumina_core | core_intensity, ring_count, noise_scale, glow_strength |
| voice_flow | flow_speed, wave_depth, breathe_intensity, line_count |
| spectrum_genesis | bar_count, wave_intensity, glow_radius, beat_flash |
| speech_focus | line_thickness, vu_segments, response_speed, accent_color |
| bass_temple | bass_intensity, strobe_threshold, shockwave_speed |
| orchestral_swell | swell_intensity, particle_count, dynamics_response |

### Audio-Analyse erweitern

**ACHTUNG**: Die `analyze()` Methode in `analyzer.py` sollte NICHT geändert werden (Caching!).
Stattdessen neue Features hinzufügen:
1. Neues Feature in `AudioFeatures` Model (`src/types.py`) ergänzen
2. Extraktion in `analyzer.py` hinzufügen
3. Caching-Logik bleibt unverändert
4. Feature in `GPUBatchRenderer`/`GPUPreviewRenderer` `features_dict` hinzufügen

## Lizenz

MIT License - Siehe LICENSE-Datei
