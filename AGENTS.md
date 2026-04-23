# Audio Visualizer Pro - Agent Guide

Dieses Dokument enthält alle relevanten Informationen für KI-Code-Agents, die an diesem Projekt arbeiten.

## Projekt-Übersicht

**Audio Visualizer Pro** ist ein modulares, KI-optimiertes Audio-Visualisierungs-System für professionelle Musikvideos, Podcast-Visuals und kreative Projekte.

### Kern-Features
- **10 integrierte Visualizer**: Pulsing Core, Spectrum Bars, Chroma Field, Particle Swarm, Typographic, Neon Oscilloscope, Sacred Mandala, Liquid Blobs, Neon Wave Circle, Frequency Flower
- **Plugin-System**: Einfache Erweiterung mit `@register_visualizer` Decorator
- **Intelligente Audio-Analyse**: Beat-Erkennung, Key-Erkennung, Chroma-Features, Mode-Detection (speech/music/hybrid)
- **KI-gestützter Auto-Modus**: Smart Matcher analysiert Audio und empfehlt automatisch Visualizer + Farbpalette + Parameter
- **Streamlit-GUI**: Web-basierte Oberfläche mit Drag & Drop, Live-Analyse und One-Click-Render
- **Aggressives Caching**: Analysiere einmal, rendere millionenmal
- **Professionelle Codecs**: FFmpeg-basiert mit libx264 und AAC
- **Post-Processing**: LUTs, Film Grain, Vignette, Chromatic Aberration

## Technologie-Stack

| Komponente | Bibliothek | Zweck |
|------------|------------|-------|
| Audio-Analyse | librosa>=0.10.0 | Feature-Extraktion (RMS, Onset, Chroma, etc.) |
| Bildverarbeitung | Pillow>=9.0.0 | Frame-Generierung |
| Datenvalidierung | pydantic>=1.10.0 | Konfiguration-Models |
| CLI | click>=8.0.0 | Kommandozeilen-Interface |
| Numerik | numpy>=1.21.0 | Array-Operationen |
| Testing | pytest>=7.0.0 | Test-Framework |
| Video-Encoding | FFmpeg (system) | H.264/AAC Encoding |

**System-Voraussetzung**: FFmpeg muss system-seitig installiert sein.
- Ubuntu: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: https://ffmpeg.org/download.html

## Projektstruktur

```
audio_visualizer_pro/
├── main.py                 # CLI Entry Point (Click-basiert)
├── requirements.txt        # Python-Abhängigkeiten
├── config/                 # Konfigurations-Presets und Validierung
│   ├── schemas.py          # Pydantic-Schemas für Config-Validierung
│   ├── default.json        # Standard-Konfiguration
│   ├── music_aggressive.json
│   ├── podcast_minimal.json
│   └── ...
├── src/
│   ├── __init__.py
│   ├── analyzer.py         # AudioAnalyzer mit Caching
│   ├── ai_matcher.py       # SmartMatcher - KI-gestützte Visualizer-Empfehlung
│   ├── pipeline.py         # RenderPipeline, PreviewPipeline
│   ├── types.py            # Pydantic Models (AudioFeatures, VisualConfig, etc.)
│   ├── postprocess.py      # PostProcessor für Color Grading
│   ├── quote_overlay.py    # QuoteOverlayRenderer für Key-Zitat Text-Overlays
│   ├── gemini_integration.py # Gemini KI für Transkription und Zitat-Extraktion
│   ├── visuals/            # Plugin-System
│   │   ├── __init__.py
│   │   ├── base.py         # BaseVisualizer (abstrakte Basisklasse)
│   │   ├── registry.py     # VisualizerRegistry mit @register_visualizer
│   │   ├── 01_pulsing_core.py
│   │   ├── 02_spectrum_bars.py
│   │   ├── 03_chroma_field.py
│   │   ├── 04_particle_swarm.py
│   │   ├── 05_typographic.py
│   │   ├── 06_neon_oscilloscope.py
│   │   ├── 07_sacred_mandala.py
│   │   ├── 08_liquid_blobs.py
│   │   ├── 09_neon_wave_circle.py
│   │   └── 10_frequency_flower.py
│   └── renderers/
│       ├── __init__.py
│       └── pil_renderer.py # PILRenderer für Frame-Generierung
├── gui.py                  # Streamlit-GUI für One-Click-Bedienung
└── tests/
    ├── __init__.py
    ├── test_analyzer.py    # Tests für AudioAnalyzer
    ├── test_ai_matcher.py  # Tests für SmartMatcher
    ├── test_quote_overlay.py # Tests für Quote Overlay Renderer
    └── test_visuals.py     # Tests für alle Visualizer
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

# Verfügbare Visualizer anzeigen
python main.py list-visuals

# 5-Sekunden Vorschau rendern
python main.py render song.mp3 --visual pulsing_core --preview

# Volles Video rendern
python main.py render song.mp3 --visual spectrum_bars -o output.mp4

# Mit KI-Auto-Modus (Config wird automatisch generiert)
python main.py render song.mp3 --config config/auto_recommended.json -o output.mp4

# GUI starten
streamlit run gui.py

# Mit Config-Datei
python main.py render song.mp3 --config config/music_aggressive.json

# Neues Visualizer-Template erstellen
python main.py create-template mein_visualizer

# Beispiel-Config erstellen
python main.py create-config --output meine_config.json
```

### Testing
```bash
# Alle Tests ausführen
pytest tests/ -v

# Spezifische Tests
pytest tests/test_visuals.py -v
pytest tests/test_analyzer.py -v
```

## Architektur

### 3-Schichten-Architektur

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Quote Overlays (Key-Zitat Text-Overlays)         │
│  → QuoteOverlayRenderer.apply(frame, time_seconds)         │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Post-Processing (LUTs, Grain, Vignette)          │
│  → PostProcessor.apply(frame)                              │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Visualization (Frame-Generierung)                │
│  → BaseVisualizer.render_frame(frame_idx)                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Audio-Analyse (Feature-Extraktion)               │
│  → AudioAnalyzer.analyze(audio_path, fps)                  │
└─────────────────────────────────────────────────────────────┘
```

### Datenfluss

1. **Audio-Analyse** (`analyzer.py`):
   - Extrahiert Features: RMS, Onset, Chroma, Spectral Centroid, etc.
   - Caching in `.cache/audio_features/` (NPZ-Format)
   - Deterministisch und thread-safe

2. **Visualization** (`visuals/`):
   - Jeder Visualizer erbt von `BaseVisualizer`
   - Registrierung via `@register_visualizer("name")`
   - `render_frame(frame_idx)` gibt RGB-Array zurück

3. **Rendering** (`pipeline.py`):
   - `RenderPipeline` steuert den kompletten Flow
   - FFmpeg-Subprozess für Video-Encoding
   - Quote Overlays werden zeitbasiert auf Frames angewendet
   - Audio-Muxing zum Schluss

4. **KI-Integration** (`gemini_integration.py`):
   - `GeminiIntegration.transcribe_audio()` - Audio-Transkription
   - `GeminiIntegration.extract_quotes()` - Key-Zitate mit Zeitstempeln
   - Quotes werden in der GUI reviewt, editiert und gefiltert

## Code-Style Guidelines

### Visualizer erstellen

**WICHTIG**: Neue Visualizer MÜSSEN diese Struktur folgen:

```python
from .base import BaseVisualizer
from .registry import register_visualizer

@register_visualizer("einzigartiger_name")  # 1. Decorator
class MeinVisualizer(BaseVisualizer):        # 2. Erbe von BaseVisualizer
    """Dokumentation hier."""
    
    def setup(self):                         # 3. setup() implementieren
        """Einmalige Initialisierung."""
        self.center = (self.width // 2, self.height // 2)
    
    def render_frame(self, frame_idx: int) -> np.ndarray:  # 4. render_frame() implementieren
        """Rendert EINEN Frame als RGB-Array (H, W, 3), dtype uint8."""
        # Features holen
        f = self.get_feature_at_frame(frame_idx)
        rms = f['rms']        # 0.0-1.0 Lautstärke
        onset = f['onset']    # 0.0-1.0 Beat-Trigger
        chroma = f['chroma']  # 12 Werte für Halbtöne
        
        # Deine Zeichen-Logik...
        img = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        
        return np.array(img)
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
| `progress` | 0.0-1.0 | Zeit-Fortschritt |

### Konfiguration

Pfade und Einstellungen werden in `src/types.py` als Pydantic-Models definiert:

```python
# AudioFeatures: Schema für alle Audio-Features
# VisualConfig: Jeder Visualizer hat diese Konfiguration
# ProjectConfig: Gesamtkonfiguration einer Render-Job
```

JSON-Configs werden in `config/schemas.py` validiert.

## Testing Strategie

### Test-Dateien

- **`test_analyzer.py`**: Testet Audio-Feature-Extraktion
  - Feature-Shapes validieren
  - Caching-Verhalten testen
  - Wertebereiche prüfen (0-1)

- **`test_ai_matcher.py`**: Testet den SmartMatcher
  - Empfiehlt korrekte Visualizer basierend auf Audio-Features
  - Validiert KI-Parameter und Farbpaletten

- **`test_visuals.py`**: Testet alle Visualizer
  - Rückgabe muss `np.ndarray` sein
  - Shape muss `(H, W, 3)` sein
  - `dtype` muss `uint8` sein
  - Werte müssen in 0-255 liegen

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
4. **Niedrigere Auflösung**: PreviewPipeline nutzt automatisch 480p

## Wichtige Dateien für KI-Agents

| Datei | Beschreibung |
|-------|--------------|
| `src/visuals/base.py` | Muss gelesen werden für neue Visualizer |
| `src/visuals/registry.py` | Plugin-System verstehen |
| `src/types.py` | Alle Pydantic Models |
| `config/schemas.py` | Config-Validierung |
| `src/analyzer.py` | Audio-Feature-Extraktion (NICHT ÄNDERN, nur erweitern) |
| `src/ai_matcher.py` | KI-Empfehlungslogik (Smart Matcher) |
| `src/quote_overlay.py` | Text-Overlay Rendering für Quotes |
| `src/gemini_integration.py` | Gemini KI Integration |
| `src/pipeline.py` | Render-Flow verstehen |

## Sprache und Kommentare

- **Code-Kommentare**: Deutsch
- **Dokumentation**: Deutsch
- **README**: Deutsch
- **Commit-Messages**: Deutsch (empfohlen)

## Häufige Aufgaben

### Neuen Visualizer hinzufügen

1. `python main.py create-template mein_visualizer` ausführen
2. `src/visuals/mein_visualizer.py` implementieren
3. In `test_visuals.py` automatisch getestet (Registry-Autoload)

### Neue Config-Preset erstellen

1. `python main.py create-config --output config/mein_preset.json`
2. Werte anpassen
3. Schema in `config/schemas.py` bei Bedarf erweitern

### KI-Parameter für Visualizer nutzen

Jeder Visualizer liest in `setup()` Parameter aus `self.config.params` (mit Defaults):

```python
def setup(self):
    p = self.config.params
    self.num_particles = p.get('particle_count', 150)
    self.explosion_threshold = p.get('explosion_threshold', 0.4)
```

Wichtige KI-Parameter pro Visualizer:

| Visualizer | Parameter | Default |
|-----------|-----------|---------|
| pulsing_core | pulse_intensity, glow_layers, glow_radius, trail_length | 0.8, 3, 20, 20 |
| spectrum_bars | bar_count, smoothing | 40, 0.3 |
| chroma_field | field_resolution | 100 |
| particle_swarm | particle_count, explosion_threshold | 150, 0.4 |
| typographic | text_size, animation_speed, bar_width, bar_spacing | 72, 0.2, 3, 1 |
| neon_oscilloscope | line_thickness, trail_length, num_points | 4, 8, 200 |
| sacred_mandala | rotation_speed | 0.005 |
| liquid_blobs | blob_count, fluidity | 6, 0.7 |
| neon_wave_circle | circle_count, wave_amplitude | 5, 1.0 |
| frequency_flower | num_petals, layer_count | 8, 3 |

### Audio-Analyse erweitern

**ACHTUNG**: Die `analyze()` Methode in `analyzer.py` sollte NICHT geändert werden (Caching!).
Stattdessen neue Features hinzufügen:
1. Neues Feature in `AudioFeatures` Model (`src/types.py`) ergänzen
2. Extraktion in `analyzer.py` hinzufügen
3. Caching-Logik bleibt unverändert

## Lizenz

MIT License - Siehe LICENSE-Datei
