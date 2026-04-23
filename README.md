# Audio Visualizer Pro 🎵✨

Ein modulares, KI-optimiertes Audio-Visualisierungs-System für professionelle Musikvideos, Podcast-Visuals und kreative Projekte.

**Mit KI-gestütztem Auto-Modus, intelligenten Key-Zitat-Overlays und 10 einzigartigen Visualizern.**

---

## 🚀 Schnellstart

**Neu hier?** Siehe [QUICKSTART.md](QUICKSTART.md) für eine vollständige Schritt-für-Schritt-Anleitung!

### Option 1: Grafische Oberfläche (Empfohlen für Einsteiger)
```bash
# Windows: Doppelklicke auf start_gui.bat
# Oder überall:
python start_gui.py

# Öffnet automatisch http://localhost:8501 im Browser
```

### Option 2: Kommandozeile
```bash
# Installation
pip install -r requirements.txt

# FFmpeg muss system-seitig installiert sein (siehe QUICKSTART.md)

# 5-Sekunden Vorschau rendern
python main.py render song.mp3 --visual pulsing_core --preview

# Volles Video rendern
python main.py render song.mp3 --visual spectrum_bars -o output.mp4
```

---

## ✨ Features

- **🖥️ Grafische Oberfläche**: Moderne Web-GUI mit Streamlit — keine Kommandozeile nötig!
- **🤖 KI-Auto-Modus**: Smart Matcher analysiert dein Audio und empfehlt automatisch den besten Visualizer + Farbpalette + Parameter
- **💬 KI-Zitat-Extraktion**: Gemini KI findet automatisch die besten Key-Zitate aus deinem Audio (mit Zeitstempeln)
- **✏️ Zitat-Review**: Bearbeite, aktiviere/deaktiviere Zitate bevor sie als Overlay gerendert werden
- **🎙️ Podcast-Genre-Presets**: Optimierte Presets für News, Interview, Storytelling und Mixed-Formate
- **10 integrierte Visualizer**: Pulsing Core, Spectrum Bars, Chroma Field, Particle Swarm, Typographic, Neon Oscilloscope, Sacred Mandala, Liquid Blobs, Neon Wave Circle, Frequency Flower
- **🔌 Plugin-System**: Einfache Erweiterung mit `@register_visualizer` Decorator
- **📊 Intelligente Audio-Analyse**: Beat-Erkennung, Key-Erkennung, Chroma-Features, Mode-Detection (Musik/Sprache/Hybrid)
- **⚡ Aggressives Caching**: Analysiere einmal, rendere millionenmal
- **🎬 Professionelle Codecs**: FFmpeg-basiert mit libx264 und AAC
- **🎨 Post-Processing**: LUTs, Film Grain, Vignette, Chromatic Aberration
- **📈 Echte Fortschrittsanzeige**: Live-Rendering-Fortschritt in der GUI

---

## 🎬 Verfügbare Visualizer

| Visualizer | Beschreibung | Best für |
|------------|--------------|----------|
| `pulsing_core` | Pulsierender Kreis mit Chroma-Farben | EDM, Pop |
| `spectrum_bars` | 40-Balken Equalizer | Rock, Hip-Hop |
| `chroma_field` | Partikel-Feld basierend auf Tonart | Ambient, Jazz |
| `particle_swarm` | Physik-basierte Partikel-Explosionen | Dubstep, Trap |
| `typographic` | Minimalistisch mit Wellenform | Podcasts, Sprache |
| `neon_oscilloscope` | Retro-futuristischer Oszilloskop mit Neon-Effekten | Synthwave, Cyberpunk |
| `sacred_mandala` | Heilige Geometrie mit rotierenden Mustern | Meditation, Ambient |
| `liquid_blobs` | Flüssige MetaBall-ähnliche Blob-Animation | House, Techno |
| `neon_wave_circle` | Konzentrische Neon-Ringe mit Wellen | EDM, Techno |
| `frequency_flower` | Organische Blumen mit Audio-reaktiven Blütenblättern | Indie, Folk, Pop |

---

## 🎙️ Podcast-Workflow (Neu!)

Der perfekte Workflow für Podcast-Visuals auf YouTube:

```bash
# 1. GUI starten
python start_gui.py

# 2. Audio hochladen → KI analysiert automatisch

# 3. "Podcast-Genre" wählen (News/Interview/Story/Mixed)
#    ODER "Auto-Modus" für KI-Empfehlung

# 4. "Key-Zitate extrahieren" klicken → Gemini findet die besten Zitate

# 5. Zitate reviewen: Text editieren, aktivieren/deaktivieren

# 6. Video rendern → Zitate erscheinen automatisch als elegante Overlays
```

**Die Zitat-Overlays sind:**
- **Zeitbasiert**: Erscheinen bei `start_time`, verschwinden bei `end_time`
- **Animiert**: Sanftes Fade-In/Out
- **Stilvoll**: Abgerundete Box mit Schatten, zentriert unten
- **Editierbar**: Vor dem Rendering im GUI bearbeiten

---

## 🤖 KI-Auto-Modus

Der Smart Matcher analysiert dein Audio und empfiehlt automatisch:

```bash
# In der GUI: "Auto-Modus (KI empfiehlt)" wählen
# Oder CLI mit generierter Config:
python main.py render podcast.mp3 --config config/auto_recommended.json
```

**Was die KI analysiert:**
- Dynamik-Range (RMS-Verteilung)
- Beat-Dichte (Onset-Rate)
- Tempo (BPM)
- Key & Mode (Dur/Moll, Musik/Sprache)

**Was die KI empfiehlt:**
- Besten Visualizer (z.B. `typographic` für Sprache, `spectrum_bars` für Musik)
- Farbpalette passend zum Key
- Optimierte Parameter (Partikel-Anzahl, Geschwindigkeit, etc.)

---

## 📋 CLI-Referenz

```bash
# Hauptbefehle
python main.py render <audio> [options]
python main.py analyze <audio>
python main.py list-visuals
python main.py create-template <name>
python main.py create-config [options]

# Render-Optionen
--visual, -v        Visualizer-Typ (default: pulsing_core)
--output, -o        Output-Datei (default: output.mp4)
--config, -c        Config-JSON verwenden
--resolution, -r    Auflösung (default: 1920x1080)
--fps               FPS (default: 60)
--preview           5-Sekunden-Vorschau
--preview-duration  Vorschau-Dauer in Sekunden
```

### Beispiele

```bash
# Audio analysieren
python main.py analyze song.mp3

# Vorschau mit Auto-Modus Config
python main.py render podcast.mp3 --config config/podcast_interview.json --preview

# Volles Video mit Custom Visualizer
python main.py render song.mp3 --visual particle_swarm -o output.mp4

# Mit Config-Preset
python main.py render song.mp3 --config config/music_aggressive.json
```

---

## ⚙️ Konfiguration

### Beispiel-Config erstellen

```bash
python main.py create-config --output meine_config.json
```

### Config-Struktur (mit Quotes)

```json
{
  "audio_file": "song.mp3",
  "output_file": "output.mp4",
  "visual": {
    "type": "pulsing_core",
    "resolution": [1920, 1080],
    "fps": 60,
    "colors": {
      "primary": "#FF0055",
      "secondary": "#00CCFF",
      "background": "#0A0A0A"
    },
    "params": {
      "particle_intensity": 2.0
    }
  },
  "postprocess": {
    "contrast": 1.1,
    "saturation": 1.2,
    "grain": 0.05,
    "vignette": 0.3
  },
  "quotes": [
    {
      "text": "Das ist ein Key-Zitat aus dem Audio!",
      "start_time": 10.5,
      "end_time": 15.2,
      "confidence": 0.95
    }
  ]
}
```

---

## 🧪 Tests

```bash
# Alle Tests ausführen (42 Tests)
pytest tests/ -v

# Spezifische Test-Suites
pytest tests/test_visuals.py -v
pytest tests/test_analyzer.py -v
pytest tests/test_ai_matcher.py -v
pytest tests/test_quote_overlay.py -v
pytest tests/test_gemini_integration.py -v
```

---

## 📁 Projektstruktur

```
audio_visualizer_pro/
├── config/                     # 14 Konfigurations-Presets
│   ├── default.json
│   ├── music_aggressive.json
│   ├── podcast_minimal.json
│   ├── podcast_news.json       # 🆕 Podcast News Preset
│   ├── podcast_interview.json  # 🆕 Podcast Interview Preset
│   ├── podcast_story.json      # 🆕 Podcast Storytelling Preset
│   ├── podcast_mixed.json      # 🆕 Podcast Mixed Preset
│   └── ...
├── src/
│   ├── analyzer.py             # Audio-Feature-Extraktion
│   ├── ai_matcher.py           # 🆕 SmartMatcher - KI-Empfehlung
│   ├── gemini_integration.py   # 🆕 Gemini KI für Zitate
│   ├── quote_overlay.py        # 🆕 Text-Overlay Renderer
│   ├── pipeline.py             # Haupt-Orchestrator
│   ├── types.py                # Pydantic Models
│   ├── visuals/                # 10 Visualizer + Plugin-System
│   ├── renderers/
│   └── postprocess.py          # Color Grading
├── tests/                      # 42 Tests
├── gui.py                      # Streamlit-GUI
├── main.py                     # CLI Entry Point
└── requirements.txt
```

---

## 🔧 System-Voraussetzungen

- **Python**: 3.9+
- **FFmpeg**: System-seitig installiert
  - Ubuntu: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- **API-Key** (optional): `GEMINI_API_KEY` für KI-Zitat-Extraktion
  - Kostenlos bei [Google AI Studio](https://aistudio.google.com/)

---

## 🎯 Performance-Tipps

1. **Vorschau zuerst**: Nutze `--preview` für schnelles Testen (5 Sekunden, 480p)
2. **Caching**: Audio-Analyse wird automatisch gecached (`.cache/audio_features/`)
3. **Niedrigere FPS**: 30fps statt 60fps für schnelleres Rendering
4. **KI-Auto-Modus**: Spart Zeit bei der Visualizer-Auswahl

---

## 🆕 Changelog

### v2.0 — KI-Release
- **🤖 Smart Matcher**: KI-gestützte Visualizer-Empfehlung basierend auf Audio-Analyse
- **💬 Gemini Integration**: Automatische Key-Zitat-Extraktion aus Audio
- **✏️ Zitat-Review**: Editieren, aktivieren/deaktivieren vor dem Rendering
- **🎙️ Podcast-Presets**: 4 neue Genre-spezifische Presets
- **📈 Echte Fortschrittsanzeige**: Live-Progress in der GUI
- **⚡ 42 Tests**: Vollständige Test-Abdeckung

---

## 📄 Lizenz

MIT License — Siehe [LICENSE](LICENSE) Datei

## 🙏 Credits

- **Audio-Analyse**: [Librosa](https://librosa.org/)
- **Bildverarbeitung**: [Pillow](https://python-pillow.org/)
- **Video-Encoding**: [FFmpeg](https://ffmpeg.org/)
- **KI-Integration**: [Google Gemini](https://ai.google.dev/)
