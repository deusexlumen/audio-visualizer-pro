<div align="center">

# 🎵 Audio Visualizer Pro

**Verwandle Audio in GPU-gerenderte Musikvideos und Podcast-Visuals.**

[![Version](https://img.shields.io/badge/version-v3.2.0-blue)](https://github.com/deusexlumen/audio-visualizer-pro/releases/latest)
[![Download](https://img.shields.io/badge/⬇%20Download-Windows--Installer-success)](https://github.com/deusexlumen/audio-visualizer-pro/releases/latest)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-282%20passed-brightgreen)](#-tests)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Audio Visualizer Pro analysiert deine Audiodatei mit **librosa**, rendert sie über
**ModernGL/OpenGL**-Shader in einer **Float16-HDR-Pipeline** und encodiert das
Ergebnis mit **FFmpeg** zu einem fertigen Video — gesteuert über eine
**PyQt6-Desktop-App** oder die Kommandozeile.

</div>

---

## ✨ Highlights

- **26 GPU-Visualizer** (10 Classic + 8 Signature Pro + 8 Archetypen) plus **Visualizer-Studio**: eigene Visualizer aus Bausteinen zusammenklicken, ganz ohne Code.
- **HDR-Render-Pipeline**: Float16 + 4×-MSAA, echter HDR-Bloom, ACES-Tonemapping, 3D-LUTs (`.cube`), Vignette, chromatische Aberration, Film-Grain, Dithering — kein Banding, keine harten Clips.
- **Szenen-Timeline**: Visualizer wechseln automatisch über die Zeit, aus der Songstruktur abgeleitet (mit Crossfades).
- **KI-Unterstützung (Gemini, optional)**: Transkription, Zitat-Extraktion mit Zeitstempeln, automatische Visualizer-/Parameter-Empfehlung und ein Voll-KI-Modus.
- **Moderne GUI**: Dark-Studio-Oberfläche mit Wellenform-Timeline, Live-Vorschau, Drag & Drop und Projektdateien (`.avproj`).
- **Multi-Codec-Export**: H.264, HEVC, ProRes über FFmpeg.
- **Windows-Installer**: fertige `.exe`, kein Python nötig — FFmpeg wird bei Bedarf automatisch nachgeladen.

---

## ⬇ Installation

### Windows (empfohlen, kein Python nötig)

1. **[Neuestes Release herunterladen](https://github.com/deusexlumen/audio-visualizer-pro/releases/latest)** → `AudioVisualizerPro-Setup-<version>.exe`.
2. Ausführen. **Kein Administrator nötig** (Per-User-Installation).
3. Beim ersten Start bietet die App an, **FFmpeg** automatisch herunterzuladen (~90 MB), falls es nicht schon im System vorhanden ist.

> Windows Defender meldet die unsignierte EXE eventuell als unbekannt → „Weitere Informationen" → „Trotzdem ausführen". Vollständige Anleitung: **[docs/INSTALLATION.md](docs/INSTALLATION.md)**.

### Aus dem Quellcode (Entwickler / macOS / Linux)

```bash
git clone https://github.com/deusexlumen/audio-visualizer-pro.git
cd audio-visualizer-pro

# Abhängigkeiten (exakt gepinnt — empfohlen)
pip install -r requirements.lock
# Alternativ mit losen Versionsgrenzen:
pip install -r requirements.txt

# FFmpeg muss systemweit installiert und im PATH sein:
#   Ubuntu/Debian: sudo apt-get install ffmpeg
#   macOS:         brew install ffmpeg
#   Windows:       https://ffmpeg.org/download.html

python gui.py        # GUI starten
```

---

## 🔑 Gemini-API-Key einrichten (optional)

Die KI-Features (Transkription, Zitat-Extraktion, Auto-Modus) brauchen einen
**Gemini-API-Key**. Ohne Key laufen Rendering und alle Visualizer unverändert
weiter — nur das KI-Panel bleibt inaktiv.

Key kostenlos erstellen: **https://aistudio.google.com/apikey**

**Variante A — `.env`-Datei** (einfachster Weg):
Lege im Programm-/Projektordner eine Datei `.env` an:

```
GEMINI_API_KEY=dein-key-hier
```

*(Installierte App: Start-Menü-Eintrag → Rechtsklick → „Dateipfad öffnen".)*

**Variante B — Windows-Umgebungsvariable:**
Start → „Umgebungsvariablen bearbeiten" → Benutzervariable `GEMINI_API_KEY`
anlegen, App neu starten.

---

## 🚀 Schnellstart (CLI)

```bash
# Audio analysieren (Features werden gecached)
python main.py analyze song.mp3 --fps 30

# Verfügbare Visualizer auflisten
python main.py list-visuals

# 5-Sekunden-Vorschau in 480p rendern
python main.py render song.mp3 --visual lumina_core --preview

# Vollständiges Video rendern
python main.py render song.mp3 --visual spectrum_bars -o output.mp4 \
  --resolution 1920x1080 --fps 60

# Mit Preset
python main.py render song.mp3 --config config/music_aggressive.json

# Mit eigenen Parametern
python main.py render song.mp3 --visual particle_swarm \
  --param particle_count=200 --param trail_length=15 -o custom.mp4

# Mehrere Auflösungen / Batch
python main.py render-multi song.mp3 --visual nebula_drift --resolutions 1920x1080,1280x720
python main.py batch jobs.json
```

Alle Befehle: `python main.py --help`.

---

## 🎨 Visualizer

### Classic (10)

| Name | Beschreibung | Ideal für |
|------|--------------|-----------|
| `spectrum_bars` | Klassischer Balken-Equalizer | Rock, Hip-Hop, Pop |
| `pulsing_core` | Pulsierender Kern mit Glow & Beat-Schockwellen | EDM, Techno, House |
| `particle_swarm` | Physik-basierte Partikel-Schwärme | Dubstep, Trap, Bass |
| `neon_oscilloscope` | Retro-Oszilloskop mit Neon-Trails | Synthwave, Cyberpunk |
| `chroma_field` | Partikelfeld basierend auf der Tonart | Jazz, Ambient, Klassik |
| `typographic` | Nächtliche Skyline; Fenster leuchten im Takt | Podcasts, Sprache |
| `sacred_mandala` | Rotierende geometrische Muster | Meditation, Spiritual |
| `liquid_blobs` | Flüssige MetaBall-Animation | Deep House, Liquid DnB |
| `neon_wave_circle` | Konzentrische Neon-Ringe | Trance, Progressive |
| `frequency_flower` | Organische Blüten-Animation | Indie, Folk, Acoustic |

### Signature Pro (8)

| Name | Beschreibung | Ideal für |
|------|--------------|-----------|
| `lumina_core` | Intelligenter Hybrid-Visualizer | Allrounder |
| `voice_flow` | Sprach-optimierte Visualisierung | Podcasts, Interviews |
| `spectrum_genesis` | Evolvierendes Spektrum (Peak-Hold, Reflexion) | Elektronische Musik |
| `speech_focus` | Stimm-Linie, im Musik-Modus ein Spektrum-Band | Hörbücher, Vorträge |
| `bass_temple` | Bass-zentrierte Tempel-Architektur | Bass Music, Trap |
| `orchestral_swell` | Aufsteigende Licht-Vorhänge | Filmmusik, Klassik |
| `aurora_voice` | Ruhige Aurora-Bänder, kein Beat-Blitzen | Lange Podcasts, Hörbücher |
| `nebula_drift` | Treibende Nebelwolken + Sternenfeld | Ambient, Big-Room-EDM |

### Archetypen (8, neu in v3.2)

Jeder eine eigene Bildwelt statt einer weiteren Kreis- oder Wellenvariante.
Alle laufen in beiden Modi: bei Sprache ändert sich die Empfindlichkeit,
nicht die Optik. Und alle lassen ein Hintergrundbild durchscheinen.

| Name | Beschreibung | Ideal für |
|------|--------------|-----------|
| `retro_sun` | Sonne am Horizont über einem Gitterboden | Synthwave, Retro |
| `dna_helix` | Doppelhelix; jede Querstrebe ist ein Ton | Elektronisch, Ambient |
| `kaleidoscope` | Winkel-Faltung, Sektorfarbe aus der Tonart | Psychedelic, Trance |
| `spirograph` | Kurvenfigur, deren Form aus der Tonart entsteht | Klassik, Melodisches |
| `voronoi_cells` | Wanderndes Zellnetz, Zellen gehören je einem Ton | Minimal, Techno |
| `ink_bloom` | Tinte in Wasser: Schlieren und Fäden | Ambient, Downtempo |
| `silk_ribbons` | Wehende Bänder mit Glanzkante | Chill, Neo-Soul |
| `scissor_lattice` | Scherengitter, schnappt auf dem Beat auf | Industrial, Beat-Musik |

### 🧩 Visualizer-Studio

Im **Studio-Tab** der GUI kombinierst du GLSL-Bausteine (Ring, Kern-Glow, Welle,
Balken, Partikel, Nebelfeld) zu einem eigenen Visualizer: Parameter per Regler,
Audio-Verknüpfungen, Live-Vorschau. Gespeichert wird als deklaratives
JSON-„Rezept", das sofort überall als vollwertiger Visualizer erscheint —
im Dropdown, in der CLI und in der Timeline. Ein KI-Assistent erzeugt auf Wunsch
aus einer Textbeschreibung einen Rezept-Entwurf. **Keine Zeile Code nötig.**

---

## 🏗️ Architektur

```
┌──────────────────────────────────────────────────────────────┐
│  Quote-Overlays   → gecachtes Text-Rendering mit Fade         │
├──────────────────────────────────────────────────────────────┤
│  HDR-Post-FX      → Bloom · Exposure · ACES · LUT · Vignette  │
│                     · Chromatic Aberration · Grain · Dither   │
├──────────────────────────────────────────────────────────────┤
│  GPU-Visualizer   → ModernGL-Shader, Float16-HDR + 4×-MSAA    │
│                     18 Visualizer + Studio-Rezepte + Timeline │
├──────────────────────────────────────────────────────────────┤
│  Audio-Analyse    → librosa: RMS, Onset, Chroma, MFCC, Beat,  │
│                     Voice-Clarity, Tempo (NPZ-gecached)       │
└──────────────────────────────────────────────────────────────┘
```

**Datenfluss:** Audio-Analyse → GPU-Rendering (Float16-HDR) → HDR-Post-FX →
Quote-Overlay → FFmpeg-Encoding (PBO-Readback + paralleler Encoder-Thread).

Tiefergehende Architektur-Doku, die Feature-Key-Tabelle und die Vorlage zum
Bau eigener Visualizer stehen in **[AGENTS.md](AGENTS.md)**.

---

## ⚙️ Konfiguration

Presets liegen als JSON in `config/` und werden gegen Pydantic-Schemas validiert:

- **Musik:** `default`, `music_aggressive`, `chromatic_dream`, `neon_cyberpunk`, `sacred_geometry`, `liquid_blobs`, `neon_circle`, `flower_bloom`
- **Podcast:** `podcast_minimal`, `podcast_news`, `podcast_interview`, `podcast_story`, `podcast_mixed`

```bash
# Eigenes Preset-Gerüst erzeugen
python main.py create-config --output mein_preset.json

# Preset validieren
python -c "from config.schemas import load_and_validate_config; load_and_validate_config('config/mein_preset.json')"
```

Nutzerdaten (Cache, Logs, eigene Studio-Rezepte) liegen unter
`%APPDATA%` / `%LOCALAPPDATA%\AudioVisualizerPro` — nie im Installationsordner.

---

## 🧪 Tests

```bash
pytest tests/ -v                      # gesamte Suite (282 Tests)
pytest tests/test_gpu_renderer.py -v  # einzelne Suite
```

GPU-abhängige Tests werden auf Systemen ohne OpenGL-fähige GPU automatisch
übersprungen (`-m "not gpu"`), sodass die Suite auch headless durchläuft.

---

## 💻 Systemanforderungen

| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| Betriebssystem | Windows 10/11 (Installer) · Python 3.10+ (Quellcode) | Windows 11 |
| RAM | 8 GB | 16 GB |
| GPU | OpenGL 3.3+ | dedizierte GPU |
| Speicher | 500 MB | 1 GB+ |
| FFmpeg | 4.0+ (Auto-Download möglich) | 6.0+ |

---

## 🛠️ Eigenen Build erstellen

```bash
pip install -r requirements.lock
pip install pyinstaller

python build/build.py            # → dist/AudioVisualizerPro/ (onedir)

# Windows-Installer (benötigt Inno Setup):
ISCC build/installer.iss /DMyAppVersion=3.2.0
# → dist/installer/AudioVisualizerPro-Setup-3.2.0.exe
```

Build-Details und bekannte Stolpersteine (librosa/numba/soundfile-Bundling)
stehen als Kommentare in `build/avp.spec`.

---

## 🤝 Mitwirken

```bash
pip install -e ".[dev]"
black src/ tests/
flake8 src/ tests/
pytest tests/ -v
```

- Kommentare, Docstrings und Commit-Messages auf **Deutsch** (passend zum Bestand).
- Black-konform (88 Zeichen), Type Hints für neue Funktionen.
- Neue Visualizer folgen der Vorlage in [AGENTS.md](AGENTS.md); jeder registrierte Visualizer wird automatisch vom Smoke-Test abgedeckt.

---

## 💡 Entwicklungshintergrund

Ein Solo-Projekt, entwickelt von einem Projektinitiator **ohne klassische
Programmierkenntnisse** (Fokus: Vision, Testing, Dokumentation) gemeinsam mit
**KI-Assistenten** (Code-Generierung, Refactoring, Debugging). Leitprinzipien:
Stabilität vor Feature-Menge, alles ausführlich getestet und dokumentiert,
Bedienbarkeit ohne Code.

---

## 📄 Lizenz & Credits

MIT — siehe [LICENSE](LICENSE).

Aufgebaut auf [ModernGL](https://moderngl.readthedocs.io/) ·
[librosa](https://librosa.org/) ·
[PyQt6](https://www.riverbankcomputing.com/software/pyqt/) ·
[FFmpeg](https://ffmpeg.org/) ·
[Pydantic](https://docs.pydantic.dev/) ·
[Gemini API](https://ai.google.dev/) ·
[Inter](https://rsms.me/inter/) (OFL).

---

## 📬 Support

- **Download:** [Releases](https://github.com/deusexlumen/audio-visualizer-pro/releases/latest)
- **Fehler melden:** [GitHub Issues](https://github.com/deusexlumen/audio-visualizer-pro/issues)
- **Fragen & Austausch:** [GitHub Discussions](https://github.com/deusexlumen/audio-visualizer-pro/discussions)
- **Weiterführend:** [Installation](docs/INSTALLATION.md) · [Changelog](CHANGELOG.md) · [Architektur (AGENTS.md)](AGENTS.md)

<div align="center">
<sub>Audio Visualizer Pro v3.2.0 · mit ❤️ und KI erstellt</sub>
</div>
