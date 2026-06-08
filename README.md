[![Version](https://img.shields.io/badge/SOTA-v2.1.0-blue)](https://github.com/audio-visualizer-pro/audio-visualizer-pro)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-134%20passed-brightgreen)](https://github.com/audio-visualizer-pro/audio-visualizer-pro/actions)
[![Coverage](https://img.shields.io/badge/coverage-77%25-yellowgreen)](https://github.com/audio-visualizer-pro/audio-visualizer-pro/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# Audio Visualizer Pro v2.1.0

**Professionelles Audio-Visualisierungs-System mit GPU-Beschleunigung und KI-Unterstützung**

Erstelle atemberaubende Musikvideos, Podcast-Visuals und kreative Projekte mit 16 GPU-beschleunigten Visualizern, KI-gestützter Zitat-Extraktion und professionellem Video-Encoding.

---

## 🎯 Überblick

Audio Visualizer Pro ist ein modulares System zur Erstellung hochwertiger Audio-Visualisierungen. Es kombiniert GPU-beschleunigtes Rendering (ModernGL/OpenGL), KI-gestützte Audio-Analyse (Gemini) und eine professionelle DearPyGui-Oberfläche.

### Kernfunktionen

- **🎨 16 GPU-Visualizer**: Shader-basierte Visualisierung mit ModernGL (10 Classic + 6 Signature Pro)
- **🤖 KI-Integration**: Automatische Transkription und Zitat-Extraktion mit Gemini 3.1 Flash-Lite
- **🖥️ DearPyGui GUI**: Premium Dark UI mit Echtzeit-Vorschau
- **✨ Post-Processing**: Bloom, Film Grain, Vignette, Chromatic Aberration, LUTs
- **🎬 Multi-Codec**: H.264, HEVC, ProRes Encoding via FFmpeg
- **🎵 Beat-Sync**: Synchronisierte Zitat-Einblendungen und Visual-Effekte
- **🔌 Plugin-System**: Einfache Erweiterung um eigene Visualizer
- **🧪 134 Tests**: Umfassende Testabdeckung mit 77% Coverage
- **🧠 Evo-Agent Framework**: State Ledger, Root Orchestrator & Skill-Dispatcher

---

## 🚀 Schnellstart
### Voraussetzungen
```bash
# Python 3.10+ erforderlich
python --version
# FFmpeg installieren (systemweit)
# Ubuntu/Debian:
sudo apt-get install ffmpeg
# macOS:
brew install ffmpeg
# Windows: https://ffmpeg.org/download.html
```
### Installation
```bash
# Repository klonen
git clone https://github.com/audio-visualizer-pro/audio-visualizer-pro.git
cd audio-visualizer-pro
# Abhängigkeiten installieren
pip install -r requirements.txt
```
### GUI starten
```bash
# DearPyGui Oberfläche starten
python gui.py
```
### CLI Nutzung
```bash
# Audio analysieren
python main.py analyze dein_audio.mp3
# Vorschau rendern (5 Sekunden, 480p)
python main.py render dein_audio.mp3 --visual lumina_core --preview
# Vollständiges Video rendern
python main.py render dein_audio.mp3 --visual spectrum_bars -o output.mp4 --resolution 1920x1080 --fps 60
# Mit benutzerdefinierten Parametern
python main.py render dein_audio.mp3 --visual neon_wave_circle \
  --param viz_scale=1.2 \
  --param color_mode=chroma \
  -o custom.mp4
```

---

## 🎨 Verfügbare Visualizer
### Classic Visualizer (10)
| Name | Beschreibung | Ideal für |
|------|--------------|-----------|
| `spectrum_bars` | Klassischer 40-Balken Equalizer | Rock, Hip-Hop, Pop |
| `pulsing_core` | Pulsierender Kern mit Glow-Effekten | EDM, Techno, House |
| `particle_swarm` | Physik-basierte Partikel-Schwärme | Dubstep, Trap, Bass |
| `neon_oscilloscope` | Retro Oszilloskop mit Neon-Trails | Synthwave, Cyberpunk |
| `chroma_field` | Partikel-Feld basierend auf Tonart | Jazz, Ambient, Klassik |
| `typographic` | Minimalistische Wellenform-Darstellung | Podcasts, Sprache |
| `sacred_mandala` | Rotierende geometrische Muster | Meditation, Spiritual |
| `liquid_blobs` | Flüssige MetaBall-Animation | Deep House, Liquid DnB |
| `neon_wave_circle` | Konzentrische Neon-Ringe | Trance, Progressive |
| `frequency_flower` | Organische Blumen-Petal Animation | Indie, Folk, Acoustic |
### Signature Pro Visualizer (6) — Neu in v2.0+
| Name | Beschreibung | Ideal für |
|------|--------------|-----------|
| `lumina_core` | Intelligenter Hybrid-Visualizer | Allrounder |
| `voice_flow` | Sprach-optimierte Visualisierung | Podcasts, Interviews |
| `spectrum_genesis` | Evolvierendes Spektrum-Design | Elektronische Musik |
| `speech_focus` | Fokus auf Sprachfrequenzen | Hörbücher, Vorträge |
| `bass_temple` | Bass-zentrierte Tempel-Architektur | Bass Music, Trap |
| `orchestral_swell` | Orchestrale Wellenbewegungen | Filmmusik, Klassik |

---

## 🤖 KI-Features
### Automatisierte Zitat-Extraktion
Nutzt Gemini 3.1 Flash-Lite für:
- **Audio-Transkription**: Wandelt Sprache zu Text mit Zeitstempeln
- **Key-Zitat-Erkennung**: Identifiziert die wichtigsten Passagen
- **Beat-Sync**: Synchronisiert Zitate mit musikalischen Highlights
```python
from src.gemini_integration import GeminiIntegration
gemini = GeminiIntegration()
# Transkription
transcript = gemini.transcribe_audio("podcast.mp3")
# Zitate extrahieren (max. 5 Key-Zitate)
quotes = gemini.extract_quotes("podcast.mp3", max_quotes=5)
for quote in quotes:
    print(f"[{quote.start_time:.1f}s - {quote.end_time:.1f}s]")
    print(f"{quote.text} ({quote.confidence*100:.0f}% Confidence)")
```
### Smart Parameter Matching
Die KI analysiert Audio-Eigenschaften und empfiehlt:
- Passenden Visualizer-Typ
- Optimierte Farbpaletten (basierend auf erkannter Tonart)
- Angepasste Parameter (Partikel-Dichte, Geschwindigkeit, Intensität)

---

## 🏗️ Architektur
```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Quote Overlays                                    │
│  → GPUTextRenderer mit SDF-Fonts                            │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Post-Processing                                   │
│  → Bloom, Grain, Vignette, Chromatic Aberration, LUTs       │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: GPU Visualization                                 │
│  → ModernGL Shader, 16 Visualizer, Real-time Preview        │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Audio Analysis                                    │
│  → librosa Features, Beat Detection, Voice Clarity          │
└─────────────────────────────────────────────────────────────┘
```
### Datenfluss
1. **Audio-Analyse**: Extrahiert RMS, Onset, Chroma, MFCC, Tempogram
2. **GPU-Rendering**: ModernGL Shader verarbeiten Features in Echtzeit
3. **Quote Overlay**: SDF-basiertes Text-Rendering mit Fade-Animation
4. **Post-Processing**: Color Grading und Effekte
5. **Video-Encoding**: FFmpeg mit Multi-Codec Support

---

## ⚙️ Konfiguration
### Config-Presets
Vordefinierte Presets im `config/` Ordner:
#### Musik-Presets
- `default.json` — Ausgewogene Standardeinstellungen
- `music_aggressive.json` — Hoher Kontrast, intensive Effekte
- `chromatic_dream.json` — Weiche Farben, Chromatic Aberration
- `neon_cyberpunk.json` — Cyan/Magenta Neon-Effekte
- `sacred_geometry.json` — Spirituelle Farbpalette
- `liquid_blobs.json` — Flüssige Blau/Pink Animation
- `neon_circle.json` — Grün/Rot konzentrische Ringe
- `flower_bloom.json` — Sanfte Pastellfarben
#### Podcast-Presets
- `podcast_minimal.json` — Sauber, minimalistisch
- `podcast_news.json` — Sachlich, professionell
- `podcast_interview.json` — Warm, einladend
- `podcast_story.json` — Dramatisch, atmosphärisch
- `podcast_mixed.json` — Ausgewogen für gemischte Formate
### Eigene Parameter
```bash
# Beispiel: Custom Parameter setzen
python main.py render audio.mp3 --visual particle_swarm \
  --param particle_count=200 \
  --param explosion_threshold=0.5 \
  --param trail_length=15 \
  -o custom.mp4
```

---

## 🧪 Testing
```bash
# Alle Tests ausführen (134 Tests)
pytest tests/ -v
# Spezifische Test-Suiten
pytest tests/test_gpu_renderer.py -v        # GPU Rendering
pytest tests/test_visuals.py -v             # Visualizer Tests
pytest tests/test_gemini_integration.py -v  # KI Integration
pytest tests/test_postprocess.py -v         # Post-Processing
pytest tests/test_quote_overlay.py -v       # Quote Overlays
```
### Coverage
| Modul | Coverage |
|-------|----------|
| `postprocess.py` | 100% |
| `gpu_preview.py` | 95% |
| `gpu_text_renderer.py` | 78% |
| `quote_overlay.py` | 93% |
| `types.py` | 100% |
| **Gesamt** | **77%** |

---

## 📁 Projektstruktur
```
audio-visualizer-pro/
├── main.py                     # CLI Entry Point
├── gui.py                      # DearPyGui Frontend
├── pyproject.toml              # Project Configuration
├── requirements.txt            # Python Dependencies
├── config/                     # JSON Presets
│   ├── schemas.py              # Pydantic Validation
│   ├── default.json
│   ├── music_aggressive.json
│   ├── podcast_interview.json
│   └── ...
├── src/
│   ├── __init__.py
│   ├── analyzer.py             # Audio Feature Extraction
│   ├── ai_matcher.py           # KI Parameter Matching
│   ├── beat_sync.py            # Beat Synchronization
│   ├── gemini_integration.py   # Gemini KI Client
│   ├── gpu_preview.py          # Live Preview Renderer
│   ├── gpu_renderer.py         # Batch GPU Renderer
│   ├── gpu_text_renderer.py    # SDF Text Rendering
│   ├── gpu_visualizers/        # 16 GPU Visualizer
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── spectrum_bars.py
│   │   ├── pulsing_core.py
│   │   ├── particle_swarm.py
│   │   ├── neon_oscilloscope.py
│   │   ├── chroma_field.py
│   │   ├── typographic.py
│   │   ├── sacred_mandala.py
│   │   ├── liquid_blobs.py
│   │   ├── neon_wave_circle.py
│   │   ├── frequency_flower.py
│   │   ├── lumina_core.py         # Signature Pro
│   │   ├── voice_flow.py          # Signature Pro
│   │   ├── spectrum_genesis.py    # Signature Pro
│   │   ├── speech_focus.py        # Signature Pro
│   │   ├── bass_temple.py         # Signature Pro
│   │   └── orchestral_swell.py    # Signature Pro
│   ├── local_transcription.py  # Lokale Transkription
│   ├── postprocess.py          # Post-Processing Effects
│   ├── quote_cache.py          # Quote Caching
│   ├── quote_overlay.py        # Quote Overlay Logic
│   ├── quote_refiner.py        # Quote Timestamp Refinement
│   └── types.py                # Pydantic Models
├── tests/                      # Test Suite (134 Tests)
│   ├── conftest.py
│   ├── test_ai_matcher.py
│   ├── test_analyzer.py
│   ├── test_beat_sync.py
│   ├── test_e2e.py
│   ├── test_gemini_integration.py
│   ├── test_gpu_preview.py
│   ├── test_gpu_renderer.py
│   ├── test_gpu_text_renderer.py
│   ├── test_postprocess.py
│   ├── test_quote_overlay.py
│   ├── test_quote_refiner.py
│   └── test_visuals.py
└── cognitive_core/             # Evo-Agent Framework
    ├── agents.md
    ├── system_prompt.md
    └── tool.md
```

---

## 💻 Systemanforderungen
| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| Python | 3.10 | 3.12 |
| RAM | 8 GB | 16 GB |
| GPU | OpenGL 3.3+ | Vulkan/DX12 |
| VRAM | 2 GB | 4 GB+ |
| Speicher | 500 MB | 1 GB+ |
| FFmpeg | 4.0+ | 6.0+ |

---

## 🛣️ Roadmap

### ✅ Abgeschlossen (v2.0 - v2.5)

#### 2024-2025 — Foundation & Core Features
- ✅ GPU-basiertes Rendering mit ModernGL/OpenGL
- ✅ 16 GPU-Visualizer implementiert (10 Classic + 6 Signature Pro)
- ✅ DearPyGui Premium UI mit Live-Preview
- ✅ Gemini KI-Integration (Transkription & Zitat-Extraktion)
- ✅ Test-Suite: 134 Tests mit 77% Coverage
- ✅ Post-Processing Pipeline (Bloom, Grain, Vignette, LUTs, Chromatic Aberration)
- ✅ SDF-basiertes Text-Rendering für Quote Overlays
- ✅ Multi-Codec Support (H.264, HEVC, ProRes)
- ✅ Evo-Agent Framework (`cognitive_core/`)
- ✅ 5 Skill-Spezifikationen implementiert

---

### 🎯 Aktuell (v2.6 - v3.0) — 2026

#### Q1-Q2 2026 — Stabilisierung & Usability ✓ Im Fokus
Da dieses Projekt von einem Ein-Personen-Team (mit KI-Unterstützung) entwickelt wird, liegt der Fokus auf **pragmatischen Verbesserungen** statt Feature-Creep:

- ✅ **Code-Qualität**: Refactoring für bessere Wartbarkeit ohne Programmier-Kenntnisse
- ✅ **Dokumentation**: Ausführliche Anleitungen für Nicht-Entwickler
- ✅ **Fehlerbehandlung**: Robuste Error-Messages für Endnutzer
- [ ] **One-Click Installer**: Vereinfachte Installation ohne manuelle Dependency-Konfiguration
- [ ] **GUI-Verbesserungen**: Intuitivere Bedienung für Power-User-Funktionen
- [ ] **Preset-Bibliothek**: 10+ zusätzliche vordefinierte Presets für häufige Anwendungsfälle

#### Q3-Q4 2026 — Geplante Erweiterungen (realistisch für Solo-Entwicklung)
- [ ] **Batch-Rendering**: Mehrere Audio-Dateien nacheinander verarbeiten
- [ ] **Auto-Update Mechanismus**: Einfache Updates ohne manuelle Git-Operationen
- [ ] **Video-Tutorials**: Schritt-für-Schritt Anleitungen für alle Kernfunktionen
- [ ] **Community-Support**: Discord/Forum für Nutzeraustausch

---

### 🔮 Langfristige Vision (v3.0+) — 2027+

#### Priorisierte Features (nur wenn wirklich benötigt)
Diese Features werden **nur implementiert**, wenn konkrete Nachfrage besteht:

- [ ] **Echtzeit-Audio-Input**: Mikrofon/Live-Stream Unterstützung
- [ ] **Multi-Track Support**: Separates Rendering von Audio-Stems
- [ ] **Custom Shader Editor**: GUI-basierter Editor für Visual-Anpassungen
- [ ] **Mobile Companion App**: iOS/Android für Remote-Steuerung

#### Experimentelle Features (KI-getrieben)
- [ ] **Style Transfer**: Musikvideo-Stil von Referenzvideos lernen
- [ ] **Generative Visuals**: KI-generierte Visualizer basierend auf Text-Beschreibung
- [ ] **Automatische Schnittsetzung**: KI-generierte Cuts synchron zur Musik

---

### ⚠️ Nicht Geplant (bewusste Entscheidungen)

Als Ein-Personen-Projekt mit Fokus auf **Stabilität und Einfachheit** werden folgende Features bewusst **nicht** verfolgt:

- ❌ Cloud-Rendering Pipeline (zu komplex, zu teuer)
- ❌ VR/AR Visualizer (zu spezialisiert)
- ❌ Unreal Engine Integration (Overkill für Use-Case)
- ❌ WebAssembly-Export (Performance-Einbußen inakzeptabel)
- ❌ Plugin Marketplace (Wartungsaufwand zu hoch)
- ❌ Trainierbare KI-Modelle (API-Lösung ist pragmatischer)

---

### 📊 Entwicklungs-Prinzipien

| Prinzip | Beschreibung |
|---------|--------------|
| **KI-First** | Alle Code-Änderungen werden primär durch KI generiert |
| **No-Code Friendly** | Features müssen ohne Programmierkenntnisse nutzbar sein |
| **Stability > Features** | Lieber weniger Features, dafür stabil und gut getestet |
| **Documentation Driven** | Jede Funktion wird ausführlich dokumentiert |
| **Pragmatic Evolution** | Nur Features implementieren, die wirklich gebraucht werden |

---

### 📈 Meilensteine

| Version | Ziel | Status | Zeitraum |
|---------|------|--------|-----------|
| v2.0 | GPU-Rendering Launch | ✅ Abgeschlossen | 2024 |
| v2.1 | Testing & Stability | ✅ Abgeschlossen | Q1 2025 |
| v2.2-v2.5 | Evo-Agent Framework, Quality Improvements | ✅ Abgeschlossen | 2025 |
| v2.6 | Usability & One-Click Setup | 🔄 In Arbeit | Q2 2026 |
| v2.7 | Batch Processing & Auto-Updates | 📅 Geplant | Q3 2026 |
| v3.0 | Documentation Complete & Community Ready | 📅 Geplant | Q4 2026 |
| v3.1+ | Community-Driven Features | 💭 Evaluierung | 2027+ |

---

### 💡 Entwicklungshintergrund

**Team-Struktur:**
- 👤 **Projektinitiator**: Keine Programmierkenntnisse, fokussiert auf Vision, Testing & Dokumentation
- 🤖 **KI-Assistenten**: Code-Generierung, Refactoring, Testing, Debugging

**Entwicklungs-Geschwindigkeit:**
- Realistische Feature-Umsetzung: 1-2 größere Features pro Quartal
- Fokus auf Qualität statt Quantität
- Alle Änderungen werden umfassend getestet (77%+ Coverage)
- Dokumentation hat gleiche Priorität wie Code

**Warum dieser Ansatz?**
Dieses Projekt beweist, dass moderne Softwareentwicklung auch ohne traditionelle Programmierkenntnisse möglich ist. Durch die Kombination aus menschlicher Vision, domänenspezifischem Wissen und KI-gestützter Implementierung entstehen robuste, professionelle Tools – in realistischer Geschwindigkeit und mit nachhaltiger Wartbarkeit.

---


## 🤝 Contributing

Wir freuen uns über Beiträge! Bitte beachte folgende Richtlinien:

### Entwicklungsumgebung einrichten
```bash
# Fork klonen
git clone https://github.com/YOUR_USERNAME/audio-visualizer-pro.git
cd audio-visualizer-pro
# Development Dependencies
pip install -e ".[dev]"
# Pre-Commit Hooks
black src/ tests/
flake8 src/ tests/
```

### Pull Request Prozess
1. Issue erstellen oder existierendes kommentieren
2. Feature-Branch erstellen (`git checkout -b feature/mein-feature`)
3. Änderungen commiten (`git commit -m 'feat: neues Feature hinzugefügt'`)
4. Tests ausführen (`pytest tests/ -v`)
5. Branch pushen (`git push origin feature/mein-feature`)
6. Pull Request öffnen

### Code Style
- **Sprache**: Kommentare und Dokumentation auf Deutsch
- **Formatierung**: Black-konform (88 Zeichen pro Zeile)
- **Typisierung**: Type Hints für alle Funktionen
- **Tests**: Mindestens 80% Coverage für neue Features

---

## 📄 Lizenz
MIT License — Siehe [LICENSE](LICENSE) für Details.

---

## 🙏 Credits
| Projekt | Zweck |
|---------|-------|
| [ModernGL](https://moderngl.readthedocs.io/) | GPU Rendering Engine |
| [librosa](https://librosa.org/) | Audio-Analyse |
| [DearPyGui](https://dearpygui.readthedocs.io/) | GUI Framework |
| [Gemini API](https://ai.google.dev/) | KI Transkription |
| [FFmpeg](https://ffmpeg.org/) | Video Encoding |
| [Pydantic](https://docs.pydantic.dev/) | Datenvalidierung |

---

## 📬 Support
- **Issues**: [GitHub Issues](https://github.com/audio-visualizer-pro/audio-visualizer-pro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/audio-visualizer-pro/audio-visualizer-pro/discussions)
- **E-Mail**: support@audio-visualizer.pro

---
<div align="center">
**Audio Visualizer Pro v2.1.0**
Mit ❤️ erstellt vom Audio Visualizer Pro Team
[Documentation](https://github.com/audio-visualizer-pro/audio-visualizer-pro/blob/main/README.md) · [Changelog](CHANGELOG.md) · [Quickstart](QUICKSTART.md)
</div>
