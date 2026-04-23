# Audio Visualizer Pro - Schnellstart 🚀

Dein KI-gestütztes Audio-Visualisierungs-System ist bereit! Diese Anleitung zeigt dir, wie du sofort loslegen kannst.

---

## 🖥️ Grafische Oberfläche (GUI) — Einfachste Option!

Die GUI ist die benutzerfreundlichste Art, den Visualizer zu nutzen — mit Drag & Drop, Live-Analyse und One-Click-Render.

### Starten

**Windows:**
```bash
# Doppelklicke auf:
start_gui.bat

# Oder via Python:
python start_gui.py
```

**macOS / Linux:**
```bash
python start_gui.py
```

Die GUI öffnet sich automatisch in deinem Browser unter `http://localhost:8501`

---

## ✅ Was wurde bereits eingerichtet

- [x] **10 integrierte Visualizer** (Pulsing Core, Spectrum Bars, Chroma Field, Particle Swarm, Typographic, Neon Oscilloscope, Sacred Mandala, Liquid Blobs, Neon Wave Circle, Frequency Flower)
- [x] **🤖 KI-Auto-Modus** — Smart Matcher analysiert Audio und empfiehlt Visualizer + Farben + Parameter
- [x] **💬 Gemini KI-Integration** — Automatische Key-Zitat-Extraktion aus Audio
- [x] **✏️ Zitat-Review** — Bearbeite, aktiviere/deaktiviere Zitate vor dem Rendering
- [x] **🎙️ Podcast-Genre-Presets** — News, Interview, Storytelling, Mixed
- [x] **🔌 Plugin-System** mit `@register_visualizer` Decorator
- [x] **📊 Audio-Analyse** mit Beat-Erkennung, Key-Erkennung, Mode-Detection
- [x] **⚡ Aggressives Caching** (`.cache/` Ordner)
- [x] **🎨 Post-Processing** (Film Grain, Vignette, Chromatic Aberration)
- [x] **🎬 FFmpeg-Integration** für professionelles Video-Encoding
- [x] **📈 Echte Fortschrittsanzeige** in der GUI
- [x] **🧪 42 Tests** — Vollständige Test-Abdeckung

---

## 🛠️ Voraussetzungen

### FFmpeg muss installiert sein

**Windows:**
1. Lade FFmpeg von https://ffmpeg.org/download.html herunter
2. Entpacke es und füge den `bin` Ordner zu deinem PATH hinzu
3. Teste: `ffmpeg -version`

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

### Python-Abhängigkeiten

```bash
pip install -r requirements.txt
```

### Optional: Gemini API Key (für KI-Zitate)

1. Erstelle einen kostenlosen Key bei [Google AI Studio](https://aistudio.google.com/)
2. Speichere ihn in einer `.env` Datei im Projekt-Ordner:
   ```
   GEMINI_API_KEY=dein_key_hier
   ```

---

## 🚀 Erste Schritte

### Weg 1: GUI (Empfohlen für Einsteiger)

```bash
python start_gui.py
```

**Schritt-für-Schritt im Browser:**

1. **Audio hochladen** — Ziehe eine MP3/WAV/FLAC-Datei in die Drop-Zone
2. **Modus wählen:**
   - **🤖 Auto-Modus** — Die KI analysiert dein Audio und wählt alles automatisch
   - **🎙️ Podcast-Genre** — Optimierte Presets für Podcast-Formate
   - **🎨 Manuell** — Alle 10 Visualizer einzeln auswählbar
   - **📋 Config-Preset** — JSON-Presets aus dem `config/`-Ordner
3. **Einstellungen:** Auflösung und FPS wählen
4. **(Optional) Key-Zitate extrahieren:**
   - Klicke auf "🔮 Key-Zitate extrahieren"
   - Gemini analysiert dein Audio und findet die besten Zitate
   - Bearbeite, aktiviere oder deaktiviere Zitate
5. **🎬 Render-Button** klicken
6. **Fertig!** Video ansehen und herunterladen

### Weg 2: Kommandozeile

```bash
# 1. Audio analysieren
python main.py analyze dein_lied.mp3

# 2. Verfügbare Visualizer anzeigen
python main.py list-visuals

# 3. Schnelle Vorschau (5 Sekunden, 480p)
python main.py render dein_lied.mp3 --visual pulsing_core --preview

# 4. Volles Video rendern
python main.py render dein_lied.mp3 --visual spectrum_bars -o output.mp4

# 5. Mit Config-Preset rendern
python main.py render dein_lied.mp3 --config config/music_aggressive.json
```

---

## 🎨 Verfügbare Visualizer

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

## ⚙️ Config-Presets

### Musik-Presets

| Preset | Beschreibung |
|--------|--------------|
| `default.json` | Ausgewogene Einstellungen für den Allgemeingebrauch |
| `music_aggressive.json` | Hoher Kontrast, Film Grain, Vignette für aggressive Musik |
| `chromatic_dream.json` | Weiche Farben, Chromatic Aberration für Ambient |
| `neon_cyberpunk.json` | Cyan/Magenta Neon-Effekte für Synthwave |
| `sacred_geometry.json` | Lila/Orange für spirituelle/ambient Musik |
| `liquid_blobs.json` | Flüssige Blau/Pink Blobs für elektronische Musik |
| `neon_circle.json` | Grün/Rot konzentrische Ringe für EDM |
| `flower_bloom.json` | Sanfte Pastellfarben für Indie/Folk |

### 🎙️ Podcast-Presets (Neu!)

| Preset | Beschreibung |
|--------|--------------|
| `podcast_minimal.json` | Sauber, minimalistisch mit Wellenform |
| `podcast_news.json` | Sachlich, professionell, hoher Kontrast |
| `podcast_interview.json` | Warm, einladend, Gespräch-optimiert |
| `podcast_story.json` | Dramatisch, atmosphärisch, Storytelling-optimiert |
| `podcast_mixed.json` | Ausgewogen für gemischte Formate |

---

## 🧪 Tests ausführen

```bash
# Alle Tests (42 Stück)
pytest tests/ -v

# Nur Visualizer-Tests (schnell)
pytest tests/test_visuals.py -v

# Nur Analyzer-Tests (braucht länger)
pytest tests/test_analyzer.py -v

# KI-Tests
pytest tests/test_ai_matcher.py -v

# Quote Overlay Tests
pytest tests/test_quote_overlay.py -v

# Gemini Integration Tests
pytest tests/test_gemini_integration.py -v
```

---

## 🤖 KI-Auto-Modus nutzen

### In der GUI
1. "🤖 Auto-Modus (KI empfiehlt)" wählen
2. Audio hochladen
3. Die KI analysiert automatisch und zeigt die Empfehlung
4. Mit Klick auf "🎬 Video rendern" loslegen

### Über CLI
```bash
# Config mit KI-Empfehlung generieren (via GUI oder Script)
# Dann rendern:
python main.py render podcast.mp3 --config config/auto_recommended.json
```

**Was die KI analysiert:**
- Dynamik-Range, Beat-Dichte, Tempo (BPM)
- Key & Mode (Dur/Moll, Musik/Sprache/Hybrid)

**Was die KI empfiehlt:**
- Besten Visualizer (z.B. `typographic` für Sprache)
- Farbpalette passend zum Key
- Optimierte Parameter (Partikel-Anzahl, Geschwindigkeit, etc.)

---

## 💬 KI-Zitat-Workflow (Neu!)

Der perfekte Workflow für Podcast-Visuals mit Text-Overlays:

### Schritt 1: Zitate extrahieren
```bash
# In der GUI: "🔮 Key-Zitate extrahieren" klicken
# Oder per Code:
from src.gemini_integration import GeminiIntegration

gemini = GeminiIntegration()
quotes = gemini.extract_quotes("podcast.mp3", max_quotes=5)
for q in quotes:
    print(f"[{q.start_time:.1f}s] {q.text} ({q.confidence*100:.0f}%)")
```

### Schritt 2: Zitate reviewen (GUI)
- **Bearbeiten**: Text direkt im Textfeld korrigieren
- **Aktivieren/Deaktivieren**: Checkbox pro Zitat
- **Nur aktivierte Zitate** werden ins Video gerendert

### Schritt 3: Video mit Overlays rendern
- Die Zitate erscheinen automatisch als elegante Text-Overlays
- **Zeitbasiert**: Erscheinen bei `start_time`, verschwinden bei `end_time`
- **Animiert**: Sanftes Fade-In/Out
- **Stilvoll**: Abgerundete Box mit Schatten

---

## 🎨 Eigenen Visualizer erstellen

### Template generieren

```bash
python main.py create-template mein_visualizer
```

### Implementieren

Bearbeite `src/visuals/mein_visualizer.py`:

```python
@register_visualizer("mein_visualizer")
class MeinVisualizer(BaseVisualizer):
    def setup(self):
        self.center = (self.width // 2, self.height // 2)
    
    def render_frame(self, frame_idx: int) -> np.ndarray:
        # Features holen
        f = self.get_feature_at_frame(frame_idx)
        rms = f['rms']        # 0.0-1.0 Lautstärke
        onset = f['onset']    # 0.0-1.0 Beat-Trigger
        chroma = f['chroma']  # 12 Werte für Halbtöne
        
        # Deine Zeichen-Logik...
        img = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Beispiel: Kreis mit RMS-Größe
        radius = int(50 + rms * 100)
        draw.ellipse([self.center[0]-radius, self.center[1]-radius,
                      self.center[0]+radius, self.center[1]+radius],
                     fill=(255, 0, 100))
        
        return np.array(img)
```

### Testen

```bash
python main.py render dein_lied.mp3 --visual mein_visualizer --preview
```

---

## 📊 Feature-Keys Referenz

| Key | Bereich | Verwendung |
|-----|---------|------------|
| `rms` | 0.0-1.0 | Lautstärke → Größe/Opazität |
| `onset` | 0.0-1.0 | Beats → Trigger/Explosionen |
| `chroma` | Array[12] | Tonart → Farben (C, C#, D, ...) |
| `spectral_centroid` | 0.0-1.0 | Helligkeit/Detail |
| `spectral_rolloff` | 0.0-1.0 | Bandbreite |
| `zero_crossing_rate` | 0.0-1.0 | Noise vs Tonal |
| `progress` | 0.0-1.0 | Zeit-Fortschritt |

---

## 🎯 Workflow-Beispiele

### Podcast-Visual für YouTube erstellen

```bash
# 1. GUI starten
python start_gui.py

# 2. Im Browser:
#    - Audio hochladen
#    - "Podcast-Genre" → z.B. "Interview" wählen
#    - "Key-Zitate extrahieren" klicken
#    - Zitate reviewen und aktivieren
#    - "Video rendern" klicken

# Oder komplett via CLI:
python main.py render podcast.mp3 --config config/podcast_interview.json -o podcast_visual.mp4
```

### Musikvideo erstellen

```bash
# 1. Audio analysieren
python main.py analyze song.mp3

# 2. Vorschau mit verschiedenen Visualizern testen
python main.py render song.mp3 --visual pulsing_core --preview
python main.py render song.mp3 --visual spectrum_bars --preview

# 3. Besten Visualizer wählen und volles Video rendern
python main.py render song.mp3 --visual spectrum_bars -o music_video.mp4
```

### Kreatives Projekt mit Custom Config

```bash
# Config-Template erstellen
python main.py create-config --output my_config.json

# Config anpassen (Farben, Effekte, etc.)
# ... editiere my_config.json ...

# Mit Custom Config rendern
python main.py render song.mp3 --config my_config.json
```

---

## 💡 Performance-Tipps

1. **Immer Vorschau zuerst**: Nutze `--preview` für schnelles Testen (5 Sekunden, 480p)
2. **Caching**: Audio-Analyse wird automatisch gecached (`.cache/audio_features/`)
3. **Niedrigere FPS**: 30fps statt 60fps für schnelleres Rendering
4. **Niedrigere Auflösung**: Starte mit 1280x720 für schnellere Tests
5. **KI-Auto-Modus**: Spart Zeit bei der Visualizer-Auswahl

---

## 🆘 Troubleshooting

### FFmpeg nicht gefunden
```
Fehler: FFmpeg nicht installiert oder nicht im PATH
```
**Lösung**: FFmpeg installieren und zu PATH hinzufügen (siehe Voraussetzungen)

### Audio-Datei nicht gefunden
```
FileNotFoundError: Audio nicht gefunden
```
**Lösung**: Überprüfe den Dateipfad, verwende absolute Pfade wenn nötig

### ImportError: No module named 'librosa'
```
ModuleNotFoundError: No module named 'librosa'
```
**Lösung**: `pip install -r requirements.txt`

### Visualizer wird nicht gefunden
```
ValueError: Unbekannter Visualizer: xxx
```
**Lösung**: Überprüfe den Namen mit `python main.py list-visuals`

### Gemini API Key fehlt
```
ValueError: Gemini API Key fehlt
```
**Lösung**: Erstelle einen Key bei [Google AI Studio](https://aistudio.google.com/) und speichere ihn in `.env`:
```
GEMINI_API_KEY=dein_key_hier
```

---

**🎉 Fertig! Viel Spaß beim Erstellen von KI-gestützten Audio-Visualisierungen!**
