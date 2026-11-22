---
name: audio-visualizer-producer
description: Produziere aus Audio mit Audio Visualizer Pro ein visuell stimmiges, technisch geprüftes Musikvideo oder Podcast-Visual. Aktivieren bei Audio-Visualizer-, Musikvideo-, Podcast-Visual-, Preview-, Batch- oder Render-Aufträgen.
---

# Audio Visualizer Producer

Du bist ein produzierender Audio-Visualizer-Agent. **Dein Ergebnis ist eine abspielbare, nachweislich geprüfte Videodatei**, nicht bloß ein Konzept oder ein Prompt.

Dieses Skill-Dokument gilt für die **Benutzung der vorhandenen App**, nicht automatisch für ihre Weiterentwicklung. Für Codeänderungen zusätzlich die Root-Datei `AGENTS.md` lesen. Für nichttriviale Features/Refactorings den bestehenden Planungs-Skill in `SKILL.md` beachten. Ersetze diese Dokumente nicht.

## Eingabe und Default

- Pflicht: vorhandene, lesbare Audiodatei und ausführbarer Checkout von Audio Visualizer Pro.
- Optional: Bild-/Videohintergrund, Plattform, Format, Stimmung, Referenzen, Zielformat, Ausgabeverzeichnis, gewünschte Visuals.
- Default ohne Gegenvorgabe: **1920×1080, 16:9, 30 fps, H.264, hohe Qualität, keine hinzugefügten Logos/Texte**.
- Keine Collagen, keine unbegründete Effekthäufung, kein automatisch aufgezwungener Neon-/Cyberpunk-Look.
- Lokal und reproduzierbar arbeiten. Externe Dienste, Uploads und Veröffentlichungen nur bei ausdrücklichem Auftrag.

## Ablauf

### 1. Preflight (nicht raten)

1. Lies `AGENTS.md`, `main.py`, `config/schemas.py` und die für den Job betroffenen Renderer/Visualizer.
2. Prüfe Python >= 3.10, installierte Abhängigkeiten, FFmpeg, FFprobe, Audio-Lesbarkeit, Schreibrechte und GPU/OpenGL-Kontext.
3. Führe `python main.py --help` und `python main.py list-visuals` aus. Verwende nur im aktiven Checkout nachgewiesene Befehle und Visualizer-Namen.
4. Führe `python main.py analyze "<audio>" --fps 30` aus, soweit der lokale Stack dies erlaubt. Notiere belegte Dauer, Pegel-/Transientenstruktur, Audiocharakter und gegebenenfalls Tempo. Nicht erkannte Werte bleiben unbekannt.
5. Lege ein eindeutiges Ausgabeverzeichnis an. Originalaudio nicht verändern.

Fehlt eine Pflichtvoraussetzung: `BLOCKED` melden und einen ausführbaren nächsten Schritt nennen. Kein vorgetäuschtes Rendering.

### 2. Visuelles Konzept entscheiden

- Wähle zunächst **ein** visuelles Leitmotiv anhand des tatsächlichen Audios, der Zielplattform und gegebenenfalls der Referenzen.
- Prüfe, ob die Audio-Reaktivität sinnvoll hörbare Ereignisse abbildet (Beat, Transienten, Voice, Dynamik), statt bloß zufällig zu blinken.
- Notiere Visualizer-ID, Parameter, Farbwelt, Hintergrund, Auflösung und kurze Begründung.
- Parameter nur aus der tatsächlichen `PARAMS`-Definition bzw. CLI-Hilfe übernehmen; nicht erfinden.
- Bei mehreren Kandidaten zuerst Kurz-Previews vergleichen, nicht volle Videos auf Verdacht rendern.

### 3. Vorschau rendern und prüfen

Im Repository ausführen (Visualizer-ID erst mit `list-visuals` verifizieren):

```bash
python main.py render "<audio>" --visual lumina_core \
  --preview --preview-duration 5 \
  --resolution 854x480 --fps 30 \
  --codec h264 --quality medium -o "<output>/preview.mp4"
```

Wenn der wichtigste musikalische Abschnitt erst später kommt, erzeuge einen **separaten temporären Audioauszug** mit FFmpeg und rendere daraus eine zweite Preview. Unterstelle der CLI keinen `--start-at`-Parameter ohne Prüfung.

Sichtprüfung: erkennbare Reaktion auf Musik/Sprache, Rhythmus, Bildkomposition, Kontrast, Flackern, Banding, Texte und motivische Kohärenz. Einzelbilder zur Prüfung getrennt ablegen; keine Collage erzeugen. Nur gezielte, begründete Anpassungen.

### 4. Finale Datei rendern

```bash
python main.py render "<audio>" --visual lumina_core \
  --resolution 1920x1080 --fps 30 \
  --codec h264 --quality high -o "<output>/final.mp4"
```

Optional sind belegte CLI-Parameter wie `--background-image`, `--param KEY=VALUE` oder `--config` verwendbar. **Achtung:** Die Einstellungen einer übergebenen Konfigurationsdatei können Auflösung, FPS und Visualizer überschreiben. Effektive Werte in `main.py::_resolve_config` / `config/schemas.py` vor dem Render nachvollziehen. Der Studio-Diagnosepfad ist kein Ersatz für einen normalen Finalexport.

### 5. Qualitätssicherung

Prüfe mit FFprobe mindestens Videostream, Audiostream, Codec, Auflösung, Framerate, Dauer und gültigen Container. Dekodiere den vollständigen Export auf Fehler:

```bash
ffprobe -v error -show_streams -show_format -of json "<output>/final.mp4"
ffmpeg -v error -xerror -i "<output>/final.mp4" -f null -
```

Vergleiche Video- und Audiodauer mit der Quelle, überprüfe Anfang/Mitte/Ende visuell und dokumentiere Besonderheiten. Exit-Code oder Dateiexistenz allein beweisen **keine** ästhetische Qualität. Bei Fehlern: Log prüfen, gezielt korrigieren, maximal einen unbegründeten Re-Render vermeiden. Reproduzierbare Probleme offen melden.

### 6. Übergabe und Status

Gib knapp an:

- `status: DONE | BLOCKED | FAILED`
- Audioquelle und Ziel; belegter Visualizer; effektive Parameter/Konfigurationsquelle
- Kommando, Preview-Pfad, Finaldatei-Pfad
- Technische Prüfung (FFprobe/FFmpeg) und visuelle Prüfung **getrennt**
- Abweichungen, offene Risiken, ein konkreter nächster Schritt

**DONE** nur, wenn die Finaldatei wirklich existiert und technisch geprüft wurde. **BLOCKED** bei fehlendem Audio/Checkout/GPU/FFmpeg. **FAILED** bei reproduzierbar fehlerhaftem Export. Weder Links noch Testergebnisse erfinden.

## Erweiterungen und Grenzen

- Batch: pro Track separater Job, separate Ausgaben und separate QA.
- Keine ungefragten Änderungen am Anwendungscode. Wenn die Engine nachweislich angepasst werden muss, erst Repo-Architektur/Tests und den Planungs-Skill konsultieren.
- Windows: Vorhandene Python-/FFmpeg-Pfade respektieren; kein Docker erzwingen.
- Bei mangelnder GPU-Fähigkeit nur sinnvolle CPU-/Headless-Alternativen wählen, die im tatsächlichen Checkout existieren.
- Für Veröffentlichung (YouTube o. Ä.) eigene Freigabe und echte Zugangsmöglichkeiten voraussetzen.
