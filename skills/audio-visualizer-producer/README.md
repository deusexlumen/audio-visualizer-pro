# Audio Visualizer Producer: Agenten-Skill

Installierbarer Skill für KI-Agenten, die den bestehenden Audio Visualizer Pro über die Python-CLI kontrollieren und ein **prüfbares** Musikvideo/Podcast-Visual ausliefern sollen.

Basis: https://github.com/deusexlumen/audio-visualizer-pro

## Installation

1. Den Ordner `audio-visualizer-producer` mit `SKILL.md`, `references/` und `scripts/` in das Skills-Verzeichnis des jeweiligen Agenten kopieren (z. B. `.agents/skills/audio-visualizer-producer/` bzw. dessen dokumentiertes Skills-Verzeichnis).
2. Audio Visualizer Pro separat installieren/klonen und laut `requirements.lock` vorbereiten.
3. Im Agenten mit „Verwende den Skill audio-visualizer-producer für diese MP3“ aufrufen.

Der Skill enthält **nicht** die Renderer-Engine, FFmpeg, Audiodateien, Bildressourcen oder API-Schlüssel. Die Laufzeit benötigt Python 3.10+, FFmpeg/FFprobe, eine geeignete OpenGL-Umgebung sowie die Python-Abhängigkeiten des Originalprojekts. Der Skill selbst verwendet für seine Hilfsskripte nur die Python-Standardbibliothek.

## Befehle

```bash
python scripts/preflight.py --repo "C:/Pfad/audio-visualizer-pro" --audio "C:/Musik/song.mp3"
python scripts/verify_export.py --video "C:/Exports/final.mp4" --source-audio "C:/Musik/song.mp3" --width 1920 --height 1080 --fps 30 --full-decode
```

Schrittweise Verwendung siehe `SKILL.md`, reproduzierbare CLI-Details `references/cli.md`, Qualitäts- und Stop-Kriterien `references/qa.md`.

## Umfang / Abgrenzung

Dieser Skill ist speziell für `audio-visualizer-pro` (Python/ModernGL), nicht für das separate React-Projekt `audio-reactive-Visualizer-`. Er ersetzt nicht die `SKILL.md` im Wurzelverzeichnis, die `programm-design` enthält. Er führt keinen Upload aus und nimmt keine unbelegten Qualitätsgarantien vor.