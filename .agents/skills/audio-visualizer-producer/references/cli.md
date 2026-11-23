# CLI-Referenz (bestätigt am GitHub-Repository, 22.11.2026)

Quelle: `main.py`, `src/gpu_visualizers/__init__.py`, `src/studio/engine.py`, `config/*.json` im Repo `deusexlumen/audio-visualizer-pro` (Branch master). Beim nächsten Einsatz gegen den tatsächlichen Checkout prüfen.

## Vorbereitung

```bash
git clone https://github.com/deusexlumen/audio-visualizer-pro.git
cd audio-visualizer-pro
python -m pip install -r requirements.lock
python main.py --help
python main.py list-visuals
python main.py analyze "song.mp3" --fps 30
```

Ohne vorhandene GPU/OpenGL-3.3+-Kontexte kann der Produktionsrenderer scheitern. `moderngl`-Installation allein ist kein GPU-Test. FFmpeg muss für CLI-Exporte verfügbar sein. Gemini/API-Key ist für normales Rendern nicht nötig.

## Produktionskommandos

```bash
# Kurzvorschau
python main.py render "song.mp3" --visual lumina_core --preview --preview-duration 5 --resolution 854x480 --fps 30 -o "preview.mp4"

# Fertiger 16:9-Export
python main.py render "song.mp3" --visual lumina_core --resolution 1920x1080 --fps 30 --codec h264 --quality high -o "final.mp4"

# Mit Hintergrund und einem bestätigten Visualizer-Parameter
python main.py render "song.mp3" --visual pulsing_core --background-image "cover.png" --param intensity=1.2 -o "final.mp4"

# Über geprüfte Projekt-Config
python main.py render "song.mp3" --config config/music_aggressive.json -o "custom.mp4"

# Mehrere Größen (nicht gleichzeitig mehrere Layouts)
python main.py render-multi "song.mp3" --visual nebula_drift --resolutions 1920x1080,1280x720 --output-prefix export

# Batch-Jobs: JSON-Array mit audio/visual/output (plus optionale Parameter)
python main.py batch "jobs.json"
```

Verfügbare Optionen des `render`-Befehls: `--visual`, `--output`, `--config`, `--resolution`, `--fps`, `--preview`, `--preview-duration`, `--background-image`, `--background-blur`, `--background-vignette`, `--background-opacity`, `--background-color`, `--codec` (`h264`, `hevc`, `prores`), `--quality` (`low`, `medium`, `high`, `lossless`), wiederholtes `--param`, `--intro`, `--intro-fade`, `--studio`, `--studio-dry`, `--studio-strict`.

## Besonderheiten

1. `--config`: `main.py::_resolve_config` setzt `visual.type`, `visual.resolution`, `visual.fps` aus der Datei, auch wenn der Benutzer scheinbar dieselben Werte per CLI übergeben hat. Explizite `--param`-Werte werden auf `visual.params` gemerged; `-o` überschreibt den JSON-Ausgabepfad, sofern nicht der Default `output.mp4` bleibt. Bei deterministischen Agentenjobs **einzelne CLI-Flags ohne Config** bevorzugen oder effektive Config aufschreiben.
2. Das `--studio`-Verfahren analysiert und löst Metriken, erstellt ein `*.studio.json`-Sidecar, rendert aber in einem gegenwärtig auf 854×480 festgelegten Preview-Pfad. Nicht als bequeme Full-HD-Automatik behandeln. `--studio-dry` erstellt keinen Videoexport. Studio-Kommandos dürfen kein Produktions-`DONE` begründen.
3. `--preview` rendert nur den Anfang, sofern nicht vorher ein Audioauszug erstellt wird. Keinen Preview-Clip als komplettes Video deklarieren.
4. `batch` überspringt laut `main.py` fehlerhafte oder fehlende Input-Jobs zum Teil und läuft weiter. Jeder Job muss **individuell** verifiziert werden, statt auf den CLI-Prozesscode allein zu vertrauen.
5. Die Visualizer-Liste ist dynamisch; `list-visuals` ist verbindlicher als veraltete tabellarische Listen in `AGENTS.md` und `QUICKSTART.md`.
6. Repository-`SKILL.md` heißt `programm-design` und behandelt Entwicklungsplanung. Der Produktionsskill hier ist davon unabhängig.

## Mögliche Einsatzszenarien

- Musikvideo / Rap: `spectrum_bars`, `lumina_core`, `particle_swarm`
- Elektronisch / pulsierend: `pulsing_core`, `bass_temple`, `neon_wave_circle`
- Atmosphäre: `nebula_drift`, `chroma_field`, `frequency_flower`
- Podcast / Sprache: `voice_flow`, `speech_focus`, `aurora_voice`

Diese Zuordnung ist eine **Designheuristik**, keine automatische Genre-Erkennung oder Qualitätsgarantie.