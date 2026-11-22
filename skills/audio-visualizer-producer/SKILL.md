---
name: audio-visualizer-producer
description: Erstellt, prüft und liefert audio-reaktive Musikvideos oder Podcast-Visuals mit deusexlumen/audio-visualizer-pro. Verwenden bei Anfragen wie „Visualisiere diesen Track“, „Erzeuge ein Musikvideo mit dem Visualizer“, „Wähle Presets“, „Rendere MP4“, „prüfe den Videoexport“ oder „produziere Audio-Visualizer im Batch“. Führt einen messbaren Preflight-Preview-Render-QA-Workflow aus, statt nur Visualizer-Ideen zu beschreiben.
---

# Audio Visualizer Producer

## Auftrag

Erzeuge **ein tatsächlich verwendbares, technisch geprüftes Video** aus einer bereitgestellten Audiodatei, mit dem bestehenden Open-Source-Projekt `deusexlumen/audio-visualizer-pro`. Der Skill ist eine **Arbeitsanweisung für Agenten**, kein Ersatz für den Renderer. Er benutzt nachweislich vorhandene CLI-Befehle, verändert die Engine nicht und führt ohne Dateizugriff oder Rendering-Tools keine fiktiven Arbeitsschritte aus.

**Quellrepository:** https://github.com/deusexlumen/audio-visualizer-pro (Standardbranch `master` zum Zeitpunkt der Erstellung)

**Verwende diesen Skill für Produktion und Export, nicht für größere Änderungen am Anwendungscode.** Bei Engine-Änderungen zuerst `AGENTS.md` und die einschlägigen Tests im Repository lesen. Die existierende `SKILL.md` im Repo ist ein *anderer* Skill (`programm-design`); nicht ersetzen.

## Eingaben und Defaults

Benötigt: lesbare lokale Audiodatei und Zugriff auf einen ausführbaren Checkout von Audio Visualizer Pro. Ermittle zusätzlich, falls vorhanden: gewünschtes Format, Plattform, Bild/Video als Hintergrund, Mood, gewünschtes visuelles Thema, explizite Textwünsche, Exportprofil, Zielfolder.

Wenn der Auftrag keine Angaben enthält, nutze **16:9, 1920×1080, 30 fps, H.264, hohe Qualität, kein zusätzlicher Text/Logo, ein einzelnes visuelles Konzept**. Nicht automatisch externe KI-Dienste oder Upload-Plattformen einschalten. Bevorzuge lokale Werkzeuge. Erzeuge keine Collagen; Preview-Standbilder bleiben separate Dateien.

Falls kein lokaler Checkout, keine Audiodatei, kein lauffähiger Render-Stack oder keine zugängliche GPU vorhanden ist: erkläre den Blocker, liefere ein ausführbares Kommando/Jobpaket und markiere das Ergebnis `BLOCKED`, nicht `DONE`.

## Ablauf (Schritt für Schritt)

### 1. Repository und Arbeitsumgebung prüfen

1. Prüfe `main.py`, `requirements.lock`, `src/gpu_renderer.py`, `src/gpu_visualizers/__init__.py`, `AGENTS.md` im Checkout. Verlasse dich nicht auf veraltete Anweisungen in `QUICKSTART.md`.
2. Stelle sicher, dass Python >=3.10, die Projekt-Dependencies, FFmpeg und FFprobe bereitstehen. Installer-GUI ist **nicht** dasselbe wie die Python-CLI.
3. `python scripts/preflight.py --repo <repo> --audio <audio>` (Scriptpfad relativ zu **diesem Skill**); validiere den JSON-Bericht. Für automatisierte Agenten den Prozess-Abbruchcode beachten.
4. Im Repository `python main.py list-visuals` ausführen. Nenne **nur** Visualizer, die dieser konkrete Checkout wirklich auflistet.
5. `python main.py analyze "<audio>" --fps 30` ausführen; Tempo, Dauer, dynamische Intensität und Sprache/Musik dokumentieren. Nie BPM, Key oder Songstruktur erfinden.

### 2. Konzept wählen (kein Preset-Roulette)

- Ein dominanter Visualizer für ein klares Erscheinungsbild; experimentell mehrere Varianten nur mit **kleinen Previews**, nie sofort mehreren Vollrenders.
- Für rhythmische elektronische Musik `pulsing_core`, `bass_temple` oder `lumina_core` prüfen; für Hip-Hop `spectrum_bars` oder `particle_swarm`; für Ambiente `nebula_drift`/`chroma_field`; für Sprache `voice_flow`/`speech_focus`. Namen zuerst gegen `list-visuals` validieren.
- Audio-Reaktivität muss in Rhythmus und Intensität sichtbar sein. Kein bloß dekoratives Hintergrund-Flackern.
- Farben und Bildsprache richten sich nach dem konkreten Auftrag. Keine pauschalen Neon-/Cyberpunk-Standards.
- Vor dem Render **eine prüfbare Entscheidung** notieren: Visualizer, FPS, Auflösung, Palette, Parameter, Hintergrund, warum das zum Track passt.

### 3. Testen: kurzer Clip, messbares Feedback

```bash
# Arbeitsverzeichnis: Root des Visualizer-Checkouts
python main.py render "<audio>" --visual lumina_core \
  --preview --preview-duration 5 \
  --resolution 854x480 --fps 30 \
  --codec h264 --quality medium -o "<ausgabe>/preview.mp4"
```

Bei einem Track mit spätem Drop: für zusätzliche Tests zuerst mit FFmpeg einen **temporären verlustarmen Audioauszug** erstellen, diesen mit der identischen Visualizer-Einstellung rendern und den Auszug nachher nicht als Vollsong ausgeben. Die CLI hat keinen belegten `--start-at`-Parameter.

Prüfe mit `scripts/verify_export.py` und visuell an **getrennten Einzelbildern**: Reaktion auf Beat/Transienten, Schärfe, Banding, flackernde Flächen, Schriftausblendung, unleserlichen Kontrast, Motivüberdeckung. Entscheidungsgrund dokumentieren; maximal eine gezielte Korrekturrunde ohne neue Informationen.

### 4. Vollständigen Export durchführen

```bash
# Arbeitsverzeichnis: Root des Visualizer-Checkouts
python main.py render "<audio>" --visual lumina_core \
  --resolution 1920x1080 --fps 30 \
  --codec h264 --quality high -o "<ausgabe>/final.mp4"
```

Optional: `--background-image "<bild>"`, `--param KEY=VALUE` (nur Namen/Werte des konkreten Visualizers aus dessen `PARAMS` bzw. CLI kontrollieren), `--config <config.json>`.

**Wichtige echte CLI-Eigenschaften:** Bei `--config` können u. a. Visualizer, Auflösung und FPS **aus der JSON-Datei** die CLI-Defaults/Angaben übersteuern. Deshalb effektive Parameter aus `config/schemas.py` und `main.py::_resolve_config` vor dem Render nachvollziehen. `--studio`, `--studio-dry` und `--studio-strict` sind **kein normaler Produktions-Export**: der gegenwärtige Studio-Pfad hat in `src/studio/engine.py::run_studio` ein festes 854×480-Preview-Ziel und einen separaten Sidecar-Prozess. Studio für Diagnostik/Probe benutzen, das Seitenformat des Hauptvideos nicht daraus ableiten.

### 5. Export hart prüfen

```bash
python "<skill>/scripts/verify_export.py" \
  --video "<ausgabe>/final.mp4" --source-audio "<audio>" \
  --width 1920 --height 1080 --fps 30 --full-decode
```

Der Verifier prüft per FFprobe Container, Videostream, Audiostream, Auflösung, FPS, zeitliche Übereinstimmung; optional dekodiert FFmpeg **die komplette Datei**. Er misst **nicht** Ästhetik, echte Lip-Synchronität oder Urheberrechte: eigene visuelle Sichtprüfung separat dokumentieren. Details siehe `references/qa.md`.

Bei Fehler: Ursachenanalyse auf Grundlage von Logs (`logs/app.log`), gezielte Änderung, genau **einen** erneuten Export. Nicht behaupten, ein Exit-Code allein beweise Bildqualität.

### 6. Handover an den Nutzer

Melde compact: `status` (`DONE`, `BLOCKED`, `FAILED`), Quelldatei, Kommando/Parameter, gewählter Visualizer plus Begründung, Datei-/Vorschaupfad, nachweisliche technische Prüfergebnisse, visuelle Auffälligkeiten, verbleibende Risiken. Verweise auf **existierende** Dateien, keine erfundenen Downloadlinks. Bei Batch je Track separaten Job und separate Abnahme.

**Stop-/Kill-Kriterien:** Keine lesbare Eingabe; unmöglicher GPU-Context; FFmpeg fehlt; mehrfacher reproduzierbarer Renderfehler; Audio-/Video-Dauer außerhalb Toleranz; Endprodukt fehlt; Nutzerziel mit vorhandenen Visualizern prinzipiell nicht erreichbar. Fehler offen berichten, nicht mit mehr Effekten kaschieren.

## On-demand-Referenzen

- `references/cli.md`: verifizierte Befehle, Eigenheiten und Grenzen des gegenwärtigen Repos
- `references/qa.md`: Prüfschritte, Fehlerklassen, Statusregeln
- `examples/job.json`: Beispiel für einen speicherbaren Agenten-Auftrag
- `scripts/preflight.py` und `scripts/verify_export.py`: dependency-arme, ausführbare Agentenwerkzeuge (Python-Standardbibliothek, FFprobe/FFmpeg extern)