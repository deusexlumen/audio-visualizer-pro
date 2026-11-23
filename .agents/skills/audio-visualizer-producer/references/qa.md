# Abnahme- und Fehlerregeln

## Technische Gates (zwingend)

1. **Preflight:** `scripts/preflight.py` exit 0, verständliche Warnungen möglich; Audio vorhanden/nicht leer; Repo-Entrypoint, Python, ffmpeg und ffprobe verfügbar.
2. **Vorschau:** Video-Datei vorhanden, `scripts/verify_export.py` gibt `ok=true` zurück. Im Kurzclip kann die Duration naturgemäß vom ganzen Song abweichen: **ohne** `--source-audio` prüfen.
3. **Vollrender:** Container enthält mindestens einen dekodierbaren Videostream und einen Audiostream. Originaldauer und Video-/Audio-Dauer passen bis zur konfigurierten Toleranz. Originaltrack wird über `--source-audio` gegen das Video abgeglichen.
4. **Frame-Maße:** Auflösung und FPS entsprechen dem Auftrag oder dokumentierten technischen Fallback; `verify_export.py` macht diese Prüfung numerisch.
5. **Dekodierung:** `--full-decode` verlangt, dass FFmpeg die vollständige MP4 ohne fatalen Decoding-Fehler durchläuft. Für jeden finalen Export ausführen, sofern FFmpeg lokal verfügbar ist.
6. **Visueller Check:** Mindestens Anfang, energetischer Höhepunkt und Ende getrennt ansehen. Keine Collagen. Prüfen auf schwarze/steckende Frames, Pixelfehler, falsch positionierten Text, gestörten Hintergrund, verdeckte Gesichter und fehlende Beat-Reaktivität. `ffprobe` beweist diese Dinge **nicht**.
7. **Tonprüfung:** Stichproben mit hörbarem Audio; kein Stummvideo, keine ungewollte Beschleunigung, kein abgeschnittenes Outro. Der Verifier prüft nur Metadaten, nicht Klangqualität.

## Abnahmezustände

- `DONE`: Final-MP4 existiert, alle technischen Gates bestanden, visueller und akustischer Review protokolliert.
- `BLOCKED`: nötige Daten, Rechte, GPU, Eingabe oder Dependencies nicht vorhanden; keine produktionsreife MP4 behaupten.
- `FAILED`: Render ist gestartet, aber technische Gates nicht erfüllt; Fehlerquelle und konkrete Reparatur nennen.
- `PREVIEW_ONLY`: nur Clips vorhanden, keine Vollversion. Ein abgenommenes Preview ist noch kein `DONE`.

## Stop-/Kill

- Kein Audio oder ungültiger Container: Job stoppen.
- OpenGL/ModernGL kann keinen Context öffnen: abbrechen, GPU-/Treiberproblem erklären; keine unbewiesene CPU-Renderfähigkeit versprechen.
- FFmpeg/FFprobe nicht auffindbar: Installation/PATH dokumentieren; nicht blind rendern.
- Finalvideo ohne Audiostream oder mit auffälliger Dauerabweichung: nicht freigeben.
- Visuelle Qualität scheitert mehrfach am gleichen Effekt: Visualizer/Parameter gezielt wechseln, nicht wahllos Effekte addieren.
- Externe Veröffentlichungen, Datei-Löschungen, API-Aufrufe oder Cloud-Uploaddienste nur auf explizite Anweisung.

## Dokumentierter Handover

```json
{
  "status": "DONE | PREVIEW_ONLY | BLOCKED | FAILED",
  "audio": "path/to/source.mp3",
  "visualizer": "lumina_core",
  "video": "path/to/final.mp4",
  "profile": {"width": 1920, "height": 1080, "fps": 30, "codec": "h264"},
  "technical_qa": {"ok": true, "report": "path/to/report.json"},
  "review": {"visual": "checked | not_checked", "audio": "checked | not_checked"},
  "risks": [],
  "next_action": "none"
}
```