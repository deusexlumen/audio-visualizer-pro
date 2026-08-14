# Installation — Audio Visualizer Pro (Windows)

Diese Anleitung richtet sich an Endnutzer, die Audio Visualizer Pro als
fertige Windows-Anwendung installieren wollen (kein Python noetig).

## Installation

1. Neueste `AudioVisualizerPro-Setup-<version>.exe` von der [Releases-Seite](https://github.com/deusexlumen/audio-visualizer-pro/releases) herunterladen.
2. Setup starten. Es wird **kein Administrator-Rechte** benoetigt — die App
   installiert sich in dein Benutzerprofil.
3. Optional: Desktop-Icon-Haekchen setzen.
4. Nach der Installation startet die App automatisch (Haekchen abwaehlbar).

### FFmpeg

Audio Visualizer Pro benoetigt FFmpeg fuer das Video-Encoding, bringt es aber
bewusst nicht mit (Lizenz- und Groessengruende). Wird FFmpeg beim ersten Start
nicht gefunden (weder im System-PATH noch von einem frueheren Download),
fragt die App nach und laedt bei Zustimmung automatisch einen schlanken
FFmpeg-Build (~90 MB) von [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
herunter. Der Download landet unter
`%LOCALAPPDATA%\AudioVisualizerPro\ffmpeg\` und muss nur einmal erfolgen.

Alternativ kann FFmpeg auch manuell installiert werden
(https://ffmpeg.org/download.html) — liegt es im System-PATH, wird es
automatisch erkannt und der Download-Dialog erscheint nicht.

## Wo landen meine Daten?

Der Install-Ordner selbst ist read-only (Programmordner). Alle
beschreibbaren Daten liegen in deinem Benutzerprofil und bleiben bei einem
Update/Reinstall erhalten:

| Inhalt | Ort |
|---|---|
| Eigene Studio-Rezepte | `%APPDATA%\AudioVisualizerPro\recipes\` |
| Analyse-Cache, Zitat-Cache | `%LOCALAPPDATA%\AudioVisualizerPro\cache\` |
| Logs | `%LOCALAPPDATA%\AudioVisualizerPro\logs\` |
| Heruntergeladenes FFmpeg | `%LOCALAPPDATA%\AudioVisualizerPro\ffmpeg\` |
| numba-JIT-Cache (Performance) | `%LOCALAPPDATA%\AudioVisualizerPro\numba_cache\` |

## Deinstallation

Ueber "Programme hinzufuegen/entfernen" oder den Eintrag im Startmenue.
Der Uninstaller fragt separat nach, ob auch die oben genannten Nutzerdaten
(insbesondere eigene Studio-Rezepte!) geloescht werden sollen — Standard ist
"Nein", damit nichts versehentlich verloren geht.

## Bekannte Stolpersteine

- **Windows Defender/Antivirus meldet die EXE als verdaechtig**: die EXE ist
  aktuell nicht code-signiert (Signierung ist fuer eine spaetere Version
  geplant). "Weitere Informationen" → "Trotzdem ausfuehren", falls die Quelle
  vertrauenswuerdig ist (offizielle Releases-Seite).
- **GPU-Fehler beim Start** ("cannot create context" o.ae.): Audio Visualizer
  Pro braucht einen OpenGL-3.3-faehigen Grafiktreiber. Grafiktreiber
  aktualisieren (Windows Update oder Hersteller-Seite).
- **Erster Analyse-Lauf ist langsam**: numba kompiliert seine JIT-Funktionen
  beim ersten Gebrauch und cacht sie danach (`numba_cache`, siehe Tabelle
  oben). Folge-Laeufe sind deutlich schneller.

## Fuer Entwickler: selbst bauen

```bash
# Aus einem venv mit requirements.lock (exakte numba/llvmlite-Version wichtig)
pip install -r requirements.lock
pip install pyinstaller

python build/build.py
# Ergebnis: dist/AudioVisualizerPro/AudioVisualizerPro.exe (onedir)

# Installer bauen (Inno Setup https://jrsoftware.org/isinfo.php installiert):
ISCC build/installer.iss /DMyAppVersion=3.2.0
# Ergebnis: dist/installer/AudioVisualizerPro-Setup-3.2.0.exe
```

Details zu Build-Fallstricken (librosa/numba/soundfile-Bundling, Qt-Modul-
Ausschluss) stehen als Kommentare in `build/avp.spec`.
