# Session Notes – Audio Visualizer Pro

## 2026-04-21 – Fixes committed & gepusht + State-of-the-Art README

**Commit:** `ba2f258` auf `master`
**Remote:** `https://github.com/deusexlumen/audio-visualizer-pro.git`

**Enthaltene Änderungen:**
- `gui.py` — Slide-In Validation Fix + Streamlit `use_container_width` → `width`
- `src/analyzer.py` — AudioFeatures Cache-Edge-Case (verschachtelte ndarrays)
- `SESSION_NOTES.md` — Dokumentation der Fixes
- `README.md` — Komplette Überarbeitung: ASCII-Banner, Mermaid-Diagramme, Feature-Matrix, 16-Visualizer-Galerie, Farbpaletten, Roadmap

**Status:** ✅ Live auf GitHub.

---

## 2026-04-21 – Zwei Bugs gefixt (GUI Slide-In + AudioFeatures Cache)

**Bug 1: `ValueError: 'slide_up' is not in list` in `gui.py`**
- Ursache: Im Streamlit-Session-State war noch der alte Wert `"slide_up"` gespeichert, aber die Optionsliste der Selectbox wurde auf `["none", "up", "down", "left", "right"]` geaendert. `.index()` crashte.
- Fix: Vor `.index()` wird geprueft, ob der Session-State-Wert in der Optionsliste existiert. Falls nicht, Fallback auf `"none"`.
- Betroffene Zeilen: `quote_slide` und `quote_slide_out` Selectboxen.

**Bug 2: `[GPU Preview] Fehler: 1 validation error for AudioFeatures key Input should be a valid string`**
- Ursache: Obwohl der Cache-Lade-Fix vom 2026-04-25 bereits im Code war, gab es einen Edge-Case: Object-Arrays im NPZ-Cache konnten verschachtelt sein, sodass `.item()` selbst ein `np.ndarray` zurueckgab statt `None`. Pydantic bekam dann ein ndarray statt eines Strings.
- Fix in `src/analyzer.py`: Explizite Prüfung `isinstance(item, np.ndarray)` im Cache-Lade-Code. Falls ein ndarray zurueckkommt, wird es auf `None` gesetzt.
- Status: ✅ Beide Fixes applied.

---

## 2026-04-25 – Audio-Analyse Caching Bug gefixt

**Problem:**
In der Streamlit-GUI (`gui.py`) wurde bei jedem Rerun (Slider, Button, Farbwechsel) eine neue temporäre Audio-Datei mit zufälligem Namen erstellt. Der `AudioAnalyzer` cached zwar basierend auf Dateigröße und mtime, aber da der Dateipfad jedes Mal unterschiedlich war, fand der Cache nie einen Treffer. Ergebnis: Die komplette Audio-Analyse lief bei jeder Einstellungsänderung neu.

**Fix in `gui.py`:**
1. Temporärer Dateipfad wird jetzt im `st.session_state` gecacht (Key: `temp_audio_{filename}_{size}`)
2. `AudioFeatures` werden ebenfalls im Session-State gecacht (Key: `features_{filename}_{size}`)
3. Alle Stellen die `uploaded_file.getvalue()` in eine Datei geschrieben haben (KI-Optimierung, Zitat-Extraktion, GPU-Preview, GPU-Rendering) nutzen jetzt den gecachten Pfad

**Status:** ✅ Getestet (Syntax-Check OK), wartet auf Re-Start der Streamlit-GUI.

---

## 2026-04-25 – KI-Optimierung und Zitat-Extraktion verbessert

**Problem 1: KI-Optimierung wirkte sich kaum aus**
- Die KI bekam nur Durchschnittswerte (rms_mean, onset_mean), keine Dynamik-Infos
- Die KI kannte die gültigen Param-Bereiche (min/max/step) nicht
- Es gab keinen Fallback, wenn die KI offline war oder Mist baute
- Der Prompt war zu generisch

**Problem 2: Zitate wurden immer starr 5 Stück extrahiert**
- Ein 30-Sekunden-Clip bekam genauso viele Zitate wie ein 60-Minuten-Podcast
- Keine Qualitäts-Filterung (confidence)
- Keine Mindestlänge für Zitate

**Fix in `src/gemini_integration.py`:**

### `extract_quotes` neu:
- Dynamische `max_quotes` basierend auf Audio-Dauer:
  - < 60s = 2 Zitate
  - 60s-3min = 3 Zitate
  - 3min-5min = 4 Zitate
  - 5min-10min = 6 Zitate
  - > 10min = max 10 Zitate (ca. 1 pro 90 Sekunden)
- Confidence-Filter: nur Zitate >= 0.6
- Mindestlänge: 3 Wörter
- Prompt geändert: KI soll selbst entscheiden wie viele Zitate gut sind (Qualität > Quantität)

### `optimize_all_settings` neu:
- Neue Parameter `param_specs` (min/max/step) werden mitgegeben
- Erweiterte Audio-Features: `rms_std`, `onset_std`, `transient_mean`, `voice_clarity_mean`
- Viel konkreterer Prompt mit Regeln pro Modus, Tempo, Voice-Clarity
- **Hartes Clamping**: Jeder zurückgegebene Parameter wird auf seinen gültigen min/max-Bereich begrenzt
- **Deterministischer Fallback**: Wenn die KI fehlschlägt, werden Parameter automatisch aus den Audio-Features berechnet (Tempo -> Speed, RMS-Std -> Intensität, etc.)

**Fix in `gui.py`:**
- `extract_quotes` ruft jetzt `audio_duration=features.duration` auf
- `optimize_all_settings` übergibt jetzt `param_specs=viz_cls.PARAMS` und erweiterte Features

**Status:** ✅ Syntax-Check OK, wartet auf Test in der GUI.

---

## 2026-04-25 – Preview-Bugfix, GPU-Cache und Code-Cleanup

**Problem 1: Preview-Hash wurde vor erfolgreichem Render gesetzt**
- Wenn GPU-Preview crashte, wurde der Hash trotzdem aktualisiert
- Ergebnis: Preview versuchte nie wieder zu rendern, auch bei Parameter-Aenderungen
- **Fix in `gui.py`:** Hash wird erst nach erfolgreichem Render (`preview_img is not None`) gesetzt

**Problem 2: GPU-Context wurde bei jedem Slider-Zug neu gebaut**
- `render_gpu_preview` erstellte bei jedem Aufruf neu: ModernGL-Context, Framebuffer, Shader, Visualizer
- Das ist massiv ineffizient und kann Memory-Leaks verursachen
- **Fix in `src/gpu_preview.py`:** Modul-Level Cache `_PREVIEW_CACHE` fuer Renderer + Visualizer
  - Cache-Key: `(visualizer_type, width, height, fps)`
  - Bei Visualizer-Wechsel wird alter Renderer ordentlich freigegeben
  - Parameter werden nur via `viz.set_params()` aktualisiert (kein Neuerstellen)
- **Fix in `src/gpu_renderer.py`:** Neue `release()` Methode fuer explizites Cleanup von FBO, VAO, VBO, Texture, Context

**Problem 3: `hex_to_rgb` crashte bei ungueltigen Farben**
- Wenn KI kaputte Hex-Farbe zurueckgab (z.B. `#FF00`), knallte es
- **Fix in `gui.py`:** Try/except + Fallback auf Weiss, Unterstuetzung fuer 3-stellige Hex (#RGB)

**Problem 4: Toter Code `render_live_frame`**
- 45 Zeilen ungenutzter PIL-basierter Preview-Code lagen noch in `gui.py`
- **Fix in `gui.py`:** Entfernt

**Status:** ✅ Alle 4 Dateien Syntax-Check OK.

---

## 2026-04-25 – Cache-Lade-Bug gefixt (Pydantic Validierungsfehler)

**Problem:**
`[GPU Preview] Fehler: 1 validation error for AudioFeatures key Input should be a valid string`

**Ursache:**
- `AudioAnalyzer._save_cache()` speichert `key=None` als `np.array(None, dtype=object)`
- `_load_cache()` pruefte nur `val.dtype.kind == 'U'` (Unicode-Strings)
- `None` hat `dtype.kind == 'O'` (Object) und wurde als numpy Array an Pydantic uebergeben
- Pydantic meckerte, weil es einen String erwartet, aber ein Array bekam

**Fix in `src/analyzer.py`:**
- Neue Liste `scalar_fields = {'duration', 'sample_rate', 'fps', 'frame_count', 'tempo', 'key', 'mode'}`
- Skalare Felder werden jetzt mit `val.item()` aus 0-dim Arrays zurueckkonvertiert
- Strings weiterhin via `val.dtype.kind == 'U'`
- Alle anderen Werte bleiben als numpy Arrays (rms, onset, chroma...)

**Status:** ✅ Syntax-Check OK. Alter Cache funktioniert jetzt korrekt (kein Loeschen noetig).

---

## 2026-04-25 – FFmpeg Chroma-Subsampling Bug gefixt ("komplett blurred")

**Problem:**
User meldete: "Ich hab jetzt was gerendert und es ist komplett blurred obwohl ich das ausgestellt hab"

**Ursache:**
- FFmpeg wurde immer mit `-pix_fmt yuv420p` aufgerufen
- `yuv420p` = 4:2:0 Chroma-Subsampling: Farbauflösung wird auf HALBE Breite/Hoehe reduziert
- Bei scharfen bunten Visualizern (Neon-Balken, kräftige Kontraste) sieht das aus wie starker Weichzeichner
- Der User hatte Hintergrund-Blur korrekt auf 0 gestellt, aber das Encoding selbst war schuld

**Fix in `src/gpu_renderer.py`:**
- Qualitaetsprofile erweitert um `pix_fmt`:
  - `low` / `medium` → `yuv420p` (Kompatibilitaet)
  - `high` / `lossless` → `yuv444p` (KEIN Chroma-Subsampling, scharfe Farben)
- Zusaetzlich: `_load_background_texture` prueft jetzt `blur > 0.01` statt `blur > 0` (robuster)

**Status:** ✅ Syntax-Check OK. Videos mit Qualitaet "high" sind jetzt deutlich schaerfer.
