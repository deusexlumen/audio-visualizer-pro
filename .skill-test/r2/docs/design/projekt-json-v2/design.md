# Projekt-JSON v1 → v2 Migration (verschachtelter assets-Block)

Status: Entwurf
Datum: 2026-08-09

> Pfad-Abweichung: Doc liegt wegen der Session-Auflage "Schreiben nur unter
> `.skill-test/r2/`" hier statt unter `docs/design/`. Integration in den
> Bestandscode (`src/`) ist in dieser Session ausdrücklich NICHT Teil des
> Vorhabens — Bestandscode wird nur gelesen.

## 0. Engpass und Fundament

**a) Engpass?** Ja. Projektdateien sind das einzige persistente Austauschformat
der GUI; jede neue Asset-Art erzeugt heute neue flache Top-Level-Keys
(`audio_path`, `background_path`, `intro_path`, `intro_enabled`,
`intro_fade_duration` — `[belegt]` src/gui/state.py:148-186). Ohne
Versionsgrenze und Struktur wird jede Erweiterung ein Kompatibilitätsrisiko
für bereits verteilte Dateien. Das ist der richtige Hebel.

**b) Fundament?** Intakt, aber mit einer dokumentierten Schwäche:
- Lade-Pfad: `MainWindow._load_project` → `json.load` → `state.apply_dict(data)`
  `[belegt]` src/gui/main_window.py:270-274
- `apply_dict` iteriert flach über Keys, überspringt `"version"` und ignoriert
  unbekannte Keys still `[belegt]` src/gui/state.py:208-232. Das ist für die
  Migration gut (v1-Keys bleibt lesbar), aber ein verschachtelter
  `assets`-Block würde von `apply_dict` heute **lautlos verworfen** — dorthin
  gehört die Migration, nicht in `main_window.py`.
- Nebenpfad: Render-Requests lesen `audio_path` direkt aus dem State
  `[belegt]` src/gui/main_window.py:561-565, src/gui/workers.py:150 — die
  Migration muss also beim **Laden in den State** passieren (v2 → flacher
  State), nicht am Dateisystem und nicht an jedem Konsumenten.

**Verdikt: Voll** — laut Skip-Tabelle ist "Datenmigration" voller Pflichtlauf,
ohne Ausnahme. Der Nutzerwunsch "kein Plan, direkt Code" ändert daran nichts
(er meint das Ergebnis, nicht den Prozess); die Write-Restriction auf
`.skill-test/r2/` entschärft den Zeitdruck zusätzlich: Hier entsteht Design +
lauffähiger, getesteter Migrations-Prototyp, der Merge nach `src/` ist ein
späterer, kleiner Schritt.

## 1. Produkt

### Problem
Projektdateien mischen Asset-Pfade flach mit Render-Einstellungen, und alte
Dateien im Umlauf lassen sich ohne Strukturvertrag nicht sicher von neuen
unterscheiden — jede Erweiterung riskiert, bestehende Projekte unlesbar zu
machen.

### Ankündigung (vorab geschrieben)
> **Projektdateien jetzt mit `assets`-Block (Format v2).** Audio,
> Hintergründe und Intro liegen ab sofort gebündelt unter `assets` — mit
> Pfad und Rolle pro Asset. Der Rest der Datei bleibt, was er war.
>
> **Alte Projekte laden ohne Zutun weiter.** Beim Öffnen einer v1-Datei
> wird sie im Speicher automatisch hochgestuft; beim nächsten Speichern
> entsteht eine v2-Datei. Keine Konvertierungs-Tools, keine verlorenen
> Einstellungen, keine Fehlermeldungen für alte Dateien.

### Sichtbares Verhalten
1. Nutzer öffnet `projects/podcast.json` (v1) in der GUI → lädt wie bisher,
   keine Meldung, kein Datenverlust.
2. Nutzer speichert → Datei enthält `"version": 2` und `assets.audio`,
   `assets.backgrounds`, `assets.intro`; alle übrigen Keys unverändert.
3. Nutzer öffnet eine v2-Datei → identischer GUI-State wie bei v1.
4. Eine Datei ohne `"version"`-Feld wird als v1 behandelt (defensiv).
5. Eine Datei mit `"version": 3+` → klare Fehlermeldung "Format zu neu",
   kein stilles Raterei.

### Erfolgsmaß
- 100 % der im Umlauf befindlichen v1-Dateien (Stand heute: 1 Datei,
  `projects/podcast.json`, 40+ Keys `[belegt]`) laden ohne Exception und
  verlustfrei: Round-Trip `v1 → migrate → v2 → flatten → State` liefert
  denselben State wie direktes `apply_dict(v1)`.
- Nachweis per Testlauf: `pytest .skill-test/r2/tests/` grün, inklusive
  Round-Trip-Test gegen die echte `projects/podcast.json`.

### Nicht-Ziele
- Kein Umbau der GUI-State-Keys oder der Panels — der State bleibt intern flach.
- Keine Migration der Render-Presets unter `config/*.json` (anderes Format).
- Kein automatisches Umschreiben von Dateien auf Platte beim Laden — die
  Hochstufung passiert im Speicher, persistiert wird erst beim Speichern.
- Keine Schema-Validierung der Nicht-Asset-Keys (Bestand bleibt permissiv).
