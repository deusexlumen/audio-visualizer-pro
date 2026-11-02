# Podcast-Folgen: Look und Prüf-Ablauf — Design

**Datum:** 2026-11-02
**Status:** Entwurf, wartet auf Freigabe durch den Nutzer
**Kanal:** YouTube „Deus Ex Lumen“

---

## 1. Absicht und Warum

Die App ist für die eigenen Podcasts des Nutzers entstanden. Er braucht zu jeder Audio-Folge ein Video für YouTube und nutzt die App dafür schon. Musik ist ein zweiter Anwendungsfall, nicht der Hauptzweck.

Heute kostet eine Folge mehr Arbeit, als sie spart. Die Zitate verschwinden zu schnell, ihr Timing stimmt oft nicht, und sie sehen schlecht aus (generische dunkle Box, enge Zeilen, verwaiste Einzelwörter, mittig über dem Visualizer). Das Bild der Folge wirkt durch Nachbearbeitung schlechter als das Original.

**Ziel in einem Satz:** Der Nutzer gibt Audio und Bild hinein und muss danach „nur noch prüfen und hochladen“.

## 2. Ziele, Nicht-Ziele, Anti-Ziele

**Ziele**

- Ein fertiger, ruhiger Look für ganze Folgen, der auf YouTube professionell wirkt.
- Die KI bereitet alles vor: Kapitel, 2–3 Bühnen-Sätze, normale Zitate.
- Prüfen vor dem Render, in etwa 10 Minuten pro Folge.
- Ein Fehler kostet keinen zweiten langen Render.

**Nicht-Ziele (dieses Projekt)**

- YouTube Shorts (später, siehe §12).
- Musik-Visuals und ENTHEA-Ports (eigenes Brainstorming danach).
- Mehrere Bilder pro Folge. Das Design darf es später erlauben, braucht es aber nicht.

**Anti-Ziele (Fehlschlag, auch wenn es technisch läuft)**

- Der Nutzer hat mit dem neuen Ablauf mehr Arbeit als heute.
- Das Bild sieht im Video schlechter aus als die ChatGPT-Datei.
- Text erscheint sichtbar vor oder nach dem gesprochenen Satz.
- Bestehende Podcast-Presets, alte Projektdateien oder der Musik-Pfad ändern ihr Verhalten.

## 3. Ausgangslage (Befunde aus Code und Render)

- `output/visualization_20260811_020404.mp4`: statisches Bild über 11:22, starke Farbsäume an allen Kanten (chromatische Aberration), keine sichtbaren Zitate.
- `background_opacity` ist standardmäßig `0.3`. Über dem Bild läuft die ganze Nachbearbeitung: Bloom, ACES-Tonemapping, chromatische Aberration, Grain. `config/podcast_story.json` setzt die Aberration auf `0.5`.
- Der Hintergrund-Unschärfe-Wert wird einmal beim Laden per PIL eingerechnet (`_load_background_texture`). Es gibt keine Kamerafahrt: `_render_background` zeichnet ein festes Vollbild-Quad.
- `Quote` (`src/types.py`) hat nur `text`, `start_time`, `end_time`, `confidence`. Es gibt keine Kapitel.
- Gemini liefert Segmente von 1–2 Sätzen (`transcribe_segments`, gecacht). Wortzeiten innerhalb eines Segments schätzt `quote_timing._interpolate` über die Zeichenposition. Sie sind Näherungen.
- `quote_overlay.py`: feste Mindestdauer 3,5 s, Höchstdauer 8 s, unabhängig von der Satzlänge.
- Mitgeliefert ist nur `assets/fonts/Inter-Variable.ttf`. Eine Serifenschrift fehlt.
- `segmentation.py` teilt nach Klang, nicht nach Thema. Für Kapitel taugt es nicht.
- `intro_renderer.py` setzt ein Intro-Video vor das Hauptvideo. Das Hauptvideo beginnt in der Enddatei bei `intro_dauer − fade_dauer`.

## 4. Vom Nutzer entschieden

- Hauptzweck: Podcast-Videos für „Deus Ex Lumen“. Podcast zuerst, Musik danach.
- Ein Bild pro Folge, gemacht mit ChatGPT.
- Alle drei Looks gemischt in einer Folge:
  - **A — Kino-Standbild** als Grundzustand.
  - **B — Zitat-Bühne** für die 2–3 stärksten Sätze.
  - **C — Kapitel-Karte** an jedem Themenanfang.
- Bühnen-Sätze (B) sind die kühnsten Behauptungen der Folge, die Sätze, die Leute zum Nachdenken bringen. Nicht die beste Zusammenfassung. Der KI-Auftrag für die Auswahl folgt genau diesem Maßstab.
- Prüfen passiert vor dem Render, als eine Liste mit Abspiel-Knopf pro Zeile (siehe §6). Der Nutzer hat den Ablauf bestätigt („its good“).

## 5. Der Look im Detail

### 5.1 A — Kino-Standbild (Grundzustand)

- Das Bild füllt den Rahmen, voll deckend und scharf. Keine Farbsäume.
- Langsame Kamerafahrt (Zoom und Verschiebung, höchstens etwa 8 % Zoom, damit das Bild scharf bleibt).
- Unten ein dünner Lichtstrich, den die Stimme treibt (`voice_band` / `rms`). Das ist die einzige Visualizer-Rolle in diesem Look.
- Kleines „DEUS EX LUMEN“ oben links.
- Normale Zitate: unten links, Serifenschrift, keine Box, weicher Schatten und Verlauf nach unten für Lesbarkeit. Die Wörter erscheinen nacheinander im Takt der Sprache.

### 5.2 B — Zitat-Bühne (2–3 Sätze pro Folge)

- Das Bild wird dunkler und unscharf (weich überblenden, etwa 0,8 s).
- Der Satz füllt die Bildmitte, groß. Ein Schlüsselwort trägt die Akzentfarbe.
- Danach blendet das Bild zurück zu A.

### 5.3 C — Kapitel-Karte (an jedem Kapitelanfang)

- Für etwa 4 s: Textspalte links mit „DEUS EX LUMEN“, Folgentitel, „Kapitel N · Name“. Das Bild rückt nach rechts.
- Danach Rückblende zu A. Ein Wechsel der Kamerafahrt passiert unter der Karte, damit er nicht springt.

### 5.4 Lesezeit-Regel (normale Zitate und Bühnen-Sätze)

- Sichtbar ab Sprechbeginn, mindestens bis Sprechende.
- Danach steht der Text so lange, bis eine Lesezeit von 15 Zeichen pro Sekunde erreicht ist, mindestens aber 1,5 s nach Sprechende.
- Obergrenze: Ausblenden spätestens 0,3 s vor dem nächsten Zitat oder der nächsten Karte.
- Die Wort-für-Wort-Einblendung ist weich und folgt der geschätzten Wortzeit. Sie verspricht keine wortgenaue Synchronität.

Diese Regel ersetzt für den neuen Look die festen 3,5 s / 8 s.

## 6. Ablauf von Audio bis Upload

1. Der Nutzer wählt die Audiodatei und das ChatGPT-Bild.
2. Die KI erstellt aus dem vorhandenen Segment-Transkript in einem Aufruf: Kapitel (Start + Name), 2–3 Bühnen-Sätze (mit Schlüsselwort), normale Zitate. Zeiten kommen immer aus den Segmenten (`locate_in_segments`), nie aus Zeitangaben der KI selbst.
3. Die App zeigt **eine Prüfliste** mit allen Kapiteln, Bühnen-Sätzen und Zitaten, nach Zeit sortiert. Jede Zeile hat:
   - Art (Kapitel / Bühne / Zitat), Zeit, Text;
   - einen Abspiel-Knopf: Audio ab dieser Stelle plus eine Vorschau mit dem Look;
   - Löschen, Text ändern, Zeit verschieben; bei Bühnen-Sätzen das Schlüsselwort wählen; eine Zeile zwischen Zitat und Bühne umstellen.
   Hinzufügen ist möglich, aber nicht nötig.
4. Der Folgentitel steht oben in der Liste, vorbefüllt aus dem Dateinamen (Unterstriche zu Leerzeichen), änderbar.
5. Der Nutzer startet den Render. Danach schreibt die App die Kapitelzeiten als Text für die YouTube-Beschreibung (siehe §8.6).

Alles aus Schritt 2–4 wird im Projekt (`.avproj`) gespeichert.

## 7. Datenmodell

Neue Felder, alle mit Standardwerten, damit alte `.avproj`-Dateien und `projects/podcast.json` unverändert laden:

- `Quote.kind`: `"quote"` | `"stage"`, Standard `"quote"`.
- `Quote.accent_word`: optional, nur für `"stage"`.
- Neu `Chapter`: `start` (Sekunden), `title`.
- `ProjectConfig.chapters`: Liste, Standard leer.
- `ProjectConfig.episode_title`: optional.
- Ein Schalter oder Preset, der den neuen Look wählt (Name dem Builder überlassen).

## 8. Technische Randbedingungen

1. **Bild bleibt original.** Das Folgenbild darf nicht durch Bloom, Tonemapping, Aberration oder Grain laufen, oder diese Stufen sind im neuen Preset neutral. Opazität 1,0. Akzeptanz siehe §9.
2. **Unschärfe über Zeit.** Für B ist eine zweite, vorab unscharf gerechnete Textur mit Überblendung ein günstiger Weg. Die Kamerafahrt braucht eine Skalierung und Verschiebung pro Frame in `_render_background`.
3. **Bildgröße.** ChatGPT-Bilder sind oft 1536×1024 oder 1024×1024. Das Bild wird auf 16:9 zugeschnitten (cover) und hochwertig hochskaliert. Ist es nach dem Zuschnitt kleiner als die Zielauflösung, zeigt die Prüfliste einen Hinweis.
4. **Schrift.** Eine Serifenschrift mit offener Lizenz (OFL), mit äöüß und „ “, wird in `assets/fonts/` aufgenommen.
5. **Text-Rendering.** Der neue Text (Wort-Einblendung, Bühne, Karte) ändert sich pro Frame. Das heutige Cache-pro-Zitat-Prinzip in `quote_overlay.py` reicht dafür nicht unverändert. Der Weg (PIL pro Wort-Zustand cachen oder `gpu_text_renderer.py`) ist dem Builder überlassen.
6. **Kapiteltext für YouTube.** Format `M:SS Titel` pro Zeile. Regeln von YouTube: erste Zeile bei `0:00`, mindestens 3 Kapitel, jedes mindestens 10 s lang. Ist ein Intro-Video gesetzt, werden alle Zeiten außer der ersten um `intro_dauer − fade_dauer` verschoben. Verletzt die Liste eine Regel, warnt die App, statt still falschen Text zu schreiben.
7. **Keine Rückschritte.** Bestehende Presets in `config/`, der alte Zitat-Look und der Musik-Pfad bleiben wie sie sind. Die vorhandene Testsuite (Stand 2026-11-02: 577 Tests grün) und das Golden-Set (`tests/golden/`, `test_golden_corpus.py`) bleiben grün.
8. **Wiederverwendung.** Die Prüfliste baut auf `src/gui/quote_editor.py` auf (Wellenform, Abspielen, Einrasten). Die Look-Vorschau pro Zeile nutzt `src/gpu_preview.py`.
9. **Umfang.** Das Projekt berührt mehr als 5 Dateien. Der Plan holt sich vor der Umsetzung ein Go.

## 9. Akzeptanzkriterien

- Von Audio + Bild bis zur gefüllten Prüfliste ohne Handarbeit (außer ggf. Titel).
- Farbtreue: In einem Testframe ohne Text und ohne Bühne weicht das Bild im Mittel höchstens 3/255 pro Kanal vom zugeschnittenen Original ab (Bereich ohne Lichtstrich).
- Lesezeit-Regel aus §5.4 per Test geprüft (kurzer, langer und dicht gefolgter Satz).
- Kapiteltext per Test geprüft, mit und ohne Intro-Video.
- Alte Projektdateien laden; alte Presets rendern wie vorher; Testsuite und Golden-Set grün.
- Renderzeit des neuen Looks höchstens so lang wie ein heutiger Podcast-Render mit Visualizer bei gleicher Auflösung.

## 10. Meine Entscheidungen (bitte beim Lesen prüfen)

Diese Punkte hat der Nutzer nicht entschieden. Sie sind meine Wahl:

- Pro Kapitel eine andere Kamerafahrt, Wechsel unter der Kapitel-Karte.
- Die KI wählt alles vor, der Nutzer löscht und korrigiert meist nur.
- Der Kapiteltext für die YouTube-Beschreibung (§8.6).
- Lesezeit 15 Zeichen/s, Nachlauf mindestens 1,5 s, Abstand 0,3 s.
- Akzentfarbe warmes Gold (`#F2B866`, wie im Mockup), änderbar.
- Zitat-Dichte: etwa ein normales Zitat alle 90 s, einstellbar.
- Zitate, die in eine Kapitel-Karte fallen würden, wählt die KI nicht; liegt trotzdem eins dort, gewinnt die Karte und die Prüfliste markiert die Zeile.
- Standardauflösung 1920×1080.
- Kapitelzahl: die KI schlägt 3–8 vor, je nach Länge der Folge.
- Folgennummer auf der Karte („Folge 12“) ist ein optionales Feld, standardmäßig leer.
- Der Lichtstrich bleibt in B und C sichtbar, aber gedimmt.
- Scheitert die KI, zeigt die Prüfliste nur den Titel und der Nutzer kann Zeilen von Hand anlegen.

## 11. Dem Builder überlassen

- Namen von Preset, Feldern und Modulen.
- Ob das Bild an der Nachbearbeitung vorbei gezeichnet wird oder neutrale Werte reichen (Hauptsache §9 Farbtreue).
- Text-Rendering-Weg (§8.5).
- Genaue Kurven von Kamerafahrt und Überblendungen.
- Wahl der Serifenschrift innerhalb der Vorgaben aus §8.4.
- Form der Prüfliste im GUI (eigener Reiter oder Umbau von `quotes_panel.py`).
- CLI: Der Render liest die neuen Felder aus der Projekt-JSON; eine Prüfliste gibt es nur im GUI.

## 12. Später (nicht in diesem Projekt)

- YouTube Shorts aus den Bühnen-Sätzen (B ist dafür gebaut).
- Ein zweites Bild pro Kapitel.
- Musik-Visuals (ENTHEA-Ports: Cymatics, Quasikristall).
