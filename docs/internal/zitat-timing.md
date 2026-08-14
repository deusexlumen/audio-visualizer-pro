# Zitat-Timing: Messung, Umbau, Grenzen

Stand: 2026-08-11. Betrifft `src/gemini_integration.py` (Extraktion),
`src/quote_timing.py` (lokale Korrektur), `src/quote_overlay.py` (Anzeige)
und `src/gui/quote_editor.py` (manuelle Korrektur).

Vorgabe des Nutzers: **exaktes Timing ist wichtiger als die Qualität der
ausgewählten Zitate.**

## Wie gemessen wurde

Zeitstempel lassen sich nicht dadurch prüfen, dass man dieselbe KI nochmal
fragt. Deshalb eine unabhängige Gegenprobe:

1. Zitat mit Zeitfenster `[start, end]` erzeugen lassen.
2. Genau dieses Fenster mit ffmpeg aus dem Audio herausschneiden.
3. Den Ausschnitt **separat** transkribieren — dabei muss das Modell nichts
   schätzen, es hört nur, was drin ist.
4. Wortüberlappung zwischen behauptetem Zitat und tatsächlichem Inhalt.

Überlappung 1.0 heißt: jedes Wort des Zitats fällt wirklich in dieses Fenster.

Skripte dazu liegen im Scratchpad der Sitzung (nicht im Repo, weil sie
API-Aufrufe kosten). Das Muster ist oben vollständig beschrieben.

## Befund vor dem Umbau

Korpus: die drei Podcasts aus `tests/golden/audio/` (je 90 s).

| | Wert |
|---|---|
| Gemessene Zitate | 7 |
| Median-Überlappung | 0.90 |
| unter 0.7 | 0 |

Das Timing war also **nicht grob falsch** — die Erwartung „die KI rät
Sekunden, das kann nur schiefgehen" hat sich am Korpus nicht bestätigt.
Die tatsächlichen Probleme lagen woanders:

- **Ränder unsauber.** Das Fenster begann mitten im ersten Wort
  („…brauchen Sie mal zwei Dinge. **Sie brauchen** gewaltige Ressourcen").
- **Text nicht wörtlich.** Der Prompt verlangte „max. 15 Wörter,
  konzentriert und prägnant" — das Modell kürzte und formulierte um. Ein
  eingeblendeter Satz, der so nie gefallen ist, wirkt zwangsläufig
  unsynchron, egal wie exakt die Sekunden sind.
- **Nicht wiederholbar.** Zwei Läufe auf derselben Datei lieferten
  unterschiedliche Zitate und unterschiedliche Anzahl.
- **Anzeige lag hinter der Sprache.** Der Fade-In begann bei `start_time`,
  der Text war also erst `fade_duration` (0.6 s) nach dem Satzanfang voll
  lesbar. Bei einem 2.2-s-Zitat blieb kaum ein Moment volle Deckung.

Ebenfalls geprüft: **Segment-Transkripte mit Zeitstempeln sind genau.**
Bei 10-s- und 3-s-Fenstern lag die Überlappung zwischen behauptetem
Segment-Text und tatsächlich gehörtem Inhalt bei 0.92–1.00. Das trägt als
Zeitquelle.

## Umbau

### 1. Zwei Stufen statt einer

```
Audio ──> Segment-Transkript mit Zeiten ──> Zitat-Auswahl auf reinem Text
                    (einmal)                        (kein Audio)
                       │                                   │
                       └────────► Zeit lokal berechnen ◄────┘
```

Die KI gibt **keine Sekunden mehr aus**. Sie wählt Textstellen; die Zeit
entsteht aus dem Segment-Transkript, über die Zeichenposition innerhalb der
beteiligten Segmente interpoliert (`quote_timing.locate_in_segments`).

Steht ein Zitat nicht wörtlich im Transkript, wird es **verworfen** — seine
Zeit wäre geraten. Das ist gleichzeitig die Absicherung gegen umformulierte
Zitate.

Beide Stufen laufen mit `temperature = 0.0`, damit zwei Klicks dasselbe
Ergebnis liefern.

Fällt Stufe 1 aus, greift weiterhin der alte Ein-Stufen-Pfad direkt am Audio.

### 2. Plausibilitätsprüfung

Ein berechnetes Fenster muss zur Textmenge passen: mindestens 0.15 s und
höchstens 1.2 s pro Wort, mindestens 0.8 s insgesamt. Das fängt kaputte
Zuordnungen ab — im ersten Testlauf entstand ein Fenster von 0.03 s, weil
die Segment-Zeiten am Audio-Ende zusammenfielen.

### 3. Lokales Einrasten auf Sprech-Kanten

`quote_timing.speech_segments` bestimmt aus dem bereits berechneten RMS,
wo überhaupt gesprochen wird (Schwelle zwischen 10.- und 90.-Perzentil,
Wortpausen unter 0.30 s werden geschlossen, Lautinseln unter 0.18 s
verworfen). `snap_to_speech` zieht Start und Ende auf die nächste Kante,
höchstens 1.2 s weit, mit 0.12 s Vorlauf, damit das erste Wort ganz drin
ist. Kein Netzwerk, deterministisch, testbar.

Läuft automatisch bei der Extraktion (wenn `features` übergeben werden) und
manuell über den Knopf „Zeiten einrasten" bzw. im Zitat-Editor.

### 4. Anzeige: Fade vor die Sprechzeit

`lead_in_fade` (Standard an) legt Ein- und Ausblenden **außerhalb** von
`[start, end]`. Der Text steht damit voll, wenn der Satz beginnt, statt
0.6 s danach. `min_display_duration` (3.5 s) verlängert kurze Zitate
ausschließlich **nach hinten** — Lesezeit ohne den Anfang zu verschieben.
`display_duration` bleibt die harte Obergrenze.

### 5. Manuelle Korrektur

Vorher gab es **keine Möglichkeit**, den Zeitpunkt eines Zitats in der GUI
zu ändern — nur den Text (`QInputDialog`). Neu: `QuoteEditorDialog` mit

- Wellenform des Ausschnitts, erkannte Sprech-Abschnitte hinterlegt
- ziehbare Griffe für Start und Ende
- Spinboxen mit ±0.1 / ±0.5-Schritten
- Ausschnitt abspielen (QtMultimedia, mit 0.4 s Vorlauf)
- „Auf Sprechgrenzen einrasten"

Doppelklick in der Zitat-Liste öffnet ihn. Neue Zitate werden direkt darin
angelegt.

## Ergebnis der Gegenprobe nach dem Umbau

Gleiche Messung, gleicher Korpus: Median-Überlappung **0.93**, Texte sind
jetzt wörtlich. Zwei Ausreißer im ersten Lauf gingen auf ein schwaches
Segment-Transkript und ein entartetes Fenster zurück; beide Ursachen sind
adressiert (Temperatur 0, Plausibilitätsprüfung).

## Grenzen

- Drei Podcast-Dateien à 90 s sind eine dünne Grundlage.
- Die Zeiten der Segment-Transkripte sind weiterhin Modell-Ausgaben, keine
  Messung. Bei langen Aufnahmen ist unklar, ob die Genauigkeit hält —
  gemessen wurde nur bis 90 s.
- Das Einrasten arbeitet auf Energie, nicht auf Phonemen. Bei Musikbett
  unter der Stimme sind die Kanten unschärfer.
- Wer echte Frame-Genauigkeit braucht, kommt um einen Forced Aligner
  (z. B. WhisperX) nicht herum — das wäre eine neue, schwere Abhängigkeit
  und ist bewusst nicht eingebaut.
