# KI-Optimierung: warum sie nichts gebracht hat

Stand: 2026-08-11. Betrifft `GeminiIntegration.optimize_all_settings` und
`_validate_optimized_result` in `src/gemini_integration.py`.

Nutzer-Rückmeldung: „KI-Optimierung — das Feature ist gefühlt null hilfreich
momentan in der Form."

Der Eindruck war zutreffend, und zwar wörtlich: **die Funktion hat noch nie
einen einzigen Parameter gesetzt.** Drei Fehler hintereinander, jeder für
sich schon ausreichend.

## Befund

Gemessen durch einen echten Aufruf gegen die API mit
`music_severance.m4a` und `podcast_macy.m4a`.

### 1. Jeder API-Aufruf ist an einer Exception gestorben

```
[Gemini] All-Settings-Optimierung fehlgeschlagen:
additionalProperties is not supported in the Gemini API., verwende Fallback
```

`OPTIMIZE_RESPONSE_SCHEMA` beschrieb `params` und `quotes` als offene Maps
über `additionalProperties`. Genau das akzeptiert die Gemini-API nicht. Der
Aufruf schlug also **immer** fehl — unabhängig von Audio, Modell oder Prompt.

### 2. Der Fallback holte die Werte aus der falschen Datei

Im Fehlerfall lud der Code `config/default.json` und übernahm dessen
`params`/`colors`. Diese Parameter gehören zu einem **anderen Visualizer**
und wurden anschließend vollständig weggefiltert. Ergebnis: `params = {}`.
Auch die vom SmartMatcher aus der Tonart abgeleitete Palette wurde dabei
durch die Standardfarben `#FF0055 / #00CCFF / #0A0A0A` ersetzt — deshalb sah
jede „Optimierung" gleich aus.

Der bereits vorhandene, deterministische Fallback (`_fallback_params`, kennt
die echten Parameter-Specs) wurde dabei übersprungen.

### 3. Selbst mit funktionierendem Aufruf wäre nichts angekommen

Nach dem Entfernen des Schemas antwortete das Modell so:

```json
{ "visualizer": "frequency_flower",
  "parameters": { "num_petals": 6, ... },
  "post_process": { "contrast": 1.2, ... },
  "quote": { ... } }
```

Der Code liest `params`, `postprocess`, `quotes`. Der Prompt nannte die
erwarteten Schlüssel **nirgends** — das sollte das Schema erledigen, und das
Schema war ja kaputt. Also erneut: alles verworfen.

### 4. Nebenbefund: Blur war auf 1.0 gedeckelt

`background.blur` wurde wie `opacity` und `vignette` auf `[0, 1]` geclamped.
Der Regler im Assets-Panel geht bis 20 (Radius). Ein von der KI
vorgeschlagener Blur von 12 kam als 1.0 an — praktisch kein Weichzeichnen.

## Korrektur

- **Kein `response_schema` mehr** für diesen Aufruf. Eine offene
  Parameter-Map ist im Gemini-Schema-Subset nicht ausdrückbar; die Antwort
  wird stattdessen in `_validate_optimized_result` geprüft, geclamped und
  auf bekannte Parameternamen gefiltert.
- **Antwortformat steht jetzt im Prompt**, mit den exakten Schlüsselnamen
  und den erlaubten Wertebereichen für den Hintergrund.
- **Alias-Toleranz** in `_validate_optimized_result`: `parameters`,
  `post_process`, `colour` usw. werden als das erkannt, was sie sind.
  Ein umbenannter Block darf nicht das ganze Ergebnis kosten.
- **Fallback ohne `default.json`.** Bei einem Fehler greift die
  deterministische Berechnung, die die Specs des aktuell gewählten
  Visualizers kennt. Die SmartMatcher-Palette bleibt erhalten.
- **Blur bis 20**, Opacity und Vignette weiterhin 0..1.
- **Ehrliche Statusmeldung.** Das Ergebnis trägt `_source` (`gemini` oder
  `fallback`); die GUI meldet „N Parameter von der KI optimiert" bzw.
  „KI nicht erreichbar — N Parameter aus der Audio-Analyse berechnet".
  Vorher stand dort in beiden Fällen „Parameter optimiert!".
- Die Farb-Regel im Prompt sagt jetzt, dass die vorgeschlagene Palette der
  Ausgangspunkt ist. Vorher standen dort drei feste Paletten, die die
  Tonart-Analyse überschrieben haben.

## Gegenprobe nach der Korrektur

`music_severance` / `frequency_flower`: alle 13 Parameter kommen an
(`num_petals`, `rotation_speed`, `petal_width`, …), Farben bleiben die aus
der Tonart abgeleitete Palette, `_source = "gemini"`.

## Was damit noch nicht beantwortet ist

Ob die von der KI gewählten Werte **gut** sind, ist eine andere Frage als
ob sie ankommen. Das lässt sich nur im Bild beurteilen und braucht eine
Sichtung durch den Nutzer. Bis dahin ist offen, ob der Prompt (Regeln,
Parameter-Beschreibungen) inhaltlich taugt.
