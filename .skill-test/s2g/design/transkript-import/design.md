# Transkript-Import-Pipeline (Zitate → quotes.json)
Status: Entwurf
Datum: 2026-08-09

> Ablage-Hinweis: Der Skill schreibt `docs/design/<slug>/` vor; der Nutzer hat
> Schreibzugriffe auf `.skill-test/s2g/` begrenzt, daher liegt das Doc hier.

## 0. Engpass und Fundament

**Engpass:** Das angefragte Feature existiert bereits weitgehend fertig:
`.skill-test/s2/transcript_import.py` [belegt, gesamte Datei, 128 Zeilen]
implementiert genau die geforderte Pipeline — Zeilenparser
`[MM:SS] Sprecher: Text` (inkl. `H:MM:SS`), Zitat-Erkennung über
Anführungszeichen (`"`/`„“`/`«»`), Zeitstempel→Sekunden, Duplikat-Erkennung
(normalisierter Text), JSON-Export und argparse-CLI mit `-o`. Der vom Nutzer
geschilderte Stand („nur `load_transcript` + TODO") ist überholt. Der echte
Engpass ist daher nicht Implementierung, sondern **Konsolidierung**: die
fertige Lösung aus `s2` nach `s2g` überführen (s2 bleibt read-only), dabei die
Schema-Lücke zum Projekt schließen und mit Tests belegen. Ein Neubau von Grund
auf wäre Duplikation bereits abgenommenen Codes.

**Fundament:** Das Projekt hat ein bestehendes Zitat-Ökosystem:
`src/types.py:11` [belegt] definiert `Quote` mit `start_time`, `end_time`,
`confidence`; `src/gemini_integration.py:775-950` [belegt] erzeugt Quotes mit
`start_time`/`end_time` als Float-Sekunden. Die s2-Pipeline schreibt nur
`start_time` (int) und weder `end_time` noch `confidence` — ihre `quotes.json`
ist damit **nicht direkt** vom bestehenden Quote-Schema ladbar. Das ist kein
Blocker für die CLI als Standalone-Tool, aber ein bekannter Defekt relativ zum
Projekt-Fundament und wandert als P2 in den Lückenreport (Stufe 5).

**Verdikt:** Weitermachen — aber als Überführung + Härtung des Bestehenden,
nicht als Neubau. Zielverzeichnis `.skill-test/s2g/`, Quelle `.skill-test/s2/`
nur lesend.

## 1. Produkt
_(folgt nach Checkpoint Stufe 0)_

## 2. Architektur
_(folgt)_

## 3. Programm-Design
_(folgt)_

## 4. Slice-Plan
_(folgt)_

## 5. Konfidenz- und Lückenreport
_(folgt)_
