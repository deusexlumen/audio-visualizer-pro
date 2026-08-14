# Quality-Gate mit Auto-Fix (Visualizer Studio, Slice 1)

**Status:** Entwurf, **unverifiziert** — Stufe 0–2 ausgefüllt, Stufe 3–5 offen

> **Beleglage:** In dieser Session war das Repository nicht erreichbar. Kein einziger Punkt ist `[belegt]`.
> Alle Aussagen über Bestandscode stammen aus Projektnotizen vom 2026-07-27 und sind `[erinnert]`.
> Die Verifikationsliste in Abschnitt 5 ist damit nicht Kür, sondern Voraussetzung für Stufe 3.
**Datum:** 2026-08-09
**Slug:** quality-gate

---

## 0. Engpass-Einordnung

**Verdikt: bedingt ja — mit einem ernsten Vorbehalt.**

Der Engpass in diesem Projekt ist die Zeit zwischen "Parameter gesetzt" und "ich weiß, dass es falsch war". Heute läuft diese Schleife über einen vollständigen Render oder über Einzelframe-Previews, die einzelne Momente zeigen, aber keine Kombinationsfehler über die Zeit. Ein Gate, das schlechte Kombinationen vor dem Render meldet, greift genau dort an. Das ist ein echter Engpass, kein Komfort.

**Vorbehalt (P1, blockiert die Sinnhaftigkeit von Slice 1):** Der verifizierte Bestandsstand sagt, dass `CompositeVisualizer` hart `alpha = 1.0` ausgibt (`gpu_visualizers/composite.py:137`), während der Mischpunkt `mix(bg, viz.rgb, viz_alpha)` rechnet (`gpu_renderer.py:1235-1264`). Composite-Visualizer können den Hintergrund damit strukturell nicht durchscheinen lassen — unabhängig von jedem Parameter. Ein Gate, das Parameterkorridore prüft, während der Mischpfad nicht mischt, validiert die falsche Schicht und erzeugt Vertrauen, das nicht gedeckt ist.

**Empfehlung:** Alpha-Pfad zuerst klären (eigener kleiner Vorgang, kein Gate-Thema), dann dieses Vorhaben. Wenn stattdessen zuerst das Gate gebaut wird, gehört "Alpha-Pfad ungeklärt" als bekannte Lücke in die Gate-Ausgabe, damit es nicht als grün durchgeht.

---

## 1. Produkt

*Kein Tech in diesem Abschnitt.*

### Problem

Fehlgriffe bei Visualizer-, Farb- und Parameterwahl fallen erst am fertigen Video auf — nach Minuten Renderzeit, oft erst beim zweiten Ansehen.

### Ankündigung (vorab geschrieben)

> **Visualizer Studio: das Quality-Gate**
>
> Bevor der Render startet, prüft das Studio den Aufbau gegen die Regeln des gewählten Modus. Podcast-Projekte mit einem Musik-Visualizer, Parameter außerhalb ihres Korridors, Farbkombinationen ohne ausreichenden Kontrast zum Hintergrund, Post-Processing-Stufen, die sich gegenseitig aufheben: All das steht als Liste da, bevor die erste Sekunde gerendert ist.
>
> Was sich rechnerisch korrigieren lässt, korrigiert das Studio auf Knopfdruck — mit sichtbarem Vorher/Nachher und einem Klick zum Zurücknehmen. Was Geschmack ist, bleibt Geschmack: Das Gate warnt, es verbietet nichts.

### Sichtbares Verhalten

```
Studio-Panel, vor dem Render:

  [ Prüfen ]   Modus: Podcast

  ● KRITISCH  Visualizer "spectrum_bars" nicht für Podcast freigegeben
              → Vorschlag: "waveform_line"                [Übernehmen]
  ● WARNUNG   Partikeldichte 4800 (Korridor Podcast: 200–1500)
              → Vorschlag: 1500                           [Übernehmen]
  ● WARNUNG   Kontrast Visualizerfarbe/Hintergrund zu gering
              → Vorschlag: Helligkeit +18 %               [Übernehmen]
  ○ HINWEIS   Bloom aktiv bei Post-Grad "dezent"
              → nicht automatisch korrigierbar

  [ Alle korrigierbaren übernehmen ]   [ Erneut prüfen ]

  [ Render starten ]   ← bleibt immer klickbar
```

Ablauf: Nutzer klickt Prüfen → Befundliste erscheint in unter zwei Sekunden → einzeln oder gesammelt übernehmen → Vorher/Nachher sichtbar → erneut prüfen → rendern. Jede Übernahme ist einzeln rücknehmbar.

### Erfolgsmaß

| Kriterium | Zielwert | Wie gemessen |
|---|---|---|
| Wiederholungs-Renders wegen Parameterfehlern | von heutigem Ausgangswert auf ≤ 1/3 | Zählung über 20 Projekte vor/nach; Ausgangswert **muss vorher gemessen werden**, sonst existiert kein Erfolg |
| Falsch-Positive | < 20 % der Befunde | Protokoll über die ersten 20 Läufe: Nutzer markiert jeden Befund mit berechtigt/nicht berechtigt |
| Auto-Fix-Bestandsquote | ≥ 80 % der übernommenen Korrekturen bleiben bestehen | Zählung der Rücknahmen innerhalb derselben Sitzung |
| Prüfdauer | p95 < 2 s, ohne GPU-Render | Messung im Panel, geloggt |
| Falsch-Negative bei harten Regeln | 0 | Testsuite: jede Modus-Whitelist- und Korridorverletzung wird von einem Test erzwungen und muss gemeldet werden |

### Nicht-Ziele

- Keine ästhetische Bewertung ("sieht gut aus") — das Gate prüft Regeln, nicht Geschmack
- Keine ML-basierte Qualitätsbewertung in Slice 1
- Kein Blockieren des Renders; das Gate warnt
- Kein automatisches Umschreiben von Rezeptdateien unter `config/recipes/`
- Keine Echtzeit-Prüfung während der Parameteränderung — nur auf Anforderung
- Keine Prüfung der Audioqualität selbst

---

## 2. Architektur

### Leitentscheidung

**Regeln sind Daten, Prüfer sind eine schmale Maschine.** Modus-Whitelists, Parameterkorridore und Post-Processing-Grade leben deklarativ unter `config/rules/`, nicht als `if`-Kaskade im Panel. Berechnete Eigenschaften, die sich deklarativ nicht ausdrücken lassen — Kontrast, Wechselwirkungen zwischen Post-FX — kommen als registrierte Code-Prüfer dazu.

Begründung: Ein Regelsatz wird sich über Monate ändern, der Prüfmechanismus nicht. Regeln in Panel-Code wären nach der dritten Anpassung verstreut und nicht mehr testbar — das ist P2 aus Stufe 1.

Zweite Leitentscheidung: **Das Gate ist eine reine Funktion auf einem Snapshot.** Kein Qt, kein GPU-Kontext, kein Dateizugriff im Prüfpfad. Alles, was I/O braucht — insbesondere ein repräsentativer Hintergrundframe — wird vorher vom Panel beschafft und in den Snapshot gelegt. Damit ist der Prüfer ohne Oberfläche testbar und das Ziel p95 < 2 s wird zur Trivialität statt zum Risiko.

### Komponenten

```
                  ┌─────────────────────────────────────────┐
   Bestand        │  Qt-Studio-Panel                         │
   [erinnert]     │  sammelt Snapshot, zeigt Befunde         │
                  └───────────────┬─────────────────────────┘
                                  │ ProjectSnapshot (reine Daten)
                                  ▼
                  ┌─────────────────────────────────────────┐
   NEU            │  src/studio/gate.py                      │
   Quality-Layer  │  evaluate(snapshot, ruleset) -> Report    │
                  └───┬──────────────────┬───────────────────┘
                      │                  │
        ┌─────────────▼──────┐   ┌───────▼─────────────────┐
        │ studio/rules/      │   │ studio/checks/          │
        │ Regelmodell +      │   │ Registry:               │
        │ Laden (Pydantic v2)│   │  - deklarative Prüfer   │
        └─────────┬──────────┘   │  - berechnete Prüfer    │
                  │              └─────────────────────────┘
        ┌─────────▼──────────┐
        │ config/rules/      │   NEU, versioniert
        │  music.json        │
        │  podcast.json      │
        └────────────────────┘

                  ┌─────────────────────────────────────────┐
   NEU            │  studio/fixes.py    Vorschlag -> Patch   │
                  │  studio/protocol.py Befund-Protokoll     │
                  └─────────────────────────────────────────┘

   Unverändert:  gpu_renderer.py · gpu_preview.py · ai_matcher.py · Analyzer/NPZ-Cache
                 Der Bestand bleibt Ausführungsschicht. Das Gate schreibt nie in ihn hinein.
```

### Datenfluss

```
[Prüfen] im Panel
  └─ Panel baut ProjectSnapshot
       ├─ Modus (gesetzt, oder Vorschlag aus SmartMatcher-Mode-Detection)  [erinnert]
       ├─ gewählter Visualizer + Parameter
       ├─ Farbwerte
       ├─ PostProcessConfig                                                [erinnert]
       └─ Hintergrund-Repräsentant: mittlere Luminanz + dominante Farbe
          (aus Standbild, bei Video aus einem bereits geladenen Frame)
  └─ gate.evaluate(snapshot, ruleset)
       ├─ Regelsatz laden und validieren -> bei Fehler: genau ein Befund, Abbruch
       ├─ deklarative Prüfer   (Whitelist, Korridore, Post-Grad)
       └─ berechnete Prüfer    (Kontrast, Post-FX-Konflikte, Alpha-Dauerbefund)
  └─ Report -> Panel rendert Befundliste
  └─ Nutzer übernimmt einzeln | gesammelt
       └─ fixes.apply(snapshot, finding) -> (neue Werte, UndoToken)
       └─ nach Sammelübernahme: automatisch erneut prüfen
  └─ protocol.record(...) für jedes Befund-Ereignis
```

Der Prüfpfad löst **keine** Audioanalyse aus. Vorhandene NPZ-Features werden gelesen, nie erzeugt — sonst hängt das Gate an einer Analyse und die 2-Sekunden-Zusage fällt.

### Schnittstellen (neu)

| Schnittstelle | Eingabe | Ausgabe | Anmerkung |
|---|---|---|---|
| `gate.evaluate` | `ProjectSnapshot`, `RuleSet` | `Report` | rein, deterministisch, ohne Seiteneffekt |
| `rules.load` | Modus | `RuleSet` | Pydantic-v2-validiert, wie der Bestand |
| `checks.register` | Prüfer-Callable | — | Registry, ein Prüfer pro Regel-ID |
| `fixes.apply` | `ProjectSnapshot`, Befund-ID | `FixResult` mit `UndoToken` | jede Übernahme einzeln umkehrbar |
| `fixes.revert` | `UndoToken` | `ProjectSnapshot` | |
| `protocol.record` | Befund-Ereignis | — | anhängend, für die Messkriterien aus Stufe 1 |
| Qt-Signal `pruefungFertig` | — | `Report` | Panel-Seite, kein Studio-Wissen über Qt |

### Persistenz

```
config/rules/music.json          NEU   Whitelist, Korridore, Post-Grad
config/rules/podcast.json        NEU   dito
<nutzerdaten>/gate_log.jsonl     NEU   Befund-Protokoll, anhängend
config/recipes/*.json            UNBERÜHRT  (Nicht-Ziel aus Stufe 1)
```

Protokollzeile: Zeitstempel, Regel-ID, Schweregrad, Nutzerurteil berechtigt/nicht, übernommen ja/nein, zurückgenommen ja/nein. Genau die Felder, aus denen sich Falsch-Positiv-Quote und Bestandsquote ohne Nacharbeit rechnen lassen.

### Externe Abhängigkeiten

| Abhängigkeit | Wofür | Verhalten bei Ausfall |
|---|---|---|
| Pydantic v2 (Bestand) | Regelsatz-Validierung | — |
| stdlib | Kontrast per WCAG-Relativluminanz, keine neue Bibliothek | — |
| Hintergrundframe (bei Video über FFmpeg-Pfad des Bestands) | Kontrastprüfung | Befund **"nicht prüfbar"** statt geratenem Ergebnis |
| Regeldatei | alles Deklarative | genau ein Befund "Regelsatz nicht ladbar", Prüfung bricht ab |

Grundhaltung: **laut ausfallen.** Ein Gate, das bei einem internen Fehler schweigend eine leere Befundliste zeigt, ist gefährlicher als kein Gate — der Nutzer liest "alles in Ordnung".

### Ist-Zustand (Bestandscode)

| Aussage | Beleg | Marke |
|---|---|---|
| Mischpunkt `mix(bg, viz.rgb, viz_alpha)` in `_init_composite_shader` | `gpu_renderer.py:1235-1264` | [erinnert] |
| `CompositeVisualizer` gibt hart `alpha = 1.0` | `gpu_visualizers/composite.py:137` | [erinnert] |
| Post-FX nach dem Blend: Bloom, dann Final-Pass | `gpu_renderer.py:511-534` | [erinnert] |
| `PostProcessConfig` als Pydantic-Schema | `config/schemas.py:188-199` | [erinnert] |
| SmartMatcher liefert Mode-Detection speech/music/hybrid | `ai_matcher.py` | [erinnert] |
| Rezepte liegen als JSON vor | `config/recipes/*.json` | [erinnert] |
| Audio-Features im NPZ-Cache abrufbar ohne Neuanalyse | Analyzer | [erinnert], **Verhalten unklar** |
| Qt-Panel kann einen Hintergrundframe synchron liefern | — | [vermutet] |

## 3. Programm-Design

*Offen.*

## 4. Slice-Plan

*Offen.*

## 5. Konfidenz- und Lückenreport

### Konfidenz (Zwischenstand)

| Stufe | Konfidenz | Woran es hängt |
|---|---|---|
| Engpass | Mittel | Hängt daran, ob der Alpha-Pfad vorher geklärt wird |
| Produkt | Mittel | Ausgangswert für Wiederholungs-Renders ist nicht gemessen |
| Architektur | **Niedrig-mittel** | Kein Punkt ist [belegt]; Snapshot-Grenze steht und fällt mit Punkt 4 der Verifikationsliste |

### Annahmen

| Annahme | Was kippt, wenn sie falsch ist |
|---|---|
| Modus-Whitelists und Parameterkorridore sind als Daten formulierbar, nicht als Code | Gate wird zur Regelmaschine statt zur Prüftabelle; Aufwand mindestens verdoppelt |
| SmartMatcher-Mode-Detection ist zuverlässig genug, um den Prüfmodus vorzugeben | Nutzer muss den Modus immer manuell setzen; ein Befundtyp weniger |
| Der Bestandsstand vom 2026-07-27 gilt unverändert | Zeilennummern und Mischpunkt-Befund müssen neu verifiziert werden |

### Lücken

- Ausgangswert für Wiederholungs-Renders fehlt.
- Unklar, ob die Korridore je Visualizer oder je Modus definiert sind. Betrifft die Form von `config/rules/*.json`.
- Ablageort des Befund-Protokolls (nutzerweit vs. projektweit) noch offen.

### Verifikationsliste

Vor Stufe 3 am echten Code nachzusehen. Alles darunter ist heute `[erinnert]` oder `[vermutet]`.

| # | Zu prüfen | Wo suchen | Was kippt, wenn es anders ist |
|---|---|---|---|
| 1 | Gibt `CompositeVisualizer` weiterhin hart `alpha = 1.0` aus | `gpu_visualizers/composite.py`, Suche `alpha` | P1 aus Stufe 0 entfällt oder bleibt Dauerbefund |
| 2 | Felder und Wertebereiche von `PostProcessConfig` | `config/schemas.py`, Suche `class PostProcessConfig` | Regelmodell für Post-Grad muss neu geschnitten werden |
| 3 | Rückgabeform der SmartMatcher-Mode-Detection | `ai_matcher.py`, Suche `mode` / `detect` | Modus-Vorbelegung im Snapshot fällt weg |
| 4 | Liefert der NPZ-Cache Features ohne Neuanalyse, und woran erkennt man einen Treffer | Analyzer-Modul, Suche `npz` / `cache` | Snapshot-Aufbau wird I/O-lastig, 2-Sekunden-Zusage fällt |
| 5 | Kann das Panel synchron einen Hintergrundframe liefern (Bild und Video) | Panel-/Preview-Code, Suche `frame` / `background` | Kontrastprüfung wird asynchron, Gate braucht einen Worker |
| 6 | Existieren Parameterkorridore heute schon irgendwo implizit | `config/recipes/*.json`, Suche nach Min/Max-Werten | Regeldateien können daraus abgeleitet statt erfunden werden |

### Risiken

| Tier | Risiko | Gegenmaßnahme |
|---|---|---|
| P1 | Alpha-Pfad ungeklärt, Gate suggeriert Korrektheit auf einer Schicht, die nicht mischt | Alpha zuerst klären oder als Dauerbefund im Gate ausweisen |
| P2 | Falsch-Positive erziehen zum Wegklicken; das Gate wird Dekoration | Falsch-Positiv-Quote als hartes Abnahmekriterium, siehe oben |
| P2 | Regeln landen verstreut in Panel-Code statt in Daten | Entscheidung in Stufe 2 explizit treffen und als ADR ablegen |
| P3 | Auto-Fix-Vorschläge kollidieren untereinander bei Sammelübernahme | Reihenfolge festlegen, nach Sammelübernahme automatisch erneut prüfen |
| P2 | Gate fällt still aus und zeigt leere Befundliste als "alles gut" | Fehler des Regelsatzes und nicht prüfbare Checks erscheinen als eigene Befunde |

---

## Änderungslog

| Datum | Was sich am Design geändert hat | Warum |
|---|---|---|
| 2026-08-09 | Erstfassung, Stufe 0 und 1 | — |
| 2026-08-09 | Stufe 2, Beleglage-Marken, Verifikationsliste | Repo in dieser Session nicht erreichbar |
