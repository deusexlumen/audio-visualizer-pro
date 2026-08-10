---
name: programm-design
description: Unbedingt verwenden, sobald ein nicht-triviales Feature, ein neues Modul, ein Refactoring, eine Migration, eine Datenpipeline oder ein neues Projekt ansteht — auch wenn direkt "bau mir X", "implementier Y", "schreib mir das Tool" oder "lass uns loslegen" gesagt wurde und ausdrücklich kein Plan verlangt war. Ebenso bei Fragen wie "wie ziehe ich das auf", "in welcher Reihenfolge baue ich das", bei Slice- und Meilensteinplanung sowie ADR- und Repo-Kontextdateien. Nicht verwenden für Einzeiler, Ein-Datei-Bugfixes, reine Wissensfragen oder das Härten bereits fertigen Codes.
---

# Programm-Design — das Gate vor dem Code

> **Kein Implementierungscode in Stufe 0–5.** Signaturen ja, Bodies nein.
> **Artefakte sind Dateien**, nicht Chat-Text: `docs/design/<slug>/design.md`.
> **Jede Stufe endet mit einem menschlichen Checkpoint** (Stufe 0 läuft mit Stufe 1 zusammen). Nicht durchrauschen.

---

## The Point

Modelle lösen Probleme hervorragend und schreiben trotzdem unwartbaren Code, wenn niemand vorne mitdenkt. Der Grund ist strukturell, nicht mangelnde Intelligenz: Reinforcement Learning belohnt „Tests grün", nicht „gutes Design". Für Wartbarkeit existiert kein schnelles Orakel — Tests laufen in Sekunden, die Kosten schlechter Architektur zeigen sich in Wochen. Also wird darauf nicht optimiert. Ein Benchmark-Score von 99% sagt nichts darüber, ob der Codebase in zwei Monaten noch änderbar ist.

Daraus folgt der einzige echte Hebel: **menschliche Intuition vorne einspeisen, nicht hinten im Review verbrennen.**

- Solange nur Text existiert, kostet Umsteuern Minuten.
- Nach 800 Zeilen Code kostet es Stunden — und das Modell ist durch die eigenen frühen Entscheidungen bereits gebiast, tief im Kontextfenster, mit weniger nutzbarer Intelligenz pro Token.
- Denken ist am Anfang eines Kontextfensters am billigsten und am schärfsten. Design-Sessions bleiben deshalb klein und token-effizient, während sie die teuersten Entscheidungen treffen.

Zweiter Hebel: **Backpressure.** Ein Agent mit einer messbaren Zielgröße versetzt Berge. Ein Agent mit „mach es sauber" würfelt. Jede Stufe hier erzeugt deshalb etwas, das später prüfbar ist — deterministisch, wenn möglich.

Siehe `references/warum-modelle-slop-schreiben.md`, wenn die Begründung gegenüber jemandem verteidigt werden muss.

---

## Skip-Regel — wann dieser Skill NICHT läuft

Dieser Skill ist Leverage, kein Ritual. Er lohnt sich nicht überall:

| Situation | Entscheidung |
|---|---|
| Einzeiler, Typo, CSS-Tweak, Ein-Datei-Bugfix | **Skip.** Direkt bauen. |
| Prototyp, der ausdrücklich weggeworfen wird | **Skip.** Explosionsradius ist null. |
| Pre-PMF-Exploration, Ziel ist Lernen statt Halten | **Skip oder nur Stufe 1.** Geschwindigkeit schlägt Struktur. |
| Feature über mehr als 2 Dateien oder 2 Schichten | **Voll.** |
| Etwas, das in 6 Monaten noch laufen und änderbar sein muss | **Voll.** |
| Datenmigration, Auth, Krypto, Zahlungen, alles Irreversible | **Voll, ohne Ausnahme.** |
| Ein Refactoring, das Code anfasst, der nicht mehr verstanden wird | **Voll**, plus Stufe 2 als Ist-Aufnahme. |

Wenn geskippt wird, das in einem Satz sagen und begründen. Kein stiller Skip.

---

## Session-Setup

Vom Nutzer wird gebraucht (fehlt etwas, aus Repo und Kontext erschließen und die Annahme als solche markieren):

- **Vorhaben** — was gebaut werden soll, in Nutzersprache
- **Repo/Projekt** — Pfad oder Beschreibung; existierender Code wird gelesen, nicht geraten
- **Zwang** — Deadline, Stack, Kompatibilität, Betriebsumgebung

Fehlende Angaben blockieren nicht. Annahmen treffen, sichtbar machen, im Lückenreport führen.

---

## Beleglage — jede Aussage über Bestandscode trägt ihre Quelle

Das häufigste stille Scheitern eines Designs ist nicht ein Denkfehler, sondern eine Behauptung über den Ist-Zustand, die niemand geprüft hat. Ein Design, das auf einer falsch erinnerten Signatur steht, ist unbrauchbar — und man merkt es erst mitten im Bau.

Deshalb bekommt jede Aussage über Bestehendes eine von drei Marken:

| Marke | Bedeutung |
|---|---|
| `[belegt]` | im Repo gelesen, mit Pfad und Zeile |
| `[erinnert]` | aus Notizen, früheren Sitzungen oder Doku, nicht nachgeprüft |
| `[vermutet]` | Standardannahme, keine Quelle |

**Wenn das Repo nicht erreichbar ist** — reine Chat-Session, kein Pfad, kein Dateizugriff — wird trotzdem designt. Aber dann gilt:

1. Alles über Bestandscode ist `[erinnert]` oder `[vermutet]`. Nichts wird als `[belegt]` ausgegeben, nur weil es plausibel klingt.
2. Das Doc trägt Status **Entwurf, unverifiziert** — nicht Freigegeben.
3. Stufe 3 wird geschrieben, aber nicht als verbindlich behandelt: Signaturen gegen unbelegten Bestand sind Vorschläge.
4. Der Lückenreport enthält eine **Verifikationsliste** — konkrete Punkte, die vor dem ersten Slice am echten Code nachzusehen sind, mit Datei und Suchbegriff, nicht als vage Absicht.

Der Sinn ist nicht Bürokratie. Es ist der Unterschied zwischen "dieses Design ist fertig" und "dieses Design ist fertig, sobald drei Zeilen geprüft sind" — und der zweite Satz ist der ehrliche.

---

## Checkpoint — wie eine Stufe endet

Der Wert dieses Skills entsteht **beim Menschen**, nicht im Text. Eine Stufe zu schreiben ist billig; der teure und nützliche Moment ist der, in dem jemand "warum liegt das da" sagt.

Wenn alle Stufen in einem Zug ausgegeben werden, liest der Mensch zweitausend Wörter am Stück und sagt "passt". Das ist exakt der Fehlermodus, gegen den der ganze Skill gebaut ist — Review am Ende, nur ohne Code. Ein Design in einem Rutsch ist kein Gate, sondern ein Dokument.

**Eine Stufe pro Turn** — mit einer Einordnung: Stufe 0 ist ein Zwei-Minuten-Verdikt und läuft mit Stufe 1 im selben Turn; der erste Checkpoint-Block kommt am Ende von Stufe 1. Ab Stufe 1 gilt: Eine Stufe endet mit diesem Block und dann ist der Turn zu Ende:

```
── Checkpoint Stufe N ──
Entschieden: <zwei bis vier Zeilen, was jetzt feststeht>
Offen:       <was diese Stufe bewusst nicht entschieden hat>
Wackelig:    <die eine Entscheidung, die am ehesten kippt — und was sie kippen würde>
```

**Kein Optionsmenü zurückreichen.** Der Checkpoint ist zum Widersprechen da, nicht zum Auswählen. Wer die Stufe geschrieben hat, hat auch die Abwägung im Kopf — eine Frage der Form „A oder B?" schiebt genau diese Arbeit zurück, und zwar an jemanden mit weniger Kontext. „Nein, anders" ist ein billigerer Einwand als eine Wahl, die erst durchdacht werden muss.

Echte Fragen bleiben erlaubt, wenn die Information ausschließlich im Kopf des Nutzers existiert — Frist, Geschmack, Geschäftsentscheidung, unbekannter Betriebszwang. Dann eine, klar gestellt, und trotzdem mit einem Vorschlag daneben.

Kein Ausblick auf die nächste Stufe, keine vorweggenommene Architektur, kein "ich habe schon mal angefangen". Der Zug in diese Richtung ist stark — er fühlt sich nach Hilfsbereitschaft an und ist das Gegenteil.

**Ausnahme:** Wenn der Nutzer ausdrücklich alles auf einmal will, dann so liefern — aber einmal klar sagen, dass das Gate damit zum Dokument wird und Einwände jetzt teurer sind. Danach nicht weiter darauf herumreiten.

---

## Ablauf

### Stufe 0 — Engpass und Fundament (2 Minuten, nicht optional)

Zwei unbequeme Fragen, bevor irgendetwas designt wird.

**a) Ist das hier überhaupt der Engpass?**

Es gibt Ineffizienzen, die keine Engpässe sind. Ein schnelleres Agenten-Setup hilft nicht, wenn der Flaschenhals das Review ist. Ein weiteres Feature hilft nicht, wenn niemand das bestehende benutzt.

**b) Ist die Schicht darunter intakt?**

Ein Vorhaben kann sauber designt und trotzdem sinnlos sein, weil etwas tiefer im System nicht tut, was es soll. Ein Prüfmechanismus über einem Pfad, der die geprüfte Eigenschaft gar nicht durchreicht. Ein Cache vor einer Abfrage, die falsche Daten liefert. Eine Optimierung an einer Stelle, die nie erreicht wird.

Also: **Setzt dieses Vorhaben etwas voraus, das nachweislich nicht funktioniert?** Bekannte Defekte, offene Befunde, Stellen mit „das war schon immer komisch". Wenn ja, ist die Reihenfolge das Ergebnis dieser Stufe, nicht das Design.

Ausgabe: ein bis zwei Absätze und ein klares Verdikt. Wenn das Vorhaben nicht der Engpass ist oder auf einem kaputten Fundament steht, das benennen und die Alternative nennen. Danach trotzdem weitermachen, wenn der Nutzer das will — es ist sein Feuer, das brennen darf. Aber ein bekannter Defekt darunter wandert dann als P1 in den Lückenreport und darf nicht stillschweigend grün werden.

Stufe 0 bekommt keinen eigenen Checkpoint — das Verdikt steht im Doc über Stufe 1, und derselbe Turn liefert direkt Stufe 1 mit.

### Stufe 1 — Produkt (kein Tech)

Hier wird kein Wort über Datenbanken, Schemas oder Frameworks verloren. Fällt eine Technologie, gehört sie nach Stufe 2.

1. **Problem** — welches Nutzerproblem, in einem Satz, ohne Lösung darin.
2. **Ankündigung vorab schreiben** — die zwei Absätze, mit denen das Feature später erklärt würde (Changelog, Discord-Post, README-Abschnitt). Lässt es sich nicht attraktiv schreiben, ist es meist nicht attraktiv.
3. **Sichtbares Verhalten** — was sieht und tut der Nutzer, Schritt für Schritt. Bei UI: rohes HTML/ASCII-Mockup statt Prosa.
4. **Erfolgsmaß** — woran wird nach dem Merge erkannt, dass es funktioniert hat. Eine Zahl oder ein Ereignis, kein Gefühl.
5. **Nicht-Ziele** — was ausdrücklich nicht gebaut wird.

Erfolgsmaße sind der Punkt, an dem die meisten Pläne verwässern. `references/messbare-kriterien.md` enthält Übersetzungen von „gut" in Zahlen, aufgeschlüsselt nach Projekttyp.

**Checkpoint:** Stimmt das Problem und das Maß? Erst dann weiter.

### Stufe 2 — Systemarchitektur

Eine Ebene tiefer. Immer noch kein Implementierungscode.

- Komponenten und wie sie zusammenhängen (ASCII- oder Mermaid-Diagramm)
- Datenfluss von Auslöser bis Ergebnis
- Neue oder geänderte Schnittstellen: Endpunkte, Events, IPC-Kanäle, CLI-Kommandos — mit Ein- und Ausgabeform
- Neue oder geänderte Persistenz: Tabellen, Dateien, Keys, Schema-Skizze
- Externe Abhängigkeiten und was passiert, wenn sie ausfallen
- Bei Bestandscode: was **ist** heute, bevor beschrieben wird, was werden soll — mit Beleglage-Marke pro Aussage

**Checkpoint:** Passt der Schnitt? Erst dann weiter.

### Stufe 3 — Programm-Design (die Stufe, die alle überspringen)

Das ist die Stufe mit dem höchsten Hebel und der geringsten Verbreitung. Hier werden genau die Entscheidungen vorweggenommen, die ein Agent sonst still und nach eigenem Geschmack trifft — und die hinterher teuer sind.

Vier Artefakte, alle kompakt, alle in Codeblöcken, damit ein Mensch sie in dreißig Sekunden scannen und „falsch" sagen kann:

**a) Dateilayout** — welche Datei entsteht wo, welche wird angefasst.
```
src/
  parser/
    frontmatter.ts        NEU   — Parsing + Validierung
    frontmatter.test.ts   NEU
  pipeline/
    ingest.ts             EDIT  — ruft Parser auf, Zeile ~40
```

**b) Call-Stack** — der Aufrufpfad des Hauptszenarios, von außen nach innen.
```
CLI ingest <pfad>
  └─ loadSources(pfad)
      └─ parseFrontmatter(raw)        <- neu
          └─ validateSchema(parsed)   <- neu
      └─ writeIndex(docs)             <- bestehend, Signatur unverändert
```

**c) Typen und Signaturen** — Datenformen und Funktionsköpfe. **Keine Bodies.** Fehlerfälle gehören in die Signatur, nicht in einen späteren `catch`.
```ts
type Frontmatter = { title: string; tags: string[]; updated: Date | null };
type ParseResult =
  | { ok: true; data: Frontmatter; body: string }
  | { ok: false; reason: 'missing' | 'malformed' | 'schema'; line: number };

function parseFrontmatter(raw: string): ParseResult;
```

**d) Testskizze** — welche Tests entstehen, was sie beweisen. Ein Test, der auch vor der Änderung grün wäre, testet nichts. Diese Frage hier stellen, nicht im Review.
```
frontmatter.test.ts
  - leerer Input            -> ok:false, reason:'missing'
  - Delimiter ohne Ende     -> ok:false, reason:'malformed', line korrekt
  - unbekannter Key         -> ok:true, Key wird verworfen
  - tags als String statt Liste -> ok:false, reason:'schema'
```

**Checkpoint:** Der Nutzer sagt hier typischerweise „warum liegt das da" oder „der Fehlerfall fehlt". Genau dafür ist die Stufe da. Billiger geht Korrektur nicht.

### Stufe 4 — Vertikale Slices

Modelle bauen von sich aus **horizontal**: erst die komplette Datenschicht, dann alle Services, dann die API, dann das Frontend. Ergebnis: tausende Zeilen, und bis zum Schluss gibt es nichts, was man ausprobieren kann.

Deshalb wird der Bau in **vertikale Slices** geschnitten — jeder Slice geht durch alle Schichten und ist am Ende ausführbar:

1. **Slice 1 ist ein Tracer Bullet:** dünnster Ende-zu-Ende-Pfad, hartkodierte Daten erlaubt, ein einziger Happy Path. Er beweist, dass die Verdrahtung stimmt.
2. Jeder weitere Slice ersetzt Attrappen durch Echtes oder fügt genau einen Fall hinzu.
3. Fehlerbehandlung, Randfälle und Politur sind **eigene Slices am Ende**, nicht überall verstreut.

Pro Slice wird notiert: Name, was danach funktioniert, **wie man es prüft** (konkreter Befehl, Klickpfad, curl, Testlauf), grobe Größe.

Vorlagen für typische Stacks — Web/API, CLI, Userscript, Desktop, Datenpipeline — stehen in `references/vertikale-slices.md`.

**Checkpoint:** Reihenfolge freigegeben?

### Stufe 5 — Konfidenz- und Lückenreport

Der Abschluss jeder Design-Session, ohne Ausnahme. Die Frage „welche Entscheidungen hast du getroffen, bei denen du unsicher bist?" wird üblicherweise gestellt, *nachdem* der Code steht. Hier wird sie gestellt, solange Ändern noch nichts kostet.

Ehrlich, nicht dekorativ. Ein Report ohne einzige Unsicherheit ist ein Report, der nicht nachgedacht hat.

- **Konfidenz je Stufe** — hoch/mittel/niedrig plus ein Satz, woran es hängt
- **Wackelige Entscheidungen** — was anders gehen könnte und welches Signal für einen Wechsel spräche
- **Annahmen** — was ohne Bestätigung angenommen wurde, und was kippt, wenn die Annahme falsch ist
- **Lücken** — was nicht geklärt werden konnte und wer oder was es klären müsste
- **Verifikationsliste** — alles `[erinnert]` oder `[vermutet]`, das vor dem ersten Slice am echten Code nachzusehen ist; je Punkt Datei und Suchbegriff, damit die Prüfung Minuten dauert und nicht eine Stunde
- **Risiken in P-Tiers** — P1 zerstört Daten, Sicherheit oder Betrieb; P2 verursacht Nacharbeit über mehrere Slices; P3 ist kosmetisch oder später billig zu fixen

**Checkpoint:** Freigabe. Danach — und erst danach — wird gebaut.

---

## Nach der Freigabe — Bauregeln

- **Ein Slice pro Session.** Nach jedem Slice: ausführen, prüfen, dann der nächste.
- **Design-Doc statt Gedächtnis.** Neue Session beginnt mit dem Doc, nicht mit dem Gesprächsverlauf.
- **Abweichung meldet sich.** Wenn beim Bauen klar wird, dass das Design falsch ist: stoppen, Doc korrigieren, weiter. Nicht still danebenbauen — sonst ist das Doc nach zwei Tagen Fiktion.
- **Kontext-Hygiene.** Wird die Session lang und die Antworten flach, ist das kein Signal für mehr Prompting, sondern für einen Schnitt: Stand ins Doc, neue Session. Details in `references/kontext-hygiene.md`.
- **Kontext ins Repo.** Getroffene Entscheidungen wandern nach `docs/adr/`, Umgebungswissen nach `docs/external/`. Was im Repo steht, muss nicht in jeden Prompt.

---

## Ausgabeformat

Vollständige Vorlage: `assets/design-doc-vorlage.md`. Gerüst anlegen mit:

```bash
python scripts/scaffold.py <slug> --titel "<Titel>" --repo <pfad>
```

Struktur von `docs/design/<slug>/design.md`:

```markdown
# <Vorhaben>
Status: Entwurf | Entwurf, unverifiziert | Freigegeben | In Bau | Erledigt
Datum: YYYY-MM-DD

## 0. Engpass und Fundament
## 1. Produkt
### Problem / Ankündigung / Verhalten / Erfolgsmaß / Nicht-Ziele
## 2. Architektur
### Komponenten / Datenfluss / Schnittstellen / Persistenz / Abhängigkeiten
## 3. Programm-Design
### Dateilayout / Call-Stack / Typen und Signaturen / Testskizze
## 4. Slice-Plan
| # | Slice | Danach funktioniert | Prüfbefehl | Größe |
## 5. Konfidenz- und Lückenreport
### Konfidenz / Wackelige Entscheidungen / Annahmen / Lücken / Verifikationsliste / Risiken (P1-P3)
```

---

## Rules

- **KEIN CODE VOR DER FREIGABE** — Typen, Signaturen, Testnamen sind Design. Bodies sind es nicht.
- **EINE STUFE PRO TURN** — Stufe 0 läuft mit Stufe 1 zusammen, danach endet jede Stufe am Checkpoint-Block. Alle Stufen am Stück ausgeben heißt, das Gate selbst zu umgehen.
- **„KEIN PLAN NÖTIG" IST KEIN SKIP-GRUND** — wer „einfach loslegen" sagt, meint das Ergebnis, nicht den Prozess. Über den Skip entscheidet allein die Skip-Tabelle, und wer sie aufruft, begründet das in einem Satz.
- **ENTSCHEIDEN, NICHT AUSWÄHLEN LASSEN** — der Checkpoint nennt die Entscheidung und das Wackelige daran. Ein zurückgereichtes Optionsmenü ist keine Sorgfalt, sondern verschobene Arbeit.
- **CODEBLÖCKE STATT PROSA** — Signaturen und Aufrufpfade werden in Sekunden geprüft, Absätze in Minuten.
- **MESSBAR SCHLÄGT SCHÖN** — „soll performant sein" ist kein Kriterium. „p95 unter 200 ms bei 1000 Zeilen" ist eins.
- **ERST LESEN, DANN DESIGNEN** — und wenn nicht gelesen werden kann, wird das markiert, nicht überspielt. `[belegt]`, `[erinnert]`, `[vermutet]`.
- **VERTIKAL, NIE HORIZONTAL** — wenn ein Slice nichts Ausführbares hinterlässt, ist er falsch geschnitten.
- **UNSICHERHEIT IST ERGEBNIS** — ein Design ohne offene Punkte ist entweder trivial oder unehrlich.
- **DAS DOC IST DIE WAHRHEIT** — es lebt im Repo und wird beim Abweichen sofort nachgezogen.
- **KEIN RITUAL** — die Skip-Regel ist Teil des Skills, nicht ihr Feind.
- **DER BUCHSTABE IST DER GEIST** — wer „im Sinne des Skills" codet, bevor Stufe 5 freigegeben ist, umgeht das Gate.

---

## Häufige Ausreden — beobachtet und beantwortet

Diese Rationalisierungen treten unter Druck (Deadline, Autorität, „ist doch klar") zuverlässig auf:

| Ausrede | Realität |
|---|---|
| „Demo in einer Stunde — keine Zeit fürs Gate" | Zeitdruck ändert den Slice-Umfang, nicht das Gate. Unter Zeitdruck ist eine falsche Weiche am teuersten. |
| „Das Konzept ist schon klar" | Klarheit im Kopf des Nutzers ist nicht prüfbar. Das Doc macht sie prüfbar — in Minuten statt Stunden. |
| „Teamlead/Chef hat den Ansatz schon abgenommen" | Umso schneller sind Stufe 0–2 durch. Eine mündliche Abnahme ersetzt das prüfbare Doc nicht. |
| „Ich interpretiere die Anforderung eben selbst" | Genau diese stillen Entscheidungen sind der Fehlermodus, gegen den Stufe 3 gebaut ist. |
| „Ist doch nur ein kleines Feature" | Das entscheidet die Skip-Tabelle oben — nicht das Gefühl im Moment. |
| „Ich frage lieber gar nicht erst, das bremst nur" | Der Checkpoint ist kein Bremsen, sondern der Moment, in dem der Skill seinen Wert erzeugt. |

---

## Red Flags — STOPP, zurück zum Gate

- Implementierungscode vor der Freigabe in Stufe 5
- Alle Stufen in einem Zug, ohne dass der Nutzer es ausdrücklich verlangt hat
- „Kein Plan nötig" oder „Demo-Druck" als Skip-Begründung
- Checkpoint als Optionsmenü („A oder B?") statt als Entscheidung mit Wackelkandidat
- Aussage über Bestandscode ohne `[belegt]` / `[erinnert]` / `[vermutet]`
- Stiller Skip ohne Satz und Begründung
- Erfolgsmaß ohne Zahl oder Ereignis

Jedes davon heißt: Gate umgangen. Zurück zur aktuellen Stufe.

---

## Referenzdateien

Bei Bedarf lesen, nicht vorsorglich:

| Datei | Wann |
|---|---|
| `references/vertikale-slices.md` | Stufe 4, besonders bei unklarem Schnitt oder ungewohntem Stack |
| `references/messbare-kriterien.md` | Stufe 1, wenn das Erfolgsmaß schwammig bleibt |
| `references/kontext-hygiene.md` | lange Sessions, Handoffs, Repo-Kontextdateien, ADRs |
| `references/warum-modelle-slop-schreiben.md` | wenn die Begründung des Vorgehens gefragt oder bestritten wird |
