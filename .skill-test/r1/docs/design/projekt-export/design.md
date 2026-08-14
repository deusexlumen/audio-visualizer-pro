# Projekt-Export (ZIP-Bundle mit Manifest)

Status: Entwurf
Datum: 2026-08-09

> Hinweis zur Ablage: Das Doc liegt unter `.skill-test/r1/docs/design/…` statt
> `docs/design/…`, weil in dieser Session Schreibzugriffe auf `.skill-test/r1/`
> beschränkt sind. Bei Freigabe ins Repo verschieben.

## 0. Engpass und Fundament

**a) Ist das der Engpass?** Für den genannten Zweck — ein Projekt von einem
Rechner/Ordner auf einen anderen bringen oder archivieren — ja, plausibel:
Projekt-JSONs verweisen heute auf absolute Pfade außerhalb des Repos
(`[belegt]` projects/podcast.json:3-4 — `audio_path` und `background_path`
zeigen nach `C:\Users\Buxe\Downloads\…`). Ohne Bündelung ist ein Projekt außer
auf genau diesem Rechner faktisch nicht reproduzierbar. Das ist ein reales,
greifbares Problem, kein Luxus-Feature. Ob es *der* Engpass des Produkts
insgesamt ist, kann aus dem Repo nicht beantwortet werden — als
Einzelmaßnahme ist es aber kohärent und klein genug, um nicht zu schaden.

**b) Fundament intakt?** Das Vorhaben setzt voraus, dass Laden eines
Projekt-JSON zuverlässig funktioniert — sonst ist ein Export ohne funktionierenden
Re-Import wertlos. Speichern/Laden existiert und ist zentral an einer Stelle:
`AppState.to_dict()` `[belegt]` src/gui/state.py:145, Menüaktionen
"Projekt speichern/öffnen" `[belegt]` src/gui/main_window.py:173-186,
Serialisierung via `json.dump(self.state.to_dict(), …)` `[belegt]`
src/gui/main_window.py:251. Kein bekannter Defekt an dieser Stelle im Repo
sichtbar (keine TODOs/FIXMEs dort). **Aber:** Ein Export, der absolute Pfade
einfach mitbündelt, erzeugt ein ZIP, dessen JSON nach dem Entpacken auf die
alten absoluten Pfade zeigt — der Re-Import-Schritt ist damit implizit Teil
des Fundaments. Das wird in Stufe 2/3 entschieden (Pfade im exportierten JSON
relativ umschreiben vs. unverändert lassen).

**Verdikt:** Engpass plausibel, Fundament ausreichend intakt. Weiter mit Stufe 1.
Offener Fundament-Punkt (Pfad-Umschreibung beim Export) wandert als P2 in den
Lückenreport (Stufe 5), falls er dort nicht aufgelöst wird.

## 1. Produkt

### Problem

Ein gespeichertes Projekt ist nicht portabel: Das JSON zeigt auf Audio- und
Bilddateien irgendwo auf der Platte, und wer den Ordner weitergeben oder
archivieren will, muss von Hand raten, welche Dateien dazugehören — und merkt
fehlende Dateien erst beim nächsten Rendern.

### Ankündigung (vorab geschrieben)

> **Neu: Projekt-Export als ZIP.** Ein Klick im Studio-Bereich packt das
> komplette Projekt — JSON, Audiodatei, Hintergrundbilder, verwendete
> Configs — in ein einziges ZIP-Archiv. Bereit zum Archivieren, Teilen oder
> Umziehen auf einen anderen Rechner.
>
> Vor dem Packen prüft der Export, ob alle referenzierten Dateien existieren,
> und meldet fehlende Assets in einer klaren Liste — kein Ratespiel mehr beim
> Render-Start. Optional legt er eine `manifest.json` mit SHA256-Prüfsummen
> bei, mit der sich später maschinell verifizieren lässt, dass das Archiv
> vollständig und unverändert ist.

### Sichtbares Verhalten

Schritt für Schritt:

1. Nutzer öffnet ein Projekt (oder hat eines geladen) und klickt im
   Studio-Panel auf **„Projekt exportieren…"**.
2. Dateidialog: Ziel-ZIP wählen (Vorschlag: `<projektname>_export.zip`).
3. Der Export sammelt alle referenzierten Pfade aus dem Projekt-JSON.
4a. **Alles vorhanden:** ZIP wird geschrieben, Abschlussmeldung zeigt
    Dateianzahl und Gesamtgröße.
4b. **Assets fehlen:** Dialog mit Liste der fehlenden Pfade. Nutzer wählt:
    „Trotzdem exportieren (ohne fehlende Dateien)" oder „Abbrechen".
    Default: Abbrechen.

Rohes UI-Mockup (Studio-Panel, Ausschnitt):

```
┌ Studio ────────────────────────────────────────┐
│  [ Projekt exportieren… ]                      │
└────────────────────────────────────────────────┘

Fehlerdialog (4b):
┌ Fehlende Assets ───────────────────────────────┐
│  2 Dateien wurden nicht gefunden:              │
│    • C:\…\Downloads\musik.m4a   (Audio)        │
│    • C:\…\Downloads\bg.png      (Hintergrund)  │
│                                                │
│  [ Abbrechen ]  [ Trotzdem exportieren ]       │
└────────────────────────────────────────────────┘

Abschlussmeldung (4a):
  „Export abgeschlossen: podcast_export.zip — 5 Dateien, 41,3 MB
   (inkl. manifest.json mit SHA256-Prüfsummen)"
```

ZIP-Inhalt (flach, sprechende Namen):

```
podcast_export.zip
├── project.json            # Projekt-JSON (siehe offener Punkt: Pfade)
├── audio/<dateiname>       # Audiodatei
├── backgrounds/<dateiname> # Hintergrundbild(er)
├── configs/<dateiname>     # referenzierte Config-Presets
└── manifest.json           # optional: SHA256 je Datei + Metadaten
```

Die Manifest-Option ist eine Checkbox im Export-Dialog, Default: **an**.

### Erfolgsmaß

Nach dem Merge, an einer echten Demo oder am Beispielprojekt
`projects/podcast.json` gemessen:

1. Ein Export von `podcast.json` erzeugt ein ZIP, das Audio + Hintergrund +
   project.json enthält — **und** der Dialog meldet beim aktuellen Stand
   genau die 2 fehlenden Downloads-Dateien als fehlend (beide Pfade existieren
   `[vermutet]` nicht mehr auf diesem Rechner — zu prüfen, s. Verifikationsliste).
2. Bei vollständig vorhandenen Assets: Export einer 50-MB-Projektmappe
   (Audio ~45 MB + Bilder) dauert **unter 30 Sekunden** und das entpackte
   ZIP verifiziert gegen `manifest.json` ohne einzige Prüfsummen-Differenz
   (Ereignis: „0 Fehler beim Verify-Lauf").

### Nicht-Ziele

- **Kein Import/„Projekt aus ZIP öffnen"** in dieser Iteration — der Export
  muss ohne ihn schon nützlich sein (Archiv/Weitergabe). Import ist eine
  eigene Design-Session.
- Keine Gerenderten Outputs (`output/*.mp4`) im ZIP.
- Kein `.cache/` (Audio-Features) im ZIP — Cache ist regenerierbar.
- Keine Cloud-Uploads, keine Verschlüsselung, keine inkrementellen Archive.
- Keine Änderung am bestehenden Speicherformat des Projekt-JSON
  (Versionierung bleibt wie sie ist).
