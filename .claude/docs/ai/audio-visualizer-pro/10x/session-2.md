# 10x Analysis: Audio Visualizer Pro — Kurskorrektur Qualität & Podcast-Parität
Session 2 | Date: 2026-08-09 | Supersedes: Prioritäten aus session-1.md

## Auslöser

Nutzer-Feedback: *"Ich bin noch nicht mit der Qualität der Visualizer und dem Studio
zufrieden, und ich habe das Gefühl, es ist sehr für Music und weniger für Podcast —
es sollte genauso gut für beides sein."*

→ Bevor irgendein Expansions-Feature aus Session 1 (Shorts, Album-Mode, Live) Sinn
ergibt, muss der Kern tragen. Expansions-Features multiplizieren einen schwachen Kern
nur schneller in die Breite.

## Befund: Das Nutzergefühl ist messbar richtig

Code-Audit (Belege mit Datei:Zeile im Audit-Report; hier die Essenz):

**Die Podcast-Seite ist strukturell dünner, nicht nur gefühlt:**

- **3 von 18 Visualizern sind echt sprach-adaptiv** (typographic, voice_flow,
  aurora_voice). 15 ignorieren `voice_clarity`/`voice_band` komplett; 3 davon
  (pulsing_core, lumina_core, nebula_drift) **hardcoden `mode="music"`** — sie können
  gar nicht auf Sprache reagieren.
- **`speech_focus` ist Speech nur im Namen**: liest kein einziges Voice-Feature, das
  "Speech-Gating" ist ein simpler RMS-Smoothstep (`speech_focus.py:71,207-210`).
- **SmartMatcher**: Speech-Empfehlungsraum 4 Visualizer vs. 16 bei Music
  (`ai_matcher.py:74-85`); aurora_voice bekommt keine adaptiven Parameter, nur
  statische Defaults.
- **Studio-Profile**: Podcast-Whitelist 6 vs. 14 Visualizer (`profiles.py:34-59`) —
  und 3 der 6 sind intern music-forced, d.h. das Podcast-Profil (desaturate, ruhig)
  kämpft gegen den Visualizer selbst. Korridore sind generisch (`intensity`/`speed`),
  nicht pro Visualizer getunt.
- **Podcast-Presets**: 2 von 5 nutzen reine Musik-Visualizer (neon_wave_circle,
  sacred_mandala), Parametertiefe 1–4 vs. 10+ bei Musik-Presets, keines nutzt
  aurora_voice oder speech_focus, nur eines konfiguriert Quote-Overlays.
- **Es gibt keine Silenz/Pausen-Erkennung im gesamten Codebase** — kein Feature, das
  Sprach-Rhythmus (Sprechpausen, Betonung, Dialog-Wechsel) abbildet. Podcast-Visuals
  fahren heute einfach "RMS mit ruhigerem Profil".

**Und die Studio-Gate misst aktuell nicht vertrauenswürdig Qualität:**

- Golden Set: 36 Renders auf **einer einzigen Speech-Datei**, 0% Musik
  (`build_golden_set.py:32-46`) — Labels sind konstruiert (`good = alpha_cap ≤ 0.6`),
  **keine menschliche Validierung**. 2 von 9 Thresholds kalibriert, Rest "assumed".
- **M4 (Textkontrast — DIE Podcast-Metrik) wird in der Engine gar nicht berechnet**
  (`engine.py:78` liefert `M4=None`).
- C14 (P1, offen): Visualizer emittieren hart `alpha=1.0` → Alpha-Cap wirkt als
  Vollbild-Schleier statt Coverage-Control. Betrifft jeden Composite über Hintergrund.
- Der Solver kann nur drosseln (alpha_cap, bloom, scale, speed …) — er kann **keine
  schlechte Visualizer-Wahl, keine Shader-Qualität, kein Layout fixen**
  (`solver.py:18-28`; Layout-Hebel explizit nicht implementiert). Bei Plateau gibt er
  auf (`status="plateau"`).
- Letzte 30 Commits: 27x Gate-Maschinerie, 0x Podcast-Feature, 0x Musik-Feature.

**Kernsatz**: AVP hat eine aufwendige Messmaschine gebaut, die Drosselung misst —
aber die zu messende Qualität (Visualizer selbst, Speech-Nativität) existiert auf der
Podcast-Seite noch nicht. Gate ohne Qualität darunter ist Theater.

---

## Die eigentliche 10x-Frage (revidiert)

Nicht "was kommt obendrauf", sondern: **Was macht AVP zum Tool, das für Podcast
genauso selbstverständlich gut ist wie für Musik?**

Musik-Visualizer sind ein gelöstes Problem (jeder kennt Spectrum-Bars). **Sprach-
Visualisierung, die wirklich auf Sprache reagiert, ist ein ungelöstes Problem** —
Headliner, Descript, Wavve etc. zeigen statische Waveforms mit Captions. Wer
sprach-native GPU-Visuals baut, hat keine Konkurrenz im Segment. Das ist der 10x-Graben.

---

## Massive Opportunities (revidiert)

### 1. Sprach-native Feature-Ebene: Silence, Rhythmus, Betonung
**What**: Neue Audio-Features in `analyzer.py` (Erweiterung, nicht Änderung — siehe
AGENTS.md): Pause/Silenz-Detektion (Envelopes mit Pausen-Markern), Sprech-Rhythmus
(Silbenrate/Energie-Modulation im Voice-Band), Betonungs-Envelope (prosodische Peaks),
optional Dialog-Dynamik (Sprecherwechsel-Heuristik über MFCC-Distanz). Alles in
`AudioFeatures` + Cache + `features_dict` durchziehen.
**Why 10x**: Das ist das fehlende Fundament. Heute reagieren Podcast-Visuals auf
Lautstärke — nicht auf *Sprache*. Pausen sind visuell wertvoll (Atem, Ruhe vor
Quote), Betonung ist der Beat der Sprache. Ohne diese Features kann kein Visualizer
und kein Gate "gut für Podcast" sein. Music hatte diese Ebene immer (beat, onset,
tempo) — Podcast hat kein Äquivalent. **Das ist die Paritäts-Lücke in einem Satz.**
**Unlocks**: Alle folgenden Punkte; Visualizer, die Pausen atmen lassen; Gate-Metriken,
die Sprachqualität messen statt Pixel-Statistik; SmartMatcher mit echten
Sprach-Formeln.
**Effort**: Medium-High (librosa reicht dafür, keine neuen Deps nötig)
**Risk**: Feature-Qualität an echten Podcasts validieren (deutsche Sprache,
Studio- vs. Telefon-Qualität); Cache-Invalidierung für bestehende Analysen.
**Score**: 🔥 (der eigentliche Unlock)

### 2. Speech-Awareness-Rollout über alle Visualizer
**What**: 
- Die 3 hardcoded `mode="music"` Visualizer (pulsing_core, lumina_core, nebula_drift)
  auf `mode=f["mode"]` umstellen.
- `speech_focus` reparieren: echte Voice-Features konsumieren (es trägt den Namen!).
- `_map_features_to_uniforms(mode=f.get("mode","hybrid"))` als Standard für alle
  Classic-Visualizer — Sprache dämpft Beat-Reaktion automatisch (Mechanismus existiert
  schon in `base.py:443-495`, wird nur nicht genutzt).
**Why 10x**: Verdreifacht den Podcast-Visualizer-Raum von 3→18 ohne einen neuen
Visualizer zu schreiben. Die Infrastruktur ist gebaut — sie ist nur nicht angeschlossen.
Das ist der billigste große Qualitätssprung im gesamten Audit.
**Unlocks**: Podcast-Whitelist im Studio kann ehrlich wachsen (6→12+); SmartMatcher-
Speech-Raum wächst; Podcast-Presets hören auf, Musik-Visualizer zu verkleiden.
**Effort**: Medium (mechanisch, aber jeder Visualizer braucht visuelle Verifikation)
**Risk**: Musik-Look darf nicht regressieren — Golden Set muss vorher musik-seitig
erweitert werden (siehe 4), sonst fliegt man blind.
**Score**: 🔥

### 3. Podcast-native Visualizer-Familie (statt Musik-Hand-me-downs)
**What**: 2–3 Visualizer, die *von Sprache aus* designed sind, nicht nachträglich
gedämpft: z.B. "Caption-native" Visualizer (Transkript-Wörter als visuelles Element,
Wort-Timing aus Gemini), "Dialog-Wellen" (Sprech-Rhythmus als Topographie, Pausen als
Täler), "Quote-Stage" (ruhige Basis, die bei Quote-Fenstern Fokus/Bühne aufbaut).
**Why 10x**: Das ist die Differenzierung, die kein Wettbewerber hat — und sie macht
AVP für Podcaster zum *eigenen* Tool statt zum Musik-Tool-mit-Quote-Overlay. Erst
möglich nach Opportunity 1 (Features) und verstärkt durch 2.
**Unlocks**: Podcast-Marketing-Story ("built for voice"), neue Preset-Kategorie,
Recipe-Ökosystem bekommt Inhalt.
**Effort**: High
**Risk**: Design-Risiko — braucht Iteration an echten Episoden (`projects/podcast.json`
als Referenz-Content).
**Score**: 👍 (nach 1+2)

### 4. Golden Set v2: menschlich gelabelt, beide Modi, Qualitäts-Bar
**What**: Golden Set neu: je Modus (music/podcast/hybrid) mehrere Audiodateien,
**menschliche Labels** (du selbst, eine Session, festes Bewertungsraster), dabei auch
die 4K-Drift und M4 miteinbeziehen. Thresholds danach neu kalibrieren.
**Why 10x**: Ohne menschliche Wahrheit misst das Gate Konstruktionen
(`good = alpha_cap ≤ 0.6`). Jede weitere Studio-Arbeit auf "assumed"-Thresholds ist
Sandburg. Das ist gleichzeitig die Antwort auf "Ich bin mit der Qualität nicht
zufrieden" — weil es *definiert*, was "gut" heißt, statt es dem Zufall zu überlassen.
**Unlocks**: Vertrauenswürdiges Gate; Basis für sicheren Rollout (2); ehrliche
Regressionstests für Musik-Qualität.
**Effort**: Medium (Harness existiert: `tools/build_golden_set.py`,
`calibrate_thresholds.py`; Aufwand ist Labeling + Musik-Audio-Auswahl)
**Score**: 🔥 (Blocker für alles Studio-seitige)

---

## Medium Opportunities

### 1. C14 Alpha-Pfad fixen (P1, bekannt, dokumentiert)
**What**: Luma-abgeleitetes Alpha im Blit-Shader statt hartem `alpha=1.0`
(Spec-Zeilen 230-232, 289).
**Why**: Jeder Composite über Hintergrund betroffen; Alpha-Cap aktuell ein
Vollbild-Schleier. Visueller Defekt, der *jedes* Podcast-Video mit Hintergrundbild
degradiert.
**Effort**: Medium | **Score**: 🔥

### 2. M4 in der Engine berechnen + Podcast-Metriken
**What**: `engine.py:78` — M4 (Textkontrast) real berechnen statt `None`; zusätzlich
1–2 sprach-native Metriken (z.B. Vitalität in Pausen-Fenstern: Visuals sollen in
Sprechpausen abklingen — messbar erst nach Massive-1).
**Why**: M4 ist die einzige Metrik, die direkt die Podcast-Kern-Experience (lesbare
Quotes) schützt — und sie ist ausgerechnet die, die nicht läuft.
**Effort**: Low-Medium | **Score**: 🔥

### 3. Solver-Hebel an echte PARAMS binden
**What**: `_VIZ_PARAM_LEVERS` (`engine.py:18-19`) mappt auf viz_scale/glow/speed/
beat_response/intensity/chroma_modulation — die die meisten `PARAMS`-Dicts gar nicht
definieren → No-ops. Entweder PARAMS-Konvention (Base-Klasse erzwingt Standard-Hebel)
oder Levers pro Visualizer deklarieren.
**Why**: Sonst "löst" der Solver Probleme, indem er Schrauben dreht, die nicht
verbunden sind — und meldet Erfolg/Plateau auf Messung eines unveränderten Bildes.
**Effort**: Medium | **Score**: 👍

### 4. Podcast-Presets neu bauen (nach Rollout 2)
**What**: 5 Presets ersetzen durch 5, die Speech-aware Visualizer + Quote-Overlay +
ruhige Post-FX konsequent nutzen (Vorbild: podcast_interview, das einzige gute).
**Why**: Presets sind das erste, was ein neuer Podcast-Nutzer sieht. Heute sieht er
sacred_mandala mit `rotation_speed: 0.15`.
**Effort**: Low | **Score**: 👍

### 5. Musik-Qualitäts-Pass (nicht vergessen!)
**What**: Parität heißt nicht nur Podcast hochziehen — die Musik-Seite hat eigene
offene Qualitätsthemen (C16 4K-Drift durch Pixel-basierte Bloom-Radien, C15
Grain-Verunreinigung der Messung, Anti-Aliasing-Konsistenz). Ein gezielter Polish-Pass
über die Top-5-Musik-Visualizer mit Golden-Set-v2 als Bar.
**Why**: "Genauso gut für beides" impliziert eine definierte Qualitäts-Bar für beide —
aktuell existiert sie für keine Seite wirklich.
**Effort**: Medium | **Score**: 👍

---

## Small Gems

### 1. Modus-Badge überall sichtbar
**What**: GUI zeigt permanent erkannten Modus (music/podcast/hybrid) + Confidence;
ein Klick erklärt "warum dieser Visualizer empfohlen wurde".
**Why powerful**: Macht die Paritäts-Arbeit sichtbar und debuggbar; Nutzer lernen dem
Tool zu vertrauen. ModeGate + SmartMatcher haben die Daten schon.
**Effort**: Low | **Score**: 🔥

### 2. "Podcast-Projekt" vs "Musik-Projekt" als Projekt-Typ
**What**: Beim Anlegen eines Projekts Typ wählen → GUI filtert Visualizer-Liste,
Presets und KI-Panel auf den Typ (statt einer 18er-Liste für alle).
**Why powerful**: Ein Dropdown, das die gefühlte Parität sofort herstellt und
Fehlgriffe (sacred_mandala für Interview) strukturell verhindert.
**Effort**: Low | **Score**: 🔥

### 3. Pausen-Marker in der Timeline
**What**: Sobald Silenz-Features existieren: Sprechpausen in `timeline_widget.py`
einzeichnen (neben Beat-Markern bei Musik).
**Why powerful**: Podcaster *sehen* ihre Sprechstruktur; Grundlage für Quote-Review
und später Shorts-Cuts.
**Effort**: Low (nach Massive-1) | **Score**: 👍

### 4. Speech-Test-Corpus festlegen
**What**: 3–4 repräsentative Dateien (Interview, Monolog, Musik+Sprache-Hybrid,
schlechte Mikro-Qualität) als feste Testbasis in `tests/golden/` dokumentiert.
**Why powerful**: Beendet die Ein-Dateien-Kalibrierung strukturell.
**Effort**: Low | **Score**: 👍

---

## Recommended Priority (ersetzt Session 1)

### Do Now — Fundament & Vertrauen
1. **Golden Set v2 (menschliche Labels, beide Modi)** — definiert "gut", beendet
   Kalibrierung auf Konstruktionen. Ohne das ist jede weitere Qualitätsarbeit blind.
2. **C14 Alpha-Fix** — aktiver visueller Defekt in jedem Hintergrund-Composite.
3. **M4 in Engine berechnen** — die Podcast-Schutz-Metrik läuft aktuell nicht.
4. **Small Gems 1+2 (Modus-Badge, Projekt-Typ)** — sofortige gefühlte Parität, Low Effort.

### Do Next — Parität herstellen
1. **Sprach-native Feature-Ebene** (Silenz, Rhythmus, Betonung) — der eigentliche
   Paritäts-Unlock. Warum: Musik hatte Beat/Onset/Tempo immer; Podcast hat nichts.
2. **Speech-Awareness-Rollout** (3→18 Visualizer) — größter Qualitätssprung pro
   Aufwand; Golden-Set-v2 schützt vor Musik-Regression.
3. **Solver-Hebel an echte PARAMS binden** — macht das Gate wirksam statt dekorativ.
4. **Podcast-Presets neu** — sichtbares Ergebnis der Parität.

### Explore — Differenzierung
1. **Podcast-native Visualizer-Familie** (Caption-native, Dialog-Wellen, Quote-Stage) —
   Risk: Design-Iteration nötig; Upside: Alleinstellung im Podcast-Segment.
2. **Musik-Qualitäts-Pass mit Golden-Set-Bar** — Parität heißt auch Musik-Seite
   messbar gut.
3. **(aus Session 1 weiterhin valide, aber danach):** Auto-Captions/SRT und
   YouTube-Export-Paket — sie *profitieren* massiv von der Sprach-Feature-Ebene
   (Wort-Timing, Pausen) und sollten nach ihr gebaut werden, nicht vor.

### Backlog / explizit verschoben aus Session 1
- Shorts-Engine, Album-Modus, Live-Modus, Recipe-Ökosystem — alles richtig, aber alles
  multipliziert den Kern. Kern zuerst.

---

## Der strategische Kern

Session 1 hat gefragt: "Was kommt obendrauf?" Die ehrliche Antwort nach dem Audit:
**Nichts — noch nicht.** Die 10x-Chance liegt darin, dass *sprach-native Visualisierung
ein unbesetztes Feld ist* und AVP mit Gemini-Transkript + Voice-Features + GPU-Stack
bereits 70% der Zutaten besitzt. Der Graben ist nicht "mehr Visualizer", sondern
**"das Tool, das Sprache versteht"**. Parität ist dabei kein Fairness-Projekt, sondern
die Marktposition.

Messbarer Zielzustand für "Parität erreicht" (als Scorecard ins Repo):
- Speech-aware Visualizer: 18/18 (via mode-driven Mapping) + ≥3 podcast-nativ
- SmartMatcher Speech-Raum ≥ 8 mit adaptiven Params für alle
- Studio-Whitelist podcast ≥ 10, Korridore pro Visualizer
- Golden Set: ≥3 Audios pro Modus, 100% menschlich gelabelt, 0 "assumed"-Thresholds
- M4 berechnet, C14 geschlossen
- 5 Podcast-Presets auf Niveau von podcast_interview

## Questions

### Answered
- **Q**: Ist der Music-Bias real oder Einbildung? **A**: Real und messbar (siehe Befund:
  3/18 speech-aware, 4-vs-16 Matcher-Raum, 6-vs-14 Whitelist, Ein-Datei-Kalibrierung).
- **Q**: Kann das Studio-Gate die Qualität retten? **A**: Nicht in aktueller Form — es
  drosselt nur, fixt keine Struktur, und misst teilweise Konstruktionen statt Qualität.
- **Q**: Braucht Parität neue Dependencies? **A**: Nein — librosa + vorhandene
  Voice-Features reichen für Silenz/Rhythmus/Betonung.

### Entscheidungen (2026-08-09, delegiert an Strategie-Empfehlung — "was am sinnvollsten ist")
- **Golden Set v2 Labeling: Selbst-Labeling durch den Maintainer, mit festem Raster.**
  Begründung: Single-Maintainer-Produkt — sein Geschmack *ist* die Qualitäts-Bar;
  Test-Hörer kosten Organisationsaufwand, ohne die Bar zu schärfen. Fremd-Labels werden
  erst relevant, wenn externe Nutzer/Community anstehen (Recipe-Ökosystem, Session 1).
  Raster vorab schriftlich fixieren (M1–M6-Kriterien in Worte übersetzt), damit Labels
  reproduzierbar bleiben.
- **Musik-Look: strikt im Music-Mode, pragmatisch in Speech/Hybrid.**
  Begründung: Das mode-driven Mapping in `base.py:443-495` verzweigt pro Modus —
  Music-Mode-Output kann bei sauberer Umstellung exakt identisch bleiben, ohne die
  Speech-Seite einzuschränken. "Strikt identisch" wird also billig, sobald Golden Set
  v2 Musik-Referenzen enthält: Musik-Renders vor/nach Rollout müssen pixelnah
  übereinstimmen (Gate: bestehende M1/M3/M6-Drift-Checks). Speech/Hybrid darf frei
  getunt werden — dort gibt es keinen Bestand, den man verschlechtern könnte.

### Blockers
- (keine offenen Blocker)

## Next Steps
- [ ] Golden Set v2: Musik-Audios auswählen (3), Podcast-Audios auswählen (3),
      Labeling-Raster schriftlich definieren, Selbst-Labeling-Session einplanen
- [ ] C14-Fix als eigenen Slice spec'en (Spec-Zeilen 230-232, 289 als Vorlage)
- [ ] M4-Implementierung in `engine.py:78` als Slice
- [ ] Sprach-Feature-Ebene: Design-Doc (Silenz-Detektion, Rhythmus, Betonung →
      `AudioFeatures`-Erweiterung, Cache-Strategie)
- [ ] Scorecard (oben) als `docs/internal/podcast-parity-scorecard.md` anlegen und
      bei jedem Release abhaken
- [ ] Rollout-Absicherung: Musik-Renders vor Speech-Rollout als Referenz einfrieren
      (Vorher/Nachher-Diff via M1/M3/M6)
