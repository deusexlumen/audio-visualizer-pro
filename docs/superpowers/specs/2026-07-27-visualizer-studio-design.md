# Visualizer Studio — Systemkonzept & Pipeline-Spezifikation **v2.1**

**Datum:** 2026-07-27
**Status:** Revision von v2. Code-Lücken 1–3 aus v2-§19 gegen den Bestandscode verifiziert — die Verifikation hat **vier neue P1-Defekte** aufgedeckt (C14–C17), die hier eingearbeitet sind. Die beiden v2.1-Code-Lücken (Seeding, Bloom-Radius) sind ebenfalls verifiziert (siehe §3.2.1, §3.4, §19).
**Schema-Version dieses Dokuments:** `studio-spec/2.1`
**Kernänderung gegenüber v1:** Die Qualitätsschleife wandert **vor** den Full-Render. Statt „rendern → messen → 3× neu rendern" gilt: **Probe (billig, iterativ) → Commit (einmalig) → Verify (assertiv, ohne Schleife)**.
**Kernänderung gegenüber v2:** Das Messfundament wird gegen drei reale Eigenschaften des Bestandscodes gehärtet — Alpha-Fallback, stochastisches Post-FX und Auflösungsabhängigkeit. Ohne diese drei Korrekturen misst der Differenz-Render systematisch das Falsche.

---

## 0. Änderungslog v1 → v2

| # | Änderung | Grund (Defekt in v1) |
|---|----------|----------------------|
| C1 | Zwei-Phasen-Loop: Probe-Solve auf Stichprobenframes, danach **ein** Commit-Render | v1 impliziert bis zu 4 Full-Renders → Laufzeit ×4 |
| C2 | Feasibility-Precheck vor jedem Render | v1 erkennt unlösbare Fälle erst nach 4 Renders |
| C3 | Kontinuierliche Metriken (Energie-Integral) statt Schwellwert-Flächenanteil | Flächen-Metrik reagiert sprunghaft auf Alpha-Cap → Solver konvergiert nicht |
| C4 | Messung per **Differenz-Render** (mit/ohne Visualizer) an der Composite-Endstufe | v1 klemmt Alpha *vor* Post-FX, misst aber *nach* Post-FX (Bloom hebt Deckung wieder an) |
| C5 | Skalarer Penalty-Solver mit Step-Ladder + Plateau-Abbruch statt Prioritätstabelle | v1-Fixes können oszillieren (Deckung ↓ vs. Bewegungsenergie ↑) |
| C6 | Ereignisgetriebenes, stratifiziertes Sampling statt 1 Frame/s | 1 Hz aliast gegen den Beat-Grid (120 BPM = 2 Hz) → Peaks unsichtbar |
| C7 | Normalisierter Messraster (854 px lange Kante, Linear-Light) | Metriken sonst auflösungsabhängig → Preview ≠ Batch trotz gleichem ConstraintSet |
| C8 | Maske im **Quellbildraum** + geteilte UV-Transform-Funktion | v1 bricht bei Crop/Pan-Zoom/Ken-Burns-Hintergründen |
| C9 | Schwellwerte als versionierte Daten + Kalibrier-Harness | v1-Zahlen (0.6 / 60 % / 10 %) sind unkalibrierte Magie |
| C10 | Provenance-Block im Sidecar (`schema_version`, Masken-Provider, Modell-Hash, Threshold-Version) | v1 verspricht Nachvollziehbarkeit, ML-Fallback macht Ergebnis maschinenabhängig |
| C11 | Textzonen aus Salienz/Maske abgeleitet, Kontrast per Glyphenmaske + p5-Worst-Case | Statische Zonen + Bounding-Box-Kontrast messen das Falsche |
| C12 | Perf-Budgets als harte Akzeptanzkriterien | v1 hat Live-Badge ohne Latenzbudget → UI-Tod bei ML-Maske |
| C13 | Video-Hintergründe: **Degradation statt Ablehnung** | v1 lässt das Loch unbenannt; harter Abbruch wäre Feature-Regression |

### Änderungslog v2 → v2.1 (aus der Code-Verifikation)

| # | Änderung | Grund (neu entdeckter Defekt) |
|---|----------|-------------------------------|
| **C14** | **Luminanz-abgeleiteter Visualizer-Alpha vor dem Cap** (§6.1) | Der Composite-Shader setzt `viz_alpha = 1.0`, wenn ein Visualizer kein Alpha liefert. Ein Cap auf diesen Mix-Faktor ist dann **kein Deckungsregler, sondern ein Vollbild-Schleier** — er dimmt Bild*teile* nicht, sondern das ganze Bild gleichmäßig. M1/M3/Feasibility messen dann Unsinn. |
| **C15** | **Deterministischer Rausch-Seed / grain-freie Messrenders** (§3.2) | Film-Grain und Dither sind stochastisch pro Frame. In `contrib = |A−B|` heben sie sich nur auf, wenn A und B mit identischem Seed laufen. Sonst hat `contrib` einen Rauschboden in derselben Größenordnung wie die M5-Musikschwelle (0.02) — die Vitalitätsmetrik misst dann Filmkorn. |
| **C16** | **Probe-Auflösung gekoppelt an Zielauflösung + Drift-Kalibrierung** (§3.4) | Shader-Größen (Partikel, Linienbreiten, Glow-/Bloom-Radien) sind pixelbasiert. Probe @480p und Commit @4K liefern damit *systematisch* andere Metriken. Die v2-Regel „Drift = Implementierungsbug → Abbruch" hätte den Normalfall zum Dauerabbruch gemacht. |
| **C17** | **Video-Hintergrund degradiert statt abgelehnt** (§14, §17) | Video-Hintergründe sind im Bestand implementiert. Ein harter Abbruch unter `--studio` macht ein funktionierendes Feature unbenutzbar. Der „kein Hintergrundbild"-Pfad existiert bereits und deckt den Fall vollständig ab. |

---

## 1. Invarianten (das, was nicht verhandelbar ist)

1. **Kein stiller Schlecht-Output.** Jeder Output ist entweder gate-konform oder wird mit Report abgebrochen.
2. **Was die Preview zeigt, misst das Gate.** Gleiches ConstraintSet, gleicher Messraster, gleiche Metrikimplementierung — testgesichert.
3. **Ein Commit-Render pro Auftrag.** Re-Render nur auf explizite Nutzeranforderung, nie automatisch.
4. **Bestand bleibt Ausführungsschicht.** `GPUBatchRenderer` bekommt Hooks, keine Logik.
5. **Determinismus bei fixierter Provenance.** Gleiche Inputs + gleicher Masken-Provider + gleiche Threshold-Version → bitgleiche Entscheidungen.

---

## 2. Gesamtarchitektur

```
┌──────────────────────────────────────────────────────────────────────┐
│  src/studio/                                                         │
│                                                                      │
│  analyze ─→ ModeGate ─→ PresetFactory ─→ ConstraintSet               │
│                │             │                 │                     │
│                │             │                 ▼                     │
│                │             │        FeasibilityCheck ──infeasible──┼─→ Layout-Fallback
│                │             │                 │                     │      │
│                │             │                 ▼                     │      │
│                │             │        ┌────────────────┐             │◄─────┘
│                │             │        │ PROBE-LOOP     │             │
│                │             │        │ N Sample-Frames│             │
│                │             │        │ @probe_res     │             │
│                │             │        │ Differenz-Rend.│             │
│                │             │        │  ↓ Metrics     │             │
│                │             │        │  ↓ Penalty J   │             │
│                │             │        │  ↓ Solver-Step │  ≤8 Iter.   │
│                │             │        └───────┬────────┘             │
│                │             │                │ J ≤ 0 (feasible)     │
│                ▼             ▼                ▼                      │
│           StudioDecision ◄──────────── COMMIT-RENDER (1×, full res)  │
│                                               │                      │
│                                               ▼                      │
│                                          VERIFY (assertiv)           │
│                                          pass → Encode + Sidecar     │
│                                          fail → Abort + Report       │
└──────────────────────────────────────────────────────────────────────┘
```

**Module:** `engine.py`, `mode_gate.py`, `profiles.py`, `constraints.py`, `mask_service.py`, `metrics.py`, `sampling.py`, `solver.py`, `feasibility.py`, `preset_factory.py`, `thresholds.py`, `types.py`, `provenance.py`

**Warum Probe/Commit/Verify statt v1-Loop:** Ein 4-min-Track @60 fps = 14 400 Frames. v1 im Worst Case = 57 600 gerenderte Frames. v2.1 = 14 400 + (8 Iterationen × [1 B-Render + 18 Samples × 2 A-Renders]) ≈ 14 400 + 296 Billigframes (Kostenmodell siehe §3.2.2). **Faktor ~4 Laufzeitersparnis im Fehlerfall, ~0 Overhead im Gutfall.**

---

## 3. Messfundament (neu, Phase 0 — alles andere hängt daran)

### 3.1 Normalisierter Messraster

Alle Metriken rechnen auf einem kanonischen Raster:

- Downscale auf **lange Kante 854 px** (Area-Filter, kein Bilinear), Seitenverhältnis erhalten.
- sRGB → **Linear-Light** vor jeder Luminanz-/Differenzrechnung.
- Float32, Wertebereich [0, 1], NaN/Inf werden vor der Metrik als Integritätsverletzung erfasst, nicht geklemmt.

Konsequenz: Preview (Einzelframe, evtl. 720p) und Batch (4K) liefern **identische Metrikwerte** ± ε = 0.01. Das ist testbar und wird getestet.

### 3.2 Differenz-Render (die zentrale Messtechnik)

Pro Sample-Zeitpunkt `t` werden zwei Frames erzeugt:

- **A(t)** = vollständiger Frame (Hintergrund + Visualizer + Post-FX)
- **B(t)** = identische Pipeline, Visualizer-Beitrag auf 0 (`u_viz_alpha_cap = 0`), **Post-FX-Kette unverändert aktiv**

Daraus:

```
contrib(p) = clamp( mean_c | A_lin(p,c) − B_lin(p,c) | , 0, 1 )
```

`contrib` ist der **post-FX-wirksame Visualizer-Einfluss pro Pixel**. Bloom, Grain, Vignette sind automatisch enthalten — genau der Defekt, an dem v1 vorbeimisst.

#### 3.2.1 Rauschfreiheit ist Voraussetzung, nicht Detail (C15)

Grain und Dither sind **stochastisch pro Frame**. In der Differenz `|A − B|` heben sie sich **nur dann exakt auf**, wenn beide Renders denselben Rauschzustand haben. Andernfalls trägt `contrib` einen Rauschboden in der Größenordnung der Grain-Amplitude — und die liegt typisch bei 0.02–0.05, also **exakt auf der M5-Musikuntergrenze**. Ohne diese Regel misst die Vitalitätsmetrik Filmkorn statt Bewegung.

**Pflichtregeln für alle Messrenders (Probe, Preview-Badge, Verify):**

1. Grain- und Dither-RNG werden **deterministisch aus dem Frame-Zeitpunkt** geseedet, nie aus Aufrufzähler, Wanduhr oder GPU-Zustand.
2. A(t) und B(t) laufen mit **identischem Seed**.
3. **M5 wird zusätzlich auf einem grain-freien Renderpaar berechnet** (`film_grain = 0`, `dither = 0`). Für M1–M4 genügt Regel 1+2; für eine zeitliche Differenzmetrik ist selbst perfekt aufgehobenes Korn zwischen `t` und `t+Δ` nicht mehr korreliert.
4. Falls der Renderer keinen frame-basierten Seed anbietet: Grain/Dither für **alle** Messrenders hart deaktivieren und den Abschlag im Report vermerken (`noise_suppressed: true`). Das ist die zulässige Notlösung, nicht der Normalfall.

> **Code-verifiziert (2026-07-27):** Regeln 1+2 sind **bereits erfüllt, keine Codeänderung nötig.** Grain seedet aus `fract(u_time * 100.0)` (`gpu_renderer.py:1132-1134`), Dither aus `fract(u_time)` (`:1142`), und `u_time` ist die deterministische Frame-Zeit aus dem Render-Loop (`:528`, `time = frame_idx / fps`) — kein Wall-Clock, kein Aufrufzähler. A(t) und B(t) am selben Zeitpunkt teilen damit automatisch denselben Seed. Regel 3 bleibt Pflicht (Korn zwischen t und t+Δ ist unkorreliert); die Notlösung Regel 4 entfällt im Bestandscode.

#### 3.2.2 Kostenmodell

Der Hintergrund ist statisch (Bild, harter Resize, keine Bewegung — siehe §6.2), Post-FX ist bei fixiertem Seed zeitinvariant. Daraus folgt: **B ist über die gesamte Probe-Runde konstant und wird einmal gerendert und gecacht.**

| Phase | Renders pro Probe-Runde |
|-------|-------------------------|
| B (Hintergrund + FX, `alpha_cap = 0`) | 1 (gecacht; bei Video-Hintergrund: 1 pro Sample) |
| A(t) je Sample | 18 |
| A(t+Δ) je Sample (nur M5, grain-frei) | 18 |
| **Summe** | **37** statt naiv 54 |

Im B-Render kann der Visualizer-Pass vollständig übersprungen werden (Mix-Faktor ist 0) — reine Ersparnis, kein Semantikunterschied.

**Reihenfolge im Shader ist bindend:** Alpha-Fallback und Luma-Ableitung (§6.1) laufen **vor** der Cap-Anwendung. Nur so gilt `u_viz_alpha_cap = 0 ⇒ contrib ≡ 0`, worauf die gesamte Messtechnik aufsetzt.

> **Code-verifiziert (2026-07-27):** Die Post-FX-Kette existiert und ist steuerbar: Bloom (HDR, additiv, vor Tonemapping, `gpu_renderer.py:511-517`) und Final-Pass Exposure → ACES-Tonemap → Grading → LUT → Vignette → Grain → Dither (`gpu_renderer.py:519-534`). Alle Parameter kommen aus dem `postprocess`-Dict (`PostProcessConfig`, `config/schemas.py:188-199`). Der Differenz-Render muss B(t) daher lediglich mit `u_viz_alpha_cap = 0` durch dieselbe Kette schicken — kein Renderer-Umbau nötig.

> **Entscheidung:** Differenz-Render statt zusätzlichem MRT-Contribution-Buffer. Grund: kein Shader-Umbau, keine Sonderbehandlung von Post-FX-Spreading, exakt gleiche Semantik in Preview und Probe.

### 3.3 Metrikkatalog (`src/studio/metrics.py`)

| ID | Metrik | Definition | Typ | Default-Schwelle |
|----|--------|-----------|-----|------------------|
| **M1** | Overlay-Energie | `mean_p(contrib)` | hart, kontinuierlich | ≤ 0.22 |
| **M2** | Overlay-Deckung | `share_p(contrib > 0.5)` | weich (warn) | ≤ 0.60 |
| **M3** | Subjekt-Störung | `Σ(contrib·mask) / Σ(mask)` | hart, kontinuierlich | ≤ 0.10 |
| **M4** | Text-Kontrast | `min_t( p5( ratio(glyph_px, ring_bg) ) )` | hart | ≥ 4.5 (PODCAST) |
| **M5** | Vitalität | `mean_t( mean_p |contrib(t+Δ) − contrib(t)| )`, Δ = 40 ms | hart (Korridor) | MUSIC ≥ 0.02 · PODCAST ≤ 0.09 |
| **M6** | Integrität | NaN/Inf-Anteil > 0 · p99-Luminanz < 0.02 (Blackframe) · Clipping-Anteil > 0.15 | hart, binär | keine Verletzung |

**M1 statt v1-Flächenmetrik:** Das Energie-Integral ist **monoton und stetig** in `u_viz_alpha_cap`. Damit hat der Solver eine verwertbare Ableitungsrichtung. Die alte Flächenmetrik bleibt als M2 erhalten — aber nur noch als `warn`, weil sie an der Schwelle springt.

**M5-Warnung:** Die Untergrenze 0.02 für MUSIC liegt in derselben Größenordnung wie typische Grain-Amplituden. M5 ist deshalb **ausschließlich** auf grain-freien Renderpaaren gültig (§3.2.1, Regel 3). Ein M5-Wert aus einem Render mit aktivem Korn ist kein Messwert, sondern ein Artefakt.

**M4 präzisiert:** Vordergrund = **Glyphen-Deckungsmaske** aus dem Text-Renderer (nicht Bounding-Box). Hintergrund = mittlere Luminanz eines 3-px-dilatierten Rings um die Glyphen. Kontrastformel WCAG 2.x auf relativen Luminanzen. Aggregation: **p5 innerhalb des Frames** (Worst-Case-nah, robust gegen Einzelpixel), dann **Minimum über alle Sample-Frames**. Mittelwerte sind hier verboten — ein einziger unlesbarer Frame ist der Defekt.

### 3.4 Probe-Auflösung & Drift-Budget (C16)

**Der Defekt:** v2 probt @480p und committet @Zielauflösung, verlässt sich dabei auf den normalisierten Messraster (§3.1). Der Raster normalisiert aber nur die *Messung*, nicht das *Rendering*. Partikelgrößen, Linienbreiten, Glow- und Bloom-Radien sind in Shadern typisch **pixelbasiert**: derselbe Effekt belegt bei 480p relativ ein Vielfaches der Fläche wie bei 2160p. Metrikwerte driften damit systematisch — und die v2-Regel „Probe↔Verify-Abweichung = Implementierungsbug → Abbruch" hätte den **Normalfall zum Dauerabbruch** gemacht.

> **Code-verifiziert (2026-07-27):** Der Bloom-Radius ist bestätigt **pixelbasiert**: `u_texel = radius / src_size` im Upsample-Pass (`gpu_bloom.py:169-172, 181-183`). C16 trifft Bloom damit direkt; die Probe-Skalierung muss `bloom_radius × (probe_res / target_res)` anwenden. Grain/Dither sind per `gl_FragCoord.xy` ebenfalls auflösungsabhängig, fallen aber unter §3.2.1 (Messrenders), nicht unter den Drift-Budget-Mechanismus.

**Regelung:**

- **Probe-Auflösung** `probe_res = max(480p, Zielauflösung / 4)`, **identisches Seitenverhältnis** wie das Ziel. Nie eine andere Aspect Ratio proben.
- **Bloom-/Glow-Radien und alle pixelbasierten Größen werden mit `probe_res / target_res` skaliert.** Wo das nicht möglich ist, wird der betroffene Visualizer als `resolution_dependent` markiert.
- **Drift-Kalibrierung ist Teil von P0:** Der Kalibrier-Harness rendert das Golden-Set bei Probe- und Zielauflösung und protokolliert je Visualizer und Metrik den Drift `d = |m_probe − m_commit|`.
  - `d ≤ 0.02` → Visualizer gilt als auflösungsstabil.
  - `d > 0.02` → Eintrag in `config/studio_drift.v1.json`; der Solver rechnet mit einem **Sicherheitsabschlag** auf die harten Schwellen (`τ_effektiv = τ − d`), statt die Drift zu ignorieren.
  - `d > 0.10` → Visualizer wird für den Studio-Pfad gesperrt, bis er auflösungsunabhängig gemacht ist. Ehrlicher Ausschluss schlägt falsche Zahlen.
- **Verify-Abbruch nur oberhalb des kalibrierten Drifts:** `drift_max > d_kalibriert + 0.02` ⇒ Implementierungsbug. Alles darunter ist erwartete Physik und landet als `warn` im Sidecar.

Ohne diesen Abschnitt ist §9 („Probe-Commit-Drift ⇒ Abbruch") eine Falle: sie feuert bei korrekter Implementierung.

### 3.5 Kalibrierung statt Magie (`src/studio/thresholds.py`)

- Schwellwerte liegen als versionierte Datei `config/studio_thresholds.v1.json`, nicht als Konstanten im Code.
- **Kalibrier-Harness** `tools/calibrate_thresholds.py`: läuft über ein Golden-Set (≥ 20 Referenz-Renders, manuell als gut/schlecht gelabelt), gibt je Metrik Trennschärfe (Sensitivität/Spezifität bei Schwellwert-Sweep) aus.
- Jede Schwelle trägt ein Feld `provenance: "calibrated@<set-hash>" | "assumed"`. Nicht kalibrierte Schwellen erzeugen im Report eine sichtbare Warnung.
- Die Threshold-Version landet im Sidecar. Ohne diese Kopplung ist kein Report reproduzierbar.

---

## 4. Sampling (`src/studio/sampling.py`)

**v1-Defekt:** 1 Frame/s koppelt mit dem Beat-Grid. Bei 120 BPM trifft man systematisch dieselbe Beat-Phase — die Peaks, die das Gate reißen, sind unsichtbar.

**v2-Strategie — stratifiziert + ereignisgetrieben, `N = 18` Samples (Default):**

| Anteil | Auswahl |
|--------|---------|
| 6 | Gleichverteilt über die Dauer, mit **Jitter** ±0.5 · Intervall (deterministisch aus Audio-Hash geseedet) |
| 6 | Top-k Onset-/RMS-Peaks (Worst-Case für Overlay-Energie) |
| 3 | Minimal-RMS-Fenster (Worst-Case für Vitalität/Blackframe) |
| 3 | Frames mit aktivem Quote-Overlay (nur PODCAST/HYBRID; sonst auf Peaks aufgefüllt) |

- Seed = Hash(Audio-Feature-Cache-Key) → **deterministisch reproduzierbar**, aber nicht beat-phasenverriegelt.
- Die Sample-Liste liegt im Sidecar. Ein Report ohne Sample-Zeitpunkte ist wertlos.
- Verify-Phase nutzt **denselben** Sample-Satz plus 6 zusätzliche Zufallspunkte (Overfitting-Kontrolle: der Solver darf nicht nur die Punkte fixen, die er sieht).

---

## 5. Modus-Weiche (ModeGate)

Unverändert in der Idee, präzisiert in den Rändern:

- Ausgabe `MUSIC | PODCAST | HYBRID` + Konfidenz + begründende Feature-Werte.
- **HYBRID-Auflösung numerisch statt verbal:** `speech_score = 0.5·norm(voice_clarity) + 0.3·norm(voice_band) − 0.2·norm(onset_density)`. `speech_score ≥ 0.55` → Podcast-Regelwerk mit Musik-Parameteranteil, sonst Musik-Regelwerk mit reduzierter Quote-Präsenz.
- **Hysterese:** liegt `speech_score` in [0.50, 0.60], wird die Entscheidung des letzten Laufs für dieselbe Datei (aus dem Feature-Cache) beibehalten. Verhindert Klassenflattern bei minimalen Reanalysen.
- Schwellwerte kommen aus `thresholds.json`, nicht aus dem Modul.
- Whitelist-Keys werden gegen `VISUALIZER_MAP` beim Profil-Load geprüft → Fail-fast.

**Profile** (`profiles.py`, Pydantic) wie v1, ergänzt um: `postfx_budget` (Bloom-/Grain-Obergrenzen als Teil des ConstraintSets), `vitality_corridor`, `subject_strength`, `threshold_set`.

---

## 6. Constraints (`src/studio/constraints.py`)

### 6.1 Anwendungspunkt

**Der Cap greift an der letzten Composite-Stufe vor dem Encode, und die Post-FX-Parameter sind Teil des ConstraintSets.** In v1 wurde vor Bloom geklemmt und nach Bloom gemessen — das ist strukturell nicht erfüllbar. Konkret:

> **Code-verifiziert (2026-07-27, zweiter Durchlauf — korrigiert die erste Fassung):** Der einzige reale Mischpunkt von Visualizer über Hintergrund ist der **Blit-Shader** (`_init_blit_shader`, `gpu_renderer.py:1285-1303`: `f_color = vec4(tex.rgb, tex.a * u_opacity)` mit `SRC_ALPHA`-Blending), aufgerufen aus `_blit_viz_to_fbo` (`gpu_renderer.py:1415`) im Batch-Loop (`:498`) und im Preview (`gpu_preview.py:161`). `_init_composite_shader`/`_composite_viz_over_bg` (`:1235-1283`) wird **nirgends aufgerufen** (toter Code). Der Blit-Shader hat **keinen** Alpha-Fallback — er nutzt `tex.a` direkt. Konsequenzen: (1) `u_viz_alpha_cap`, Luma-Ableitung und Subjekt-Maske gehören in den **Blit-Shader**; (2) die Luma-Ableitung muss im Studio-Modus **unbedingt** gelten (nicht nur bei `tex.a < 0.01`), weil `CompositeVisualizer` hart `alpha = 1.0` ausgibt — der Hauptfall des C14-Defekts würde von einer `viz.a < 0.01`-Bedingung nicht erfasst; (3) die Subjekt-Maske muss im **Bildschirmraum** gesampelt werden (`gl_FragCoord.xy / u_resolution`), weil der Blit-Quad Offset/Scale-UVs hat.

#### Der Alpha-Fallback macht den Cap wirkungslos — C14

Der reale Mischpunkt (Blit-Shader, siehe oben) nutzt `tex.a` direkt und ohne Fallback. Damit deckt **jeder Visualizer mit `alpha = 1.0` — insbesondere jeder `CompositeVisualizer`** (hart `alpha = 1.0`, `composite.py:137`) — die **gesamte Bildfläche** ab, auch dort, wo er nichts zeichnet und sein Layer schwarz ist. (Der Alpha-Fallback im toten Composite-Shader dokumentiert nur, dass das Problem dem ursprünglichen Autor bereits aufgefallen war.)

Konsequenzen, wenn man darauf einfach einen Cap setzt:

- `mix(bg, viz, 0.6)` ist **kein Deckungsregler**, sondern ein gleichmäßiger 60-%-Schleier über das ganze Bild. Der Hintergrund wird nirgends „freigegeben", er wird überall halb ausgeblendet.
- Die Subjekt-Maske moduliert einen bereits flächendeckenden Faktor — Subjektschutz wird zu einem lokalen Aufhellen des Visualizer-Schwarzes, nicht zu einer Freistellung.
- M1 (Overlay-Energie) und M3 (Subjekt-Störung) liefern für praktisch jedes Rezept Höchstwerte. Der Feasibility-Precheck schickt in der Folge **jeden** Job in den Layout-Fallback, und die Schwellenkalibrierung kalibriert auf Rauschen.

**Regelung (Pflicht, gehört in Phase P0):**

```glsl
// im Blit-Shader, VOR der Cap-Anwendung (Studio-Modus: u_viz_alpha_from_luma = 1)
float a_viz = tex.a;
if (u_viz_alpha_from_luma > 0.5) {
    float luma = dot(tex.rgb, vec3(0.2126, 0.7152, 0.0722));
    a_viz = smoothstep(u_luma_knee_lo, u_luma_knee_hi, luma);  // Default 0.02 / 0.25
}
vec2 screen_uv = gl_FragCoord.xy / u_resolution;               // Maske im Bildschirmraum
float a_eff = min(a_viz, u_viz_alpha_cap)
            * (1.0 - u_subject_strength * texture(u_subject_mask, screen_uv).r);
f_color = vec4(tex.rgb, a_eff * u_opacity);                    // SRC_ALPHA-Blending wie Bestand
```

- Semantisch korrekt: Visualizer ohne belastbares Alpha sind faktisch **additive Emitter auf Schwarz**. Ihre Helligkeit *ist* ihre Deckung. Der Soft-Knee (`smoothstep`) verhindert, dass sehr dunkle Bereiche als harte Kante wegbrechen.
- **Unbedingt, nicht konditional:** Die Ableitung gilt im Studio-Modus auch bei `tex.a = 1.0` — sonst bleibt der Composite-Stack-Hauptfall (alpha = 1.0, `composite.py:137`) ein Vollbild-Schleier. Der Blit-Shader hat keinen Alpha-Fallback, der rettet nichts.
- **Verhaltensänderung gegenüber dem Bestand.** Deshalb hinter einem Flag: `u_viz_alpha_from_luma` ist im Studio-Pfad an, im Direkt-Render-Pfad aus. Bestehende Renders bleiben bit-identisch (`f_color = vec4(tex.rgb, tex.a * u_opacity)`).
- `u_luma_knee_lo/hi` sind Profilparameter und **Teil der Kalibrierung** — sie bestimmen, wo „nichts gezeichnet" endet und „gezeichnet" beginnt, und beeinflussen damit jede Metrik.

#### Hebel

- `u_viz_alpha_cap` im Composite-Shader als *erster* Hebel — jetzt mit definierter Wirkung.
- Zusätzlich: `bloom_intensity`, `bloom_threshold`, `film_grain` (bestehende `PostProcessConfig`-Felder) sind Solver-Hebel mit Profil-Obergrenzen. **Ausnahme:** `film_grain` ist während aller Messrenders fixiert (§3.2.1); als Solver-Hebel wirkt es nur auf den Commit-Render und wird deshalb konservativ (Profil-Obergrenze, kein Feintuning) gesetzt.
- Konfigwerte oberhalb der Caps werden **geklemmt + geloggt**, nie verworfen (v1-Verhalten beibehalten, ist korrekt).

### 6.2 Subjekt-Maske (`mask_service.py`)

- Erzeugung pro Hintergrundbild, Cache als NPZ in `.cache/subject_masks/`.
- **Cache-Key = `sha256(Bilddatei) + provider_id + model_hash + service_version`.** v1s „Datei-Hash + Modell-Version" reicht nicht: die Fallback-Kette produziert bei gleichem Modell-Stand unterschiedliche Masken je nach installierten Paketen.
- **Maske wird im Quellbildraum gespeichert**, nicht im Framebuffer-Raum. Der Shader sampelt sie über **dieselbe UV-Transformfunktion wie den Hintergrund** (gemeinsame GLSL-Funktion `bg_uv(vec2)`), damit Crop, Letterbox und Pan-Zoom automatisch mitlaufen.
- **Code-Stand (2026-07-27):** Es gibt heute **keine** Hintergrund-UV-Transform: `_load_background_texture()` resized das Bild hart auf Zielauflösung (`gpu_renderer.py:698-721`, LANCZOS) — kein Cover-Crop, kein Letterbox, kein Pan/Zoom/Ken-Burns. Damit reduziert sich C8 vorerst auf „Maske im Quellbildraum + geometrisch identischer Resize". Die gemeinsame `bg_uv()`-Funktion wird eingeführt, **sobald** Cover-Crop/Letterbox/Bewegung implementiert wird — bis dahin ist sie YAGNI.
- **Identisch ist die Geometrie, nicht der Kernel.** LANCZOS hat negative Nebenkeulen und erzeugt an Maskenkanten Überschwinger (< 0 und > 1) und Halos um das Subjekt — bei einer nahezu binären Maske ist das ein Messfehler, kein Detail. Die Maske wird deshalb mit **AREA (Downscale) bzw. BILINEAR (Upscale)** auf **dieselbe Zielgröße mit derselben Aspect-Behandlung** gebracht und anschließend auf [0, 1] geklemmt. Der Paritätstest prüft Geometrie (Pixelkoordinaten), nicht Filterwahl. Video-Hintergründe existieren dagegen bereits (`gpu_renderer.py:723-771`, FFmpeg-Extraktion + Prefetch-Thread) → Degradationspfad siehe §14, Nicht-Ziele §17.
- Fallback-Kette: rembg/u2net → OpenCV GrabCut/Saliency → Zentrums-Gauß. Jeder Fallback: Warnung + Eintrag in Provenance.
- Alpha-Modulation: `alpha_final = alpha_eff · (1 − subject_strength · mask)`.
- Kein Hintergrundbild → M3 deaktiviert, M1/M2/M5/M6 bleiben aktiv.

### 6.3 Textzonen (PODCAST)

- **Zonen werden abgeleitet, nicht statisch gesetzt:** Kandidatenrechtecke aus `1 − mask` und niedriger Hintergrund-Detailvarianz; Auswahl der größten kollisionsfreien Zone pro Overlay-Slot. Fallback auf statische Zonen aus dem Profil, wenn keine Kandidatenzone die Mindestfläche erreicht.
- In Textzonen zusätzlicher Cap `text_zone_alpha` (Default 0.15).
- **Text-Scrim ist Pflichtbestandteil, nicht Notfallmaßnahme:** ein halbtransparentes, an die Glyphenmaske gebundenes Backplate mit berechneter Opazität garantiert M4 **konstruktiv**. Hintergrund-Abdunkeln/Blur sind nur noch kosmetische Zweit-Hebel. (v1 hatte die Reihenfolge invers — die beiden ersten Hebel garantieren gar nichts.)

### 6.4 Composite-Visualizer / Layer-Stacking (neu, code-verifiziert)

Der `CompositeVisualizer` (`src/gpu_visualizers/composite.py`) baut aus Rezepten (`config/recipes/*.json`) **einen** Fragment-Shader, dessen Ebenen sequenziell auf einen Akkumulator geblendet werden — Ausgabe ist hart `f_color = vec4(acc, 1.0)` (`composite.py:137`). Konsequenzen:

- **Cap greift erst durch die Luma-Ableitung sinnvoll:** `alpha = 1.0` bedeutet beim Composite-Stack „vollflächig deckend", obwohl der Akkumulator in unbespielten Regionen schwarz ist. Ohne C14 (§6.1) wäre der Cap hier ein Vollbild-Schleier. **Der Composite-Stack ist damit der Hauptgrund für C14, nicht ein Sonderfall daneben.** Ein per-Layer-Alpha existiert nicht und wird nicht eingeführt (YAGNI).
- **Messung:** Der Differenz-Render misst den Gesamtstapel automatisch korrekt (A mit Stapel, B ohne) — kein Sonderfall nötig.
- **Hebel-Zuordnung bei Stacks:** Solver-Hebel, die „den Visualizer" adressieren (Scale, Offset, Intensity), werden auf die **dominante Ebene** angewendet (Ebene mit größtem Beitrag zum Stack, aus Rezept-Gewichten abgeleitet); globale Hebel (alpha_cap, Post-FX) wirken ohnehin stapelweit.

---

## 7. Feasibility-Precheck (`src/studio/feasibility.py`) — neu

Läuft **vor** dem ersten Probe-Render, rein analytisch auf Maskenstatistik + Profil:

| Prüfung | Bedingung | Reaktion |
|---------|-----------|----------|
| Subjektfläche | `area(mask > 0.5) > 0.75` und M3-Schwelle aktiv | **Layout-Fallback**: Visualizer-Auswahl auf peripher-geometrische Whitelist einschränken (Rahmen-/Rand-/Ecken-Visualizer) |
| Vitalitätskorridor | erreichbare Restfläche `1 − area(mask)` reicht rechnerisch nicht für M5-Untergrenze | Layout-Fallback, sonst M5 auf `warn` degradieren + Report |
| Textzone | keine Zone ≥ Mindestfläche mit Hintergrund-Luminanzvarianz < Grenze | Scrim erzwingen, `text_zone_alpha` = 0.05 |
| Zielkonflikt | M3-Schwelle und M5-Untergrenze zugleich hart und geometrisch unvereinbar | **Abbruch vor Render** mit klarer Diagnose |

Wert: v1 rendert bis zu 4× ein Video, das prinzipiell nie bestehen kann. v2.1 sagt das in < 200 ms.

---

## 8. Solver statt AutoFix (`src/studio/solver.py`)

**v1-Defekt:** Prioritätstabelle ohne Konvergenzgarantie. „Deckung zu hoch → Alpha ↓" und „Bewegung zu niedrig → Intensity ↑" ziehen gegeneinander; drei Iterationen können zwischen zwei Verletzungen pendeln und dann abbrechen, obwohl eine gültige Lösung existiert.

### 8.1 Skalares Zielmaß

```
J = Σ_i  w_i · max(0, (m_i − τ_i) / τ_i)        # Obergrenzen
  + Σ_j  w_j · max(0, (τ_j − m_j) / τ_j)        # Untergrenzen
```

Gewichte `w`: harte Metriken (M1, M3, M4, M6) = 1.0; Korridormetriken (M5) = 0.4; M2 = 0.0 (nur Report). **`J = 0` ⇔ gate-konform.**

### 8.2 Hebel-Leiter (deterministisch, geordnet)

| Verletzung | Hebel in Reihenfolge, feste Schrittweiten |
|-----------|-------------------------------------------|
| M1/M2 Energie/Deckung | `alpha_cap` −0.08 → `bloom_intensity` ×0.75 → `viz_scale` ×0.9 → `glow` −0.1 |
| M3 Subjekt | `subject_strength` +0.1 → Visualizer-Offset aus Subjektzentrum → `alpha_cap` −0.05 → peripherer Visualizer aus Whitelist |
| M4 Kontrast | `scrim_opacity` +0.12 (garantierend) → `text_zone_alpha` −0.05 → `background_blur` +1 → Textfarbe auf Profil-Kontrastpaar wechseln |
| M5 zu hoch | `speed` ×0.85 → `beat_response` ×0.8 → Amplitudenkorridor Richtung Mitte |
| M5 zu niedrig | `beat_response` ×1.2 → **Farb-/Chroma-Modulation ↑** (bewegt ohne Fläche zu kosten) → `intensity` +0.08 |
| M6 Integrität | Deterministischer Reset auf Profil-Default + Blackframe-Ursachenanalyse (Audio-Stille vs. Shader-NaN) |

Bei M5-zu-niedrig steht die **Chroma-Modulation vor der Intensity** — sie erhöht Vitalität, ohne M1/M3 zu verschlechtern. Genau die Entkopplung, die die v1-Oszillation auflöst.

### 8.3 Ablaufregeln

1. Verletzung mit größtem Einzelbeitrag zu `J` bestimmt den aktiven Hebel.
2. Schritt anwenden → Probe-Render → `J'` berechnen.
3. **Akzeptanz nur bei `J' < J − 0.01`.** Sonst: Schritt verwerfen, nächster Hebel derselben Zeile.
4. Alle Hebel einer Zeile ohne Verbesserung → **Plateau**: Metrik als `infeasible` markieren, Abbruch mit Report.
5. Besuchte Parametervektoren werden gehasht; Wiederholung = sofortiger Plateau-Abbruch (Zyklusschutz).
6. Iterationslimit **8 Probe-Runden** (statt 3 Full-Renders). Kosten pro Runde: 37 Billigframes @probe_res (§3.2.2).

**Garantie:** `J` ist über die akzeptierten Schritte streng monoton fallend → keine Oszillation, terminierender Ablauf.

---

## 9. Commit & Verify

- **Commit:** ein einziger Full-Render mit dem gelösten ConstraintSet.
- **Verify:** Post-Render-Hook liefert die Sample-Frames (gleiche Zeitpunkte + 6 Kontrollpunkte). Metriken werden neu berechnet.
  - Alle hart `pass` → Encode + Sidecar.
  - Abweichung Probe↔Verify > `d_kalibriert + 0.02` auf einer harten Metrik → **Abbruch mit Diagnose „Probe-Commit-Drift"**. Das ist ein Implementierungsbug (Preview/Batch-Divergenz), kein Nutzerproblem, und muss laut werden. Drift **innerhalb** des kalibrierten Budgets (§3.4) ist erwartete Auflösungsphysik und wird nur als `warn` protokolliert — sonst bricht der Normalfall ab.
  - Kein automatischer Re-Render. GUI bietet „Erneut lösen mit strengeren Startwerten" als **expliziten** Nutzerklick.

---

## 10. Preset-Factory (`preset_factory.py`)

Wie v1 (baut auf SmartMatcher auf, ersetzt ihn nicht), ergänzt:

- Erzeugte Presets tragen `schema_version` und `threshold_set`-Referenz; ohne die ist ein exportiertes Preset in sechs Monaten nicht mehr interpretierbar.
- Parameter-Klemmung erfolgt gegen den **Profil-Korridor und** das ConstraintSet — Presets sind per Konstruktion gate-konform im ersten Probe-Durchlauf (Ziel: ≥ 70 % der Läufe lösen mit `J = 0` ohne Solver-Schritt; diese Quote ist eine Akzeptanzkennzahl, kein Wunsch).
- Farbpalette: Key-basiert (`KEY_COLORS`), PODCAST entsättigt; zusätzlich wird pro Palette ein **garantiertes Kontrastpaar** (Text/Scrim) mitgeführt, damit M4 einen deterministischen Endhebel hat.

---

## 11. Preview, GUI, CLI

### 11.1 Preview

- PreviewWorker rendert mit identischem ConstraintSet **und** identischem Messraster; Qualitäts-Badge zeigt M1/M3/M4/M5 live.
- Badge nutzt denselben Differenz-Render (2 Frames pro Preview) — bei debouncetem Einzelframe vernachlässigbar.
- A/B-Vergleich: letzter Frame als „Vorher", inkl. **Metrik-Delta** (nicht nur Bild) — der eigentliche Nutzen des A/B.
- Maskenerzeugung **niemals** im UI-Thread; erster Aufruf pro Bild ist ein Hintergrundtask mit Fortschritt und deaktiviertem Badge (`mask: pending`).

### 11.2 GUI

- `studio_panel`: Modus-Badge (+ Konfidenz), Profil-/Preset-Auswahl, Quality-Badge, **Solver-Trace** (welcher Hebel, welcher Schritt, `J`-Verlauf) als aufklappbare Liste.
- `ki_panel`: „Studio-Preset anwenden".
- Diagnose-Dialog bei Abbruch: verletzte Metriken, Messwerte vs. Schwellen, Sample-Zeitpunkte mit Thumbnails, angewandte Hebel, Feasibility-Befund.

### 11.3 CLI

```
python main.py render song.mp3 --studio
python main.py render song.mp3 --studio --studio-dry      # Analyse + Solve, kein Commit-Render
python main.py render song.mp3 --studio --studio-strict   # Fallback-Maskenprovider = Fehler statt Warnung
```

Sidecar: `<output>.studio.json`. Exit-Code ≠ 0 bei Gate-Abbruch, Report-Pfad auf stderr.

---

## 12. Provenance & Sidecar-Schema (`provenance.py`)

```json
{
  "schema_version": "studio-decision/2.1",
  "created_utc": "2026-07-26T18:00:00Z",
  "input": { "audio_sha256": "...", "background_sha256": "...", "duration_s": 214.3 },
  "mode": { "value": "PODCAST", "confidence": 0.87, "speech_score": 0.63, "hysteresis_applied": false },
  "profile": { "name": "podcast_default", "version": 3 },
  "thresholds": { "set": "config/studio_thresholds.v1.json", "sha256": "...", "calibrated": true },
  "mask": { "provider": "rembg:u2net", "model_sha256": "...", "fallback_chain_used": [], "cache_hit": true },
  "sampling": { "n": 18, "seed": "...", "timestamps_s": [ ... ] },
  "measurement": {
    "probe_res": "960x540", "target_res": "3840x2160",
    "luma_alpha": { "enabled": true, "knee": [0.02, 0.25] },
    "noise": { "seed_mode": "frame_time", "noise_suppressed": false },
    "drift_budget": { "set": "config/studio_drift.v1.json", "per_metric": { "M1": 0.011, "M3": 0.007 } }
  },
  "solver": { "iterations": 3, "j_trace": [0.41, 0.18, 0.04, 0.0], "steps": [ ... ], "final_constraints": { ... } },
  "verify": { "metrics": { "M1": 0.19, "M3": 0.06, "M4": 5.1, "M5": 0.05 }, "status": "pass", "drift_max": 0.004, "drift_within_budget": true },
  "renderer": { "gpu": "...", "driver": "...", "app_version": "..." }
}
```

`schema_version` ist Pflicht — v1s Sidecar wäre nach der ersten Formatänderung unlesbar gewesen.

---

## 13. Performance-Budgets (Akzeptanzkriterien, keine Ziele)

| Vorgang | Budget |
|---------|--------|
| Metrikberechnung, ein Sample @854 px | ≤ 15 ms CPU |
| Probe-Runde (37 Renders @probe_res + Metriken, §3.2.2) | ≤ 2.5 s auf Referenz-GPU |
| Vollständiger Solve (Worst Case 8 Runden) | ≤ 20 s |
| Feasibility-Precheck | ≤ 200 ms |
| Maskenerzeugung, Cache-Miss, ML-Pfad | ≤ 6 s, Hintergrundthread, abbrechbar |
| Preview-Badge-Overhead gegenüber v1-Preview | ≤ +40 ms |
| Commit-Render-Overhead durch Studio-Hooks | ≤ 3 % |

Überschreitung = P2-Defekt, nicht „halt langsam".

---

## 14. Fehlerbehandlung

| Fall | Verhalten |
|------|-----------|
| rembg nicht installiert | Fallback-Kette + Warnung; unter `--studio-strict` Abbruch |
| **Video-Hintergrund übergeben** | **Degradation statt Abbruch:** M3 (Subjekt) und die Maskenregeln werden deaktiviert — identisch zum bereits spezifizierten „kein Hintergrundbild"-Pfad. M1/M2/M4/M5/M6 bleiben aktiv, `mask.provider = "none:video_background"` im Sidecar, sichtbare Warnung im Report. Zusätzlich: B(t) ist nicht mehr zeitinvariant und wird pro Sample gerendert (§3.2.2). Ein harter Abbruch wäre eine Regression an einem funktionierenden Bestandsfeature (`gpu_renderer.py:723-771`). |
| Solver-Plateau | Abbruch vor Commit-Render, Report mit blockierender Metrik und ausgeschöpften Hebeln |
| Feasibility-Konflikt | Abbruch vor jedem Render, Diagnose nennt die geometrische Ursache |
| Probe-Commit-Drift | Abbruch, Diagnose als **Implementierungsfehler** markiert, Sample-Vergleichsbilder im Report |
| Kein Hintergrundbild | M3 aus, Maskenregeln aus, Rest aktiv |
| Unbekannter Visualizer im Profil | Fail-fast beim Load |
| Threshold-Set nicht kalibriert | Lauf erlaubt, Report trägt sichtbares `calibrated: false` |
| Audio kürzer als Sample-Bedarf | Sample-Anzahl adaptiv reduzieren, `n` im Sidecar |

---

## 15. Testing

**Phase-0-Tests (blockierend für alles Weitere):**

- `test_studio_metrics.py` — Metriken auf konstruierten Frames mit analytisch bekannten Sollwerten; NaN/Blackframe/Clipping-Erkennung.
- `test_studio_metric_invariance.py` — **derselbe Frame in 480p/1080p/4K ⇒ Metriken innerhalb ε = 0.01.** Ohne diesen Test ist „Preview = Batch" eine Behauptung.
- `test_studio_diff_render.py` — Headless-GL: Alpha-Cap 0 ⇒ `contrib ≡ 0`; Bloom aktiv ⇒ `contrib` größer als bei Bloom aus (beweist, dass Post-FX mitgemessen wird).
- `test_studio_luma_alpha.py` (**C14, blockierend**) — Visualizer mit `alpha = 1.0` und überwiegend schwarzem Layer: `contrib` muss in den schwarzen Regionen ≈ 0 sein und darf **nicht** flächendeckend dem Cap folgen. Gegenprobe mit deaktiviertem `u_viz_alpha_from_luma` muss den Vollbild-Schleier zeigen — der Test dokumentiert den Defekt, den er verhindert. Composite-Stack durchläuft denselben Test.
- `test_studio_noise_cancellation.py` (**C15, blockierend**) — bei `film_grain > 0` und identischem Seed (gegeben durch `u_time`, `:1132`, `:1142`): `contrib` auf einem visualizer-freien Frame ≡ 0 (Toleranz 1e-4). Mit unterschiedlichem `u_time` muss der Test **fehlschlagen** (Negativkontrolle, sonst prüft er nichts). M5 auf grain-freiem Paar gegen M5 auf grain-behaftetem Paar: Differenz wird protokolliert und gegen die Grain-Amplitude plausibilisiert.
- `test_studio_resolution_drift.py` (**C16, blockierend**) — je Visualizer der Whitelist: Metriken bei `probe_res` vs. Zielauflösung; Drift wird gemessen und nach `config/studio_drift.v1.json` geschrieben. Der Test schlägt nicht bei Drift fehl, sondern **wenn kein Driftwert erfasst wurde** — unbekannte Drift ist der Defekt, nicht Drift selbst.

**Weitere:**

- `test_studio_mode_gate.py` — Klassifikation, Determinismus, Hysterese an Grenzfällen.
- `test_studio_sampling.py` — Beat-Aliasing-Regressionstest: synthetisches 120-BPM-Signal, Sample-Verteilung darf nicht beat-phasenverriegelt sein.
- `test_studio_constraints.py` — Klemmung mit Warnung, Maskeninjektion, **geometrische** Masken-Resize-Parität zum Hintergrund-Pfad (gleiche Zielgröße/Aspect) **bei nicht-negativem Kernel**: Maskenwerte nach Resize bleiben in [0, 1], keine Kanten-Überschwinger.
- `test_studio_feasibility.py` — Subjekt 90 % Fläche ⇒ Layout-Fallback bzw. Abbruch **ohne** Render (Render-Mock zählt Aufrufe: erwartet 0).
- `test_studio_solver.py` — **Property-Test: `J` fällt über akzeptierte Schritte streng monoton**; Zyklusschutz greift; Plateau-Abbruch statt Endlosschleife; Konfliktszenario M1↔M5 löst über Chroma-Hebel auf.
- `test_studio_preset_factory.py` — Schema-Validierung, Whitelist-Konformität, Korridor-Einhaltung, `schema_version` gesetzt.
- `test_studio_provenance.py` — Sidecar vollständig, Pflichtfelder, deterministischer Seed.
- `test_studio_integration.py` — 2-s-Mini-Render @480p mit gemocktem FFmpeg: **genau ein Commit-Render-Aufruf**, Sidecar geschrieben, Verify grün.
- `test_studio_video_background.py` — Video-Hintergrund unter `--studio`: Lauf **erfolgreich**, M3 deaktiviert, Warnung im Report, `mask.provider = "none:video_background"`. Explizit kein Abbruch.
- `test_studio_perf.py` — Budgets aus §13 als Assertions (mit großzügigem CI-Faktor).

---

## 16. Umsetzungsreihenfolge

| Phase | Inhalt | Definition of Done |
|-------|--------|--------------------|
| **P0** | `metrics.py`, Messraster, Differenz-Render, **Luma-Alpha-Ableitung im Composite-Shader (C14)**, **Rausch-Regeln für Messrenders (C15, Regel 3 — Regeln 1+2 sind Bestand)**, **Drift-Kalibrierung (C16)**, `thresholds.py`, Kalibrier-Harness | Invarianz-, Diff-Render-, Luma-Alpha-, Rausch- und Drift-Tests grün; `studio_drift.v1.json` erzeugt; Golden-Set angelegt |
| **P1** | `mask_service.py` (Source-Space, geometrische Resize-Parität, nicht-negativer Kernel), `constraints.py`, restliche Shader-Uniforms | Maske überlebt Paritäts- und Kantentest; Cap wirkt post-FX nachweisbar — auch bei Composite-Stacks |
| **P2** | `sampling.py`, `feasibility.py` | Aliasing-Test grün; unlösbarer Fall bricht mit 0 Renders ab |
| **P3** | `solver.py`, `engine.py` (Probe/Commit/Verify), `provenance.py` | Monotonie-Property-Test grün; Integrationstest zählt 1 Commit-Render |
| **P4** | `mode_gate.py`, `profiles.py`, `preset_factory.py` | ≥ 70 % der Golden-Set-Läufe lösen mit `J = 0` ohne Solver-Schritt |
| **P5** | GUI-Badges, Solver-Trace, Diagnose-Dialog, CLI-Flags | Perf-Budget Preview eingehalten |

**P0 ist nicht verhandelbar zuerst — und C14/C15/C16 gehören hinein, nicht in eine spätere Phase.** Jede Zeile Solver-Code vor einem verifizierten Messfundament optimiert gegen Rauschen; im Fall von C15 wörtlich.

---

## 17. Nicht-Ziele (YAGNI)

- Kein Echtzeit-Audio-Playback, keine Echtzeit-Renderloop im GUI.
- Keine deklarative DAG-Pipeline, kein Plugin-Stage-System.
- Kein neues Hauptfenster.
- Kein Training eigener ML-Modelle.
- **Keine Subjekt-Maskierung auf Video-Hintergründen** — das Maskenkonzept setzt ein statisches Quellbild voraus. Video-Hintergründe bleiben unter `--studio` **voll nutzbar**, nur ohne M3 (§14). Keyframe-Masken alle N Frames + zeitliche Glättung sind v3-Scope. *(Korrektur gegenüber v2: dort war ein harter Abbruch vorgesehen — das wäre eine Feature-Regression gewesen.)*
- **Keine Hintergrund-UV-Transform (Cover-Crop/Letterbox/Pan-Zoom)** — existiert im Bestand nicht (harter Resize); die gemeinsame `bg_uv()`-Funktion wird erst mit Einführung solcher Transforms gebaut.
- **Kein automatischer Re-Render nach Commit.** Bewusst: die Schleife gehört vor den teuren Schritt.
- **Kein per-Layer-Alpha im CompositeVisualizer** — der Cap am luma-abgeleiteten Mix-Faktor (§6.1) deckt Stacks ab.
- Keine Mehrsprachigkeit der Diagnose-Texte.

---

## 18. Neue Dateien & Eingriffspunkte

**Neu (`src/studio/`):** `__init__.py`, `engine.py`, `mode_gate.py`, `profiles.py`, `constraints.py`, `mask_service.py`, `metrics.py`, `sampling.py`, `feasibility.py`, `solver.py`, `preset_factory.py`, `thresholds.py`, `provenance.py`, `types.py`

**Neu (sonstige):** `config/studio_thresholds.v1.json`, **`config/studio_drift.v1.json`**, `tools/calibrate_thresholds.py`, **`tools/measure_drift.py`**, `tests/golden/` (Referenzframes), `.cache/subject_masks/`

**Geändert (minimal):**

- `src/gpu_renderer.py` — im `_init_blit_shader` (`:1285-1303`, Aufruf via `_blit_viz_to_fbo` `:1415`): **Luma-Alpha-Ableitung hinter Flag `u_viz_alpha_from_luma` (C14, unbedingt im Studio-Modus)**, Uniforms `u_viz_alpha_cap`, `u_luma_knee_lo/hi`, `u_subject_strength`, `u_subject_mask`, `u_resolution`; Masken-Textur mit geometrischer Resize-Parität; **grain-freier Messmodus für M5 (C15, Regel 3 — Seeding selbst ist Bestand)**; **skalierbare Bloom-/Glow-Radien für Probe-Auflösung (C16, Bloom-Radius bestätigt pixelbasiert)**; Post-Render-Sample-Hook; Post-FX-Parameter von außen setzbar. *(Korrektur: nicht `_init_composite_shader` — der ist ungenutzter Code.)*
- `src/gpu_preview.py` — ConstraintSet + Differenz-Render-Modus
- `src/gui/workers.py`, `studio_panel.py`, `ki_panel.py`, `preview_widget.py` — Badges, Metrik-Delta-A/B, Solver-Trace, Diagnose-Dialog
- `main.py` — `--studio`, `--studio-dry`, `--studio-strict`
- `requirements.txt` — `rembg` als Extra

---

## 19. Konfidenz- und Lückenreport

| Bereich | Konfidenz | Anmerkung |
|---------|-----------|-----------|
| Probe/Commit/Verify-Umbau | **hoch** | Rein struktureller Gewinn, unabhängig von Codedetails |
| Differenz-Render als Messtechnik | **hoch** | Semantisch korrekt; Kosten nur in Probe/Preview |
| Kontinuierliche Metriken → Solver-Konvergenz | **hoch** | Monotonie ist konstruktiv, nicht empirisch |
| Konkrete Schwellenwerte (0.22 / 0.10 / 4.5 / M5-Korridor) | **niedrig** | **Bewusst geraten.** Ohne Golden-Set nicht belastbar — deshalb ist die Kalibrierung Teil von P0 |
| Misch-/Composite-Shader, Post-FX-Kette, Hintergrund-Pfad, Layer-Stacking | **verifiziert (2026-07-27, 2 Durchläufe)** | Realer Mischpunkt = Blit-Shader (`_init_blit_shader` `:1285-1303`, via `_blit_viz_to_fbo` `:1415`, Batch `:498`, Preview `gpu_preview.py:161`); `_init_composite_shader`/`_composite_viz_over_bg` (`:1235-1283`) ist **toter Code**. Blit-Shader ohne Alpha-Fallback → Luma-Ableitung unbedingt (siehe §6.1). Bloom (`:511-517`) und Final-Pass (`:519-534`) nach dem Blend über `PostProcessConfig`; harter Resize ohne UV-Transform (`:698-721`); Video-Hintergründe vorhanden (`:723-771`); `CompositeVisualizer` mit `alpha = 1.0` (`composite.py:137`). |
| C14 Luma-Alpha-Ableitung | **hoch (Diagnose) / mittel (Parametrierung)** | Dass der Cap ohne Luma-Ableitung ein Vollbild-Schleier ist, folgt zwingend aus dem Fallback. Die Knee-Werte 0.02/0.25 sind **geraten** und Kalibrierungsgegenstand. |
| C15 Rauschaufhebung | **verifiziert (2026-07-27)** | Seeding ist frame-zeitbasiert und deterministisch: Grain `fract(u_time * 100.0)` (`:1132-1134`), Dither `fract(u_time)` (`:1142`), `u_time` aus dem Render-Loop (`:528`). A/B-Seed-Gleichheit ist Bestand; nur Regel 3 (grain-freies M5-Paar) ist neu zu bauen. |
| C16 Auflösungsdrift | **hoch (Existenz) / niedrig (Größenordnung)** | Bloom-Radius bestätigt pixelbasiert (`gpu_bloom.py:169-172, 181-183`). **Wie stark** die Drift insgesamt ist, ist unbekannt und genau deshalb Messgegenstand statt Annahme. Die Schwellen 0.02/0.10 sind Platzhalter bis zur ersten Messung. |
| C17 Video-Degradation | **hoch** | Nutzt einen bereits spezifizierten Pfad; kein neuer Mechanismus. |
| Perf-Budgets | **mittel** | Plausibel; abhängig von realer Shader-Komplexität. Das Kostenmodell in §3.2.2 ist gegenüber v2 um ~30 % gesunken (B-Caching). |
| Textzonen-Ableitung aus Salienz | **mittel** | Heuristik; Fallback auf statische Zonen ist deshalb Pflicht |
| Alle numerischen Schwellen (M1–M5, Knee, Drift) | **niedrig** | Durchgängig geraten. P0 ohne Golden-Set liefert ein funktionierendes Messsystem mit willkürlichen Grenzen. |
| Aufwandsschätzung | **nicht erstellt** | Bewusst ausgelassen |

**Verbleibende offene Lücken (Nutzer-Input erforderlich):**

1. **Golden-Set:** Existieren gelabelte Referenz-Renders? Ohne die bleiben alle Schwellen `calibrated: false` — das Messsystem funktioniert, die Grenzen sind Konvention.
2. **Lizenz der Segmentierungsmodelle:** Bei kommerzieller Kanalverwertung ist der Modell-Lizenzstatus (u2net vs. isnet-Varianten) zu prüfen — nicht Teil dieser Spezifikation.
