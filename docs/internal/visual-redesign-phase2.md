# Visual-Qualität — Phase 2: Prinzipien und Vorgehen

Stand: 2026-08-09. Ergänzt `10x/session-2.md` (dort: technische Befunde).
Hier: die vom Nutzer vorgegebenen Design-Prinzipien und das Vorgehen für den Neu-/Umbau.

## Nutzer-Verdikt zur Sichtung (2026-08-09, bewegte Previews)

**Umbau oder Kill (unfertig / unpassend / billig):**
orchestral_swell, liquid_blobs, speech_focus, particle_swarm, pulsing_core, typographic

**OK (bleiben):** alle anderen 12 (aurora_voice, bass_temple, chroma_field,
frequency_flower, lumina_core, neon_oscilloscope, neon_wave_circle, nebula_drift,
sacred_mandala, spectrum_bars, spectrum_genesis, voice_flow)

**Zitate:** gut — Einblendung/Timing der Overlays passt. Anforderung: Zitate müssen
zum gesprochenen Moment erscheinen (Sync auf Sprechzeit, nicht willkürlich).

**Neues Prinzip Hintergrundbild-Harmonie:** In den meisten Fällen liegt ein
Hintergrundbild unter der Visualisierung. Design muss so arbeiten, dass das Bild
sichtbar bleibt und die Visualisierung sich harmonisch darüberlegt
(Feintuning über Parameter später). Konkret: transparente/dunkle Anteile statt
Vollfläche, additive Leuchtelemente, zurückhaltende Flächenfüllung.

## Referenz-Projekt (Nutzer-Vorgabe)

github.com/deusexlumen/audio-reactive-Visualizer- (React/Canvas-Prototyp,
Analyse in `.cache/ref-visualizer`). Übernommene Kern-Ideen:

- **Feature-Mapping-Muster:** Bass = Stoß/Puls/Radius, Treble = Textur/Jitter/Funken,
  Mid = Drift/Verschiebung, Energy = Menge/Tempo. Nichtlinear: pow(v, 1.5–3).
- **Stil × Theme-Trennung:** Stil = Geometrie/Algorithmus; Theme = 2 Farben,
  sensitivity, lineWidth, backgroundFade (Trail), glowIntensity — als generische
  Parameter-Schicht auf unser PARAMS-System mappbar.
- **Trail-Effekt:** Ping-Pong-Framebuffer, altes Bild pro Frame abdunkeln statt Clear.
- **Compositing über Hintergrund:** Visualizer-Layer additiv ('lighter'/screen) über
  dem Hintergrundbild; Schwarz = transparent. Passt zu unseren background_*-Feldern.
- **Stil-Ideen-Pool:** Neon-Tunnel, Sunburst, Metropolis-Skyline, Retro-Sun,
  DNA-Helix, Spirograph, Vortex, Galaxy, Plasma, String-Theory, Equalizer mit
  Peak-Hold, Kaleidoskop (Winkel-Fold im Shader).

## Design-Prinzipien (verbindlich)

1. **Grundlegend verschiedene Archetypen.**
   Nicht N Varianten von Kreisen/Wellen. Jeder Visualizer ist eine eigene Welt:
   Partikel, Geometrie, Typografie, Felder, Linien-Systeme etc.
2. **Reiche Reaktionen.**
   Mehr als „wird größer bei Lautstärke". Beispiel aus Nutzer-Referenz:
   Farbwechsel gekoppelt an Bass. Jeder Visualizer braucht mehrere,
   verschiedenartige Audio-Reaktionen (Größe, Farbe, Form, Dichte, Geschwindigkeit).
3. **Modus = Empfindlichkeit, nicht Visualisierung.**
   Derselbe Visualizer kann Musik UND Podcast. Was sich unterscheidet:
   auf welche Features er wie reagiert.
   - Musik: Beats, Onset, Chroma, Bass, Tempo.
   - Sprache: Voice-Clarity, Voice-Band, Pausen/Silenz, Betonung, Sprechrhythmus.
4. **Parametrisierbarkeit.**
   Neue Looks als Parameter-Presets, nicht als neue Code-Pfade.

## Vorgehen

1. **Phase 1 (läuft)**: Preview-Harness rendert pro Visualizer je einen
   Musik- und Podcast-Clip (12 s, 720p, Podcast mit Zitat-Overlay)
   nach `output/previews/` + `index.html` zur Sichtung.
2. **Sichtung durch Nutzer**: Verdikt pro Visualizer = keep / redesign / kill.
   (Erwartung nach bisherigem Feedback: überwiegend redesign/kill.)
3. **Archetyp-Map**: Für redesign/kill je Ziel-Archetyp festlegen,
   Dopplungen (mehrere Kreis-/Wellen-Varianten) zusammenlegen oder streichen.
4. **Umbau in Wellen** (2–3 Visualizer pro Welle), jede Welle:
   umbauen → Preview rendern → Nutzer-Sichtung → erst dann nächste Welle.
5. Erst nach Freigabe der neuen Basis: Golden Set v2 erneut labeln
   (dann sinnvoll, weil die Basis dem Nutzer gefällt).

## Stand Umbau (2026-08-10)

Alle sechs Umbau-Kandidaten sind neu gebaut, Tests grün (400 passed),
Previews aktuell in `output/previews/` (Standbilder zur Sichtung:
`output/previews/_frames/welle2/`).

| Visualizer | Neuer Archetyp | Zustand |
|---|---|---|
| pulsing_core | Neon-Tunnel | fertig, Sichtung offen |
| particle_swarm | Galaxie/Vortex | fertig, Sichtung offen |
| speech_focus | Stimm-Linie / Spektrum-Band | fertig, Sichtung offen |
| typographic | Metropolis-Skyline | fertig, Sichtung offen |
| liquid_blobs | Plasma-Metaballs | fertig, Sichtung offen |
| orchestral_swell | Swell-Vorhänge | fertig, Sichtung offen |

Beim Nachprüfen der Standbilder gefundene und behobene Fehler:

- `orchestral_swell`: Vorhänge hingen von oben herab — im Renderer ist
  `gl_FragCoord.y = 0` **oben** im fertigen Bild, der Shader muss y spiegeln
  (`uv.y = 1.0 - uv.y`, so macht es auch `typographic`). Zusätzlich:
  Schwellkurve wird jetzt auf das 95%-Perzentil des Clips normiert (leise
  gemasterte Stücke ließen die Vorhänge dauerhaft bei ~20 % stehen),
  Grundhöhen stärker gestreut, Strahl-Abfall nach oben.
- `liquid_blobs`: Blobs wirkten hohl (Rand heller als Kern). Kern-Normierung
  und Beat-Kopplung liegen jetzt beim Kern statt beim Rand.

- `speech_focus`: gleicher y-Spiegel-Fehler — die Ruhezone für die Zitate lag
  auf der falschen Bildhälfte.

Zitat-Lesbarkeit im Podcast-Modus für alle sechs geprüft: in Ordnung.

## speech_focus: Musik-Modus als Spektrum-Band (2026-08-10)

Im Musik-Modus öffnet sich die Stimm-Linie zu einer symmetrischen Schleife um
die Bildmitte; die halbe Höhe pro x-Position folgt einem Frequenz-Profil
(links tief, rechts hoch), zwischen den Kanten liegt ein zarter Schleier.
Im Sprach-Modus bleiben die Werte flach und klein (`u_band_gain` 0.12), die
Schleife fällt optisch auf die dünne Linie zusammen — Prinzip 3 bleibt gewahrt.

Das Profil ist **kein FFT-Spektrum**: `analyze()` bleibt unangetastet (Caching).
Es entsteht aus vorhandenen Kanälen — Bass aus `rms`/`transient`, Mitten aus
`rms`, Höhen aus `spectral_centroid`/`zero_crossing_rate`, dazu die 12
Chroma-Werte als bewegte Feinstruktur (multiplikativ, damit die Grobform
bleibt). Anschließend auf die eigene Spitze normiert, Kontrast über `pow`,
Pegel getrennt aus Lautstärke. **Kein Peak-Hold** — das zieht alle
Stützstellen auf ihr jeweiliges Maximum und bügelt die Silhouette flach.

~~Offener Punkt: `podcast_macy.m4a` wird vom Analyzer als `mode = "music"`
klassifiziert.~~ Erledigt 2026-08-10, siehe `mode-detection.md`. Die
Podcast-Previews laufen jetzt über den Sprach-Zweig; Standbilder dazu in
`output/previews/_frames/speech-modus/`. Alle sechs verhalten sich wie
entworfen (ruhiger als im Musik-Modus, Zitate überall lesbar).

## Offene technische Punkte (aus Session 2, weiterhin gültig)

- M4 (Quote-Lesbarkeit) in `src/studio/engine.py` berechnen.
- ~~C14-Alpha-Fix (Luma-Alpha im Blit-Shader).~~ Erledigt 2026-08-10:
  Der Shader-Code war fertig, aber nur im Studio-Pfad verdrahtet. Im
  normalen Render-Pfad wurde die Visualizer-Ebene mit `alpha = 1.0`
  geblittet — jedes schwarze Pixel war deckend und uebermalte ein
  Hintergrundbild vollstaendig. Betraf alle 19 Visualizer, auch die
  unveraenderten. Luma-Alpha wird jetzt automatisch aktiv, sobald ein
  Hintergrundbild oder -video gesetzt ist (Renderer und Live-Vorschau).
  Ohne Hintergrund bleibt es beim deckenden Blit.
  Lehre: Previews auf Schwarz zeigen diesen Fehler nicht — "dunkel =
  gedacht transparent" sieht dort genauso aus wie "dunkel = deckendes
  Schwarz". Visualizer immer auch mit Hintergrundbild sichten.
- ~~Silhouetten verschwinden ueber dem Hintergrundbild.~~ Erledigt
  2026-08-11: Luma-Alpha kann "dunkel UND deckend" nicht ausdruecken, die
  Metropolis-Skyline loeste sich damit in ein Fenster-Raster auf. Neuer
  Opt-in `WRITES_OCCLUSION_ALPHA` (Klassen-Flag in `base.py`): der Shader
  schreibt seine Deckung selbst nach `f_color.a`, der Blit verknuepft sie
  per `max()` mit der Luma-Deckung. Nur `typographic` nutzt das; die
  anderen 18 Visualizer bleiben unveraendert. Neuer Parameter
  `silhouette_opacity` (0.88) steuert, wieviel Bild durchscheint.
  Im Crossfade nur aktiv, wenn beide Szenen eine Deckung schreiben —
  `_xfade_prog` mischt auch den Alpha-Kanal.
  Nebeneffekt: ohne Hintergrundbild zeigt der Himmel jetzt die gewaehlte
  `background_color` statt hartem Schwarz.
- Golden Set v2 wurde mit `mode = "music"` für alle sechs Audios gelabelt.
  Die Labels der Podcast-Renders beziehen sich damit auf den falschen Zweig
  und sind für Sprach-Vergleiche nicht mehr gültig.
- Sprach-Feature-Ebene in `src/analyzer.py` ERWEITERN (nie ändern, Caching!):
  Silenz/Pausen, Sprechrhythmus, Betonung — Voraussetzung für Prinzip 3.
- `speech_focus` rendert schwarz (auf allen Audios) — beim Umbau ersetzen oder reparieren.
- Quote-Overlay-Hex-Farben-Bug: GEFIXT 2026-08-09 (`_normalize_color` in `src/quote_overlay.py`).

## Welle 3: drei neue Archetypen (2026-08-11)

Der Umbau hat die sechs abgelehnten Visualizer ersetzt, aber nichts
Neues hinzugefuegt. Aus dem Stil-Ideen-Pool wurden drei Archetypen
gebaut, die in der Sammlung bisher fehlten:

| Visualizer | Archetyp | Kernidee |
|---|---|---|
| `retro_sun` | Landschaft/Horizont | Sonne mit waagerechten Schlitzen ueber einem perspektivischen Gitter |
| `dna_helix` | Struktur/Gitter | Doppelhelix, Querstreben leuchten pro Chroma-Ton |
| `kaleidoscope` | Symmetrie/Spiegelung | Winkel-Faltung in N Sektoren, Sektorfarbe aus Chroma |

Bewusst NICHT gebaut: Sunburst (waere eine weitere Radialform neben
`sacred_mandala`/`frequency_flower`), String-Theory (liegt zu nah an
`neon_oscilloscope`). Prinzip 1 verlangt eigene Welten, keine Varianten.

### Neue automatische Pruefung

`tests/test_visuals_welle3.py` prueft, was sich am Bild messen laesst:

- **Prinzip 3**: derselbe Visualizer laeuft in `music`/`speech`/`hybrid`
  und ist bei Sprache messbar ruhiger.
- **Prinzip 5**: mindestens 35 % der Flaeche bleiben praktisch schwarz —
  dort blendet der Blit-Shader die Visualizer-Ebene aus und ein
  Hintergrundbild bleibt sichtbar.
- **Zeitspruenge**: das Bild bei t = 3.0 s haengt nicht davon ab, welche
  Frames vorher gerendert wurden (sonst flackert die Vorschau beim
  Scrubben).

Die Regel gilt bewusst nur fuer die neuen Visualizer. Bei den aelteren
waere sie eine Nachruestung mit offenem Ausgang.

### Beim Bauen gefundene Fehler

- `kaleidoscope` deckte anfangs 93 % der Flaeche zu — der Prinzip-5-Test
  hat das gefunden, bevor ein Frame gesichtet wurde. Ursache waren zwei
  Dinge: ein verdrehter Parameter (`line_sharpness` machte die Linien
  breiter statt schmaler) und ein fbm-Schleier ueber dem ganzen Bild.
- `retro_sun`: das perspektivische Gitter lief nahe am Horizont in eine
  geschlossene Flaeche. Ursache ist die Perspektive selbst — `1/depth`
  waechst dort so schnell, dass die Linienfolge dichter wird als das
  Pixelraster. Die Gitterlinie blendet jetzt aus, sobald `fwidth` zu
  gross wird.
- `retro_sun`: die Chroma-Faerbung machte aus der Sonne je nach Tonart
  einen gruenen Klecks. Der Farbton wird jetzt zu 85 % auf die
  Sonnenuntergangs-Palette gezogen; Chroma verschiebt nur noch.
- `dna_helix`: im Sprach-Modus verschwanden die Querstreben ganz, weil
  Chroma dort durch den Stimmwert ersetzt wird. Jetzt mit Grundwert —
  ruhiger ja, unsichtbar nein.

Sichtung: `output/previews/_frames/welle3/` (je Musik und Podcast,
gerendert MIT Hintergrundbild).

## Welle 4: zwei weitere Archetypen (2026-08-12)

| Visualizer | Archetyp | Kernidee |
|---|---|---|
| `spirograph` | Kurvenzeichnung | Hypotrochoide; das Radienverhaeltnis (= Zackenzahl) kommt aus dem staerksten Chroma-Ton, drei Echos ziehen eine Spur |
| `voronoi_cells` | Zerlegung/Mosaik | Wanderndes Zellnetz, nur die Kanten leuchten; zwoelf Zellen gehoeren je einem Chroma-Ton |

**Vortex und Galaxy aus dem Ideen-Pool entfallen**: `particle_swarm` ist
seit dem Welle-2-Umbau genau das (Docstring: "Galaxy/Vortex Visualizer").
Sie zu bauen waere die Dopplung, die Prinzip 1 verbietet.

Damit ist der Ideen-Pool aus dem Referenz-Projekt weitgehend abgearbeitet.
Offen und noch nicht bewertet: Mechanik/Zahnraeder, Baender/Tuch,
Tinten-/Rauchfahne (Fluid-Advektion, teuer).

Beim Bauen gefunden:

- `spirograph` kostet bei 96 Stuetzstellen rund 17 ms pro Frame bei 720p.
  Ohne die Abkuerzung "ausserhalb des Huellkreises gar nicht rechnen"
  waere es ein Vielfaches. Bei 4K ist der Visualizer entsprechend teuer.
- `voronoi_cells` fiel zuerst durch den Dunkelflaechen-Test (26 %):
  Kanten zu breit, Halo zu weit, Flaechentoenung zu kraeftig.

Ausserdem nachgebessert: das Gitter von `retro_sun` wirkte flach. Linien
haben jetzt einen weichen Saum, werden nach vorne hin heller (Tiefe) und
unter der Sonne liegt eine gestreifte Spiegelung.

## Welle 5: Fluid und Stoff — ohne Simulation (2026-08-12)

| Visualizer | Archetyp | Kernidee |
|---|---|---|
| `ink_bloom` | Fluid/Diffusion | Tinte in Wasser: Schlieren und Faeden aus verschachteltem Domain Warping |
| `silk_ribbons` | Tuch/Band | Wehende Baender mit Glanzkante aus einer Pseudo-Normalen |

### Warum keine echte Simulation

Beide Archetypen waeren die klassischen Kandidaten fuer einen Solver:
Advektion fuer die Tinte, Massepunkte und Federn fuer das Tuch. Beides
braucht **Zustand ueber Frames** (Ping-Pong-Framebuffer bzw. Integration
der Partikelpositionen). Das kollidiert mit zwei Eigenschaften, die das
Projekt bisher hat:

1. Der Offline-Render soll reproduzierbar sein.
2. Die Vorschau springt beim Scrubben an beliebige Zeitpunkte. Mit
   Frame-Zustand saehe derselbe Zeitpunkt jedes Mal anders aus — genau
   das prueft `test_zeitsprung_ist_reproduzierbar`.

Statt der Simulation zwei Tricks:

**Tinte — Domain Warping.** fbm-Rauschen wird mit sich selbst verzerrt:

    q = fbm(p);  r = fbm(p + 4*q);  d = fbm(p + 4*r)

Die Wirbel und Faeden entstehen aus der Verschachtelung. Fuenf
Rauschabfragen pro Pixel, 9 ms pro Frame bei 720p, zustandslos.
Ein Dichte-Tor (`density_gate`) laesst nur die dichten Faeden leuchten —
ohne das legt sich ein geschlossener Schleier ueber das Bild.

**Stoff — Glanz statt Geometrie.** Ein Band wirkt erst dann wie Stoff,
wenn es eine Oberflaeche hat. Aus der **analytischen** Ableitung der
Mittellinie wird eine Pseudo-Normale gebaut:

    n = normalize(vec2(-dy/dx, 1))

Mit fester Lichtrichtung wandert der Glanzstreifen ueber die Faltung —
die Woelbung wird sichtbar, ohne dass eine Flaeche berechnet wird. Die
Ableitung muss analytisch sein; numerisch verrauscht sie bei hohen
Wellenzahlen sichtbar.

Was dabei verloren geht, ehrlich benannt: echte Tinte reagiert auf ihre
eigene Vorgeschichte (eine Wolke, die einmal da war, bleibt und wird
weitergetragen). Domain Warping hat kein Gedaechtnis — es sieht aus wie
Stroemung, ist aber ein wanderndes Muster. Bei langsamen Passagen faellt
das nicht auf, bei einem einzelnen harten Stoss schon: die Fahne baut
sich nicht auf, sie ist einfach da.
