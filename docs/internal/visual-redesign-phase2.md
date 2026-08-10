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
| speech_focus | Stimm-Linie (kein Schwarzbild mehr) | fertig, Sichtung offen |
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

Zitat-Lesbarkeit im Podcast-Modus für alle sechs geprüft: in Ordnung.

## Offene technische Punkte (aus Session 2, weiterhin gültig)

- M4 (Quote-Lesbarkeit) in `src/studio/engine.py` berechnen.
- C14-Alpha-Fix (Luma-Alpha im Blit-Shader).
- Sprach-Feature-Ebene in `src/analyzer.py` ERWEITERN (nie ändern, Caching!):
  Silenz/Pausen, Sprechrhythmus, Betonung — Voraussetzung für Prinzip 3.
- `speech_focus` rendert schwarz (auf allen Audios) — beim Umbau ersetzen oder reparieren.
- Quote-Overlay-Hex-Farben-Bug: GEFIXT 2026-08-09 (`_normalize_color` in `src/quote_overlay.py`).
