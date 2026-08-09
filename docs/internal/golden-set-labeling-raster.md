# Golden-Set Labeling-Raster (golden-set/2)

Zweck: menschliche Qualitäts-Labels für die Studio-Schwellenkalibrierung.
Die Konstruktions-Labels (alpha_cap ≤ 0.6) sind nur Platzhalter; erst
menschliche Labels machen Schwellen "calibrated" statt "assumed".

## Workflow

1. `python tools/build_golden_set.py` (baut Frames + Contact Sheets)
2. Pro Audio das Contact Sheet ansehen: `tests/golden/contact_sheet_<audio>.png`
   (36 Frames: 6 Visualizer × 3 alpha_caps × 2 Masken)
3. Labels setzen: `python tools/label_golden.py --set "<muster>=good|bad"`
   (einzeln per id, oder Muster wie `podcast_macy__*_cap10*=bad`;
   vorher `--dry-run` zum Prüfen)
4. Fortschritt: `python tools/label_golden.py --stats`
5. Wenn alle 216 gelabelt: `python tools/calibrate_thresholds.py`

## Bewertungsfrage (einzige Frage pro Frame)

**"Würde ich diesen Frame so in einem fertigen Video veröffentlichen?"**

- good = ja, so freigebefähig
- bad = nein — im Zweifel immer bad (keine "meh"-Kategorie)

## Worauf achten (Brücke zu den Metriken)

- **M1 Overlay-Energie**: "Frisst der Visualizer das Bild?" — good, wenn
  Hintergrund/Motiv noch erkennbar ist und der Frame nicht zugekleistert wirkt.
- **M3 Subjekt-Störung**: Im Maskenzentrum (da, wo ein Sprecher/Gesicht läge)
  soll es ruhig bleiben. bad, wenn genau die Bildmitte unruhig/verdeckt ist.
- **M4 Kontrast**: Wäre Quote-Text auf diesem Frame lesbar? bad, wenn der
  Hintergrund zu unruhig/kontrastarm für Text ist.
- **M5 Vitalität (modusabhängig)**:
  - music: genug Bewegung/Energie — statisch-langweilig ist bad.
  - podcast: Ruhe — hektisches Flackern/Zucken ist bad.
- **M2 Coverage / M6 Integrität**: keine eigenen Label-Kriterien. M2 (wie viel
  der Fläche der Visualizer bedeckt) und M6 (Rendering-Fehler/Artefakte)
  fließen indirekt in die Bewertungsfrage ein: ein Frame mit sichtbaren
  Render-Artefakten oder leerem/schwarzem Bild ist bad, ohne dass du eine
  Metrik dafür kennen musst.

## Regeln für konsistente Labels

- Immer ein Audio am Stück labeln (Kontextwechsel vermeiden).
- Nicht die Metrikwerte in labels.json ansehen, bevor das Label steht
  (sonst labelt man die Metrik, nicht das Bild).
- ~2-3 Sekunden Pro Frame; nicht grübeln — Bauchentscheid nach der
  Bewertungsfrage.
- Die id unter jedem Frame nennt Visualizer/alpha_cap/Maske; das
  Konstruktions-Tag (CONSTR: ...) ignorieren — es ist bewusst simpel.
