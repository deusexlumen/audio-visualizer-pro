# Golden-Set für die Studio-Schwellenkalibrierung

Schema golden-set/2: 6 Referenz-Audios (3 Musik, 3 Podcast — siehe
`corpus.json`) × 6 Visualizer × alpha_cap {0.3, 0.6, 1.0} × Maske
{keine, Zentrums-Gauß} = 216 Renders.

Je Eintrag in labels.json: id, good (Konstruktions-Label),
human_label (null|"good"|"bad"), audio, mode, visualizer, alpha_cap,
mask, metrics {M1, M3, M4, M5}, frame (PNG-Pfad).
Metrikwerte erzeugt der ProbeRenderer; **Labels vergibt der Mensch**
(Raster: `docs/internal/golden-set-labeling-raster.md`).
Ohne ≥ 20 menschliche Labels bleiben alle Schwellen "assumed".

## Workflow

    python tools/build_golden_set.py            # Frames + Sheets + labels.json
    python tools/label_golden.py --stats        # Label-Fortschritt
    python tools/calibrate_thresholds.py        # Trennschärfe prüfen

## Stand 2026-08-09 (v2)

Neuaufbau auf Multi-Audio-Korpus. v1 (36 Einträge, ein Speech-Audio,
reine Konstruktions-Labels) liegt als `labels.v1.backup.json`.
