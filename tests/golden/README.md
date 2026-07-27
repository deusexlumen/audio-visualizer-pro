# Golden-Set für die Studio-Schwellenkalibrierung

≥ 20 Referenz-Renders als gut/schlecht labeln (Spec §3.5, §19 Lücke 1).
Je Eintrag in labels.json: id, good (bool), metrics {M1, M3, M4, M5}.
Metrikwerte erzeugt der ProbeRenderer; Labels vergibt der Mensch.
Ohne ≥ 20 Labels bleiben alle Schwellen "assumed" und Reports tragen
calibrated: false.

## Stand 2026-07-27

36 Einträge via `tools/build_golden_set.py` (Sweep: 6 Visualizer ×
alpha_cap {0.3, 0.6, 1.0} × Maske {keine, Zentrums-Gauß}, Sprach-Audio).
**Achtung: `construction_labels: true`** — Labels per Parameterwahl, keine
menschliche Beurteilung. Kalibrier-Report liegt in
`config/studio_thresholds.v1.json` unter `calibration_report`; operative
Schwellen bleiben "assumed" (Trennschärfe zu schwach).
Review: `contact_sheet.png` durchsehen, Labels in `labels.json`
korrigieren, dann `python tools/calibrate_thresholds.py` erneut laufen
lassen und Schwellen auf "calibrated@<set-hash>" setzen.
