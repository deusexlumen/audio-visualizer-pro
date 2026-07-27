# Golden-Set für die Studio-Schwellenkalibrierung

≥ 20 Referenz-Renders als gut/schlecht labeln (Spec §3.5, §19 Lücke 1).
Je Eintrag in labels.json: id, good (bool), metrics {M1, M3, M4, M5}.
Metrikwerte erzeugt der ProbeRenderer; Labels vergibt der Mensch.
Ohne ≥ 20 Labels bleiben alle Schwellen "assumed" und Reports tragen
calibrated: false.
