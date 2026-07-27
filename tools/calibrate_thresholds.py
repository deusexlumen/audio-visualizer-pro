"""Schwellen-Kalibrierung über das Golden-Set (Spec §3.5).

Liest labels.json {"renders": [{"id": str, "good": bool,
"metrics": {"M1": float, "M3": float, "M4": float, "M5": float}}]},
sweept Kandidaten-Schwellen je Metrik und gibt Sensitivität/Spezifität aus.
"""

import argparse
import hashlib
import json
from pathlib import Path


def set_hash(labels_path: str) -> str:
    """sha256 der Label-Datei — Anker für 'calibrated@<set-hash>'."""
    return hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()


def sweep_threshold(values, labels, higher_is_bad, candidates=None):
    """Beste Schwelle nach Youden-Index (Sensitivität + Spezifität − 1).

    values: Metrikwerte, labels: True = gut. higher_is_bad: Wert über der
    Schwelle gilt als schlecht (für M4 übergeben: False).
    """
    if candidates is None:
        lo, hi = min(values), max(values)
        candidates = [lo + (hi - lo) * i / 100 for i in range(1, 100)]
    best = None
    for t in candidates:
        tp = fp = tn = fn = 0
        for v, good in zip(values, labels):
            bad = v > t if higher_is_bad else v < t
            if bad and not good:
                tp += 1
            elif bad and good:
                fp += 1
            elif not bad and good:
                tn += 1
            else:
                fn += 1
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        score = sens + spec - 1.0
        if best is None or score > best["score"]:
            best = {"threshold": t, "sensitivity": sens,
                    "specificity": spec, "score": score}
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="tests/golden/labels.json")
    args = parser.parse_args()

    data = json.loads(Path(args.labels).read_text())
    renders = data.get("renders", [])
    if len(renders) < 20:
        print(f"WARNUNG: nur {len(renders)} gelabelte Renders "
              f"(Minimum 20) — Schwellen bleiben 'assumed'.")
    for metric, higher_is_bad in [("M1", True), ("M3", True), ("M4", False)]:
        # None-Werte (z.B. M3 ohne Maske, M4 ohne Quotes) herausfiltern
        pairs = [(r["metrics"][metric], r["good"]) for r in renders
                 if r["metrics"].get(metric) is not None]
        if not pairs:
            print(f"{metric}: keine Werte — übersprungen.")
            continue
        values = [v for v, _ in pairs]
        labels = [g for _, g in pairs]
        best = sweep_threshold(values, labels, higher_is_bad)
        print(f"{metric}: t={best['threshold']:.3f} "
              f"sens={best['sensitivity']:.2f} spec={best['specificity']:.2f}")


if __name__ == "__main__":
    main()
