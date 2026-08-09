"""Schwellen-Kalibrierung über das Golden-Set (Spec §3.5).

Liest labels.json {"renders": [{"id": str, "good": bool,
"human_label": "good"|"bad"|null, "metrics": {...}}]}, sweept
Kandidaten-Schwellen je Metrik und gibt Sensitivität/Spezifität aus.
Menschliche Labels (human_label) schlagen Konstruktions-Labels (good).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def set_hash(labels_path: str) -> str:
    """sha256 der Label-Datei — Anker für 'calibrated@<set-hash>'."""
    return hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()


def effective_label(render):
    """Menschliches Label schlägt Konstruktions-Label.

    Rückgabe (label, quelle): label True = gut; quelle 'human' oder
    'construction'.
    """
    human = render.get("human_label")
    if human in ("good", "bad"):
        return human == "good", "human"
    return bool(render["good"]), "construction"


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
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="tests/golden/labels.json")
    args = parser.parse_args()

    data = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    renders = data.get("renders", [])
    labeled = [effective_label(r) for r in renders]
    human_count = sum(1 for _, src in labeled if src == "human")
    if human_count < 20:
        print(f"WARNUNG: nur {human_count} menschliche Labels "
              f"(Minimum 20) — Schwellen bleiben 'assumed'.")
    for metric, higher_is_bad in [("M1", True), ("M3", True), ("M4", False)]:
        # None-Werte (z.B. M3 ohne Maske, M4 ohne Quotes) herausfiltern
        pairs = [(r["metrics"][metric], lbl)
                 for r, (lbl, _src) in zip(renders, labeled)
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
