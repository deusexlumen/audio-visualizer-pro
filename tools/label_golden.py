"""Menschliche Labels für das Golden-Set setzen (Spec §3.5).

Workflow: Contact Sheets (tests/golden/contact_sheet_<audio>.png) ansehen,
dann Einträge einzeln oder per Muster labeln:

    python tools/label_golden.py --stats
    python tools/label_golden.py --list
    python tools/label_golden.py --set "music_severance__pulsing_core_cap03=good"
    python tools/label_golden.py --set "podcast_macy__*_cap10*=bad" --dry-run
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path

VALID_LABELS = {"good", "bad"}


def set_labels(data, pattern, label):
    """Setzt human_label für alle Render-ids, die auf das fnmatch-Muster passen.

    Rückgabe: Liste der geänderten ids.
    """
    if label not in VALID_LABELS:
        raise ValueError(f"Label muss einer von {sorted(VALID_LABELS)} sein.")
    changed = []
    for r in data["renders"]:
        if fnmatch.fnmatchcase(r["id"], pattern):
            r["human_label"] = label
            changed.append(r["id"])
    return changed


def label_stats(data):
    """Zählt human_label-Belegung: (good, bad, offen)."""
    good = bad = open_ = 0
    for r in data["renders"]:
        if r.get("human_label") == "good":
            good += 1
        elif r.get("human_label") == "bad":
            bad += 1
        else:
            open_ += 1
    return good, bad, open_


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser.add_argument("--labels", default="tests/golden/labels.json")
    parser.add_argument("--list", action="store_true",
                        help="Ungelabelte ids ausgeben")
    parser.add_argument("--stats", action="store_true",
                        help="Label-Statistik ausgeben")
    parser.add_argument("--set", dest="assignments", action="append",
                        default=[], metavar="MUSTER=good|bad",
                        help="human_label per fnmatch-Muster setzen")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur anzeigen, nicht schreiben")
    args = parser.parse_args()

    path = Path(args.labels)
    data = json.loads(path.read_text(encoding="utf-8"))

    changed_any = False
    for assignment in args.assignments:
        pattern, _, label = assignment.partition("=")
        changed = set_labels(data, pattern.strip(), label.strip())
        changed_any = changed_any or bool(changed)
        print(f"{pattern.strip()}: {len(changed)} Eintraege -> {label.strip()}")
        if args.dry_run:
            for cid in changed:
                print(f"  {cid}")

    good, bad, open_ = label_stats(data)
    data["human_labels_pending"] = open_ > 0
    if args.stats or args.assignments:
        print(f"Labels: {good} good / {bad} bad / {open_} offen")
    if args.list:
        for r in data["renders"]:
            if r.get("human_label") not in VALID_LABELS:
                print(r["id"])
    if changed_any and not args.dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"Gespeichert: {path}")


if __name__ == "__main__":
    main()
