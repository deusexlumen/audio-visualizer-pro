"""Import-Pipeline für Podcast-Transkripte.

Liest ein Transkript im Format ``[MM:SS] Sprecher: Text``, extrahiert
Zitate (Text in Anführungszeichen), erkennt Duplikate und schreibt das
Ergebnis als JSON (quotes.json).

CLI:
    python transcript_import.py transcript.txt -o quotes.json
"""

import argparse
import json
import re
import sys

# Zeilen wie: [12:34] Max: Das ist "ein Zitat" im Satz.
# MM:SS oder H:MM:SS, Sprecher bis zum ersten Doppelpunkt.
LINE_RE = re.compile(
    r"^\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*"
    r"(?P<speaker>[^:\[\]]+?)\s*:\s*"
    r"(?P<text>.+?)\s*$"
)

# Zitat-Abschnitte in typografischen oder geraden Anführungszeichen.
QUOTE_RE = re.compile(r"[\"„«](.+?)[\"“»]")


def load_transcript(path):
    """Liest eine Transkript-Datei als UTF-8-Text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def timestamp_to_seconds(ts):
    """Wandelt 'MM:SS' oder 'H:MM:SS' in Sekunden (int) um."""
    parts = [int(p) for p in ts.split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def normalize_text(text):
    """Normalisiert Text für den Duplikat-Vergleich (Kleinschreibung, Whitespace)."""
    return re.sub(r"\s+", " ", text.strip().lower())


def find_quotes(text):
    """Extrahiert Zitate mit Zeitstempel aus einem Transkript-Text.

    Erkennt Zeilen im Format ``[MM:SS] Sprecher: Text`` und darin
    Abschnitte in Anführungszeichen. Gibt eine Liste von Dicts zurück:
    ``speaker``, ``text``, ``timestamp`` (Original) und
    ``start_time`` (Sekunden, kompatibel zum Quote-Schema des Projekts).
    """
    quotes = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        speaker = m.group("speaker").strip()
        start_time = timestamp_to_seconds(m.group("ts"))
        for qm in QUOTE_RE.finditer(m.group("text")):
            quote_text = qm.group(1).strip()
            if quote_text:
                quotes.append(
                    {
                        "speaker": speaker,
                        "text": quote_text,
                        "timestamp": m.group("ts"),
                        "start_time": start_time,
                    }
                )
    return quotes


def dedupe_quotes(quotes):
    """Entfernt Duplikate anhand des normalisierten Zitat-Texts.

    Behält das erste Vorkommen. Gibt (eindeutige Liste, Anzahl Duplikate) zurück.
    """
    seen = set()
    unique = []
    duplicates = 0
    for q in quotes:
        key = normalize_text(q["text"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(q)
    return unique, duplicates


def save_quotes_json(quotes, path):
    """Schreibt die Zitat-Liste als JSON-Datei."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(quotes, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extrahiert Zitate aus einem Podcast-Transkript ([MM:SS] Sprecher: Text)."
    )
    parser.add_argument("input", help="Pfad zur Transkript-Datei (UTF-8)")
    parser.add_argument(
        "-o",
        "--output",
        default="quotes.json",
        help="Ziel-JSON-Datei (Standard: quotes.json)",
    )
    args = parser.parse_args(argv)

    text = load_transcript(args.input)
    quotes = find_quotes(text)
    unique, duplicates = dedupe_quotes(quotes)
    save_quotes_json(unique, args.output)

    print(
        f"{len(unique)} Zitate nach {args.output} geschrieben "
        f"({duplicates} Duplikate entfernt)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
