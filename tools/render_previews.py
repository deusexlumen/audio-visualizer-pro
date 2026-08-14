"""Rendert bewegte Preview-Videos aller Visualizer zur aesthetischen Bewertung.

Pro Visualizer zwei Clips:
- Musik-Modus (ohne Text) aus tests/golden/audio/music_severance.m4a
- Podcast-Modus (mit sichtbarem Zitat-Overlay) aus tests/golden/audio/podcast_macy.m4a

Aufruf:
    python tools/render_previews.py                 # alles
    python tools/render_previews.py --only spectrum_bars --mode podcast
    python tools/render_previews.py --list
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUSIC_AUDIO = ROOT / "tests" / "golden" / "audio" / "music_severance.m4a"
PODCAST_AUDIO = ROOT / "tests" / "golden" / "audio" / "podcast_macy.m4a"
OUT_DIR = ROOT / "output" / "previews"
DURATION = 12.0
RESOLUTION = "1280x720"
FPS = 30

# Beispiel-Zitate fuer den Podcast-Modus (Deckung ueber die ganze Preview-Laenge)
PODCAST_QUOTES = [
    {
        "text": "Die eigentliche Frage ist nicht, ob die Technik es kann — sondern was wir daraus machen.",
        "start_time": 0.5,
        "end_time": 4.5,
        "confidence": 0.95,
    },
    {
        "text": "Wer den Prozess nicht versteht, kann ihn auch nicht verbessern.",
        "start_time": 5.0,
        "end_time": 8.5,
        "confidence": 0.9,
    },
    {
        "text": "Am Ende zaehlt nur, ob es sich richtig anfuehlt.",
        "start_time": 9.0,
        "end_time": 11.8,
        "confidence": 0.85,
    },
]


def visualizer_names() -> list[str]:
    """Alle registrierten Visualizer-Namen."""
    sys.path.insert(0, str(ROOT))
    from src.gpu_visualizers import list_visualizers

    return sorted(list_visualizers())


def _podcast_config(visual: str, path: Path) -> Path:
    """Schreibt eine temporaere Podcast-Config mit Zitaten fuer einen Visualizer."""
    cfg = {
        "audio_file": str(PODCAST_AUDIO),
        "output_file": "unused.mp4",
        "visual": {
            "type": visual,
            "resolution": [int(x) for x in RESOLUTION.split("x")],
            "fps": FPS,
            "colors": {
                "primary": "#FF0055",
                "secondary": "#00CCFF",
                "background": "#0A0A0A",
            },
            "params": {},
        },
        "quotes": PODCAST_QUOTES,
        "quote_overlay": {
            "enabled": True,
            "font_size": 44,
            "position": "bottom",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def _is_valid_mp4(path: Path) -> bool:
    """True, wenn die Datei existiert und ffprobe sie lesen kann.

    Abgebrochene Renders hinterlassen trunkierte mp4 ohne moov-Atom —
    ffprobe schlaegt dann fehl und die Datei wird neu gerendert.
    """
    if not path.exists() or path.stat().st_size < 50_000:
        return False
    proc = subprocess.run(
        ["ffprobe", "-v", "error", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def render_one(visual: str, mode: str, log_dir: Path, skip_existing: bool = False) -> tuple[bool, str]:
    """Rendert einen Preview-Clip. Gibt (erfolg, output_pfad) zurueck."""
    out = OUT_DIR / mode / f"{visual}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and _is_valid_mp4(out):
        return True, str(out)

    if mode == "music":
        cmd = [
            sys.executable, str(ROOT / "main.py"), "render", str(MUSIC_AUDIO),
            "-v", visual, "-r", RESOLUTION, "--fps", str(FPS),
            "--preview", "--preview-duration", str(DURATION),
            "-o", str(out),
        ]
    else:
        cfg_path = _podcast_config(visual, OUT_DIR / "_configs" / f"{visual}.json")
        cmd = [
            sys.executable, str(ROOT / "main.py"), "render", str(PODCAST_AUDIO),
            "-c", str(cfg_path),
            "--preview", "--preview-duration", str(DURATION),
            "-o", str(out),
        ]

    log_path = log_dir / f"{mode}_{visual}.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
    ok = proc.returncode == 0 and out.exists() and out.stat().st_size > 0
    return ok, str(out)


def write_index(names: list[str]) -> Path:
    """Schreibt eine HTML-Uebersicht aller Previews (lokal im Browser abspielbar)."""
    rows = []
    for name in names:
        rows.append(
            f"<tr><th>{name}</th>"
            f'<td><video src="music/{name}.mp4" controls muted preload="metadata"></video></td>'
            f'<td><video src="podcast/{name}.mp4" controls muted preload="metadata"></video></td></tr>'
        )
    html = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<title>Visualizer-Previews</title>
<style>
  body {{ background: #111; color: #eee; font-family: sans-serif; margin: 24px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #333; padding: 8px; vertical-align: top; }}
  th {{ text-align: left; font-family: monospace; }}
  video {{ width: 100%; max-width: 480px; background: #000; }}
  h1 {{ font-size: 1.3em; }}
</style></head><body>
<h1>Visualizer-Previews — Musik (ohne Text) vs. Podcast (mit Zitat)</h1>
<table>
<tr><th>Visualizer</th><th>Musik</th><th>Podcast</th></tr>
{chr(10).join(rows)}
</table></body></html>
"""
    out = OUT_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview-Videos aller Visualizer rendern")
    parser.add_argument("--only", help="Nur ein Visualizer")
    parser.add_argument("--mode", choices=["music", "podcast", "both"], default="both")
    parser.add_argument("--list", action="store_true", help="Visualizer auflisten")
    parser.add_argument("--index", action="store_true", help="Nur index.html neu schreiben")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Fertige, lesbare Clips ueberspringen")
    parser.add_argument("--shard", metavar="I/N",
                        help="Nur Shard I von N bearbeiten (fuer Parallel-Laeufe)")
    args = parser.parse_args()

    names = visualizer_names()
    if args.list:
        print("\n".join(names))
        return 0
    if args.index:
        print(f"Index: {write_index(names)}")
        return 0

    if args.only:
        if args.only not in names:
            print(f"Unbekannter Visualizer: {args.only}", file=sys.stderr)
            return 2
        names = [args.only]

    if args.shard:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            print("Shard-Format: I/N (z.B. 0/3)", file=sys.stderr)
            return 2
        names = names[shard_i::shard_n]

    modes = ["music", "podcast"] if args.mode == "both" else [args.mode]
    log_dir = OUT_DIR / "logs"

    total = len(names) * len(modes)
    done = failed = 0
    failures = []
    for i, name in enumerate(names, 1):
        for mode in modes:
            ok, out = render_one(name, mode, log_dir, skip_existing=args.skip_existing)
            done += ok
            failed += not ok
            status = "OK " if ok else "FEHLER"
            print(f"[{done + failed}/{total}] {status} {mode}/{name} -> {out}", flush=True)
            if not ok:
                failures.append(f"{mode}/{name}")

    print(f"\nFertig: {done} ok, {failed} fehlgeschlagen")
    print(f"Index: {write_index(visualizer_names())}")
    if failures:
        print("Fehlgeschlagen: " + ", ".join(failures))
        print(f"Logs: {log_dir}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
