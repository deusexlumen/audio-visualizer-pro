"""Baut das Golden-Set für die Studio-Schwellenkalibrierung (Spec §3.5, §19).

Sweeped pro Referenz-Audio aus dem Korpus (``tests/golden/corpus.json``)
eine Matrix aus Visualizer × alpha_cap × Subjekt-Maske. Pro Variante werden
die Probe-Metriken (M1, M3, M4, M5) via ``evaluate_params`` gemessen, ein
Vorschau-Frame als PNG abgelegt und ein Konstruktions-Label vergeben
(good = alpha_cap ≤ 0.6). Ergebnis: ``tests/golden/labels.json`` (Schema
golden-set/2, mit leeren ``human_label``-Feldern) plus ein Contact Sheet
pro Audio zur visuellen Kontrolle und menschlichen Label-Vergabe.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Projekt-Root auf sys.path (Skript liegt in tools/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzer import AudioAnalyzer
from src.gpu_visualizers import get_visualizer
from src.render_common import build_features_dict
from src.studio.engine import evaluate_params
from src.studio.mask_service import _center_gauss
from src.studio.probe import ProbeRenderer
from src.studio.types import MeasureConstraints

from tools.golden_corpus import load_corpus, missing_audio_files

VISUALIZERS = ["spectrum_bars", "lumina_core", "voice_flow",
               "speech_focus", "neon_wave_circle", "pulsing_core"]
ALPHA_CAPS = [0.3, 0.6, 1.0]
MASK_VARIANTS = [False, True]

PROBE_W, PROBE_H = 854, 480
FPS = 30
SAMPLE_FRACTIONS = [0.2, 0.5, 0.8]
GOOD_CAP_MAX = 0.6  # Konstruktions-Regel: cap ≤ 0.6 gilt als "good"


def _cap_token(cap: float) -> str:
    """Formatiert alpha_cap als ID-Token: 0.3 -> 'cap03', 1.0 -> 'cap10'."""
    return f"cap{int(round(cap * 10)):02d}"


def variant_id(audio_id: str, viz_name: str, cap: float,
               use_mask: bool) -> str:
    """Eindeutige Varianten-ID: '<audio>__<viz>_capXX[_mask]'."""
    vid = f"{audio_id}__{viz_name}_{_cap_token(cap)}"
    if use_mask:
        vid += "_mask"
    return vid


def make_label_entry(vid: str, audio_entry: dict, viz_name: str,
                     cap: float, use_mask: bool, metrics: dict,
                     good: bool) -> dict:
    """Baut einen labels.json-v2-Eintrag (Konstruktions-Label, Human-Feld leer)."""
    note = (
        f"Konstruktions-Label: alpha_cap={cap} "
        f"{'<=' if good else '>'} {GOOD_CAP_MAX} -> "
        f"{'good' if good else 'bad'}; "
        f"Maske {'Zentrums-Gauß' if use_mask else 'keine'}."
    )
    return {
        "id": vid,
        "good": good,
        "human_label": None,
        "audio": audio_entry["id"],
        "mode": audio_entry["mode"],
        "visualizer": viz_name,
        "alpha_cap": cap,
        "mask": use_mask,
        "metrics": {
            "M1": metrics["M1"],
            "M3": metrics["M3"],
            "M4": metrics["M4"],
            "M5": metrics["M5"],
        },
        "construction_note": note,
    }


def build_contact_sheet(frame_base: Path, entries: list[dict],
                        out_path: Path, cols: int = 6,
                        thumb_w: int = 320) -> None:
    """Grid aller Vorschau-Frames mit ID + Label unter jedem Frame.

    ``entries[i]["frame"]`` ist der PNG-Pfad relativ zu ``frame_base``.
    Ein gesetztes ``human_label`` schlägt das Konstruktions-Label in der
    Anzeige (HUMAN: GOOD/BAD vs. CONSTR: GOOD/BAD).
    """
    label_h = 34
    thumb_h = int(thumb_w * PROBE_H / PROBE_W)
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)),
                      (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for i, entry in enumerate(entries):
        img = Image.open(frame_base / entry["frame"]).convert("RGB")
        img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
        x, y = (i % cols) * thumb_w, (i // cols) * (thumb_h + label_h)
        sheet.paste(img, (x, y))
        human = entry.get("human_label")
        if human in ("good", "bad"):
            tag = f"HUMAN: {human.upper()}"
            color = (120, 220, 120) if human == "good" else (230, 110, 110)
        else:
            tag = "CONSTR: " + ("GOOD" if entry["good"] else "BAD")
            color = (120, 180, 220) if entry["good"] else (220, 160, 110)
        draw.text((x + 4, y + thumb_h + 2), entry["id"], fill=(220, 220, 220))
        draw.text((x + 4, y + thumb_h + 16), tag, fill=color)
    sheet.save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Windows-Konsole (cp1252) vertraegt kein "→"/"≤" — UTF-8 erzwingen
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser.add_argument("--corpus", default="tests/golden/corpus.json",
                        help="Korpus-Manifest (golden-corpus/1)")
    parser.add_argument("--audio", default=None,
                        help="Einzelnes Audio statt Korpus (Modus via --mode)")
    parser.add_argument("--mode", default="podcast",
                        choices=["music", "podcast", "hybrid"],
                        help="Modus fuer --audio")
    parser.add_argument("--out-dir", default="tests/golden")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.audio:
        audios = [{"id": Path(args.audio).stem, "path": args.audio,
                   "mode": args.mode, "description": "Einzelnes Audio",
                   "source": args.audio}]
    else:
        audios = load_corpus(args.corpus)
        missing = missing_audio_files(audios)
        if missing:
            raise SystemExit(
                f"Fehlende Audio-Dateien: {', '.join(missing)}")

    gauss_mask = _center_gauss(PROBE_W, PROBE_H)
    entries: list[dict] = []
    analyzer = AudioAnalyzer()
    total = (len(audios) * len(VISUALIZERS)
             * len(ALPHA_CAPS) * len(MASK_VARIANTS))
    idx = 0
    t0 = time.time()
    for audio in audios:
        print(f"[golden] Analysiere {audio['id']} ({audio['path']}) ...",
              flush=True)
        features = analyzer.analyze(audio["path"], fps=FPS)
        features_dict = build_features_dict(features, features.frame_count,
                                            FPS)
        duration = float(features_dict["duration"])
        timestamps = [duration * f for f in SAMPLE_FRACTIONS]
        t_mid = duration * 0.5

        frame_dir = out_dir / "frames" / audio["id"]
        frame_dir.mkdir(parents=True, exist_ok=True)
        audio_entries: list[dict] = []
        for viz_name in VISUALIZERS:
            viz_cls = get_visualizer(viz_name)
            for cap in ALPHA_CAPS:
                for use_mask in MASK_VARIANTS:
                    idx += 1
                    t_var = time.time()
                    vid = variant_id(audio["id"], viz_name, cap, use_mask)
                    local = vid.split("__", 1)[1]
                    mask = gauss_mask if use_mask else None
                    constraints = MeasureConstraints(
                        alpha_cap=cap, alpha_from_luma=True,
                        subject_strength=0.8)
                    probe = ProbeRenderer(width=PROBE_W, height=PROBE_H,
                                          fps=FPS)
                    try:
                        viz = viz_cls(probe.ctx, PROBE_W, PROBE_H)
                        metrics = evaluate_params(
                            probe, viz, features_dict, timestamps, {},
                            constraints, subject_mask=mask)
                        frame = probe.render_frame(
                            viz, features_dict, t_mid, None, {},
                            constraints, subject_mask=mask)
                    finally:
                        probe.release()

                    rel_frame = f"frames/{audio['id']}/{local}.png"
                    Image.fromarray(frame).save(out_dir / rel_frame)

                    good = cap <= GOOD_CAP_MAX
                    entry = make_label_entry(vid, audio, viz_name, cap,
                                             use_mask, metrics, good)
                    entry["frame"] = rel_frame
                    entries.append(entry)
                    audio_entries.append(entry)
                    m3_txt = (f"{metrics['M3']:.4f}"
                              if metrics["M3"] is not None else "n/a")
                    print(f"[golden] ({idx}/{total}) {vid}: "
                          f"M1={metrics['M1']:.4f} M3={m3_txt} "
                          f"M5={metrics['M5']:.4f} "
                          f"M6={metrics['M6_violations']} "
                          f"({time.time() - t_var:.1f}s)", flush=True)

        sheet_path = out_dir / f"contact_sheet_{audio['id']}.png"
        build_contact_sheet(out_dir, audio_entries, sheet_path)
        print(f"[golden] Contact Sheet -> {sheet_path}", flush=True)

    # labels.json schreiben (Schema golden-set/2)
    labels = {
        "version": "golden-set/2",
        "construction_labels": True,
        "human_labels_pending": True,
        "renders": entries,
    }
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"[golden] {len(entries)} Eintraege -> {labels_path}", flush=True)
    print(f"[golden] Gesamtlaufzeit: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
