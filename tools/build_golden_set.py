"""Baut das Golden-Set für die Studio-Schwellenkalibrierung (Spec §3.5, §19).

Analysiert ein Referenz-Audio einmal und sweeped eine Matrix aus
Visualizer × alpha_cap × Subjekt-Maske. Pro Variante werden die
Probe-Metriken (M1, M3, M4, M5) via ``evaluate_params`` gemessen, ein
Vorschau-Frame als PNG abgelegt und ein Konstruktions-Label vergeben
(good = alpha_cap ≤ 0.6 — die cap-1.0-Varianten demonstrieren die
Regelverletzung). Ergebnis: ``tests/golden/labels.json`` plus
``tests/golden/contact_sheet.png`` zur visuellen Kontrolle.
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

DEFAULT_AUDIO = (
    "assets/user_uploads/"
    "Die_Macy-Stiftung_als_Fundament_der_Kontrolle.m4a_33410687_"
    "Die_Macy-Stiftung_als_Fundament_der_Kontrolle.m4a"
)

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


def build_contact_sheet(frame_dir: Path, entries: list[dict],
                        out_path: Path, cols: int = 6,
                        thumb_w: int = 320) -> None:
    """Grid aller Vorschau-Frames mit ID + Label unter jedem Frame."""
    label_h = 34
    thumb_h = int(thumb_w * PROBE_H / PROBE_W)
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)),
                      (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for i, entry in enumerate(entries):
        img = Image.open(frame_dir / f"{entry['id']}.png").convert("RGB")
        img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
        x, y = (i % cols) * thumb_w, (i // cols) * (thumb_h + label_h)
        sheet.paste(img, (x, y))
        tag = "GOOD" if entry["good"] else "BAD"
        color = (120, 220, 120) if entry["good"] else (230, 110, 110)
        draw.text((x + 4, y + thumb_h + 2), entry["id"], fill=(220, 220, 220))
        draw.text((x + 4, y + thumb_h + 16), tag, fill=color)
    sheet.save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Windows-Konsole (cp1252) verträgt kein "→"/"≤" — UTF-8 erzwingen
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser.add_argument("--audio", default=DEFAULT_AUDIO,
                        help="Referenz-Audio (Sprache empfohlen)")
    parser.add_argument("--out-dir", default="tests/golden")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # 1) Audio-Analyse (einmalig; Cache in .cache/ beschleunigt Folgeläufe)
    t0 = time.time()
    print(f"[golden] Analysiere {args.audio} ...", flush=True)
    features = AudioAnalyzer().analyze(args.audio, fps=FPS)
    print(f"[golden] Analyse fertig in {time.time() - t0:.1f}s "
          f"(Dauer {features.duration:.1f}s, {features.frame_count} Frames)",
          flush=True)
    features_dict = build_features_dict(features, features.frame_count, FPS)
    duration = float(features_dict["duration"])
    timestamps = [duration * f for f in SAMPLE_FRACTIONS]
    t_mid = duration * 0.5

    # 2) Sweep-Matrix: Visualizer × alpha_cap × Maske
    gauss_mask = _center_gauss(PROBE_W, PROBE_H)
    entries: list[dict] = []
    total = len(VISUALIZERS) * len(ALPHA_CAPS) * len(MASK_VARIANTS)
    idx = 0
    for viz_name in VISUALIZERS:
        viz_cls = get_visualizer(viz_name)
        for cap in ALPHA_CAPS:
            for use_mask in MASK_VARIANTS:
                idx += 1
                t_var = time.time()
                var_id = f"{viz_name}_{_cap_token(cap)}"
                if use_mask:
                    var_id += "_mask"
                mask = gauss_mask if use_mask else None
                constraints = MeasureConstraints(
                    alpha_cap=cap, alpha_from_luma=True,
                    subject_strength=0.8)

                probe = ProbeRenderer(width=PROBE_W, height=PROBE_H, fps=FPS)
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

                Image.fromarray(frame).save(frame_dir / f"{var_id}.png")

                good = cap <= GOOD_CAP_MAX
                note = (
                    f"Konstruktions-Label: alpha_cap={cap} "
                    f"{'≤' if good else '>'} {GOOD_CAP_MAX} → "
                    f"{'good' if good else 'bad'}; "
                    f"Maske {'Zentrums-Gauß' if use_mask else 'keine'}."
                )
                entries.append({
                    "id": var_id,
                    "good": good,
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
                })
                m3_txt = (f"{metrics['M3']:.4f}"
                          if metrics["M3"] is not None else "n/a")
                print(f"[golden] ({idx}/{total}) {var_id}: "
                      f"M1={metrics['M1']:.4f} M3={m3_txt} "
                      f"M5={metrics['M5']:.4f} "
                      f"M6={metrics['M6_violations']} "
                      f"({time.time() - t_var:.1f}s)", flush=True)

    # 3) labels.json schreiben
    labels = {
        "version": "golden-set/1",
        "construction_labels": True,
        "renders": entries,
    }
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"[golden] {len(entries)} Einträge → {labels_path}", flush=True)

    # 4) Contact Sheet
    sheet_path = out_dir / "contact_sheet.png"
    build_contact_sheet(frame_dir, entries, sheet_path)
    print(f"[golden] Contact Sheet → {sheet_path}", flush=True)
    print(f"[golden] Gesamtlaufzeit: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
