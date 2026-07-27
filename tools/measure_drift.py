"""Drift-Messung: Metriken bei probe_res vs. Zielauflösung (C16, Spec §3.4).

Schreibt config/studio_drift.v1.json. Visualizer mit d > 0.10 werden als
resolution_dependent markiert (Studio-Sperrung, bis auflösungsfest).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.render_common import build_features_dict
from src.gpu_visualizers import get_visualizer
from src.studio.metrics import overlay_energy, vitality
from src.studio.probe import ProbeRenderer, probe_resolution
from src.studio.types import MeasureConstraints

DRIFT_VERSION = "studio-drift/1"
DRIFT_LOCK_THRESHOLD = 0.10


def measure_visualizer_drift(viz_name, features_dict, probe_size, target_size,
                             times, postprocess=None):
    """Misst |m_probe − m_commit| für M1 und M5 eines Visualizers."""
    constraints = MeasureConstraints(alpha_from_luma=True)
    energies, vitalities = [], []
    for size in (probe_size, target_size):
        renderer = ProbeRenderer(width=size[0], height=size[1])
        try:
            viz_cls = get_visualizer(viz_name)
            e_values, c_pairs = [], []
            for t in times:
                viz = viz_cls(renderer.ctx, size[0], size[1])
                a, b = renderer.render_pair(
                    viz, features_dict, t, None, postprocess or {}, constraints
                )
                contrib = renderer.contribution_map(a, b)
                e_values.append(overlay_energy(contrib))
                c_pairs.append(contrib)
            energies.append(float(np.mean(e_values)))
            deltas = [vitality(c_pairs[i], c_pairs[i + 1])
                      for i in range(len(c_pairs) - 1)]
            vitalities.append(float(np.mean(deltas)) if deltas else 0.0)
        finally:
            renderer.release()
    return {
        "M1": abs(energies[0] - energies[1]),
        "M5": abs(vitalities[0] - vitalities[1]),
    }


def write_drift_file(entries: dict, path: str) -> None:
    locked = [name for name, e in entries.items()
              if max(e.values()) > DRIFT_LOCK_THRESHOLD]
    payload = {
        "version": DRIFT_VERSION,
        "per_visualizer": entries,
        "resolution_dependent": locked,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualizers", required=True,
                        help="Kommagetrennte Visualizer-Namen")
    parser.add_argument("--target", default="1920x1080")
    parser.add_argument("--audio", required=True, help="Referenz-Audio")
    parser.add_argument("--out", default="config/studio_drift.v1.json")
    args = parser.parse_args()

    from src.analyzer import AudioAnalyzer
    features = AudioAnalyzer().analyze(args.audio, fps=30)
    features_dict = build_features_dict(features, features.frame_count, 30)

    tw, th = (int(x) for x in args.target.split("x"))
    probe_size = probe_resolution(tw, th)
    entries = {}
    for name in args.visualizers.split(","):
        entries[name.strip()] = measure_visualizer_drift(
            name.strip(), features_dict, probe_size, (tw, th),
            times=[0.2, 0.5, 0.8],
        )
    write_drift_file(entries, args.out)
    print(f"Drift geschrieben nach {args.out}")


if __name__ == "__main__":
    main()
