"""Penalty-Solver (Spec §8).

Skalares Zielmaß J, deterministische Hebel-Leiter, Akzeptanz nur bei
strenger Verbesserung (J' < J - 0.01), Zyklusschutz über gehashte
Parametervektoren, Plateau-Abbruch statt Endlosschleife.
Garantie: J fällt über akzeptierte Schritte streng monoton.
"""

import hashlib
import json
from dataclasses import dataclass, field

from .thresholds import ThresholdSet

# Hebel-Leiter (Spec §8.2), feste Schrittweiten.
# Hinweis P3: M3-Hebel „Offset aus Subjektzentrum" und „peripherer
# Visualizer" brauchen P4-Profile und sind hier nicht enthalten.
LEVER_LADDER: dict[str, list[tuple[str, str]]] = {
    "M1": [("alpha_cap", "-0.08"), ("bloom_intensity", "*0.75"),
           ("viz_scale", "*0.9"), ("glow", "-0.1")],
    "M3": [("subject_strength", "+0.1"), ("alpha_cap", "-0.05")],
    "M4": [("scrim_opacity", "+0.12"), ("text_zone_alpha", "-0.05"),
           ("background_blur", "+1.0")],
    "M5_high": [("speed", "*0.85"), ("beat_response", "*0.8")],
    "M5_low": [("beat_response", "*1.2"), ("chroma_modulation", "+0.1"),
               ("intensity", "+0.08")],
    "M6": [("__reset__", "")],
}

ACCEPT_DELTA = 0.01


@dataclass
class SolveResult:
    """Ergebnis des Solve-Laufs (landet im Sidecar)."""

    params: dict
    j_trace: list[float]
    steps: list[dict] = field(default_factory=list)
    iterations: int = 0
    status: str = "plateau"  # "solved" | "plateau"
    infeasible_metrics: list[str] = field(default_factory=list)


def compute_j(metrics: dict, ts: ThresholdSet, mode: str = "music") -> float:
    """Skalares Zielmaß (Spec §8.1). J = 0 <=> gate-konform."""
    j = 0.0
    m1 = metrics.get("M1", 0.0)
    j += max(0.0, (m1 - ts.m1_overlay_energy_max) / ts.m1_overlay_energy_max)
    if metrics.get("M3") is not None:
        j += max(0.0, (metrics["M3"] - ts.m3_subject_max) / ts.m3_subject_max)
    if metrics.get("M4") is not None:
        j += max(0.0, (ts.m4_contrast_min - metrics["M4"]) / ts.m4_contrast_min)
    m5 = metrics.get("M5", 0.0)
    if mode == "music":
        j += 0.4 * max(0.0, (ts.m5_music_min - m5) / ts.m5_music_min)
    else:
        j += 0.4 * max(0.0, (m5 - ts.m5_podcast_max) / ts.m5_podcast_max)
    j += float(metrics.get("M6_violations", 0))
    # M2: Gewicht 0.0 — nur Report (Spec §8.1)
    return j


def _contributions(metrics: dict, ts: ThresholdSet, mode: str) -> dict[str, float]:
    """Einzelbeiträge je Metrik (für die Wahl des aktiven Hebels)."""
    c: dict[str, float] = {}
    c["M1"] = max(0.0, (metrics.get("M1", 0.0) - ts.m1_overlay_energy_max)
                  / ts.m1_overlay_energy_max)
    if metrics.get("M3") is not None:
        c["M3"] = max(0.0, (metrics["M3"] - ts.m3_subject_max) / ts.m3_subject_max)
    if metrics.get("M4") is not None:
        c["M4"] = max(0.0, (ts.m4_contrast_min - metrics["M4"]) / ts.m4_contrast_min)
    m5 = metrics.get("M5", 0.0)
    if mode == "music":
        c["M5_low"] = 0.4 * max(0.0, (ts.m5_music_min - m5) / ts.m5_music_min)
    else:
        c["M5_high"] = 0.4 * max(0.0, (m5 - ts.m5_podcast_max) / ts.m5_podcast_max)
    if metrics.get("M6_violations", 0) > 0:
        c["M6"] = float(metrics["M6_violations"])
    return c


def _hash_params(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _apply_step(params: dict, lever: str, op: str) -> dict:
    """Wendet einen Leiter-Schritt an; Werte bleiben >= 0 (Klemmung)."""
    out = dict(params)
    old = float(out.get(lever, 0.0))
    if op.startswith("*"):
        new = old * float(op[1:])
    else:
        new = old + float(op)
    out[lever] = max(0.0, new)
    return out


def solve(metrics_fn, initial_params: dict, ts: ThresholdSet,
          mode: str = "music", max_iterations: int = 8) -> SolveResult:
    """Probe-Solve-Loop (Spec §8.3).

    metrics_fn(params) -> dict mit M1/M3(optional)/M4(optional)/M5/
    M6_violations(optional). Deterministisch bei deterministischem metrics_fn.
    """
    params = dict(initial_params)
    start_params = dict(initial_params)
    visited = {_hash_params(params)}
    j_trace = [compute_j(metrics_fn(params), ts, mode)]
    steps: list[dict] = []

    for iteration in range(max_iterations):
        j = j_trace[-1]
        if j <= 0.0:
            return SolveResult(params, j_trace, steps, iteration, "solved", [])
        metrics = metrics_fn(params)
        contribs = _contributions(metrics, ts, mode)
        if not any(v > 0.0 for v in contribs.values()):
            return SolveResult(params, j_trace, steps, iteration, "solved", [])
        key = max(contribs, key=lambda k: contribs[k])

        improved = False
        for lever, op in LEVER_LADDER.get(key, []):
            # Abweichung vom ursprünglichen Plan-Entwurf: Ein Hebel wird
            # innerhalb einer Iteration wiederholt angewendet, solange er
            # streng verbessert (Plan-Entwurf: 1 Schritt/Iteration — damit
            # wären die Plan-Tests bei max_iterations=8 unlösbar, z.B.
            # braucht alpha_cap 1.0 -> <=0.275 zehn -0.08-Schritte).
            # max_iterations begrenzt weiterhin die Leiter-Durchläufe;
            # der Zyklusschutz terminiert die innere Schleife garantiert.
            while True:
                candidate = start_params if lever == "__reset__" else _apply_step(params, lever, op)
                cand_hash = _hash_params(candidate)
                if cand_hash in visited:
                    break  # Zyklusschutz (Spec §8.3 Regel 5)
                visited.add(cand_hash)
                j_new = compute_j(metrics_fn(candidate), ts, mode)
                if j_new < j - ACCEPT_DELTA:
                    steps.append({"lever": lever, "op": op,
                                  "j_before": j, "j_after": j_new})
                    params = candidate
                    j_trace.append(j_new)
                    j = j_new
                    improved = True
                else:
                    break
        if not improved:
            # Plateau: alle Hebel der Zeile ohne Verbesserung (Spec §8.3 Regel 4)
            return SolveResult(params, j_trace, steps, iteration + 1,
                               "plateau", [key])

    return SolveResult(params, j_trace, steps, max_iterations,
                       "plateau", ["iteration_limit"])
