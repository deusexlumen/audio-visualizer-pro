"""Studio-Engine (Spec §2, §9): Probe → Solve → Commit → Verify.

P3-Scope: Visualizer, Parameter und ConstraintSet sind explizite
Eingaben (ModeGate/Profile/PresetFactory kommen in P4).
"""

import numpy as np

from .constraints import ConstraintSet
from .metrics import (integrity_violations, overlay_coverage, overlay_energy,
                      subject_disturbance, to_measure_raster, vitality)
from .sampling import SamplePlan
from .solver import SolveResult, solve
from .thresholds import ThresholdSet
from .types import MeasureConstraints

# Solver-Hebel, die in Visualizer-Parameter (viz.set_params) greifen
_VIZ_PARAM_LEVERS = ("viz_scale", "glow", "speed", "beat_response",
                     "intensity", "chroma_modulation")


def evaluate_params(probe, viz, features_dict, timestamps, postprocess,
                    constraints: MeasureConstraints,
                    subject_mask: np.ndarray | None = None) -> dict:
    """Rendert A/B je Sample und aggregiert M1/M2/M3/M5/M6 (Spec §3.3).

    Kosten: 2 Renders je Sample (A, B) + 2 grain-freie A-Renders für M5
    (B ist grain-frei zeitinvariant, wird einmal gerendert).
    """
    energies, coverages, disturbances = [], [], []
    contribs_gf = []
    b_gf = None
    violations = 0
    delta = 1.0 / float(features_dict.get("fps", 30))  # ~40 ms-Raster

    for t in timestamps:
        a, b = probe.render_pair(viz, features_dict, t, None, postprocess,
                                 constraints, subject_mask=subject_mask)
        contrib = probe.contribution_map(a, b)
        energies.append(overlay_energy(contrib))
        coverages.append(overlay_coverage(contrib))
        if subject_mask is not None:
            from .mask_service import resize_mask
            mask_scaled = resize_mask(subject_mask, contrib.shape[1],
                                      contrib.shape[0])
            disturbances.append(subject_disturbance(contrib, mask_scaled))
        violations += len(integrity_violations(to_measure_raster(a)))

        # M5: grain-freies Paar (C15 Regel 3)
        gf = MeasureConstraints(alpha_cap=constraints.alpha_cap,
                                alpha_from_luma=constraints.alpha_from_luma,
                                grain_free=True)
        a_t = probe.render_frame(viz, features_dict, t, None, postprocess, gf,
                                 subject_mask=subject_mask)
        a_d = probe.render_frame(viz, features_dict, t + delta, None,
                                 postprocess, gf, subject_mask=subject_mask)
        if b_gf is None:
            b_gf = probe.render_frame(
                viz, features_dict, t, None, postprocess,
                MeasureConstraints(alpha_cap=0.0, grain_free=True),
                subject_mask=subject_mask)
        ras = probe.contribution_map  # Kurzform
        contribs_gf.append((ras(a_t, b_gf), ras(a_d, b_gf)))

    m5_values = [vitality(c0, c1) for c0, c1 in contribs_gf]
    return {
        "M1": float(np.mean(energies)),
        "M2": float(np.mean(coverages)),
        "M3": float(np.mean(disturbances)) if disturbances else None,
        "M4": None,  # Quote-Kontrast kommt mit P4/Podcast-Profil
        "M5": float(np.mean(m5_values)) if m5_values else 0.0,
        "M6_violations": violations,
    }


def solve_constraints(probe, viz_factory, features_dict, plan: SamplePlan,
                      postprocess: dict, constraints: ConstraintSet,
                      ts: ThresholdSet, mode: str,
                      subject_mask=None) -> tuple[dict, SolveResult, dict]:
    """Führt den Solver über der echten Probe aus (Spec §8.3)."""
    postprocess = dict(postprocess or {})

    def metrics_fn(params: dict) -> dict:
        mc = MeasureConstraints(
            alpha_cap=params.get("alpha_cap", constraints.max_overlay_alpha),
            alpha_from_luma=constraints.alpha_from_luma,
            subject_strength=params.get("subject_strength",
                                        constraints.subject_strength),
        )
        pp = dict(postprocess)
        if "bloom_intensity" in params:
            pp["bloom_intensity"] = params["bloom_intensity"]
        viz = viz_factory()
        viz_overrides = {k: v for k, v in params.items()
                         if k in _VIZ_PARAM_LEVERS}
        if viz_overrides:
            viz.set_params(viz_overrides)
        return evaluate_params(probe, viz, features_dict, plan.timestamps,
                               pp, mc, subject_mask=subject_mask)

    initial = {"alpha_cap": constraints.max_overlay_alpha,
               "subject_strength": constraints.subject_strength}
    result = solve(metrics_fn, initial, ts, mode=mode)
    # Probe-Metriken des gelösten Zustands (Drift-Vergleich in Verify, §9)
    final_metrics = metrics_fn(result.params)
    return result.params, result, final_metrics
