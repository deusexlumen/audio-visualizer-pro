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


# --- Commit, Verify, Orchestrierung (Spec §9) ---

import hashlib
import json
from pathlib import Path

from .feasibility import check_feasibility
from .provenance import build_sidecar, write_sidecar
from .sampling import build_sample_plan, verification_extras
from .thresholds import load_thresholds


def load_drift_budget(visualizer: str,
                      path: str = "config/studio_drift.v1.json") -> dict:
    """Kalibriertes Drift-Budget je Metrik; Default 0.02 (Spec §3.4)."""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return data.get("per_visualizer", {}).get(visualizer, {})


def verify_commit(probe_target, viz_factory, features_dict, timestamps,
                  postprocess, params, constraints: ConstraintSet,
                  drift_budget: dict, subject_mask=None) -> dict:
    """Verify auf Zielauflösung (Spec §9): gleiche Samples + Extras."""
    mc = MeasureConstraints(
        alpha_cap=params.get("alpha_cap", constraints.max_overlay_alpha),
        alpha_from_luma=constraints.alpha_from_luma,
        subject_strength=params.get("subject_strength",
                                    constraints.subject_strength),
    )
    viz = viz_factory()
    metrics = evaluate_params(probe_target, viz, features_dict, timestamps,
                              postprocess, mc, subject_mask=subject_mask)
    return metrics


def run_studio(audio_path, visualizer, features, features_dict, output_path,
               params=None, postprocess=None, constraints=None,
               thresholds=None, mode="music", background_image=None,
               subject_mask=None) -> dict:
    """End-to-End: Feasibility → Solve → 1× Commit → Verify → Sidecar."""
    from ..gpu_renderer import GPUBatchRenderer
    from ..gpu_visualizers import get_visualizer
    from .probe import ProbeRenderer, probe_resolution

    constraints = constraints or ConstraintSet()
    ts = thresholds or load_thresholds()
    postprocess = dict(postprocess or {})

    # 1) Feasibility (vor jedem Render, Spec §7)
    feas = check_feasibility(subject_mask,
                             requires_text_zone=(mode == "podcast"))
    if not feas.should_render:
        raise RuntimeError(f"Feasibility: {feas.reason}")

    # 2) Probe-Solve
    plan = build_sample_plan(features_dict)
    target_w, target_h = 854, 480  # P3: feste Zielauflösung (Preview-Pfad)
    pw, ph = probe_resolution(target_w, target_h)
    probe = ProbeRenderer(width=pw, height=ph)
    try:
        viz_factory = lambda: get_visualizer(visualizer)(probe.ctx, pw, ph)
        solved_params, solve_result, probe_metrics = solve_constraints(
            probe, viz_factory, features_dict, plan, postprocess,
            constraints, ts, mode, subject_mask=subject_mask)
    finally:
        probe.release()

    # 3) Commit: GENAU ein Render (Spec §9, Invariante 3)
    mc = MeasureConstraints(
        alpha_cap=solved_params.get("alpha_cap",
                                    constraints.max_overlay_alpha),
        alpha_from_luma=constraints.alpha_from_luma,
        subject_strength=solved_params.get("subject_strength",
                                           constraints.subject_strength),
    )
    commit_error = None
    renderer = None
    try:
        renderer = GPUBatchRenderer(width=target_w, height=target_h)
        renderer.render(audio_path, visualizer, output_path,
                        features=features, params=params,
                        postprocess=postprocess, preview_mode=True,
                        studio_constraints=mc)
    except Exception as e:  # z.B. gemocktes FFmpeg im Test
        commit_error = str(e)
    finally:
        # GL-Kontext freigeben — sonst bricht unter WGL die Currency
        # nachfolgender Tests/Renders (vgl. tests/conftest.py).
        if renderer is not None:
            try:
                renderer.release()
            except Exception:
                pass

    # 4) Verify auf Zielauflösung: gleiche Samples + 6 Extras (Spec §9)
    extras = verification_extras(plan, float(features_dict["duration"]))
    probe_t = ProbeRenderer(width=target_w, height=target_h)
    try:
        verify_metrics = verify_commit(
            probe_t,
            lambda: get_visualizer(visualizer)(probe_t.ctx, target_w, target_h),
            features_dict, plan.timestamps + extras, postprocess,
            solved_params, constraints,
            load_drift_budget(visualizer), subject_mask=subject_mask)
    finally:
        probe_t.release()

    drift_budget = load_drift_budget(visualizer)
    drift_max = 0.0
    drift_ok = True
    for key in ("M1", "M3", "M5"):
        p, c = probe_metrics.get(key), verify_metrics.get(key)
        if p is None or c is None:
            continue
        d = abs(c - p)
        drift_max = max(drift_max, d)
        if d > drift_budget.get(key, 0.02) + 0.02:
            drift_ok = False
    verify_status = "pass" if drift_ok else "drift_abort"

    # 5) Sidecar (Spec §12)
    sidecar = build_sidecar({
        "input": {"audio_sha256": hashlib.sha256(
            Path(audio_path).read_bytes()).hexdigest(),
                  "duration_s": float(features_dict["duration"])},
        "mode": {"value": mode.upper(), "confidence": 1.0,
                 "note": "P3: manueller Modus, ModeGate kommt in P4"},
        "profile": {"name": "manual", "version": 0},
        "thresholds": {"set": "config/studio_thresholds.v1.json",
                       "sha256": ts.file_sha256, "calibrated": False},
        "mask": {"provider": "provided" if subject_mask is not None else "none",
                 "cache_hit": False},
        "sampling": {"n": plan.n, "seed": plan.seed,
                     "timestamps_s": plan.timestamps},
        "solver": {"iterations": solve_result.iterations,
                   "j_trace": solve_result.j_trace,
                   "steps": solve_result.steps,
                   "status": solve_result.status,
                   "final_constraints": mc.__dict__},
        "verify": {"metrics": verify_metrics, "status": verify_status,
                   "drift_max": drift_max, "drift_within_budget": drift_ok,
                   "commit_error": commit_error},
        "renderer": {"app_version": "dev"},
    })
    write_sidecar(output_path, sidecar)
    return sidecar


def run_studio_auto(audio_path, features, features_dict, output_path,
                    profile_name=None, params_override=None,
                    postprocess_override=None, background_image=None,
                    subject_mask=None) -> dict:
    """Auto-Flow (Spec §2): ModeGate → Profil → Preset → run_studio."""
    from .mode_gate import classify_mode
    from .preset_factory import build_preset
    from .profiles import load_profile

    mode_result = classify_mode(features_dict)
    profile = load_profile(profile_name or f"{mode_result.resolved}_default")
    preset = build_preset(features, profile)

    params = dict(preset.params)
    params.update(params_override or {})
    postprocess = dict(preset.postprocess)
    postprocess.update(postprocess_override or {})

    sidecar = run_studio(
        audio_path, preset.visualizer, features, features_dict, output_path,
        params=params, postprocess=postprocess,
        constraints=preset.constraints,
        mode=mode_result.resolved,
        background_image=background_image, subject_mask=subject_mask,
    )
    # Echte ModeGate-Werte statt P3-Platzhalter (Spec §12)
    sidecar["mode"] = {
        "value": mode_result.value,
        "confidence": mode_result.confidence,
        "speech_score": mode_result.speech_score,
        "hysteresis_applied": mode_result.hysteresis_applied,
    }
    sidecar["profile"] = {"name": profile.name, "version": profile.version}
    return sidecar
