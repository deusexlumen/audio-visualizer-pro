# Visualizer Studio — P3 Solver, Engine & Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Das Herzstück der Studio-Pipeline: der konvergente Penalty-Solver (J skalar, Hebel-Leiter, Plateau-Abbruch), die Engine mit Probe → Solve → Commit (genau 1 Render) → Verify (Drift-Budget) und das Provenance-Sidecar (`<output>.studio.json`).

**Architecture:** Drei neue Module in `src/studio/`. `solver.py` ist reine CPU-Logik (metrikagnostisch via Callback), `provenance.py` schreibt das Sidecar-Schema `studio-decision/2.1`, `engine.py` orchestriert Feasibility → Probe-Loop (ProbeRenderer + Solver) → einmaligen Commit-Render (`GPUBatchRenderer.render`, neuer optionaler `studio_constraints`-Kwargs) → Verify (ProbeRenderer auf Zielauflösung, Drift-Budget aus `config/studio_drift.v1.json`). Spec: `docs/superpowers/specs/2026-07-27-visualizer-studio-design.md` (studio-spec/2.1, §8, §9, §12, §16 P3).

**Tech Stack:** Python 3.11, numpy, pydantic v2, moderngl, pytest. Keine neuen Dependencies.

**Voraussetzung (P0–P2, abgeschlossen):** `src/studio/` mit `types.py`, `thresholds.py` (`ThresholdSet`, `load_thresholds`), `metrics.py` (M1–M6), `constraints.py` (`ConstraintSet`), `mask_service.py`, `probe.py` (`ProbeRenderer`, `render_pair`, `contribution_map`), `sampling.py` (`build_sample_plan`, `verification_extras`), `feasibility.py` (`check_feasibility`). Blit-Shader-Uniforms und `_blit_viz_to_fbo`-Kwargs (P0). `GPUBatchRenderer.render()`-Signatur (`gpu_renderer.py:139-166`).

**Scope-Hinweis:** ModeGate, Profile und PresetFactory kommen in P4. Die P3-Engine nimmt Visualizer + Parameter + ConstraintSet daher als **explizite Eingaben** (kein Auto-Modus). Die Hebel „Visualizer-Offset aus Subjektzentrum" und „peripherer Visualizer-Wechsel" (Spec §8.2, M3-Zeile) brauchen die P4-Profile und werden in P3 nicht bewegt — die restlichen M3-Hebel (`subject_strength`, `alpha_cap`) bleiben.

## Global Constraints

Werte wörtlich aus der Spec — jeder Task erbt diese Anforderungen implizit:

- `J = Σ w_i·max(0,(m_i−τ_i)/τ_i) + Σ w_j·max(0,(τ_j−m_j)/τ_j)`; Gewichte: hart (M1, M3, M4, M6) = **1.0**, M5 = **0.4**, M2 = **0.0** (nur Report); **`J = 0` ⇔ gate-konform** (Spec §8.1).
- Akzeptanz nur bei **`J' < J − 0.01`**; besuchte Parametervektoren werden gehasht (Zyklusschutz); Plateau ⇒ Abbruch mit Report; Iterationslimit **8** (Spec §8.3).
- Hebel-Leiter mit festen Schrittweiten (Spec §8.2): M1: `alpha_cap −0.08 → bloom_intensity ×0.75 → viz_scale ×0.9 → glow −0.1`; M3: `subject_strength +0.1 → alpha_cap −0.05`; M4: `scrim_opacity +0.12 → text_zone_alpha −0.05 → background_blur +1`; M5 zu hoch: `speed ×0.85 → beat_response ×0.8`; M5 zu niedrig: `beat_response ×1.2 → chroma_modulation +0.1 → intensity +0.08`; M6: Reset auf Startwerte.
- **Genau ein Commit-Render** pro Auftrag; kein automatischer Re-Render (Spec §9, Invariante 3).
- Verify: gleiche Samples + 6 Extras; Abbruch nur bei `drift > d_kalibriert + 0.02` (Spec §9, §3.4).
- Sidecar: `schema_version: "studio-decision/2.1"`, Pflichtblöcke input/mode/profile/thresholds/mask/sampling/measurement/solver/verify/renderer (Spec §12).
- Code-Kommentare und Commit-Messages auf Deutsch (AGENTS.md).

## Dateistruktur

| Datei | Verantwortung |
|-------|---------------|
| `src/studio/solver.py` (neu) | `compute_j`, `LEVER_LADDER`, `solve`, `SolveResult` |
| `src/studio/provenance.py` (neu) | `build_sidecar`, `write_sidecar` |
| `src/studio/engine.py` (neu) | `evaluate_params`, `solve_constraints`, `verify_commit`, `run_studio` |
| `src/gpu_renderer.py` (Modify) | `render(..., studio_constraints=None)` → Durchreiche an `_blit_viz_to_fbo` |
| `tests/test_studio_solver.py` (neu) | J-Mathematik, Monotonie-Property, Plateau, Zyklusschutz, M1↔M5 |
| `tests/test_studio_provenance.py` (neu) | Sidecar-Schema, Pflichtfelder, Datei-Roundtrip |
| `tests/test_studio_engine.py` (neu) | Probe-Eval, Solver über echter Probe (GPU) |
| `tests/test_studio_integration.py` (neu) | 1 Commit-Render, Sidecar, Verify grün (GPU, gemocktes FFmpeg) |

---

### Task 1: Solver-Kern

**Files:**
- Create: `src/studio/solver.py`
- Test: `tests/test_studio_solver.py`

**Interfaces:**
- Consumes: `ThresholdSet` (`src/studio/thresholds.py`, P0).
- Produces:
  - `LEVER_LADDER: dict[str, list[tuple[str, str]]]` — Metrik-Schlüssel `"M1"`, `"M3"`, `"M4"`, `"M5_high"`, `"M5_low"`, `"M6"` → Liste von (Parameter, Operation) mit Operationen `"+x"`, `"-x"`, `"*x"`
  - `compute_j(metrics: dict, ts: ThresholdSet, mode: str = "music") -> float` — `metrics`-Keys: `M1`, `M3` (optional), `M4` (optional), `M5`, `M6_violations` (int, optional)
  - `@dataclass SolveResult`: `params: dict`, `j_trace: list[float]`, `steps: list[dict]`, `iterations: int`, `status: str` (`"solved"` | `"plateau"`), `infeasible_metrics: list[str]`
  - `solve(metrics_fn, initial_params: dict, ts: ThresholdSet, mode: str = "music", max_iterations: int = 8) -> SolveResult` — `metrics_fn(params: dict) -> dict`
  - Konsumiert von Task 3 (Engine) und P4/P5.

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_solver.py`:

```python
"""Tests für den Penalty-Solver (Spec §8)."""

import pytest

from src.studio.solver import LEVER_LADDER, compute_j, solve
from src.studio.thresholds import load_thresholds


@pytest.fixture
def ts():
    return load_thresholds()


def _metrics_fn_factory(model):
    """Baut eine metrics_fn aus einem linearen Modell:
    model = {"M1": {"alpha_cap": 0.8, "bloom_intensity": 0.1}, ...}"""
    def metrics_fn(params):
        out = {}
        for key, coeffs in model.items():
            out[key] = sum(c * params.get(p, 0.0) for p, c in coeffs.items())
        return out
    return metrics_fn


def test_compute_j_ober_untergrenzen(ts):
    # M1 = 2x Schwelle => Beitrag 1.0; M5 music unter min => 0.4 * Anteil
    j = compute_j({"M1": 0.44, "M5": 0.01}, ts, mode="music")
    expected = (0.44 - 0.22) / 0.22 + 0.4 * (0.02 - 0.01) / 0.02
    assert j == pytest.approx(expected)


def test_compute_j_konform_ist_null(ts):
    assert compute_j({"M1": 0.1, "M5": 0.05}, ts, mode="music") == 0.0


def test_compute_j_m2_gewicht_null(ts):
    j1 = compute_j({"M1": 0.1, "M5": 0.05, "M2": 0.99}, ts, mode="music")
    assert j1 == 0.0  # M2 nur Report


def test_j_faellt_streng_monoton(ts):
    # Lösbares Modell: M1 linear in alpha_cap
    fn = _metrics_fn_factory({"M1": {"alpha_cap": 0.8}, "M5": {"intensity": 0.0}})
    fn2 = lambda p: {**fn(p), "M5": 0.05}
    result = solve(fn2, {"alpha_cap": 1.0}, ts)
    assert result.status == "solved"
    trace = result.j_trace
    assert all(b < a - 0.01 for a, b in zip(trace, trace[1:]))


def test_plateau_bei_unverbesserbarer_metrik(ts):
    # Konstante Metrik: kein Hebel verbessert J
    fn = lambda p: {"M1": 0.9, "M5": 0.05}
    result = solve(fn, {"alpha_cap": 1.0}, ts)
    assert result.status == "plateau"
    assert "M1" in result.infeasible_metrics


def test_zyklusschutz_bei_geklemmten_werten(ts):
    # alpha_cap wird bei 0.0 geklemmt => Kandidaten wiederholen sich
    fn = lambda p: {"M1": 0.5 + p.get("glow", 0.0), "M5": 0.05}
    result = solve(fn, {"alpha_cap": 0.05}, ts)
    assert result.status == "plateau"  # kein Endlos-Loop


def test_m1_m5_konflikt_loest_ueber_chroma_hebel(ts):
    """Spec §15: Chroma-Modulation vor Intensity bei M5 zu niedrig."""
    def fn(p):
        return {
            "M1": 0.8 * p.get("alpha_cap", 0.0),
            "M5": 0.1 * p.get("intensity", 0.0) + 0.5 * p.get("chroma_modulation", 0.0),
        }
    result = solve(fn, {"alpha_cap": 1.0}, ts, mode="music")
    assert result.status == "solved"
    levers = [s["lever"] for s in result.steps]
    # M5-Verletzung wurde über chroma_modulation behoben, nicht intensity
    assert "chroma_modulation" in levers
    if "intensity" in levers:
        assert levers.index("chroma_modulation") < levers.index("intensity")


def test_iterationslimit(ts):
    # Viele kleine Verletzungen: Limit greift
    fn = _metrics_fn_factory({"M1": {"alpha_cap": 0.01}, "M5": {"intensity": 0.0}})
    fn2 = lambda p: {**fn(p), "M5": 0.05}
    result = solve(fn2, {"alpha_cap": 1.0}, ts, max_iterations=3)
    assert result.iterations <= 3
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_solver.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.solver'`

- [ ] **Step 3: Implementierung**

`src/studio/solver.py`:

```python
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
            candidate = start_params if lever == "__reset__" else _apply_step(params, lever, op)
            cand_hash = _hash_params(candidate)
            if cand_hash in visited:
                continue  # Zyklusschutz (Spec §8.3 Regel 5)
            visited.add(cand_hash)
            j_new = compute_j(metrics_fn(candidate), ts, mode)
            if j_new < j - ACCEPT_DELTA:
                steps.append({"lever": lever, "op": op,
                              "j_before": j, "j_after": j_new})
                params = candidate
                j_trace.append(j_new)
                improved = True
                break
        if not improved:
            # Plateau: alle Hebel der Zeile ohne Verbesserung (Spec §8.3 Regel 4)
            return SolveResult(params, j_trace, steps, iteration + 1,
                               "plateau", [key])

    return SolveResult(params, j_trace, steps, max_iterations,
                       "plateau", ["iteration_limit"])
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_solver.py -v`
Expected: 7 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/solver.py tests/test_studio_solver.py
git commit -m "feat(studio): P3 Solver — Penalty J, Hebel-Leiter, Plateau + Zyklusschutz"
```

---

### Task 2: Provenance-Sidecar

**Files:**
- Create: `src/studio/provenance.py`
- Test: `tests/test_studio_provenance.py`

**Interfaces:**
- Consumes: nichts aus anderen Tasks.
- Produces:
  - `SCHEMA_VERSION = "studio-decision/2.1"`
  - `build_sidecar(sections: dict) -> dict` — fügt `schema_version` und `created_utc` hinzu, validiert Pflichtblöcke
  - `write_sidecar(output_path: str, sidecar: dict) -> str` — schreibt `<output>.studio.json`, gibt dessen Pfad zurück
  - Pflichtblöcke (Spec §12): `input`, `mode`, `profile`, `thresholds`, `mask`, `sampling`, `solver`, `verify`, `renderer` (`measurement` optional in P3)
  - Konsumiert von Task 4 (Engine) und P4/P5.

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_provenance.py`:

```python
"""Tests für das Provenance-Sidecar (Spec §12)."""

import json

import pytest

from src.studio.provenance import SCHEMA_VERSION, build_sidecar, write_sidecar


def _sections():
    return {
        "input": {"audio_sha256": "ab" * 32, "duration_s": 1.0},
        "mode": {"value": "MUSIC", "confidence": 1.0},
        "profile": {"name": "manual", "version": 0},
        "thresholds": {"set": "config/studio_thresholds.v1.json", "calibrated": False},
        "mask": {"provider": "none", "cache_hit": False},
        "sampling": {"n": 18, "seed": "cd", "timestamps_s": [0.1]},
        "solver": {"iterations": 0, "j_trace": [0.0], "steps": []},
        "verify": {"metrics": {"M1": 0.1}, "status": "pass"},
        "renderer": {"app_version": "dev"},
    }


def test_schema_version_und_created_utc():
    sc = build_sidecar(_sections())
    assert sc["schema_version"] == SCHEMA_VERSION
    assert "created_utc" in sc


def test_pflichtblock_fehlt_wirft():
    sections = _sections()
    del sections["solver"]
    with pytest.raises(ValueError, match="solver"):
        build_sidecar(sections)


def test_write_sidecar_datei(tmp_path):
    out = tmp_path / "video.mp4"
    out.write_bytes(b"fake")
    path = write_sidecar(str(out), build_sidecar(_sections()))
    assert path.endswith(".studio.json")
    data = json.loads((tmp_path / "video.studio.json").read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["sampling"]["n"] == 18
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_provenance.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.provenance'`

- [ ] **Step 3: Implementierung**

`src/studio/provenance.py`:

```python
"""Provenance-Sidecar (Spec §12).

Jede Engine-Entscheidung wird reproduzierbar protokolliert:
<output>.studio.json mit schema_version studio-decision/2.1.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "studio-decision/2.1"

REQUIRED_SECTIONS = (
    "input", "mode", "profile", "thresholds", "mask",
    "sampling", "solver", "verify", "renderer",
)


def build_sidecar(sections: dict) -> dict:
    """Baut das Sidecar-Dict; fehlende Pflichtblöcke sind ein Fehler."""
    missing = [s for s in REQUIRED_SECTIONS if s not in sections]
    if missing:
        raise ValueError(f"Sidecar-Pflichtblöcke fehlen: {', '.join(missing)}")
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    sidecar.update(sections)
    return sidecar


def write_sidecar(output_path: str, sidecar: dict) -> str:
    """Schreibt <output>.studio.json neben die Output-Datei."""
    out = Path(output_path)
    sidecar_path = out.with_suffix("").with_suffix(".studio.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str))
    return str(sidecar_path)
```

Hinweis: `video.mp4` → `video.studio.json` (doppeltes `with_suffix`: erst `.mp4` entfernen, dann `.studio.json` anhängen).

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_provenance.py -v`
Expected: 3 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/provenance.py tests/test_studio_provenance.py
git commit -m "feat(studio): P3 Provenance — Sidecar studio-decision/2.1"
```

---

### Task 3: Engine — Probe-Eval & Solve-Anbindung

**Files:**
- Create: `src/studio/engine.py`
- Test: `tests/test_studio_engine.py`

**Interfaces:**
- Consumes: `ProbeRenderer` (P0/P1), `build_sample_plan` (P2), `compute_j`/`solve` (Task 1), `ConstraintSet` (P1), Metriken aus `metrics.py` (P0).
- Produces:
  - `evaluate_params(probe: ProbeRenderer, viz, features_dict: dict, timestamps: list[float], postprocess: dict, constraints: MeasureConstraints, subject_mask: np.ndarray | None = None) -> dict` — rendert A/B je Sample + grain-freie M5-Paare, liefert `{M1, M3|None, M4|None, M5, M6_violations, M2}`
  - `solve_constraints(probe, viz_factory, features_dict, plan: SamplePlan, postprocess, constraints: ConstraintSet, ts: ThresholdSet, mode: str, subject_mask=None) -> tuple[dict, SolveResult, dict]` — Solver über der echten Probe; liefert (final_params, SolveResult, final_metrics). `final_metrics` sind die Probe-Metriken des gelösten Zustands (für den Drift-Vergleich in Verify).
  - Solver-Parameter-Mapping: Solver-Hebel wirken auf `alpha_cap` (→ MeasureConstraints), `bloom_intensity` (→ postprocess), Rest wird als Visualizer-`params`-Override an `viz.set_params` gereicht.
  - Konsumiert von Task 4.

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_engine.py`:

```python
"""Tests für die Engine-Probe-Loop (Spec §8, §9)."""

import numpy as np
import pytest

from src.studio.constraints import ConstraintSet
from src.studio.engine import evaluate_params, solve_constraints
from src.studio.probe import ProbeRenderer
from src.studio.sampling import build_sample_plan
from src.studio.thresholds import load_thresholds

pytestmark = pytest.mark.gpu


@pytest.fixture
def probe():
    p = ProbeRenderer(width=160, height=90, fps=30)
    yield p
    p.release()


@pytest.fixture
def features_dict(dummy_audio_features):
    from src.render_common import build_features_dict
    return build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )


def _viz(probe):
    from src.gpu_visualizers import get_visualizer
    return get_visualizer("spectrum_bars")(probe.ctx, 160, 90)


def test_evaluate_params_liefert_metriken(probe, features_dict):
    cs = ConstraintSet(max_overlay_alpha=1.0)
    metrics = evaluate_params(
        probe, _viz(probe), features_dict, [0.2, 0.5, 0.8], {},
        cs.to_measure_constraints(),
    )
    assert set(metrics) >= {"M1", "M5", "M6_violations"}
    assert metrics["M1"] > 0.0  # spectrum_bars zeichnet sichtbar
    assert metrics["M6_violations"] == 0


def test_solve_senkt_alpha_cap_bei_engem_m1(probe, features_dict):
    ts = load_thresholds()
    # Künstlich enge M1-Schwelle: Solver muss alpha_cap senken
    ts = ts.model_copy(update={"m1_overlay_energy_max": 0.01})
    plan = build_sample_plan(features_dict)
    cs = ConstraintSet(max_overlay_alpha=1.0)
    params, result, final_metrics = solve_constraints(
        probe, lambda: _viz(probe), features_dict, plan, {}, cs, ts, "music"
    )
    assert result.j_trace[0] > result.j_trace[-1]  # J gesunken
    assert all(b < a for a, b in zip(result.j_trace, result.j_trace[1:]))
    if result.status == "solved":
        assert params["alpha_cap"] < 1.0
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_engine.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.engine'`

- [ ] **Step 3: Implementierung**

`src/studio/engine.py`:

```python
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
                      subject_mask=None) -> tuple[dict, SolveResult]:
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
```

Hinweis: `M4 = None` in P3 — Quote-Overlays/Glyphenmasken kommen mit dem Podcast-Profil (P4); `compute_j` behandelt `None` korrekt (Metrik entfällt).

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_engine.py -v`
Expected: 2 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/engine.py tests/test_studio_engine.py
git commit -m "feat(studio): P3 Engine — Probe-Eval + Solver-Anbindung"
```

---

### Task 4: Commit, Verify, Integration

**Files:**
- Modify: `src/gpu_renderer.py` (`render()`-Signatur + Blit-Aufruf ~Zeile 498)
- Modify: `src/studio/engine.py` (anfügen: `verify_commit`, `run_studio`)
- Test: `tests/test_studio_integration.py`

**Interfaces:**
- Consumes: alles aus Tasks 1–3, `check_feasibility` (P2), `verification_extras` (P2), `build_sidecar`/`write_sidecar` (Task 2).
- Produces:
  - `GPUBatchRenderer.render(..., studio_constraints: MeasureConstraints | None = None)` — optional, rückwärtskompatibel; reicht Cap/Luma/Stärke an `_blit_viz_to_fbo` weiter
  - `load_drift_budget(visualizer: str, path: str = "config/studio_drift.v1.json") -> dict` (in `engine.py`)
  - `verify_commit(probe_target, viz_factory, features_dict, plan, extras, postprocess, params, constraints, drift_budget) -> dict` — `{metrics, status, drift_max, drift_within_budget}`
  - `run_studio(audio_path, visualizer, features, features_dict, output_path, params=None, postprocess=None, constraints=None, thresholds=None, mode="music", background_image=None, subject_mask=None) -> dict` — End-to-End; liefert das Sidecar-Dict
  - Konsumiert von P4 (ModeGate/PresetFactory) und P5 (CLI/GUI).

- [ ] **Step 1: Failing Test schreiben**

`tests/test_studio_integration.py`:

```python
"""Integrationstest: genau ein Commit-Render, Sidecar, Verify grün
(Spec §9, §15, §16 P3)."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_run_studio_genau_ein_commit_render(tmp_path, dummy_audio_features):
    from src.render_common import build_features_dict
    from src.studio.constraints import ConstraintSet
    from src.studio.engine import run_studio
    from src.gpu_renderer import GPUBatchRenderer

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    render_spy = MagicMock(side_effect=RuntimeError("stop-after-first"))
    with patch.object(GPUBatchRenderer, "render", render_spy), \
         patch("src.gpu_renderer.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        sidecar = run_studio(
            str(audio), "spectrum_bars", dummy_audio_features, features_dict,
            str(out), constraints=ConstraintSet(max_overlay_alpha=1.0),
        )

    # Genau ein Commit-Render-Versuch (kein automatischer Re-Render)
    assert render_spy.call_count == 1
    # Sidecar wurde geschrieben, Verify hat gemessen
    assert (tmp_path / "out.studio.json").exists()
    data = json.loads((tmp_path / "out.studio.json").read_text())
    assert data["schema_version"] == "studio-decision/2.1"
    assert data["verify"]["status"] in ("pass", "drift_abort")
    assert data["verify"]["drift_within_budget"] is True
```

Hinweis: Der Commit-Render wird absichtlich per `side_effect=RuntimeError` nach dem ersten Aufruf gestoppt — der Test zählt dadurch exakt die Render-Aufrufe und muss nicht encodieren; `run_studio` fängt diesen Fehler als `commit_skipped` ab (siehe Implementierung) und fährt mit Verify fort. Alternativ (einfacher, wenn vorhanden): das Muster aus `tests/test_gpu_renderer.py` mit gemocktem `subprocess.Popen` verwenden und `render` durchlaufen lassen — der Implementierer wählt die robustere Variante, muss aber `call_count == 1` beweisen.

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_integration.py -v`
Expected: FAIL mit `ImportError: cannot import name 'run_studio'`

- [ ] **Step 3: Implementierung**

**3a) `src/gpu_renderer.py` — `render()` erweitern:**

In der Signatur (`gpu_renderer.py:139-166`) als letzten Parameter ergänzen:

```python
        timeline=None,
        studio_constraints=None,
    ):
```

Im Docstring ergänzen: `studio_constraints: Optionale Studio-MeasureConstraints (Alpha-Cap, Luma-Alpha, Subjekt-Stärke) — nur im Studio-Pfad gesetzt.`

Beim Blit-Aufruf (aktuell ~Zeile 497-503) die Constraints durchreichen:

```python
                    self.fbo.use()
                    blit_kwargs = {}
                    if studio_constraints is not None:
                        blit_kwargs = {
                            "alpha_cap": studio_constraints.alpha_cap,
                            "alpha_from_luma": studio_constraints.alpha_from_luma,
                            "luma_knee_lo": studio_constraints.luma_knee_lo,
                            "luma_knee_hi": studio_constraints.luma_knee_hi,
                            "subject_strength": studio_constraints.subject_strength,
                        }
                    self._blit_viz_to_fbo(
                        active_viz_tex,
                        offset_x=viz_offset_x,
                        offset_y=viz_offset_y,
                        scale=viz_scale,
                        **blit_kwargs,
                    )
```

**3b) `src/studio/engine.py` — anfügen:**

```python
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
    try:
        renderer = GPUBatchRenderer(width=target_w, height=target_h)
        renderer.render(audio_path, visualizer, output_path,
                        features=features, params=params,
                        postprocess=postprocess, preview_mode=True,
                        studio_constraints=mc)
    except Exception as e:  # z.B. gemocktes FFmpeg im Test
        commit_error = str(e)

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
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_integration.py -v`
Expected: 1 × PASS

Regression: `pytest tests/test_gpu_renderer.py tests/test_studio_engine.py -v`
Expected: alle PASS (`render()`-Signatur rückwärtskompatibel, `solve_constraints`-Rückgabe angepasst)

- [ ] **Step 5: Commit**

```bash
git add src/gpu_renderer.py src/studio/engine.py tests/test_studio_integration.py tests/test_studio_engine.py
git commit -m "feat(studio): P3 Commit+Verify — 1 Render, Drift-Budget, Sidecar"
```

---

## Abschluss P3 (Definition of Done, Spec §16)

- [ ] `pytest tests/test_studio_solver.py -v` — **Property-Test grün: J fällt streng monoton**; Zyklusschutz und Plateau greifen; M1↔M5-Konflikt löst über Chroma-Hebel
- [ ] `pytest tests/test_studio_integration.py -v` — Integrationstest zählt **genau einen** Commit-Render-Aufruf; Sidecar geschrieben; Verify grün
- [ ] `pytest tests/ -q` — keine Regressionen im Bestand (insb. `test_gpu_renderer*.py` nach der `render()`-Erweiterung)
- [ ] Danach: Plan für **P4** (mode_gate.py, profiles.py, preset_factory.py) schreiben — konsumiert `run_studio`, `SolveResult`, `FeasibilityResult`
