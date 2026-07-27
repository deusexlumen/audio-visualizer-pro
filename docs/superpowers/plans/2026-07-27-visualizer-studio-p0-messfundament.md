# Visualizer Studio — P0 Messfundament Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Das verifizierte Messfundament des Visualizer Studio bauen: normalisierter Messraster, Metriken M1–M6, Luma-Alpha-Ableitung im Blit-Shader (C14), Differenz-Render mit Rausch-Aufhebung (C15), Auflösungs-Drift-Kalibrierung (C16) und Schwellen-Kalibrier-Harness — Grundlage für alle späteren Phasen P1–P5.

**Architecture:** Neues Paket `src/studio/` neben dem Bestand. Der `GPUBatchRenderer` (Basis von `GPUPreviewRenderer`) erhält genau einen Eingriff: neue Uniforms im **Blit-Shader** `_init_blit_shader` (`gpu_renderer.py:1285-1303`) — dem einzigen realen Mischpunkt von Visualizer über Hintergrund (`_blit_viz_to_fbo`, `gpu_renderer.py:1415`; der Composite-Shader `_init_composite_shader` ist ungenutzter Code). Alle Defaults bleiben bit-identisch zum Bestand. Messung läuft über einen neuen `ProbeRenderer`, der den `GPUPreviewRenderer` wiederverwendet und den Batch-Loop exakt spiegelt. Spec: `docs/superpowers/specs/2026-07-27-visualizer-studio-design.md` (studio-spec/2.1).

**Tech Stack:** Python 3.11, numpy, Pillow, moderngl, pydantic v2, pytest. Bestehende Test-Konventionen: `@pytest.mark.gpu` + `shared_gl_context`-Fixture (`tests/conftest.py`).

**Scope-Abgrenzung:** Dieser Plan deckt **nur Phase P0** der Spec ab (§16). P1 (Maske/Constraints) bis P5 (GUI) folgen als eigene Pläne, weil sie P0-Artefakte konsumieren (`studio_drift.v1.json`, kalibrierte Schwellen).

**Abweichung zur Spec (code-verifiziert 2026-07-27, zweiter Durchlauf):**
1. Der Mischpunkt ist der **Blit-Shader**, nicht `_init_composite_shader` — letzterer wird nirgends aufgerufen (verifiziert per Volltextsuche über `src/`). Spec §6.1/§18 sind entsprechend nachgezogen.
2. Die Luma-Ableitung muss im Studio-Modus **unbedingt** gelten (nicht nur bei `tex.a < 0.01`): Der Blit-Shader hat keinen Alpha-Fallback, und `CompositeVisualizer` gibt hart `alpha = 1.0` aus — genau dieser Hauptfall würde von der Spec-GLSL-Bedingung nicht erfasst. Spec §6.1 ist korrigiert.

## Global Constraints

Werte wörtlich aus der Spec — jeder Task erbt diese Anforderungen implizit:

- Messraster: lange Kante **854 px**, Area-Filter (Pillow `Resampling.BOX`), sRGB → **Linear-Light**, float32, Wertebereich [0, 1]; Invarianz-Toleranz **ε = 0.01** (Spec §3.1).
- Default-Schwellen: **M1 ≤ 0.22**, **M2 ≤ 0.60** (nur warn), **M3 ≤ 0.10**, **M4 ≥ 4.5**, **M5: MUSIC ≥ 0.02 / PODCAST ≤ 0.09** (Δ = 40 ms) (Spec §3.3).
- Luma-Knee Defaults **0.02 / 0.25**; Flag `u_viz_alpha_from_luma` nur im Studio-Pfad an — Direkt-Render bleibt bit-identisch (Spec §6.1).
- Shader-Reihenfolge bindend: Luma-Ableitung **vor** Cap-Anwendung; es muss gelten `alpha_cap = 0 ⇒ contrib ≡ 0` (Spec §3.2.2).
- M5 wird ausschließlich auf grain-freien Renderpaaren berechnet; A- und B-Render laufen mit identischem `u_time` (Spec §3.2.1 — Seeding via `fract(u_time…)` ist Bestand, `gpu_renderer.py:1132-1142`).
- `probe_res = max(480p, Zielauflösung / 4)`, identisches Seitenverhältnis (Spec §3.4).
- Drift-Regeln: `d ≤ 0.02` stabil; `d > 0.02` Abschlag `τ_effektiv = τ − d`; `d > 0.10` Visualizer-Sperrung (Spec §3.4).
- Code-Kommentare und Commit-Messages auf Deutsch (AGENTS.md).
- `src/analyzer.py::analyze()` wird nicht angefasst (AGENTS.md).
- Keine neuen harten Dependencies in P0; `rembg` ist P1-Thema.

## Dateistruktur

| Datei | Verantwortung |
|-------|---------------|
| `src/studio/__init__.py` | Paket-Marker, re-exportiert `load_thresholds`, `MeasureConstraints` |
| `src/studio/types.py` | `MeasureConstraints` (Dataclass): Mess-ConstraintSet für Probe/Diff-Render |
| `src/studio/thresholds.py` | `ThresholdSet` (Pydantic) + `load_thresholds()` + Datei-Hash |
| `src/studio/metrics.py` | Messraster, `contribution`, M1/M2/M3/M4/M5/M6 |
| `src/studio/probe.py` | `ProbeRenderer` (A/B-Differenz-Render), `probe_resolution()` |
| `config/studio_thresholds.v1.json` | Versionierte Schwellwerte mit Provenance |
| `config/studio_drift.v1.json` | Gemessene Auflösungs-Drift je Visualizer/Metrik |
| `tools/__init__.py` | Paket-Marker für Tool-Imports in Tests |
| `tools/measure_drift.py` | Drift-Messung über Visualizer-Whitelist |
| `tools/calibrate_thresholds.py` | Schwellen-Kalibrierung über Golden-Set |
| `tests/golden/` | Golden-Set (Labels + Referenzframes, Scaffolding) |
| `src/gpu_renderer.py` (Modify) | Blit-Shader-Uniforms (C14), `_blit_viz_to_fbo(...)`-Signatur |

---

### Task 1: Studio-Grundgerüst, MeasureConstraints, ThresholdSet

**Files:**
- Create: `src/studio/__init__.py`
- Create: `src/studio/types.py`
- Create: `src/studio/thresholds.py`
- Create: `config/studio_thresholds.v1.json`
- Test: `tests/test_studio_thresholds.py`

**Interfaces:**
- Consumes: nichts (erster Task).
- Produces:
  - `MeasureConstraints` (Dataclass, `src/studio/types.py`): Felder `alpha_cap: float = 1.0`, `alpha_from_luma: bool = False`, `luma_knee_lo: float = 0.02`, `luma_knee_hi: float = 0.25`, `subject_strength: float = 0.0`, `grain_free: bool = False`. Konsumiert von Task 5/6.
  - `load_thresholds(path: str | None = None) -> ThresholdSet` (`src/studio/thresholds.py`)
  - `ThresholdSet`-Felder: `version: str`, `m1_overlay_energy_max: float`, `m2_coverage_warn: float`, `m3_subject_max: float`, `m4_contrast_min: float`, `m5_music_min: float`, `m5_podcast_max: float`, `epsilon: float`, `luma_knee_lo: float`, `luma_knee_hi: float`, `provenance: dict[str, str]`, `file_sha256: str`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_thresholds.py`:

```python
"""Tests für das versionierte Schwellwert-Set des Visualizer Studio."""

from src.studio.thresholds import ThresholdSet, load_thresholds


def test_load_thresholds_defaults():
    ts = load_thresholds()
    assert ts.version == "studio-thresholds/1"
    assert ts.m1_overlay_energy_max == 0.22
    assert ts.m2_coverage_warn == 0.60
    assert ts.m3_subject_max == 0.10
    assert ts.m4_contrast_min == 4.5
    assert ts.m5_music_min == 0.02
    assert ts.m5_podcast_max == 0.09
    assert ts.epsilon == 0.01
    assert ts.luma_knee_lo < ts.luma_knee_hi


def test_thresholds_file_sha256():
    ts = load_thresholds()
    assert len(ts.file_sha256) == 64
    int(ts.file_sha256, 16)  # gültiges Hex


def test_thresholds_provenance_present():
    ts = load_thresholds()
    for key in ("m1_overlay_energy_max", "m4_contrast_min"):
        value = ts.provenance[key]
        assert value == "assumed" or value.startswith("calibrated@")


def test_measure_constraints_defaults():
    from src.studio.types import MeasureConstraints
    mc = MeasureConstraints()
    assert mc.alpha_cap == 1.0
    assert mc.alpha_from_luma is False
    assert mc.grain_free is False
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag verifizieren**

Run: `pytest tests/test_studio_thresholds.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio'`

- [ ] **Step 3: Minimale Implementierung**

`config/studio_thresholds.v1.json`:

```json
{
  "version": "studio-thresholds/1",
  "thresholds": {
    "m1_overlay_energy_max": 0.22,
    "m2_coverage_warn": 0.60,
    "m3_subject_max": 0.10,
    "m4_contrast_min": 4.5,
    "m5_music_min": 0.02,
    "m5_podcast_max": 0.09,
    "epsilon": 0.01,
    "luma_knee_lo": 0.02,
    "luma_knee_hi": 0.25
  },
  "provenance": {
    "m1_overlay_energy_max": "assumed",
    "m2_coverage_warn": "assumed",
    "m3_subject_max": "assumed",
    "m4_contrast_min": "assumed",
    "m5_music_min": "assumed",
    "m5_podcast_max": "assumed",
    "luma_knee_lo": "assumed",
    "luma_knee_hi": "assumed"
  }
}
```

`src/studio/types.py`:

```python
"""Geteilte Datentypen des Visualizer Studio."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasureConstraints:
    """Constraints für Messrenders (Probe, Preview-Badge, Verify).

    Defaults sind bit-identisch zum Bestandsverhalten des Renderers.
    """

    alpha_cap: float = 1.0
    alpha_from_luma: bool = False
    luma_knee_lo: float = 0.02
    luma_knee_hi: float = 0.25
    subject_strength: float = 0.0
    grain_free: bool = False
```

`src/studio/thresholds.py`:

```python
"""Versionierte Schwellwerte für das Studio-Quality-Gate (Spec §3.5)."""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "studio_thresholds.v1.json"


class ThresholdSet(BaseModel):
    """Schwellwerte M1–M6 plus Messparameter, mit Provenance je Wert."""

    version: str
    m1_overlay_energy_max: float
    m2_coverage_warn: float
    m3_subject_max: float
    m4_contrast_min: float
    m5_music_min: float
    m5_podcast_max: float
    epsilon: float
    luma_knee_lo: float
    luma_knee_hi: float
    provenance: dict[str, str]
    file_sha256: str


def load_thresholds(path: str | None = None) -> ThresholdSet:
    """Lädt das Threshold-Set; ohne Pfad die Default-Datei aus config/."""
    p = Path(path) if path else _DEFAULT_PATH
    raw = p.read_bytes()
    data = json.loads(raw)
    return ThresholdSet(
        version=data["version"],
        provenance=data["provenance"],
        file_sha256=hashlib.sha256(raw).hexdigest(),
        **data["thresholds"],
    )
```

`src/studio/__init__.py`:

```python
"""Visualizer Studio — qualitätsgesicherte Render-Pipeline (Spec studio-spec/2.1)."""

from .thresholds import ThresholdSet, load_thresholds
from .types import MeasureConstraints

__all__ = ["ThresholdSet", "load_thresholds", "MeasureConstraints"]
```

- [ ] **Step 4: Tests laufen lassen, Erfolg verifizieren**

Run: `pytest tests/test_studio_thresholds.py -v`
Expected: 4 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/ config/studio_thresholds.v1.json tests/test_studio_thresholds.py
git commit -m "feat(studio): P0 Grundgerüst — ThresholdSet, MeasureConstraints"
```

---

### Task 2: Messraster (Normalisierung, sRGB→Linear)

**Files:**
- Create: `src/studio/metrics.py`
- Test: `tests/test_studio_metrics.py`

**Interfaces:**
- Consumes: nichts aus Task 1.
- Produces:
  - `srgb_to_linear(rgb: np.ndarray) -> np.ndarray` — float in [0,1], beliebige Shape (…, 3)
  - `to_measure_raster(frame: np.ndarray, long_edge: int = 854) -> np.ndarray` — uint8 (H, W, 3) → float32 linear (H', W', 3); kein Upscale
  - Konsumiert von Tasks 3–8.

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_metrics.py`:

```python
"""Tests für Messraster und Metriken M1–M6 (Spec §3.1, §3.3)."""

import numpy as np
import pytest

from src.studio.metrics import srgb_to_linear, to_measure_raster


def test_srgb_to_linear_known_values():
    assert srgb_to_linear(np.array(0.0)) == pytest.approx(0.0)
    assert srgb_to_linear(np.array(1.0)) == pytest.approx(1.0)
    # sRGB-Mittengrau 0.5 -> linear ~0.2140
    assert srgb_to_linear(np.array(0.5)) == pytest.approx(0.2140, abs=1e-3)


def test_measure_raster_long_edge_and_aspect():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = to_measure_raster(frame, long_edge=854)
    assert out.shape == (427, 854, 3)
    assert out.dtype == np.float32


def test_measure_raster_no_upscale():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = to_measure_raster(frame, long_edge=854)
    assert out.shape == (100, 200, 3)


def test_measure_raster_is_linear_light():
    frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    out = to_measure_raster(frame, long_edge=854)
    assert float(out.mean()) == pytest.approx(0.2158, abs=1e-2)
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_metrics.py -v`
Expected: FAIL mit `ImportError: cannot import name 'srgb_to_linear'`

- [ ] **Step 3: Implementierung**

`src/studio/metrics.py`:

```python
"""Messraster und Metriken M1–M6 (Spec §3.1, §3.3).

Alle Metriken rechnen auf dem normalisierten Messraster: lange Kante
854 px, Linear-Light, float32 in [0, 1].
"""

import numpy as np
from PIL import Image

MEASURE_LONG_EDGE = 854


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """sRGB (float, [0,1]) -> Linear-Light (float, [0,1])."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def to_measure_raster(frame: np.ndarray, long_edge: int = MEASURE_LONG_EDGE) -> np.ndarray:
    """uint8-RGB-Frame -> normalisiertes Messraster (float32, linear).

    Downscale per BOX (Area-Mittelung), Seitenverhältnis bleibt erhalten,
    kein Upscale.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, long_edge / max(h, w))
    if scale < 1.0:
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.BOX)
        frame = np.asarray(img)
    return srgb_to_linear(frame.astype(np.float32) / 255.0)
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_metrics.py -v`
Expected: 4 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/metrics.py tests/test_studio_metrics.py
git commit -m "feat(studio): P0 Messraster — Normalisierung + sRGB->Linear"
```

---

### Task 3: Metriken M1/M2/M3/M5/M6 + contribution + Invarianz-Test

**Files:**
- Modify: `src/studio/metrics.py` (anfügen)
- Test: `tests/test_studio_metrics.py` (anfügen)
- Test: `tests/test_studio_metric_invariance.py` (neu)

**Interfaces:**
- Consumes: `to_measure_raster` (Task 2).
- Produces:
  - `contribution(a_linear: np.ndarray, b_linear: np.ndarray) -> np.ndarray` — `clamp(mean_c |a−b|, 0, 1)`, Ergebnis (H, W) float32
  - `overlay_energy(contrib: np.ndarray) -> float` (M1)
  - `overlay_coverage(contrib: np.ndarray, threshold: float = 0.5) -> float` (M2)
  - `subject_disturbance(contrib: np.ndarray, mask: np.ndarray) -> float` (M3)
  - `vitality(contrib_t: np.ndarray, contrib_t_delta: np.ndarray) -> float` (M5, ein Zeitpaar)
  - `integrity_violations(frame_linear: np.ndarray) -> list[str]` (M6; Strings: `"nan_inf"`, `"blackframe"`, `"clipping"`)
  - Konsumiert von Tasks 6–9 und später P3 (Solver).

- [ ] **Step 1: Failing Tests anfügen**

An `tests/test_studio_metrics.py` anhängen:

```python
from src.studio.metrics import (
    contribution,
    integrity_violations,
    overlay_coverage,
    overlay_energy,
    subject_disturbance,
    vitality,
)


def test_contribution_and_m1_m2():
    a = np.full((10, 10, 3), 0.8, dtype=np.float32)
    b = np.full((10, 10, 3), 0.2, dtype=np.float32)
    contrib = contribution(a, b)
    assert overlay_energy(contrib) == pytest.approx(0.6)
    assert overlay_coverage(contrib) == pytest.approx(1.0)
    assert overlay_coverage(contrib, threshold=0.7) == pytest.approx(0.0)


def test_m1_zero_for_identical_frames():
    frame = np.random.rand(8, 8, 3).astype(np.float32)
    assert overlay_energy(contribution(frame, frame)) == pytest.approx(0.0)


def test_m3_subject_disturbance():
    contrib = np.zeros((4, 4), dtype=np.float32)
    contrib[:2, :] = 0.4  # obere Hälfte gestört
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:2, :] = 1.0     # Subjekt oben
    assert subject_disturbance(contrib, mask) == pytest.approx(0.4)
    mask_zero = np.zeros((4, 4), dtype=np.float32)
    assert subject_disturbance(contrib, mask_zero) == 0.0  # kein Subjekt -> 0


def test_m5_vitality():
    t0 = np.zeros((4, 4), dtype=np.float32)
    t1 = np.full((4, 4), 0.3, dtype=np.float32)
    assert vitality(t0, t1) == pytest.approx(0.3)
    assert vitality(t1, t1) == pytest.approx(0.0)


def test_m6_integrity():
    ok = np.full((10, 10, 3), dtype=np.float32)
    assert integrity_violations(ok) == []
    nan_frame = ok.copy(); nan_frame[0, 0, 0] = np.nan
    assert "nan_inf" in integrity_violations(nan_frame)
    black = np.zeros((10, 10, 3), dtype=np.float32)
    assert "blackframe" in integrity_violations(black)
    clipped = np.ones((10, 10, 3), dtype=np.float32)
    assert "clipping" in integrity_violations(clipped)
```

Neu: `tests/test_studio_metric_invariance.py` (Spec §15 — „derselbe Frame in 480p/1080p/4K ⇒ Metriken innerhalb ε = 0.01"):

```python
"""Metrik-Invarianz über Auflösungen (Spec §3.1, §15).

Ohne diesen Test ist 'Preview = Batch' eine Behauptung.
"""

import numpy as np
import pytest
from PIL import Image

from src.studio.metrics import contribution, overlay_energy, to_measure_raster


def test_metric_invariance_across_resolutions():
    rng = np.random.default_rng(42)
    base = (rng.random((90, 160, 3)) * 255).astype(np.uint8)
    zeros = np.zeros_like(base)

    ref = overlay_energy(
        contribution(to_measure_raster(base), to_measure_raster(zeros))
    )
    for factor in (2, 4):  # simuliert 1080p/4K-Varianten desselben Inhalts
        big = np.asarray(
            Image.fromarray(base).resize(
                (160 * factor, 90 * factor), Image.Resampling.BOX
            )
        )
        big_zeros = np.zeros_like(big)
        m = overlay_energy(
            contribution(to_measure_raster(big), to_measure_raster(big_zeros))
        )
        assert abs(m - ref) <= 0.01
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_metrics.py tests/test_studio_metric_invariance.py -v`
Expected: FAIL mit `ImportError: cannot import name 'contribution'`

- [ ] **Step 3: Implementierung anfügen**

An `src/studio/metrics.py` anhängen:

```python
# --- Differenz-Render und Metriken M1/M2/M3/M5/M6 (Spec §3.2, §3.3) ---


def contribution(a_linear: np.ndarray, b_linear: np.ndarray) -> np.ndarray:
    """Post-FX-wirksamer Visualizer-Einfluss pro Pixel (H, W, float32)."""
    diff = np.abs(a_linear.astype(np.float32) - b_linear.astype(np.float32))
    return np.clip(diff.mean(axis=-1), 0.0, 1.0)


def overlay_energy(contrib: np.ndarray) -> float:
    """M1: mittlere Overlay-Energie (kontinuierlich, hart)."""
    return float(np.mean(contrib))


def overlay_coverage(contrib: np.ndarray, threshold: float = 0.5) -> float:
    """M2: Flächenanteil oberhalb der Schwelle (weich/warn)."""
    return float(np.mean(contrib > threshold))


def subject_disturbance(contrib: np.ndarray, mask: np.ndarray) -> float:
    """M3: maskengewichtete Störung; 0.0 wenn keine Subjektfläche."""
    denom = float(mask.sum())
    if denom <= 0.0:
        return 0.0
    return float((contrib * mask).sum() / denom)


def vitality(contrib_t: np.ndarray, contrib_t_delta: np.ndarray) -> float:
    """M5: mittlere zeitliche Änderung zwischen zwei contrib-Maps."""
    return float(np.mean(np.abs(contrib_t_delta - contrib_t)))


def integrity_violations(frame_linear: np.ndarray) -> list[str]:
    """M6: binäre Integritätsprüfung (NaN/Inf, Blackframe, Clipping)."""
    violations: list[str] = []
    if not np.isfinite(frame_linear).all():
        violations.append("nan_inf")
    luma = frame_linear.mean(axis=-1)
    if float(np.percentile(luma, 99)) < 0.02:
        violations.append("blackframe")
    if float(np.mean(frame_linear >= 1.0 - 1e-6)) > 0.15:
        violations.append("clipping")
    return violations
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_metrics.py tests/test_studio_metric_invariance.py -v`
Expected: 10 × PASS (4 aus Task 2 + 5 neu + 1 Invarianz)

- [ ] **Step 5: Commit**

```bash
git add src/studio/metrics.py tests/test_studio_metrics.py tests/test_studio_metric_invariance.py
git commit -m "feat(studio): P0 Metriken M1/M2/M3/M5/M6 + contribution + Invarianz"
```

---

### Task 4: M4 Text-Kontrast (Glyphenmaske + Ring, WCAG)

**Files:**
- Modify: `src/studio/metrics.py` (anfügen)
- Test: `tests/test_studio_metrics.py` (anfügen)

**Interfaces:**
- Consumes: nichts aus anderen Tasks.
- Produces:
  - `text_contrast_wcag(frame_linear: np.ndarray, glyph_mask: np.ndarray, ring_dilate_px: int = 3) -> float` — Kontrast-Ratio ≥ 1.0; `0.0` bei leerer Glyphenmaske (kein Text = nicht messbar, Aufrufer überspringt)
  - `aggregate_text_contrast(per_frame_ratios: list[float]) -> float` — Minimum über Frames
  - Konsumiert von P3 (QualityGate) und P5 (Badge).

- [ ] **Step 1: Failing Tests anfügen**

An `tests/test_studio_metrics.py` anhängen:

```python
from src.studio.metrics import aggregate_text_contrast, text_contrast_wcag


def _glyph_mask(h=40, w=40):
    mask = np.zeros((h, w), dtype=bool)
    mask[10:30, 10:30] = True  # zentraler Textblock
    return mask


def test_m4_black_on_white():
    frame = np.ones((40, 40, 3), dtype=np.float32)  # weißer Hintergrund
    frame[10:30, 10:30] = 0.0                        # schwarze Glyphen
    ratio = text_contrast_wcag(frame, _glyph_mask())
    assert ratio == pytest.approx(21.0, abs=0.5)


def test_m4_white_on_white():
    frame = np.ones((40, 40, 3), dtype=np.float32)
    ratio = text_contrast_wcag(frame, _glyph_mask())
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_m4_empty_mask_returns_zero():
    frame = np.ones((40, 40, 3), dtype=np.float32)
    assert text_contrast_wcag(frame, np.zeros((40, 40), dtype=bool)) == 0.0


def test_m4_aggregation_is_minimum():
    assert aggregate_text_contrast([7.0, 4.6, 12.3]) == pytest.approx(4.6)
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_metrics.py -v -k m4`
Expected: FAIL mit `ImportError: cannot import name 'text_contrast_wcag'`

- [ ] **Step 3: Implementierung anfügen**

An `src/studio/metrics.py` anhängen:

```python
# --- M4: Text-Kontrast (Spec §3.3, Glyphenmaske + Ring, WCAG 2.x) ---


def _relative_luminance(rgb_linear: np.ndarray) -> np.ndarray:
    """WCAG-relative Luminanz auf bereits linearem RGB."""
    return (
        0.2126 * rgb_linear[..., 0]
        + 0.7152 * rgb_linear[..., 1]
        + 0.0722 * rgb_linear[..., 2]
    )


def text_contrast_wcag(
    frame_linear: np.ndarray,
    glyph_mask: np.ndarray,
    ring_dilate_px: int = 3,
) -> float:
    """Kontrast-Ratio Glyphen vs. dilatierter Hintergrund-Ring (WCAG 2.x).

    Vordergrund = p5 der Glyphen-Luminanz (Worst-Case-nah), Hintergrund =
    Median des Rings um die Glyphen. 0.0 bei leerer Glyphenmaske.
    """
    from PIL import ImageFilter

    glyph = np.asarray(glyph_mask, dtype=bool)
    if not glyph.any():
        return 0.0
    kernel = 2 * ring_dilate_px + 1
    dilated = np.asarray(
        Image.fromarray(glyph).filter(ImageFilter.MaxFilter(kernel)), dtype=bool
    )
    ring = dilated & ~glyph
    if not ring.any():
        return 0.0
    luma = _relative_luminance(frame_linear)
    fg = float(np.percentile(luma[glyph], 5))
    bg = float(np.median(luma[ring]))
    hi, lo = max(fg, bg), min(fg, bg)
    return (hi + 0.05) / (lo + 0.05)


def aggregate_text_contrast(per_frame_ratios: list[float]) -> float:
    """M4-Aggregation: Minimum über alle Sample-Frames (Spec §3.3)."""
    valid = [r for r in per_frame_ratios if r > 0.0]
    return min(valid) if valid else 0.0
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_metrics.py -v`
Expected: 13 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/metrics.py tests/test_studio_metrics.py
git commit -m "feat(studio): P0 Metrik M4 — WCAG-Kontrast per Glyphenmaske"
```

---

### Task 5: Luma-Alpha-Ableitung im Blit-Shader (C14)

**Kontext (verifiziert):** Der einzige reale Mischpunkt von Visualizer über Hintergrund ist `_blit_viz_to_fbo` (`gpu_renderer.py:1415`) mit dem Blit-Shader (`_init_blit_shader`, `gpu_renderer.py:1285-1303`): `f_color = vec4(tex.rgb, tex.a * u_opacity)` mit `SRC_ALPHA`-Blending. `_composite_viz_over_bg` wird **nirgends aufgerufen** (toter Code, nicht anfassen). Der Blit-Shader hat **keinen** Alpha-Fallback — Visualizer mit `alpha = 1.0` (u.a. jeder `CompositeVisualizer`) decken damit heute die gesamte Bildfläche ab, auch wo sie Schwarz zeichnen. Die Luma-Ableitung muss deshalb im Studio-Modus **unbedingt** gelten.

**Files:**
- Modify: `src/gpu_renderer.py:1285-1303` (`_init_blit_shader`), `src/gpu_renderer.py:1415-1441` (`_blit_viz_to_fbo`)
- Test: `tests/test_studio_luma_alpha.py`

**Interfaces:**
- Consumes: `MeasureConstraints` (Task 1) — wird hier noch nicht durchgereicht, nur die Shader-/Methoden-Ebene.
- Produces:
  - Neue Blit-Shader-Uniforms: `u_subject_mask` (sampler2D, unit 1), `u_resolution` (vec2), `u_viz_alpha_cap` (float, Default 1.0), `u_viz_alpha_from_luma` (float 0/1, Default 0.0), `u_luma_knee_lo/hi` (float), `u_subject_strength` (float, Default 0.0)
  - Neue Signatur: `_blit_viz_to_fbo(self, source_texture, offset_x=0.0, offset_y=0.0, scale=1.0, opacity=1.0, alpha_cap=1.0, alpha_from_luma=False, luma_knee_lo=0.02, luma_knee_hi=0.25, subject_strength=0.0, subject_mask=None) -> None`
  - **Bit-Identitäts-Garantie:** Alle Defaults ergeben exakt das Bestandsverhalten (`f_color = vec4(tex.rgb, tex.a * 1.0)`).
  - Konsumiert von Task 6 (ProbeRenderer) und P1/P3. Die bestehenden Aufrufe in `gpu_renderer.py:498` und `gpu_preview.py:161` bleiben unverändert (Defaults).

- [ ] **Step 1: Failing Test schreiben**

`tests/test_studio_luma_alpha.py`:

```python
"""Tests für die Luma-Alpha-Ableitung im Blit-Shader (C14, Spec §6.1).

Dokumentiert den Defekt: Visualizer mit alpha=1.0 (u.a. Composite-Stacks)
decken heute die gesamte Fläche ab — ein Cap darauf wäre ein
Vollbild-Schleier, keine Deckungsregel.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def renderer(shared_gl_context):
    from src.gpu_renderer import GPUPreviewRenderer
    r = GPUPreviewRenderer(width=64, height=64, fps=30)
    yield r
    r.release()


def _viz_texture(renderer):
    """64x64 RGBA: alpha=1.0 überall, aber nur ein helles Quadrat —
    exakt das Ausgabeverhalten von Composite-Stacks (composite.py:137)."""
    data = np.zeros((64, 64, 4), dtype=np.uint8)
    data[16:48, 16:48, :3] = 255
    data[..., 3] = 255
    return renderer.ctx.texture((64, 64), 4, data.tobytes())


def _blit_to_array(renderer, viz_tex, **kwargs):
    renderer.fbo.use()
    renderer.ctx.clear(0.5, 0.5, 0.5, 1.0)  # grauer „Hintergrund"
    renderer._blit_viz_to_fbo(viz_tex, **kwargs)
    # ACHTUNG: renderer.fbo ist HDR (dtype f16, gpu_renderer.py:82) —
    # read() braucht dtype="f2", ein uint8-Read wäre Datensalat.
    raw = renderer.fbo.read(components=3, dtype="f2")
    return np.frombuffer(raw, dtype=np.float16).reshape(64, 64, 3).astype(np.float32)


def test_default_behavior_documents_defect(renderer):
    out = _blit_to_array(renderer, _viz_texture(renderer))
    # Bestand: schwarze Regionen ersetzen den Hintergrund vollständig.
    assert out[4, 4].mean() < 0.05
    assert out[32, 32].mean() > 0.9


def test_luma_alpha_frees_black_regions(renderer):
    out = _blit_to_array(
        renderer, _viz_texture(renderer), alpha_from_luma=True
    )
    # Studio: schwarze Regionen bleiben Hintergrund, Quadrat bleibt sichtbar.
    assert out[4, 4].mean() == pytest.approx(0.5, abs=0.05)
    assert out[32, 32].mean() > 0.9


def test_alpha_cap_zero_yields_background(renderer):
    out = _blit_to_array(
        renderer, _viz_texture(renderer), alpha_cap=0.0, alpha_from_luma=True
    )
    # Fundament der Messtechnik: cap=0 => kein Visualizer-Beitrag.
    assert out.mean() == pytest.approx(0.5, abs=0.02)
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_luma_alpha.py -v`
Expected: Test 1 PASS (dokumentiert Bestand), Tests 2+3 FAIL mit `TypeError: _blit_viz_to_fbo() got an unexpected keyword argument 'alpha_from_luma'`

- [ ] **Step 3: Implementierung**

In `src/gpu_renderer.py` `_init_blit_shader` (Zeile ~1285-1303) — Fragment-Shader ersetzen durch:

```glsl
#version 330
uniform sampler2D u_texture;
uniform sampler2D u_subject_mask;
uniform vec2 u_resolution;
uniform float u_opacity;
uniform float u_viz_alpha_cap;        // Default 1.0 = kein Cap
uniform float u_viz_alpha_from_luma;  // 0.0 = Bestand, 1.0 = Studio (C14)
uniform float u_luma_knee_lo;
uniform float u_luma_knee_hi;
uniform float u_subject_strength;     // Default 0.0 = keine Maskierung
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 tex = texture(u_texture, v_uv);
    float a_viz = tex.a;
    // Studio-Pfad (C14): Helligkeit IST die Deckung fuer Emitter auf
    // Schwarz. Gilt UNABHAENGIG von tex.a — auch alpha=1.0-Stacks
    // (composite.py) zeichnen grossflaechig Schwarz. Laeuft VOR dem
    // Cap (Reihenfolge bindend, Spec §3.2.2).
    if (u_viz_alpha_from_luma > 0.5) {
        float luma = dot(tex.rgb, vec3(0.2126, 0.7152, 0.0722));
        a_viz = smoothstep(u_luma_knee_lo, u_luma_knee_hi, luma);
    }
    // Subjekt-Maske liegt im Bildschirmraum, nicht im Quad-UV-Raum
    // (der Blit-Quad hat Offset/Scale — v_uv waere falsch).
    vec2 screen_uv = gl_FragCoord.xy / u_resolution;
    float subject_mask = texture(u_subject_mask, screen_uv).r;
    float a_eff = min(a_viz, u_viz_alpha_cap) * u_opacity
                * (1.0 - u_subject_strength * subject_mask);
    f_color = vec4(tex.rgb, a_eff);
}
```

Und `_blit_viz_to_fbo` (Zeile ~1415-1441) ersetzen durch:

```python
def _blit_viz_to_fbo(
    self, source_texture, offset_x=0.0, offset_y=0.0, scale=1.0,
    opacity=1.0, alpha_cap=1.0, alpha_from_luma=False,
    luma_knee_lo=0.02, luma_knee_hi=0.25,
    subject_strength=0.0, subject_mask=None,
):
    """Blittet die Visualizer-Textur auf den aktuellen FBO.

    Defaults sind bit-identisch zum bisherigen Verhalten. Die Studio-
    Parameter (C14) aktivieren Luma-Alpha, Cap und Subjekt-Maskierung.
    """
    if not hasattr(self, '_blit_prog'):
        self._init_blit_shader()

    # Quad-Vertices basierend auf Offset und Skalierung berechnen
    x1 = -1.0 * scale + offset_x
    x2 =  1.0 * scale + offset_x
    y1 = -1.0 * scale + offset_y
    y2 =  1.0 * scale + offset_y

    vertices = np.array([
        x1, y1, 0.0, 0.0,
        x2, y1, 1.0, 0.0,
        x1, y2, 0.0, 1.0,
        x2, y2, 1.0, 1.0,
    ], dtype=np.float32)
    self._blit_vbo.write(vertices.tobytes())

    self._blit_prog["u_texture"].value = 0
    source_texture.use(location=0)
    # Subjekt-Maske: Default schwarz (= kein Subjekt), Dummy wiederverwenden
    if subject_mask is not None:
        subject_mask.use(location=1)
    else:
        self._dummy_black_texture.use(location=1)

    prog = self._blit_prog
    if "u_subject_mask" in prog:
        prog["u_subject_mask"].value = 1
    if "u_resolution" in prog:
        prog["u_resolution"].value = (float(self.width), float(self.height))
    if "u_viz_alpha_cap" in prog:
        prog["u_viz_alpha_cap"].value = float(alpha_cap)
    if "u_viz_alpha_from_luma" in prog:
        prog["u_viz_alpha_from_luma"].value = 1.0 if alpha_from_luma else 0.0
    if "u_luma_knee_lo" in prog:
        prog["u_luma_knee_lo"].value = float(luma_knee_lo)
    if "u_luma_knee_hi" in prog:
        prog["u_luma_knee_hi"].value = float(luma_knee_hi)
    if "u_subject_strength" in prog:
        prog["u_subject_strength"].value = float(subject_strength)
    prog["u_opacity"].value = opacity

    self.ctx.enable(moderngl.BLEND)
    self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    self._blit_vao.render(mode=moderngl.TRIANGLE_STRIP)
    self.ctx.disable(moderngl.BLEND)
```

Hinweis: `_dummy_black_texture` existiert bereits (`gpu_renderer.py:1277-1278`). Die Uniform-Guards (`if "u_x" in prog`) schützen vor GLSL-Optimierung, die ungenutzte Uniforms entfernt.

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_luma_alpha.py -v`
Expected: 3 × PASS

Zusätzlich Regression: `pytest tests/test_gpu_renderer.py tests/test_gpu_preview.py -v`
Expected: alle PASS (Bit-Identität der Defaults)

- [ ] **Step 5: Commit**

```bash
git add src/gpu_renderer.py tests/test_studio_luma_alpha.py
git commit -m "feat(studio): P0 C14 — Luma-Alpha + Cap + Subjekt-Maske im Blit-Shader"
```

---

### Task 6: ProbeRenderer (Differenz-Render)

**Files:**
- Create: `src/studio/probe.py`
- Test: `tests/test_studio_diff_render.py`

**Interfaces:**
- Consumes: `MeasureConstraints` (Task 1), `to_measure_raster`, `contribution` (Task 2/3), `_blit_viz_to_fbo`-Kwargs (Task 5), `GPUPreviewRenderer` (`src/gpu_renderer.py:1604`), `build_features_dict(features, frame_count, fps)` (`src/render_common.py`).
- Produces:
  - `probe_resolution(target_w: int, target_h: int) -> tuple[int, int]` — `max(480p, Ziel/4)`, Aspect-identisch (Spec §3.4)
  - `class ProbeRenderer(width, height, fps)`:
    - `render_frame(viz, features_dict, time_s, bg_texture, postprocess: dict, constraints: MeasureConstraints) -> np.ndarray` (uint8 HWC3)
    - `render_pair(viz, features_dict, time_s, bg_texture, postprocess, constraints) -> tuple[np.ndarray, np.ndarray]` — (A, B); B mit `alpha_cap=0`, Visualizer-Pass übersprungen
    - `contribution_map(a: np.ndarray, b: np.ndarray) -> np.ndarray` — Messraster + `contribution`
  - Konsumiert von Tasks 7–8, P3 (Solver), P5 (Badge).

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_diff_render.py`:

```python
"""Tests für den Differenz-Render (Spec §3.2, §3.4)."""

import numpy as np
import pytest

from src.studio.probe import ProbeRenderer, probe_resolution
from src.studio.types import MeasureConstraints

pytestmark = pytest.mark.gpu


def test_probe_resolution_policy():
    assert probe_resolution(3840, 2160) == (960, 540)   # Ziel/4 > 480p
    assert probe_resolution(1920, 1080) == (854, 480)   # 480p-Minimum greift
    assert probe_resolution(1280, 720) == (854, 480)
    # Aspect-identisch bei Nicht-16:9
    w, h = probe_resolution(2000, 1000)
    assert w / h == pytest.approx(2.0, abs=0.01)


@pytest.fixture
def probe():
    p = ProbeRenderer(width=160, height=90, fps=30)
    yield p
    p.release()


def test_contribution_zero_when_capped(probe, dummy_audio_features):
    from src.render_common import build_features_dict
    from src.gpu_visualizers import get_visualizer

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    viz_cls = get_visualizer("spectrum_bars")
    viz = viz_cls(probe.ctx, 160, 90)
    constraints = MeasureConstraints(alpha_from_luma=True)
    a, b = probe.render_pair(viz, features_dict, 0.5, None, {}, constraints)
    contrib = probe.contribution_map(a, b)
    assert float(contrib.max()) < 1e-3


def test_contribution_scales_with_cap(probe, dummy_audio_features):
    from src.render_common import build_features_dict
    from src.gpu_visualizers import get_visualizer

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    viz_cls = get_visualizer("spectrum_bars")
    bg = np.full((90, 160, 3), 40, dtype=np.uint8)
    bg_tex = probe.ctx.texture((160, 90), 3, bg.tobytes())

    def energy(cap):
        viz = viz_cls(probe.ctx, 160, 90)
        constraints = MeasureConstraints(alpha_cap=cap, alpha_from_luma=True)
        a = probe.render_frame(viz, features_dict, 0.5, bg_tex, {}, constraints)
        b = probe.render_frame(
            viz, features_dict, 0.5, bg_tex, {},
            MeasureConstraints(alpha_cap=0.0, alpha_from_luma=True),
        )
        return float(probe.contribution_map(a, b).mean())

    assert energy(1.0) > energy(0.3) > 0.0
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_diff_render.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.probe'`

- [ ] **Step 3: Implementierung**

`src/studio/probe.py`:

```python
"""ProbeRenderer — Differenz-Render für Probe/Preview/Verify (Spec §3.2).

Rendert A(t) (vollständig) und B(t) (Visualizer-Beitrag 0) mit identischem
u_time; Rauschaufhebung ist dadurch gegeben (Seeding via fract(u_time),
gpu_renderer.py:1132-1142). B ist bei statischem Hintergrund zeitinvariant
und kann vom Aufrufer gecacht werden (Spec §3.2.2). Spiegelt den Batch-
Loop (gpu_renderer.py:462-534) exakt: Clear -> Hintergrund -> Viz-Blit ->
Bloom -> Post-Process.
"""

import numpy as np

from ..gpu_renderer import GPUPreviewRenderer
from .metrics import contribution, to_measure_raster
from .types import MeasureConstraints

MIN_PROBE = (854, 480)


def probe_resolution(target_w: int, target_h: int) -> tuple[int, int]:
    """probe_res = max(480p, Ziel/4), Seitenverhältnis identisch (Spec §3.4)."""
    scale = max(0.25, MIN_PROBE[0] / target_w, MIN_PROBE[1] / target_h)
    return max(1, round(target_w * scale)), max(1, round(target_h * scale))


class ProbeRenderer:
    """Einzel-Frame-Renderer für Messzwecke (kein Encode, kein FFmpeg)."""

    def __init__(self, width: int, height: int, fps: int = 30):
        self._r = GPUPreviewRenderer(width=width, height=height, fps=fps)

    @property
    def ctx(self):
        return self._r.ctx

    def release(self):
        self._r.release()

    def render_frame(
        self, viz, features_dict, time_s, bg_texture,
        postprocess: dict, constraints: MeasureConstraints,
    ) -> np.ndarray:
        """Rendert ein Frame; bei alpha_cap=0 wird der Visualizer-Pass
        übersprungen (Blit-Alpha 0 — reine Ersparnis, Spec §3.2.2)."""
        r = self._r
        r.fbo.use()
        r.ctx.clear(0.0, 0.0, 0.0)
        if bg_texture is not None:
            r._render_background(bg_texture, 1.0, 0.0)
        if constraints.alpha_cap > 0.0:
            r._render_viz_into(viz, r.viz_fbo, features_dict, time_s)
            r.fbo.use()
            r._blit_viz_to_fbo(
                r.viz_fbo.color_attachments[0],
                alpha_cap=constraints.alpha_cap,
                alpha_from_luma=constraints.alpha_from_luma,
                luma_knee_lo=constraints.luma_knee_lo,
                luma_knee_hi=constraints.luma_knee_hi,
                subject_strength=constraints.subject_strength,
            )
        pp = dict(postprocess or {})
        if constraints.grain_free:
            pp["film_grain"] = 0.0  # C15 Regel 3: M5 nur grain-frei
        bloom_intensity = pp.get("bloom_intensity", 0.6)
        if r._bloom is not None and bloom_intensity > 0.0:
            r._apply_bloom(
                intensity=bloom_intensity,
                threshold=pp.get("bloom_threshold", 1.0),
                radius=pp.get("bloom_radius", 1.0),
            )
        r._apply_postprocess(
            r.fbo.color_attachments[0],
            contrast=pp.get("contrast", 1.0),
            saturation=pp.get("saturation", 1.0),
            brightness=pp.get("brightness", 0.0),
            warmth=pp.get("warmth", 0.0),
            film_grain=pp.get("film_grain", 0.0),
            time=time_s,
            exposure=pp.get("exposure", 1.0),
            vignette=pp.get("vignette", 0.0),
            chromatic_aberration=pp.get("chromatic_aberration", 0.0),
            lut_path=pp.get("lut"),
            lut_strength=pp.get("lut_strength", 1.0),
        )
        raw = r.post_fbo.read(components=3)
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(r.height, r.width, 3)
            .copy()
        )

    def render_pair(self, viz, features_dict, time_s, bg_texture,
                    postprocess, constraints) -> tuple[np.ndarray, np.ndarray]:
        """(A, B): B mit alpha_cap=0, identisches u_time für beide."""
        a = self.render_frame(viz, features_dict, time_s, bg_texture,
                              postprocess, constraints)
        b_constraints = MeasureConstraints(
            alpha_cap=0.0,
            alpha_from_luma=constraints.alpha_from_luma,
            grain_free=constraints.grain_free,
        )
        b = self.render_frame(viz, features_dict, time_s, bg_texture,
                              postprocess, b_constraints)
        return a, b

    def contribution_map(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """contrib-Map auf dem normalisierten Messraster."""
        return contribution(to_measure_raster(a), to_measure_raster(b))
```

Hinweis für den Implementierer: `width`/`height`/`fps` sind Attribute der Basisklasse (`gpu_renderer.py:68-70`), `_render_viz_into` steht bei `gpu_renderer.py:1305`, `_render_background` bei `:1443`. Alle werden im bestehenden Preview-Pfad genauso verwendet (`src/gpu_preview.py:145-194`).

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_diff_render.py -v`
Expected: 3 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/probe.py tests/test_studio_diff_render.py
git commit -m "feat(studio): P0 ProbeRenderer — Differenz-Render + probe_resolution"
```

---

### Task 7: Rausch-Aufhebung (C15)

**Files:**
- Test: `tests/test_studio_noise_cancellation.py`

**Interfaces:**
- Consumes: `ProbeRenderer` (Task 6), `MeasureConstraints.grain_free` (Task 1).
- Produces: nichts Neues — reiner Verifikations-Task (Spec §15, blockierend). Beweist: identischer `u_time` ⇒ `contrib ≡ 0` trotz Grain; unterschiedlicher `u_time` ⇒ `contrib > 0` (Negativkontrolle); `grain_free=True` erzwingt `film_grain = 0`.

- [ ] **Step 1: Test schreiben (schlägt nur fehl, wenn die Invariante verletzt ist)**

`tests/test_studio_noise_cancellation.py`:

```python
"""Rausch-Aufhebung im Differenz-Render (C15, Spec §3.2.1).

Der positive Fall muss PASSEN; die Negativkontrolle zeigt, dass der Test
überhaupt etwas prüft (unterschiedlicher Seed => Rauschboden sichtbar).
"""

import numpy as np
import pytest

from src.studio.probe import ProbeRenderer
from src.studio.types import MeasureConstraints

pytestmark = pytest.mark.gpu

GRAINY_PP = {"film_grain": 0.5, "bloom_intensity": 0.0}


@pytest.fixture
def probe():
    p = ProbeRenderer(width=160, height=90, fps=30)
    yield p
    p.release()


def _black_viz(probe):
    """Visualizer, der nichts zeichnet (Alpha 0, Farbe 0)."""
    from src.gpu_visualizers.pulsing_core import PulsingCoreVisualizer
    viz = PulsingCoreVisualizer(probe.ctx, 160, 90)
    viz.set_params({"bg_brightness": 0.0})
    return viz


def test_noise_cancels_with_identical_seed(probe, dummy_audio_features):
    from src.render_common import build_features_dict
    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    viz = _black_viz(probe)
    # Beide Renders mit cap=0 => kein Visualizer-Beitrag; Grain aktiv.
    # Identisches u_time => contrib muss trotz Grain ~0 sein.
    c = MeasureConstraints(alpha_cap=0.0)
    a = probe.render_frame(viz, features_dict, 0.5, None, GRAINY_PP, c)
    b = probe.render_frame(viz, features_dict, 0.5, None, GRAINY_PP, c)
    contrib = probe.contribution_map(a, b)
    assert float(contrib.max()) < 1e-4


def test_noise_visible_with_different_seed(probe, dummy_audio_features):
    from src.render_common import build_features_dict
    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    viz = _black_viz(probe)
    c = MeasureConstraints(alpha_cap=0.0)
    a = probe.render_frame(viz, features_dict, 0.5, None, GRAINY_PP, c)
    b = probe.render_frame(viz, features_dict, 0.516, None, GRAINY_PP, c)
    contrib = probe.contribution_map(a, b)
    # Negativkontrolle: anderer Seed => Rauschboden in derselben
    # Größenordnung wie die M5-Schwelle (0.02).
    assert float(contrib.mean()) > 0.005


def test_grain_free_forces_zero_grain(probe, dummy_audio_features):
    from src.render_common import build_features_dict
    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    viz = _black_viz(probe)
    c = MeasureConstraints(alpha_cap=0.0, grain_free=True)
    a = probe.render_frame(viz, features_dict, 0.5, None, GRAINY_PP, c)
    b = probe.render_frame(viz, features_dict, 0.516, None, GRAINY_PP, c)
    contrib = probe.contribution_map(a, b)
    # grain_free unterdrückt Grain => auch unterschiedliche Zeitpunkte
    # liefern keinen Rauschboden (C15 Regel 3).
    assert float(contrib.max()) < 1e-4
```

Hinweis: Falls `PulsingCoreVisualizer` bei `bg_brightness=0` trotzdem zeichnet, spielt das keine Rolle — bei `alpha_cap=0.0` wird der Visualizer-Pass ohnehin übersprungen; der Viz wird nur für die Signatur benötigt.

- [ ] **Step 2: Tests laufen lassen**

Run: `pytest tests/test_studio_noise_cancellation.py -v`
Expected: 3 × PASS bei korrekter Task-6-Implementierung. FAIL bei Test 1 bedeutet: Seeding ist nicht zeitbasiert (Spec-Verletzung — dann greift §3.2.1 Regel 4 als Notlösung, Abschlag im Code vermerken).

- [ ] **Step 3: Commit**

```bash
git add tests/test_studio_noise_cancellation.py
git commit -m "test(studio): P0 C15 — Rausch-Aufhebung + grain-freier M5-Modus"
```

---

### Task 8: Drift-Messung (C16)

**Files:**
- Create: `tools/__init__.py` (leer)
- Create: `tools/measure_drift.py`
- Create: `config/studio_drift.v1.json`
- Test: `tests/test_studio_resolution_drift.py`

**Interfaces:**
- Consumes: `ProbeRenderer`, `probe_resolution` (Task 6), Metriken M1/M5 (Task 3).
- Produces:
  - `measure_visualizer_drift(viz_name: str, features_dict, probe_size, target_size, times: list[float], postprocess: dict | None = None) -> dict` — `{"M1": d1, "M5": d5}` mit `d = |m_probe − m_commit|`
  - `write_drift_file(entries: dict, path: str) -> None` — schreibt `studio_drift.v1.json` (`{"version": "studio-drift/1", "per_visualizer": {...}, "resolution_dependent": [...]}`)
  - CLI: `python tools/measure_drift.py --visualizers spectrum_bars,pulsing_core --target 1920x1080 --audio <datei>`
  - Konsumiert von P3 (Solver-Abschlag `τ_effektiv = τ − d`).

- [ ] **Step 1: Failing Test schreiben**

`tests/test_studio_resolution_drift.py`:

```python
"""Drift-Messung Probe vs. Ziel (C16, Spec §3.4, §15).

Der Test schlägt nicht bei Drift fehl, sondern wenn kein Driftwert
erfasst wurde — unbekannte Drift ist der Defekt, nicht Drift selbst.
"""

import json
import pytest

from src.studio.probe import probe_resolution

pytestmark = pytest.mark.gpu


def test_drift_is_measured_and_recorded(tmp_path, dummy_audio_features):
    from src.render_common import build_features_dict
    from tools.measure_drift import measure_visualizer_drift, write_drift_file

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    target = (320, 180)
    probe = probe_resolution(*target)
    entry = measure_visualizer_drift(
        "spectrum_bars", features_dict, probe, target, times=[0.2, 0.5, 0.8]
    )
    assert "M1" in entry and "M5" in entry
    assert all(v >= 0.0 for v in entry.values())

    out = tmp_path / "studio_drift.v1.json"
    write_drift_file({"spectrum_bars": entry}, str(out))
    data = json.loads(out.read_text())
    assert data["version"] == "studio-drift/1"
    assert data["per_visualizer"]["spectrum_bars"]["M1"] == entry["M1"]
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_resolution_drift.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'tools.measure_drift'`

- [ ] **Step 3: Implementierung**

`tools/__init__.py`: leere Datei.

`tools/measure_drift.py`:

```python
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
```

`config/studio_drift.v1.json` initial anlegen:

```json
{
  "version": "studio-drift/1",
  "per_visualizer": {},
  "resolution_dependent": []
}
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_resolution_drift.py -v`
Expected: 1 × PASS

- [ ] **Step 5: Commit**

```bash
git add tools/ config/studio_drift.v1.json tests/test_studio_resolution_drift.py
git commit -m "feat(studio): P0 C16 — Drift-Messung probe_res vs. Zielauflösung"
```

---

### Task 9: Kalibrier-Harness + Golden-Set-Scaffolding

**Files:**
- Create: `tools/calibrate_thresholds.py`
- Create: `tests/golden/labels.json`
- Create: `tests/golden/README.md`
- Test: `tests/test_studio_calibration.py`

**Interfaces:**
- Consumes: nichts aus anderen Tasks (arbeitet auf Metrik-Tabellen).
- Produces:
  - `sweep_threshold(values: list[float], labels: list[bool], higher_is_bad: bool, candidates: list[float] | None = None) -> dict` — `{"threshold": t, "sensitivity": s, "specificity": sp, "score": j}`
  - `set_hash(labels_path: str) -> str` — sha256 der Golden-Set-Labels (für `calibrated@<set-hash>`)
  - CLI: `python tools/calibrate_thresholds.py --labels tests/golden/labels.json` — gibt Trennschärfe-Tabelle aus; `--write` folgt in P3 (Provenance-Update).
  - Golden-Set-Befüllung (≥ 20 gelabelte Renders) ist **Nutzer-Aufgabe** (Spec §19, offene Lücke 1) — der Harness funktioniert ab dem ersten Label, Schwellen bleiben bis dahin `"assumed"`.

- [ ] **Step 1: Failing Test schreiben**

`tests/test_studio_calibration.py`:

```python
"""Tests für den Schwellen-Kalibrier-Harness (Spec §3.5)."""

import pytest

from tools.calibrate_thresholds import set_hash, sweep_threshold


def test_sweep_perfectly_separable():
    # Perfekt trennbar: schlechte Renders haben hohe M1, gute niedrige.
    values = [0.30, 0.28, 0.35, 0.10, 0.08, 0.12]
    labels = [False, False, False, True, True, True]  # True = gut
    best = sweep_threshold(values, labels, higher_is_bad=True)
    assert best["sensitivity"] == pytest.approx(1.0)
    assert best["specificity"] == pytest.approx(1.0)
    assert 0.12 < best["threshold"] < 0.28


def test_set_hash_stable(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text('{"renders": []}')
    h1 = set_hash(str(p))
    assert len(h1) == 64
    assert h1 == set_hash(str(p))
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_calibration.py -v`
Expected: FAIL mit `ImportError: cannot import name 'sweep_threshold'`

- [ ] **Step 3: Implementierung**

`tools/calibrate_thresholds.py`:

```python
"""Schwellen-Kalibrierung über das Golden-Set (Spec §3.5).

Liest labels.json {"renders": [{"id": str, "good": bool,
"metrics": {"M1": float, "M3": float, "M4": float, "M5": float}}]},
sweept Kandidaten-Schwellen je Metrik und gibt Sensitivität/Spezifität aus.
"""

import argparse
import hashlib
import json
from pathlib import Path


def set_hash(labels_path: str) -> str:
    """sha256 der Label-Datei — Anker für 'calibrated@<set-hash>'."""
    return hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()


def sweep_threshold(values, labels, higher_is_bad, candidates=None):
    """Beste Schwelle nach Youden-Index (Sensitivität + Spezifität − 1).

    values: Metrikwerte, labels: True = gut. higher_is_bad: Wert über der
    Schwelle gilt als schlecht (für M4 übergeben: False).
    """
    if candidates is None:
        lo, hi = min(values), max(values)
        candidates = [lo + (hi - lo) * i / 100 for i in range(1, 100)]
    best = None
    for t in candidates:
        tp = fp = tn = fn = 0
        for v, good in zip(values, labels):
            bad = v > t if higher_is_bad else v < t
            if bad and not good:
                tp += 1
            elif bad and good:
                fp += 1
            elif not bad and good:
                tn += 1
            else:
                fn += 1
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        score = sens + spec - 1.0
        if best is None or score > best["score"]:
            best = {"threshold": t, "sensitivity": sens,
                    "specificity": spec, "score": score}
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="tests/golden/labels.json")
    args = parser.parse_args()

    data = json.loads(Path(args.labels).read_text())
    renders = data.get("renders", [])
    if len(renders) < 20:
        print(f"WARNUNG: nur {len(renders)} gelabelte Renders "
              f"(Minimum 20) — Schwellen bleiben 'assumed'.")
    for metric, higher_is_bad in [("M1", True), ("M3", True), ("M4", False)]:
        values = [r["metrics"][metric] for r in renders]
        labels = [r["good"] for r in renders]
        if not values:
            continue
        best = sweep_threshold(values, labels, higher_is_bad)
        print(f"{metric}: t={best['threshold']:.3f} "
              f"sens={best['sensitivity']:.2f} spec={best['specificity']:.2f}")


if __name__ == "__main__":
    main()
```

`tests/golden/labels.json`:

```json
{
  "version": "golden-set/1",
  "renders": []
}
```

`tests/golden/README.md`:

```markdown
# Golden-Set für die Studio-Schwellenkalibrierung

≥ 20 Referenz-Renders als gut/schlecht labeln (Spec §3.5, §19 Lücke 1).
Je Eintrag in labels.json: id, good (bool), metrics {M1, M3, M4, M5}.
Metrikwerte erzeugt der ProbeRenderer; Labels vergibt der Mensch.
Ohne ≥ 20 Labels bleiben alle Schwellen "assumed" und Reports tragen
calibrated: false.
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_calibration.py -v`
Expected: 2 × PASS

- [ ] **Step 5: Commit**

```bash
git add tools/calibrate_thresholds.py tests/golden/ tests/test_studio_calibration.py
git commit -m "feat(studio): P0 Kalibrier-Harness + Golden-Set-Scaffolding"
```

---

## Abschluss P0 (Definition of Done, Spec §16)

- [ ] `pytest tests/test_studio_metrics.py tests/test_studio_thresholds.py tests/test_studio_calibration.py tests/test_studio_metric_invariance.py -v` — alle PASS (CPU-Tests)
- [ ] `pytest tests/ -v -m gpu` — Invarianz-, Diff-Render-, Luma-Alpha-, Rausch- und Drift-Tests PASS
- [ ] `pytest tests/ -v` — **keine** Regressionen im Bestand (Bit-Identität der Renderer-Defaults)
- [ ] `config/studio_drift.v1.json` existiert; Golden-Set-Scaffolding liegt bereit (Befüllung = Nutzer-Aufgabe)
- [ ] Danach: Plan für **P1** (mask_service, constraints, Resize-Parität) schreiben — konsumiert `MeasureConstraints` und die Blit-Shader-Uniforms aus Task 5
