# Visualizer Studio — P2 Sampling & Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stratifiziertes, ereignisgetriebenes Sampling der Probe-Zeitpunkte (kein Beat-Aliasing) und der analytische Feasibility-Precheck, der unlösbare Fälle vor dem ersten Render abbricht (0 Render-Aufrufe).

**Architecture:** Zwei neue reine CPU-Module in `src/studio/`. `sampling.py` wählt N=18 Sample-Zeitpunkte deterministisch (Seed aus Audio-Content-Hash), stratifiziert in uniform+Jitter / Peaks / Quiet / Quotes. `feasibility.py` prüft Masken-Statistik + Schwellen rein analytisch und liefert `ok` / `layout_fallback` / `infeasible` — ohne jeden Render. Spec: `docs/superpowers/specs/2026-07-27-visualizer-studio-design.md` (studio-spec/2.1, §4, §7, §16 P2).

**Tech Stack:** Python 3.11, numpy, pydantic v2 (Bestand), pytest. Keine neuen Dependencies.

**Voraussetzung (P0+P1, abgeschlossen):** `src/studio/` mit `types.py`, `thresholds.py` (`load_thresholds`), `metrics.py`, `constraints.py`, `mask_service.py`, `probe.py`. Feature-Dict-Schema aus `src/render_common.py:55-86` (`rms`, `onset`, `beat_frames`, `tempo`, `duration`, `fps`, `frame_count`). Visualizer-Keys aus `VISUALIZER_MAP` (`src/gpu_visualizers/__init__.py:51-69`).

## Global Constraints

Werte wörtlich aus der Spec — jeder Task erbt diese Anforderungen implizit:

- Sampling: **N = 18** (Default), Stratifikation **6 uniform+Jitter / 6 Peaks / 3 Quiet / 3 Quotes** (Spec §4).
- Jitter: **±0.5 · Intervall**, deterministisch geseedet aus dem Audio-Content-Hash (Spec §4).
- Seed = Hash der Audio-Features → **deterministisch reproduzierbar, nicht beat-phasenverriegelt** (Spec §4).
- Verify-Kontrollpunkte: **6 zusätzliche** Zufallspunkte (Overfitting-Kontrolle, Spec §4).
- Feasibility: Subjektfläche `area(mask > 0.5) > 0.75` ⇒ Layout-Fallback (periphere Whitelist); Zielkonflikt M3/M5 ⇒ **Abbruch vor jedem Render**; keine Textzone ≥ Mindestfläche ⇒ Scrim erzwingen + `text_zone_alpha = 0.05` (Spec §7).
- Feasibility-Precheck-Budget: **≤ 200 ms**, rein analytisch (Spec §7).
- Audio kürzer als Sample-Bedarf ⇒ Sample-Anzahl adaptiv reduzieren, tatsächliches `n` im Ergebnis (Spec §14).
- Code-Kommentare und Commit-Messages auf Deutsch (AGENTS.md).

## Dateistruktur

| Datei | Verantwortung |
|-------|---------------|
| `src/studio/sampling.py` (neu) | `build_sample_plan`, `verification_extras`, `SamplePlan` |
| `src/studio/feasibility.py` (neu) | `check_feasibility`, `FeasibilityResult`, `FeasibilityConfig`, periphere Whitelist |
| `tests/test_studio_sampling.py` (neu) | Determinismus, Beat-Aliasing-Regression, Stratifikation, Kurz-Audio |
| `tests/test_studio_feasibility.py` (neu) | Layout-Fallback, Infeasible ohne Render, Textzonen-Regel, Whitelist-Validität |

---

### Task 1: Sampling

**Files:**
- Create: `src/studio/sampling.py`
- Test: `tests/test_studio_sampling.py`

**Interfaces:**
- Consumes: Feature-Dict im Schema von `build_features_dict` (`src/render_common.py:67-86`).
- Produces:
  - `@dataclass SamplePlan`: `timestamps: list[float]` (sortiert), `seed: str`, `categories: dict[str, list[float]]` (Keys: `"uniform"`, `"peaks"`, `"quiet"`, `"quotes"`), `n: int`
  - `build_sample_plan(features_dict: dict, n: int = 18, quote_times: list[tuple[float, float]] | None = None) -> SamplePlan`
  - `verification_extras(plan: SamplePlan, duration: float, k: int = 6) -> list[float]` — k zusätzliche deterministische Punkte, disjunkt zu `plan.timestamps`
  - Konsumiert von P3 (Engine/Solver), P5 (Badge).

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_sampling.py`:

```python
"""Tests für das stratifizierte Sampling (Spec §4)."""

import numpy as np
import pytest

from src.studio.sampling import build_sample_plan, verification_extras


def _features(duration=60.0, fps=30, bpm=120.0):
    """Synthetisches Feature-Dict: Beat-Grid bei bpm, Peaks auf den Beats."""
    frame_count = int(duration * fps)
    beat_period = 60.0 / bpm  # 0.5 s bei 120 BPM
    beat_frames = np.arange(0, frame_count, int(beat_period * fps))
    rms = np.full(frame_count, 0.2, dtype=np.float32)
    onset = np.full(frame_count, 0.05, dtype=np.float32)
    rms[beat_frames] = 0.9
    onset[beat_frames] = 1.0
    return {
        "rms": rms,
        "onset": onset,
        "beat_frames": beat_frames,
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "tempo": bpm,
    }


def test_determinismus():
    fd = _features()
    a = build_sample_plan(fd)
    b = build_sample_plan(fd)
    assert a.timestamps == b.timestamps
    assert a.seed == b.seed


def test_standard_18_samples_stratifiziert():
    plan = build_sample_plan(_features())
    assert plan.n == 18
    assert len(plan.categories["uniform"]) == 6
    assert len(plan.categories["peaks"]) == 6
    assert len(plan.categories["quiet"]) == 3
    assert len(plan.categories["quotes"]) == 3  # ohne Quotes: auf Peaks aufgefüllt
    assert plan.timestamps == sorted(plan.timestamps)


def test_kein_beat_aliasing():
    """Regression Spec §4: uniform-Samples dürfen nicht beat-phasenverriegelt sein."""
    fd = _features(duration=60.0, bpm=120.0)
    plan = build_sample_plan(fd)
    beat_period = 0.5
    phases = [t % beat_period for t in plan.categories["uniform"]]
    # Bei Verriegelung wären alle Phasen identisch
    assert len({round(p, 2) for p in phases}) >= 3


def test_peaks_treffen_onset_peaks():
    fd = _features()
    plan = build_sample_plan(fd)
    fps = fd["fps"]
    for t in plan.categories["peaks"]:
        idx = min(int(t * fps), len(fd["onset"]) - 1)
        assert fd["onset"][idx] >= 0.5  # Peak-Zeitpunkt, kein Tal


def test_quotes_stratum_nutzt_quote_zeiten():
    fd = _features()
    quotes = [(10.0, 12.0), (30.0, 32.0), (50.0, 52.0)]
    plan = build_sample_plan(fd, quote_times=quotes)
    for t in plan.categories["quotes"]:
        assert any(start <= t <= end for start, end in quotes)


def test_kurzes_audio_reduziert_adaptiv():
    fd = _features(duration=3.0)
    plan = build_sample_plan(fd)
    assert plan.n < 18
    assert plan.timestamps  # aber nicht leer


def test_verification_extras_disjunkt_und_deterministisch():
    fd = _features()
    plan = build_sample_plan(fd)
    extras_a = verification_extras(plan, fd["duration"])
    extras_b = verification_extras(plan, fd["duration"])
    assert extras_a == extras_b
    assert len(extras_a) == 6
    assert not set(extras_a) & set(plan.timestamps)
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_sampling.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.sampling'`

- [ ] **Step 3: Implementierung**

`src/studio/sampling.py`:

```python
"""Stratifiziertes, ereignisgetriebenes Sampling (Spec §4).

N=18 Samples: 6 uniform+Jitter, 6 Onset-/RMS-Peaks, 3 Quiet-Fenster,
3 Quote-Frames. Seed aus dem Audio-Content-Hash — deterministisch
reproduzierbar, aber nicht beat-phasenverriegelt (1-Frame/s würde
gegen das Beat-Grid aliasen, z.B. 120 BPM = 2 Hz).
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np

DEFAULT_N = 18
VERIFY_EXTRAS = 6


@dataclass
class SamplePlan:
    """Sample-Zeitpunkte inkl. Stratifikation und Seed (Provenance)."""

    timestamps: list[float]
    seed: str
    categories: dict[str, list[float]] = field(default_factory=dict)
    n: int = 0


def _seed_from_features(features_dict: dict) -> str:
    """Deterministischer Seed aus dem Audio-Content."""
    h = hashlib.sha256()
    h.update(np.asarray(features_dict["rms"]).tobytes())
    h.update(str(features_dict["duration"]).encode())
    return h.hexdigest()


def _rng(seed: str, salt: str = "") -> np.random.Generator:
    # hash() wäre pro Prozess randomisiert (PYTHONHASHSEED) — sha256 ist stabil
    salt_int = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "big")
    return np.random.default_rng(int(seed[:16], 16) ^ salt_int)


def _uniform_jitter(duration: float, k: int, rng: np.random.Generator) -> list[float]:
    """k gleichverteilte Punkte mit Jitter ±0.5·Intervall (Spec §4)."""
    interval = duration / k
    points = []
    for i in range(k):
        center = (i + 0.5) * interval
        jitter = float(rng.uniform(-0.5, 0.5)) * interval
        points.append(min(max(center + jitter, 0.0), duration))
    return points


def _top_k_with_separation(values: np.ndarray, fps: float, k: int,
                           min_dist_s: float = 0.5) -> list[float]:
    """Top-k Maxima mit Mindestabstand (gegen Peak-Clustering)."""
    order = np.argsort(values)[::-1]
    picked: list[float] = []
    for idx in order:
        t = float(idx) / fps
        if all(abs(t - p) >= min_dist_s for p in picked):
            picked.append(t)
            if len(picked) == k:
                break
    return picked


def _quiet_windows(rms: np.ndarray, fps: float, k: int,
                   window_s: float = 1.0) -> list[float]:
    """k Fenster minimaler RMS-Energie (Worst-Case Vitalität/Blackframe)."""
    win = max(1, int(window_s * fps))
    if len(rms) < win:
        return [0.0]
    energy = np.convolve(rms, np.ones(win) / win, mode="valid")
    return _top_k_with_separation(-energy, fps, k)


def build_sample_plan(
    features_dict: dict,
    n: int = DEFAULT_N,
    quote_times: list[tuple[float, float]] | None = None,
) -> SamplePlan:
    """Erzeugt den stratifizierten Sample-Plan (Spec §4).

    Bei Audio kürzer als der Sample-Bedarf wird die Anzahl adaptiv
    reduziert (Spec §14) — das tatsächliche n steht im Ergebnis.
    """
    duration = float(features_dict["duration"])
    fps = float(features_dict["fps"])
    rms = np.asarray(features_dict["rms"])
    onset = np.asarray(features_dict["onset"])
    seed = _seed_from_features(features_dict)
    rng = _rng(seed)

    # Stratifikation; bei Kurz-Audio anteilig reduzieren (mind. 1 je Kategorie
    # nur wenn die Dauer es hergibt)
    scale = min(1.0, duration / 18.0)
    k_uniform = max(1, round(6 * scale))
    k_peaks = max(1, round(6 * scale))
    k_quiet = max(1, round(3 * scale))
    k_quotes = max(1, round(3 * scale))

    categories: dict[str, list[float]] = {}
    categories["uniform"] = _uniform_jitter(duration, k_uniform, rng)
    categories["peaks"] = _top_k_with_separation(onset, fps, k_peaks)
    categories["quiet"] = _quiet_windows(rms, fps, k_quiet)

    if quote_times:
        quotes = []
        for start, end in quote_times[: k_quotes]:
            quotes.append(float(rng.uniform(start, max(start, end))))
        # Bei weniger Quote-Slots als k_quotes: mit Peaks auffüllen
        while len(quotes) < k_quotes:
            quotes.append(float(rng.uniform(0.0, duration)))
        categories["quotes"] = quotes
    else:
        # Ohne Quotes: auf Peaks auffüllen (Spec §4)
        categories["quotes"] = _top_k_with_separation(
            rms, fps, k_quotes, min_dist_s=1.0
        )

    timestamps = sorted({round(t, 3) for cat in categories.values() for t in cat})
    return SamplePlan(timestamps=timestamps, seed=seed,
                      categories=categories, n=len(timestamps))


def verification_extras(plan: SamplePlan, duration: float,
                        k: int = VERIFY_EXTRAS) -> list[float]:
    """k zusätzliche Kontrollpunkte für die Verify-Phase (Spec §4).

    Disjunkt zu den Probe-Punkten — Overfitting-Kontrolle: der Solver
    darf nicht nur die Zeitpunkte fixen, die er sieht.
    """
    rng = _rng(plan.seed, salt="verify")
    extras: list[float] = []
    while len(extras) < k:
        t = round(float(rng.uniform(0.0, duration)), 3)
        if t not in plan.timestamps and t not in extras:
            extras.append(t)
    return sorted(extras)
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_sampling.py -v`
Expected: 7 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/sampling.py tests/test_studio_sampling.py
git commit -m "feat(studio): P2 Sampling — stratifiziert, gejittert, beat-entkoppelt"
```

---

### Task 2: Feasibility-Precheck

**Files:**
- Create: `src/studio/feasibility.py`
- Test: `tests/test_studio_feasibility.py`

**Interfaces:**
- Consumes: nichts aus Task 1. Optional `load_thresholds` (P0) für Default-Schwellen.
- Produces:
  - `PERIPHERAL_VISUALS: tuple[str, ...]` — periphere Whitelist für den Layout-Fallback
  - `@dataclass FeasibilityConfig`: `subject_area_limit: float = 0.75`, `text_zone_min_area: float = 0.04`, `grid: int = 32`
  - `@dataclass FeasibilityResult`: `status: str` (`"ok"` | `"layout_fallback"` | `"infeasible"`), `should_render: bool`, `visualizer_whitelist: list[str] | None`, `actions: list[str]`, `reason: str`
  - `check_feasibility(mask: np.ndarray | None, requires_text_zone: bool = False, m3_active: bool = True, config: FeasibilityConfig | None = None) -> FeasibilityResult`
  - Konsumiert von P3 (Engine: vor dem ersten Probe-Render; bei `infeasible` Abbruch, bei `layout_fallback` Whitelist an die PresetFactory).

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_feasibility.py`:

```python
"""Tests für den Feasibility-Precheck (Spec §7, §15)."""

import numpy as np
import pytest

from src.studio.feasibility import (
    PERIPHERAL_VISUALS,
    FeasibilityResult,
    check_feasibility,
)


def test_ohne_maske_ok():
    result = check_feasibility(None)
    assert result.status == "ok"
    assert result.should_render is True
    assert result.visualizer_whitelist is None


def test_kleine_subjektflaeche_ok():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[:16, :16] = 1.0  # 6 % Subjekt
    result = check_feasibility(mask)
    assert result.status == "ok"


def test_grosse_subjektflaeche_layout_fallback():
    mask = np.ones((64, 64), dtype=np.float32)
    mask[:8, :8] = 0.0  # ~98 % Subjekt (> 0.75)
    result = check_feasibility(mask)
    assert result.status == "layout_fallback"
    assert result.should_render is True
    assert result.visualizer_whitelist == list(PERIPHERAL_VISUALS)


def test_zielkonflikt_infeasible_ohne_render():
    """Spec §15: unlösbarer Fall bricht mit 0 Render-Aufrufen ab."""
    mask = np.ones((64, 64), dtype=np.float32)  # 100 % Subjekt
    render_calls = []

    result = check_feasibility(mask, requires_text_zone=True)
    assert result.status == "infeasible"
    assert result.should_render is False

    # Treiber-Logik: nur rendern wenn should_render — Zähler bleibt 0
    if result.should_render:
        render_calls.append("render")
    assert render_calls == []


def test_keine_textzone_erzwingt_scrim():
    mask = np.ones((64, 64), dtype=np.float32)
    mask[0:6, 0:6] = 0.0  # winzige freie Ecke < Mindestfläche
    result = check_feasibility(mask, requires_text_zone=True, m3_active=False)
    assert "scrim" in " ".join(result.actions).lower()
    assert any("0.05" in a for a in result.actions)


def test_periphere_whitelist_ist_valide():
    from src.gpu_visualizers import VISUALIZER_MAP
    for key in PERIPHERAL_VISUALS:
        assert key in VISUALIZER_MAP, f"Unbekannter Visualizer: {key}"
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_feasibility.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.feasibility'`

- [ ] **Step 3: Implementierung**

`src/studio/feasibility.py`:

```python
"""Feasibility-Precheck (Spec §7).

Läuft VOR dem ersten Probe-Render, rein analytisch auf Masken-Statistik
(Budget ≤ 200 ms, kein GL). Unlösbare Fälle brechen hier ab — v1 hätte
bis zu 4 Full-Renders gerendert, die nie bestehen konnten.
"""

from dataclasses import dataclass, field

import numpy as np

# Peripher-geometrische Visualizer (Rahmen/Rand/Ecken) für den
# Layout-Fallback — Keys aus VISUALIZER_MAP (Spec §7).
PERIPHERAL_VISUALS: tuple[str, ...] = (
    "spectrum_bars",
    "neon_wave_circle",
    "neon_oscilloscope",
    "typographic",
)


@dataclass
class FeasibilityConfig:
    """Grenzen des Prechecks (Defaults aus Spec §7)."""

    subject_area_limit: float = 0.75
    text_zone_min_area: float = 0.04
    grid: int = 32  # Raster für die Freiflächen-Geometrie


@dataclass
class FeasibilityResult:
    """Befund des Prechecks."""

    status: str  # "ok" | "layout_fallback" | "infeasible"
    should_render: bool
    visualizer_whitelist: list[str] | None = None
    actions: list[str] = field(default_factory=list)
    reason: str = ""


def _max_rect_in_histogram(heights: np.ndarray) -> int:
    """Größtes Rechteck im Histogramm (Standard-Stack-Algorithmus)."""
    stack: list[int] = []
    best = 0
    n = len(heights)
    for i in range(n + 1):
        cur = heights[i] if i < n else 0
        while stack and heights[stack[-1]] > cur:
            h = heights[stack.pop()]
            left = stack[-1] + 1 if stack else 0
            best = max(best, int(h) * (i - left))
        stack.append(i)
    return best


def _largest_free_rect_area(mask: np.ndarray, grid: int) -> float:
    """Relativer Flächenanteil des größten freien Rechtecks.

    Raster-Approximation (grid x grid): Zellen mit Masken-Mittel > 0.5
    gelten als belegt; größtes leeres Rechteck via Histogramm-Methode.
    """
    h, w = mask.shape
    cell_h, cell_w = max(1, h // grid), max(1, w // grid)
    free = np.ones((grid, grid), dtype=bool)
    for gy in range(grid):
        for gx in range(grid):
            cell = mask[gy * cell_h:(gy + 1) * cell_h,
                        gx * cell_w:(gx + 1) * cell_w]
            if cell.size and float(cell.mean()) > 0.5:
                free[gy, gx] = False
    # Größtes freies Rechteck über zeilenweise Histogramme
    heights = np.zeros(grid, dtype=int)
    best = 0
    for row in range(grid):
        heights = np.where(free[row], heights + 1, 0)
        best = max(best, _max_rect_in_histogram(heights))
    return best / float(grid * grid)


def check_feasibility(
    mask: np.ndarray | None,
    requires_text_zone: bool = False,
    m3_active: bool = True,
    config: FeasibilityConfig | None = None,
) -> FeasibilityResult:
    """Analytischer Precheck vor jedem Render (Spec §7).

    - Subjektfläche > Limit: Layout-Fallback (periphere Whitelist)
    - 100 % Subjekt + Textpflicht: Zielkonflikt -> infeasible (0 Renders)
    - Keine Textzone >= Mindestfläche: Scrim erzwingen, text_zone_alpha 0.05
    """
    cfg = config or FeasibilityConfig()
    if mask is None:
        return FeasibilityResult("ok", should_render=True)

    mask = np.asarray(mask, dtype=np.float32)
    subject_area = float((mask > 0.5).mean())
    actions: list[str] = []

    # Textzonen-Prüfung (geometrisch, auf der Subjekt-Maske)
    if requires_text_zone:
        free_rect = _largest_free_rect_area(mask, cfg.grid)
        if free_rect < cfg.text_zone_min_area:
            actions.append(
                "scrim erzwingen: keine Textzone >= "
                f"{cfg.text_zone_min_area:.0%} (größte freie Zone "
                f"{free_rect:.1%}); text_zone_alpha=0.05"
            )

    if subject_area > cfg.subject_area_limit and m3_active:
        if subject_area >= 0.999 and requires_text_zone:
            # Zielkonflikt: Subjekt überall, aber Textpflicht —
            # geometrisch unvereinbar, Abbruch VOR jedem Render (Spec §7)
            return FeasibilityResult(
                "infeasible",
                should_render=False,
                actions=actions,
                reason=(
                    f"Subjektfläche {subject_area:.0%} bei aktivem M3 und "
                    "Textpflicht: keine freie Geometrie für Visualizer "
                    "und Textzone."
                ),
            )
        return FeasibilityResult(
            "layout_fallback",
            should_render=True,
            visualizer_whitelist=list(PERIPHERAL_VISUALS),
            actions=actions + [
                f"Subjektfläche {subject_area:.0%} > {cfg.subject_area_limit:.0%}: "
                "Visualizer auf periphere Whitelist eingeschränkt"
            ],
        )

    return FeasibilityResult("ok", should_render=True, actions=actions)
```

Hinweis für den Implementierer: `_largest_free_rect_area` ist bewusst eine einfache Raster-Approximation (32×32, Histogramm-Methode) — Präzision ist hier zweitrangig, das 200-ms-Budget ist der harte Constraint. Die Histogramm-Schleife ist die Standard-Lösung für „größtes leeres Rechteck"; bei Abweichungen bitte die beiden Textzonen-Tests als Maßstab nehmen.

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_feasibility.py -v`
Expected: 6 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/feasibility.py tests/test_studio_feasibility.py
git commit -m "feat(studio): P2 Feasibility-Precheck — Layout-Fallback, Infeasible vor Render"
```

---

## Abschluss P2 (Definition of Done, Spec §16)

- [ ] `pytest tests/test_studio_sampling.py tests/test_studio_feasibility.py -v` — alle PASS
- [ ] Beat-Aliasing-Regressionstest grün (`test_kein_beat_aliasing`)
- [ ] Unlösbarer Fall bricht mit 0 Renders ab (`test_zielkonflikt_infeasible_ohne_render`)
- [ ] `pytest tests/ -q` — keine Regressionen im Bestand
- [ ] Danach: Plan für **P3** (solver.py, engine.py Probe/Commit/Verify, provenance.py) schreiben — konsumiert `SamplePlan`, `FeasibilityResult`, `ConstraintSet`, `ProbeRenderer`
