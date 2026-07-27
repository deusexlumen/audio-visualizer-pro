# Visualizer Studio — P1 Masken-Service & Constraints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subjekt-Masken-Service mit Fallback-Kette und Cache, das Laufzeit-ConstraintSet und die End-to-End-Integration der Maske in den ProbeRenderer — sodass der Alpha-Cap und die Subjekt-Freistellung messbar wirken, auch bei Composite-Stacks.

**Architecture:** Erweitert das in P0 gebaute `src/studio/`-Paket. `mask_service.py` erzeugt Subjekt-Masken im **Quellbildraum** (rembg → OpenCV → Zentrums-Gauß, alles optional bis auf den Gauß), cached als NPZ. `constraints.py` definiert das Laufzeit-`ConstraintSet` (Pydantic) inkl. Klemm-Logik mit Warnungen. Der `ProbeRenderer` bekommt einen optionalen `subject_mask`-Parameter, der über die P0-Blit-Shader-Uniforms wirkt. Spec: `docs/superpowers/specs/2026-07-27-visualizer-studio-design.md` (studio-spec/2.1, §6, §16 P1).

**Tech Stack:** Python 3.11, numpy, Pillow, moderngl, pydantic v2, pytest. Keine neuen harten Dependencies: `rembg` und `opencv-python` bleiben optionale Extras; Tests mocken ihr Fehlen/Vorhandensein.

**Voraussetzung (P0, abgeschlossen):** `src/studio/` mit `types.py` (`MeasureConstraints` inkl. `subject_strength`), `metrics.py`, `thresholds.py`, `probe.py`; Blit-Shader mit Uniforms `u_subject_mask`, `u_resolution`, `u_viz_alpha_cap`, `u_viz_alpha_from_luma`, `u_luma_knee_lo/hi`, `u_subject_strength`; `_blit_viz_to_fbo(..., subject_strength=0.0, subject_mask=None)`.

## Global Constraints

Werte wörtlich aus der Spec — jeder Task erbt diese Anforderungen implizit:

- Maske im **Quellbildraum** speichern; Cache-Key = `sha256(Bilddatei) + provider_id + model_hash + service_version` (Spec §6.2).
- Resize der Maske: **AREA (Downscale) / BILINEAR (Upscale)**, danach Clamp auf [0, 1]; **geometrisch** identisch zum Hintergrund-Pfad (harter Resize auf Zielgröße, `gpu_renderer.py:698-721`); **nicht-negativer Kernel** — keine LANCZOS-Überschwinger (Spec §6.2).
- Alpha-Modulation: `alpha_final = alpha_eff · (1 − subject_strength · mask)`, `subject_strength` Default **0.8** (Spec §6.2).
- `max_overlay_alpha` Default **0.6**; Config-Werte oberhalb der Caps werden **geklemmt + geloggt**, nie verworfen (Spec §6.1).
- `text_zone_alpha` Default **0.15** — Feld existiert im ConstraintSet, Zonen-Ableitung ist spätere Phase (Spec §6.3).
- Fallback-Kette: rembg/u2net → OpenCV GrabCut → Zentrums-Gauß; jeder Fallback: Warnung + Provider-Eintrag (Spec §6.2).
- Code-Kommentare und Commit-Messages auf Deutsch (AGENTS.md).
- Kein Video-Hintergrund-Handling hier — Degradationspfad kommt mit der Engine (P3).
- `src/analyzer.py` wird nicht angefasst.

## Dateistruktur

| Datei | Verantwortung |
|-------|---------------|
| `src/studio/constraints.py` (neu) | `ConstraintSet` (Pydantic): Caps, Klemm-Logik, Mapping auf `MeasureConstraints` |
| `src/studio/mask_service.py` (neu) | Subjekt-Maske: Provider-Kette, NPZ-Cache, Resize-Parität |
| `src/studio/probe.py` (Modify) | `subject_mask`-Parameter in `render_frame`/`render_pair` |
| `tests/test_studio_constraints.py` (neu) | ConstraintSet-Tests |
| `tests/test_studio_mask_service.py` (neu) | Provider-Kette, Cache, Resize-Parität (CPU) |
| `tests/test_studio_mask_integration.py` (neu) | End-to-End GPU: Maske + Cap + Composite-Stack |

---

### Task 1: ConstraintSet

**Files:**
- Create: `src/studio/constraints.py`
- Test: `tests/test_studio_constraints.py`

**Interfaces:**
- Consumes: `MeasureConstraints` (`src/studio/types.py`, P0).
- Produces:
  - `class ConstraintSet(BaseModel)`: Felder `max_overlay_alpha: float = 0.6`, `alpha_from_luma: bool = True`, `luma_knee_lo: float = 0.02`, `luma_knee_hi: float = 0.25`, `subject_strength: float = 0.8`, `text_zone_alpha: float = 0.15`, `max_bloom_intensity: float = 1.0`, `max_film_grain: float = 0.5`, `grain_free: bool = False`
  - `ConstraintSet.to_measure_constraints() -> MeasureConstraints`
  - `ConstraintSet.clamp_postprocess(pp: dict) -> tuple[dict, list[str]]` — gibt (geklemmtes Dict, Warnungen) zurück
  - Konsumiert von Task 4 und später P2–P4 (Engine, Solver, Profile).

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_constraints.py`:

```python
"""Tests für das Laufzeit-ConstraintSet (Spec §6.1)."""

import pytest

from src.studio.constraints import ConstraintSet


def test_defaults_aus_spec():
    cs = ConstraintSet()
    assert cs.max_overlay_alpha == 0.6
    assert cs.subject_strength == 0.8
    assert cs.text_zone_alpha == 0.15
    assert cs.alpha_from_luma is True
    assert cs.luma_knee_lo < cs.luma_knee_hi


def test_to_measure_constraints_mapping():
    cs = ConstraintSet(max_overlay_alpha=0.4, subject_strength=0.5)
    mc = cs.to_measure_constraints()
    assert mc.alpha_cap == 0.4
    assert mc.alpha_from_luma is True
    assert mc.subject_strength == 0.5
    assert mc.luma_knee_lo == cs.luma_knee_lo


def test_clamp_postprocess_mit_warnung():
    cs = ConstraintSet(max_bloom_intensity=1.0, max_film_grain=0.5)
    pp = {"bloom_intensity": 1.8, "film_grain": 0.9, "contrast": 1.1}
    clamped, warnings = cs.clamp_postprocess(pp)
    assert clamped["bloom_intensity"] == 1.0
    assert clamped["film_grain"] == 0.5
    assert clamped["contrast"] == 1.1  # unberührt
    assert len(warnings) == 2
    assert any("bloom_intensity" in w for w in warnings)


def test_clamp_postprocess_unveraendert_keine_warnung():
    cs = ConstraintSet()
    clamped, warnings = cs.clamp_postprocess({"bloom_intensity": 0.6})
    assert clamped["bloom_intensity"] == 0.6
    assert warnings == []
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_constraints.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.constraints'`

- [ ] **Step 3: Implementierung**

`src/studio/constraints.py`:

```python
"""Laufzeit-ConstraintSet des Visualizer Studio (Spec §6.1).

Kapselt die erzwungenen Render-Regeln (Alpha-Cap, Luma-Ableitung,
Subjekt-Stärke, Post-FX-Budgets) und bildet sie auf die Messebene ab.
Config-Werte oberhalb der Caps werden geklemmt + geloggt, nie verworfen.
"""

from pydantic import BaseModel, Field

from .types import MeasureConstraints


class ConstraintSet(BaseModel):
    """Render-Constraints für Probe, Preview und Commit."""

    max_overlay_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    alpha_from_luma: bool = True
    luma_knee_lo: float = 0.02
    luma_knee_hi: float = 0.25
    subject_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    text_zone_alpha: float = Field(default=0.15, ge=0.0, le=1.0)
    max_bloom_intensity: float = Field(default=1.0, ge=0.0)
    max_film_grain: float = Field(default=0.5, ge=0.0)
    grain_free: bool = False

    def to_measure_constraints(self) -> MeasureConstraints:
        """Bildet das ConstraintSet auf die P0-Messebene ab."""
        return MeasureConstraints(
            alpha_cap=self.max_overlay_alpha,
            alpha_from_luma=self.alpha_from_luma,
            luma_knee_lo=self.luma_knee_lo,
            luma_knee_hi=self.luma_knee_hi,
            subject_strength=self.subject_strength,
            grain_free=self.grain_free,
        )

    def clamp_postprocess(self, pp: dict) -> tuple[dict, list[str]]:
        """Klemmt Post-FX-Werte auf die Budgets; gibt Warnungen zurück."""
        clamped = dict(pp or {})
        warnings: list[str] = []
        for key, cap in (
            ("bloom_intensity", self.max_bloom_intensity),
            ("film_grain", self.max_film_grain),
        ):
            if key in clamped and clamped[key] > cap:
                warnings.append(
                    f"{key}={clamped[key]} über Budget, geklemmt auf {cap}"
                )
                clamped[key] = cap
        return clamped, warnings
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_constraints.py -v`
Expected: 4 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/constraints.py tests/test_studio_constraints.py
git commit -m "feat(studio): P1 ConstraintSet — Caps, Klemm-Logik, Messebenen-Mapping"
```

---

### Task 2: MaskService Kern (Provider-Kette + Cache)

**Files:**
- Create: `src/studio/mask_service.py`
- Test: `tests/test_studio_mask_service.py`

**Interfaces:**
- Consumes: nichts aus anderen Tasks.
- Produces:
  - `SERVICE_VERSION = "mask-service/1"`
  - `@dataclass MaskResult`: `mask: np.ndarray` (float32, H×W, Quellbildraum), `provider: str`, `cache_hit: bool`, `warnings: list[str]`
  - `get_subject_mask(image_path: str, cache_dir: str = ".cache/subject_masks") -> MaskResult`
  - `resize_mask(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray` (Task 3 nutzt/testet sie, Definition hier)
  - Konsumiert von Task 3/4, P3 (Engine), P5 (GUI-Hintergrundtask).

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_mask_service.py`:

```python
"""Tests für den Subjekt-Masken-Service (Spec §6.2)."""

import numpy as np
import pytest
from PIL import Image

from src.studio import mask_service
from src.studio.mask_service import get_subject_mask, resize_mask


def _test_image(path, size=(64, 48)):
    Image.new("RGB", size, (120, 80, 40)).save(path)


def test_center_gauss_eigenschaften():
    mask = mask_service._center_gauss(64, 48)
    assert mask.shape == (48, 64)
    assert mask.dtype == np.float32
    # Zentrum deutlich höher als Rand, Werte in [0, 1]
    assert mask[24, 32] > 0.9
    assert mask[0, 0] < 0.1
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_fallback_auf_center_gauss_ohne_provider(tmp_path, monkeypatch):
    # rembg und cv2 als nicht installiert simulieren
    monkeypatch.setattr(mask_service, "_try_rembg", lambda img: None)
    monkeypatch.setattr(mask_service, "_try_opencv", lambda img: None)
    img_path = tmp_path / "bg.png"
    _test_image(img_path)
    result = get_subject_mask(str(img_path), cache_dir=str(tmp_path / "cache"))
    assert result.provider == "center_gauss"
    assert result.mask.shape == (48, 64)  # Quellbildraum
    assert any("rembg" in w or "Fallback" in w for w in result.warnings)


def test_cache_hit_bei_zweitem_aufruf(tmp_path, monkeypatch):
    monkeypatch.setattr(mask_service, "_try_rembg", lambda img: None)
    monkeypatch.setattr(mask_service, "_try_opencv", lambda img: None)
    img_path = tmp_path / "bg.png"
    _test_image(img_path)
    cache = tmp_path / "cache"
    first = get_subject_mask(str(img_path), cache_dir=str(cache))
    assert first.cache_hit is False
    second = get_subject_mask(str(img_path), cache_dir=str(cache))
    assert second.cache_hit is True
    np.testing.assert_array_equal(first.mask, second.mask)


def test_cache_schluessel_reagiert_auf_bild(tmp_path, monkeypatch):
    monkeypatch.setattr(mask_service, "_try_rembg", lambda img: None)
    monkeypatch.setattr(mask_service, "_try_opencv", lambda img: None)
    cache = tmp_path / "cache"
    img_a = tmp_path / "a.png"
    _test_image(img_a)
    get_subject_mask(str(img_a), cache_dir=str(cache))
    img_b = tmp_path / "b.png"
    _test_image(img_b, size=(32, 32))  # anderes Bild => anderer Key
    result_b = get_subject_mask(str(img_b), cache_dir=str(cache))
    assert result_b.cache_hit is False


def test_resize_mask_bleibt_in_wertebereich():
    # Extremes Schachbrett: nicht-negative Kernels bleiben in [0, 1]
    mask = np.indices((40, 40)).sum(axis=0) % 2
    mask = mask.astype(np.float32)
    out = resize_mask(mask, 100, 100)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_mask_service.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.mask_service'`

- [ ] **Step 3: Implementierung**

`src/studio/mask_service.py`:

```python
"""Subjekt-Masken-Service (Spec §6.2).

Erzeugt pro Hintergrundbild eine Salienz-Maske im Quellbildraum
(float32, HxW, [0,1]; 1 = Subjekt) und cached sie als NPZ.
Fallback-Kette: rembg/u2net -> OpenCV GrabCut -> Zentrums-Gauß.
Cache-Key = sha256(Bilddatei) + provider_id + model_hash + service_version.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

SERVICE_VERSION = "mask-service/1"
DEFAULT_CACHE_DIR = ".cache/subject_masks"


@dataclass
class MaskResult:
    """Ergebnis der Maskenerzeugung inkl. Provenance."""

    mask: np.ndarray
    provider: str
    cache_hit: bool
    warnings: list[str] = field(default_factory=list)


def _center_gauss(w: int, h: int, sigma: float = 0.35) -> np.ndarray:
    """Notfallback: Gaußsche Zentrums-Gewichtung (immer verfügbar)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dist2 = ((x - cx) / (sigma * w)) ** 2 + ((y - cy) / (sigma * h)) ** 2
    return np.exp(-0.5 * dist2).astype(np.float32)


def _try_rembg(img: Image.Image) -> np.ndarray | None:
    """ML-Segmentierung via rembg (optionale Dependency)."""
    try:
        from rembg import remove
    except ImportError:
        return None
    out = remove(img)  # RGBA, Alpha = Vordergrund
    alpha = np.asarray(out)[..., 3].astype(np.float32) / 255.0
    return alpha


def _try_opencv(img: Image.Image) -> np.ndarray | None:
    """GrabCut-Segmentierung via OpenCV (optionale Dependency)."""
    try:
        import cv2
    except ImportError:
        return None
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    grab_mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(arr, grab_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.isin(grab_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    return fg.astype(np.float32)


def _model_hash() -> str:
    """Hash des rembg-Modells, falls vorhanden (Provenance, Spec §6.2)."""
    model = Path.home() / ".u2net" / "u2net.onnx"
    if model.exists():
        return hashlib.sha256(model.read_bytes()).hexdigest()[:12]
    return "none"


def _cache_key(image_bytes: bytes, provider: str) -> str:
    raw = hashlib.sha256(image_bytes).hexdigest()[:16]
    return f"{raw}_{provider}_{_model_hash()}_{SERVICE_VERSION}.npz"


def get_subject_mask(
    image_path: str, cache_dir: str = DEFAULT_CACHE_DIR
) -> MaskResult:
    """Liefert die Subjekt-Maske eines Hintergrundbilds (gecached).

    Provider-Kette: rembg -> OpenCV -> Zentrums-Gauß. Jeder Fallback
    erzeugt eine Warnung; der genutzte Provider steht im Ergebnis.
    """
    p = Path(image_path)
    image_bytes = p.read_bytes()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    img = Image.open(p).convert("RGB")
    warnings: list[str] = []

    mask, provider = None, None
    for name, fn in (("rembg:u2net", _try_rembg), ("opencv:grabcut", _try_opencv)):
        candidate = fn(img)
        if candidate is not None:
            mask, provider = candidate, name
            break
        warnings.append(f"Provider {name} nicht verfügbar — Fallback")
    if mask is None:
        mask = _center_gauss(img.width, img.height)
        provider = "center_gauss"

    cache_file = cache / _cache_key(image_bytes, provider)
    if cache_file.exists():
        loaded = np.load(cache_file)["mask"]
        return MaskResult(loaded, provider, cache_hit=True, warnings=warnings)

    np.savez(cache_file, mask=mask)
    return MaskResult(mask.astype(np.float32), provider, cache_hit=False,
                      warnings=warnings)


def resize_mask(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Skaliert die Maske geometrisch identisch zum Hintergrund-Pfad.

    Nicht-negativer Kernel (AREA bei Downscale, BILINEAR bei Upscale),
    danach Clamp auf [0, 1] — keine LANCZOS-Überschwinger an Maskenkanten
    (Spec §6.2).
    """
    img = Image.fromarray(mask.astype(np.float32), mode="F")
    downscale = target_w < mask.shape[1] or target_h < mask.shape[0]
    kernel = Image.Resampling.BOX if downscale else Image.Resampling.BILINEAR
    out = np.asarray(img.resize((target_w, target_h), kernel), dtype=np.float32)
    return np.clip(out, 0.0, 1.0)
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_mask_service.py -v`
Expected: 5 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/mask_service.py tests/test_studio_mask_service.py
git commit -m "feat(studio): P1 MaskService — Provider-Kette, NPZ-Cache, Resize-Parität"
```

---

### Task 3: Geometrische Parität und Kanten-Härtung (Test-Task)

**Files:**
- Test: `tests/test_studio_mask_service.py` (anfügen)

**Interfaces:**
- Consumes: `resize_mask`, `_center_gauss` (Task 2).
- Produces: nichts Neues — Verifikations-Task (Spec §16 P1 DoD: „Maske überlebt Paritäts- und Kantentest").

- [ ] **Step 1: Tests anfügen**

An `tests/test_studio_mask_service.py` anhängen:

```python
def test_resize_geometrische_paritaet_zum_hintergrund():
    # Der Hintergrund-Pfad resized hart auf Zielgröße (LANCZOS). Die Maske
    # muss auf dieselbe Zielgröße/Aspect kommen — Geometrie identisch,
    # Filterwahl egal (Spec §6.2).
    from src.studio.mask_service import resize_mask
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[12:36, 16:48] = 1.0  # zentrales Rechteck (25%-75% je Achse)
    out = resize_mask(mask, 160, 90)
    assert out.shape == (90, 160)
    # Rechteck liegt an denselben relativen Koordinaten
    assert out[45, 80] > 0.9           # Zentrum bleibt Subjekt
    assert out[5, 5] < 0.1             # Ecke bleibt frei
    # Flächenanteil grob erhalten (25 % ± 3 %)
    assert (out > 0.5).mean() == pytest.approx(0.25, abs=0.03)


def test_resize_keine_ueberschwinger_an_kanten():
    # LANCZOS würde an harten Kanten über 1.0/unter 0.0 schießen;
    # der nicht-negative Kernel darf das nicht (Kanten-Härtung, Spec §6.2).
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[:, 32:] = 1.0  # harte vertikale Kante
    out = resize_mask(mask, 160, 90)
    assert out.max() <= 1.0
    assert out.min() >= 0.0
```

- [ ] **Step 2: Erfolg verifizieren**

Run: `pytest tests/test_studio_mask_service.py -v`
Expected: 7 × PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_studio_mask_service.py
git commit -m "test(studio): P1 Masken-Parität und Kanten-Härtung"
```

---

### Task 4: Probe-Integration + End-to-End (GPU)

**Files:**
- Modify: `src/studio/probe.py` (`render_frame`, `render_pair`)
- Test: `tests/test_studio_mask_integration.py`

**Interfaces:**
- Consumes: `MeasureConstraints` (P0), `ConstraintSet` (Task 1), `resize_mask` (Task 2), `_blit_viz_to_fbo(..., subject_strength, subject_mask)` (P0), `make_recipe_visualizer_class` (`src/gpu_visualizers/composite.py:237`).
- Produces:
  - `ProbeRenderer.render_frame(..., subject_mask: np.ndarray | None = None)` und `render_pair(..., subject_mask=None)` — neue optionale Keyword-Args, sonst unverändert
  - Konsumiert von P2–P5 (Engine, Solver, GUI-Badge).

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_mask_integration.py`:

```python
"""End-to-End: Subjekt-Maske + Cap wirken messbar (Spec §16 P1 DoD)."""

import numpy as np
import pytest

from src.studio.constraints import ConstraintSet
from src.studio.probe import ProbeRenderer

pytestmark = pytest.mark.gpu


@pytest.fixture
def probe():
    p = ProbeRenderer(width=160, height=90, fps=30)
    yield p
    p.release()


def _features_dict(dummy_audio_features):
    from src.render_common import build_features_dict
    return build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )


def _composite_viz(probe):
    """Composite-Stack (alpha=1.0 überall) — der C14-Hauptfall."""
    from src.gpu_visualizers.composite import make_recipe_visualizer_class
    recipe = {
        "name": "test_glow",
        "layers": [
            {"block": "core_glow", "blend": "add",
             "params": {"size": 0.22, "intensity": 1.6}, "mappings": []}
        ],
    }
    cls = make_recipe_visualizer_class(recipe)
    return cls(probe.ctx, 160, 90)


def test_maske_stellt_subjektregion_frei(probe, dummy_audio_features):
    fd = _features_dict(dummy_audio_features)
    viz = _composite_viz(probe)
    mask = np.zeros((90, 160), dtype=np.float32)
    mask[:, :80] = 1.0  # Subjekt links
    cs = ConstraintSet(max_overlay_alpha=1.0, subject_strength=1.0)
    a, b = probe.render_pair(
        viz, fd, 0.5, None, {}, cs.to_measure_constraints(),
        subject_mask=mask,
    )
    contrib = probe.contribution_map(a, b)
    h, w = contrib.shape
    assert contrib[:, : w // 2].mean() < 0.02   # Subjektregion frei
    assert contrib[:, w // 2 :].mean() > 0.05   # Rest aktiv


def test_cap_wirkt_post_fx_bei_composite_stack(probe, dummy_audio_features):
    fd = _features_dict(dummy_audio_features)
    pp = {"bloom_intensity": 1.5, "bloom_threshold": 0.5}
    viz = _composite_viz(probe)

    def energy(cap):
        cs = ConstraintSet(max_overlay_alpha=cap)
        a, b = probe.render_pair(
            viz, fd, 0.5, None, pp, cs.to_measure_constraints()
        )
        return float(probe.contribution_map(a, b).mean())

    # Cap regelt auch bei vollem Bloom und alpha=1.0-Stack (Spec §6.4)
    assert energy(1.0) > energy(0.3) > 0.0


def test_keine_maske_kein_unterschied_zu_p0(probe, dummy_audio_features):
    fd = _features_dict(dummy_audio_features)
    viz = _composite_viz(probe)
    cs = ConstraintSet(max_overlay_alpha=0.0)
    a, b = probe.render_pair(viz, fd, 0.5, None, {}, cs.to_measure_constraints())
    contrib = probe.contribution_map(a, b)
    assert float(contrib.max()) < 1e-3
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_mask_integration.py -v`
Expected: FAIL mit `TypeError: render_pair() got an unexpected keyword argument 'subject_mask'`

- [ ] **Step 3: Implementierung**

In `src/studio/probe.py`:

1. Import ergänzen: `from .mask_service import resize_mask`
2. `render_frame` — Signatur um `subject_mask: np.ndarray | None = None` erweitern und im Blit-Abschnitt die Maske injizieren:

```python
    def render_frame(
        self, viz, features_dict, time_s, bg_texture,
        postprocess: dict, constraints: MeasureConstraints,
        subject_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Rendert ein Frame; bei alpha_cap=0 wird der Visualizer-Pass
        übersprungen (Blit-Alpha 0 — reine Ersparnis, Spec §3.2.2).

        subject_mask: optionale Subjekt-Maske (float, HxW, Quellraum);
        wird auf Framebuffer-Größe skaliert (resize_mask, nicht-negativer
        Kernel) und über die Blit-Uniforms injiziert.
        """
        r = self._r
        mask_tex = None
        try:
            r.fbo.use()
            r.ctx.clear(0.0, 0.0, 0.0)
            if bg_texture is not None:
                r._render_background(bg_texture, 1.0, 0.0)
            if constraints.alpha_cap > 0.0:
                r._render_viz_into(viz, r.viz_fbo, features_dict, time_s)
                if subject_mask is not None:
                    scaled = resize_mask(subject_mask, r.width, r.height)
                    mask_tex = r.ctx.texture(
                        (r.width, r.height), 1,
                        scaled.astype("f4").tobytes(), dtype="f4",
                    )
                r.fbo.use()
                r._blit_viz_to_fbo(
                    r.viz_fbo.color_attachments[0],
                    alpha_cap=constraints.alpha_cap,
                    alpha_from_luma=constraints.alpha_from_luma,
                    luma_knee_lo=constraints.luma_knee_lo,
                    luma_knee_hi=constraints.luma_knee_hi,
                    subject_strength=constraints.subject_strength,
                    subject_mask=mask_tex,
                )
            # ... Rest (Post-FX, Readback) unverändert wie bisher ...
        finally:
            if mask_tex is not None:
                mask_tex.release()
```

(Der übrige Methodenkörper ab `pp = dict(postprocess or {})` bleibt exakt wie in P0 — nur in den `try`-Block einbetten.)

3. `render_pair` — Parameter durchreichen:

```python
    def render_pair(self, viz, features_dict, time_s, bg_texture,
                    postprocess, constraints,
                    subject_mask=None) -> tuple[np.ndarray, np.ndarray]:
        """(A, B): B mit alpha_cap=0, identisches u_time für beide."""
        a = self.render_frame(viz, features_dict, time_s, bg_texture,
                              postprocess, constraints,
                              subject_mask=subject_mask)
        b_constraints = MeasureConstraints(
            alpha_cap=0.0,
            alpha_from_luma=constraints.alpha_from_luma,
            grain_free=constraints.grain_free,
        )
        b = self.render_frame(viz, features_dict, time_s, bg_texture,
                              postprocess, b_constraints)
        return a, b
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_mask_integration.py -v`
Expected: 3 × PASS

Regression: `pytest tests/test_studio_diff_render.py tests/test_studio_noise_cancellation.py -v`
Expected: alle PASS (signatur-kompatibel, Defaults unverändert)

- [ ] **Step 5: Commit**

```bash
git add src/studio/probe.py tests/test_studio_mask_integration.py
git commit -m "feat(studio): P1 Masken-Integration im ProbeRenderer + E2E-Nachweis"
```

---

## Abschluss P1 (Definition of Done, Spec §16)

- [ ] `pytest tests/test_studio_constraints.py tests/test_studio_mask_service.py -v` — alle PASS (CPU)
- [ ] `pytest tests/test_studio_mask_integration.py -v` — PASS (GPU): Maske überlebt Paritäts-/Kantentest, Cap wirkt post-FX nachweisbar — auch bei Composite-Stacks
- [ ] `pytest tests/ -q` — keine Regressionen im Bestand
- [ ] Danach: Plan für **P2** (sampling.py, feasibility.py) schreiben — konsumiert `ConstraintSet`, `MaskResult`, `ProbeRenderer`
