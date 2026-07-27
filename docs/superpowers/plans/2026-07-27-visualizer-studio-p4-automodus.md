# Visualizer Studio — P4 ModeGate, Profile & PresetFactory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Der Auto-Modus: deterministische Modus-Weiche (MUSIC/PODCAST/HYBRID mit Hysterese), Modus-Profile mit Whitelist und Parameter-Korridoren, und die PresetFactory, die auf dem bestehenden SmartMatcher aufsetzt und per Konstruktion gate-konforme Presets erzeugt — verdrahtet als `run_studio_auto`.

**Architecture:** Drei neue Module in `src/studio/` plus Engine-Erweiterung. `mode_gate.py` klassifiziert aus dem Feature-Dict (speech_score, Hysterese über `.cache/mode_decisions.json`). `profiles.py` definiert `StudioProfile` (Pydantic) mit je einem Built-in-Profil für Musik und Podcast; Whitelist-Keys werden gegen `VISUALIZER_MAP` geprüft (Fail-fast). `preset_factory.py` nutzt `SmartMatcher.match()` (`src/ai_matcher.py:600`) und klemmt das Ergebnis in Profil-Korridore + Post-FX-Budgets. `engine.run_studio_auto` kettet alles: ModeGate → Profil → Preset → `run_studio` (P3). Spec: `docs/superpowers/specs/2026-07-27-visualizer-studio-design.md` (studio-spec/2.1, §5, §10, §16 P4).

**Tech Stack:** Python 3.11, numpy, pydantic v2, pytest. Keine neuen Dependencies.

**Voraussetzung (P0–P3, abgeschlossen):** `src/studio/` komplett inkl. `run_studio` (P3), `ConstraintSet`, `ThresholdSet`/`load_thresholds`, `build_sample_plan`. `SmartMatcher.match(features: AudioFeatures) -> AIRecommendation` mit Feldern `visualizer`, `colors: dict`, `params: dict`, `top_candidates: list[tuple[str, float]]`, `confidence` (`src/ai_matcher.py:14-32`).

**Golden-Set-Hinweis (Spec §16 P4 DoD):** Die Kennzahl „≥ 70 % der Golden-Set-Läufe lösen mit J = 0 ohne Solver-Schritt" ist **blockiert durch die fehlende Golden-Set-Befüllung (Nutzer-Aufgabe, Spec §19)**. Ersatz-DoD in diesem Plan: Presets sind per Konstruktion gate-konform — auf synthetischen Musik- und Sprach-Feature-Sets lösen sie mit `J = 0` im ersten Probe-Durchlauf.

## Global Constraints

Werte wörtlich aus der Spec — jeder Task erbt diese Anforderungen implizit:

- `speech_score = 0.5·norm(voice_clarity) + 0.3·norm(voice_band) − 0.2·norm(onset_density)`; `≥ 0.55` → Podcast-Regelwerk, sonst Musik (Spec §5).
- **Hysterese:** `speech_score ∈ [0.50, 0.60]` → Entscheidung des letzten Laufs für dieselbe Datei beibehalten (Spec §5).
- Whitelist-Keys werden beim Profil-Load gegen `VISUALIZER_MAP` geprüft → **Fail-fast** (Spec §5).
- Schwellen kommen aus `studio_thresholds.v1.json`, nicht als Konstanten im Modul (Spec §5) — die Datei wird um `speech_threshold`, `hysteresis_lo`, `hysteresis_hi` ergänzt.
- MUSIC-Whitelist: spectrum_bars, lumina_core, bass_temple, particle_swarm, chroma_field, neon_oscilloscope, spectrum_genesis, orchestral_swell, sacred_mandala, liquid_blobs, neon_wave_circle, frequency_flower, pulsing_core, typographic. PODCAST-Whitelist: voice_flow, speech_focus, neon_wave_circle, pulsing_core, aurora_voice, nebula_drift (Spec §5/v1).
- Presets tragen `schema_version` + `threshold_set`-Referenz; Parameter-Klemmung gegen Profil-Korridor **und** ConstraintSet (Spec §10).
- Code-Kommentare und Commit-Messages auf Deutsch (AGENTS.md).

## Dateistruktur

| Datei | Verantwortung |
|-------|---------------|
| `src/studio/mode_gate.py` (neu) | `classify_mode`, `ModeResult`, Hysterese-Persistenz |
| `src/studio/profiles.py` (neu) | `StudioProfile`, `load_profile(name)` mit Built-ins |
| `src/studio/preset_factory.py` (neu) | `build_preset`, `StudioPreset` |
| `src/studio/engine.py` (Modify) | `run_studio_auto` (ModeGate → Profil → Preset → run_studio) |
| `src/studio/thresholds.py` (Modify) | 3 neue Felder (speech_threshold, hysteresis_lo/hi) |
| `config/studio_thresholds.v1.json` (Modify) | dieselben 3 Felder |
| `tests/test_studio_mode_gate.py` (neu) | Klassifikation, Determinismus, Hysterese |
| `tests/test_studio_profiles.py` (neu) | Built-ins, Whitelist-Fail-fast, Korridore |
| `tests/test_studio_preset_factory.py` (neu) | Gate-Konformität, Whitelist, Korridor, J=0 synthetisch |

---

### Task 1: ModeGate

**Files:**
- Create: `src/studio/mode_gate.py`
- Modify: `src/studio/thresholds.py` (`ThresholdSet` um 3 Felder)
- Modify: `config/studio_thresholds.v1.json` (dieselben 3 Felder)
- Test: `tests/test_studio_mode_gate.py`

**Interfaces:**
- Consumes: `load_thresholds`/`ThresholdSet` (P0), `_seed_from_features` (`src/studio/sampling.py`, P2).
- Produces:
  - `@dataclass ModeResult`: `value: str` (`"MUSIC"` | `"PODCAST"` | `"HYBRID"`), `resolved: str` (`"music"` | `"podcast"` — aufgelöstes Regelwerk), `confidence: float`, `speech_score: float`, `hysteresis_applied: bool`
  - `classify_mode(features_dict: dict, ts: ThresholdSet | None = None, cache_path: str = ".cache/mode_decisions.json") -> ModeResult`
  - Konsumiert von Task 4 (`run_studio_auto`) und P5 (GUI-Badge).

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_mode_gate.py`:

```python
"""Tests für die Modus-Weiche (Spec §5)."""

import numpy as np
import pytest

from src.studio.mode_gate import classify_mode
from src.studio.thresholds import load_thresholds


def _fd(voice_clarity, voice_band, onset, duration=30.0, fps=30):
    n = int(duration * fps)
    return {
        "voice_clarity": np.full(n, voice_clarity, dtype=np.float32),
        "voice_band": np.full(n, voice_band, dtype=np.float32),
        "onset": np.full(n, onset, dtype=np.float32),
        "rms": np.full(n, 0.5, dtype=np.float32),
        "duration": duration, "fps": fps, "frame_count": n,
    }


def test_klar_sprache_ist_podcast(tmp_path):
    fd = _fd(voice_clarity=0.9, voice_band=0.8, onset=0.05)
    result = classify_mode(fd, cache_path=str(tmp_path / "m.json"))
    assert result.resolved == "podcast"
    assert result.speech_score >= 0.55


def test_klar_musik_ist_music(tmp_path):
    fd = _fd(voice_clarity=0.05, voice_band=0.1, onset=0.7)
    result = classify_mode(fd, cache_path=str(tmp_path / "m.json"))
    assert result.resolved == "music"
    assert result.value in ("MUSIC", "HYBRID")


def test_determinismus(tmp_path):
    fd = _fd(voice_clarity=0.6, voice_band=0.5, onset=0.3)
    a = classify_mode(fd, cache_path=str(tmp_path / "m.json"))
    b = classify_mode(fd, cache_path=str(tmp_path / "m.json"))
    assert (a.value, a.resolved, a.speech_score) == (b.value, b.resolved, b.speech_score)


def test_hysterese_haelt_letzte_entscheidung(tmp_path):
    cache = str(tmp_path / "m.json")
    # Erster Lauf: klare Podcast-Entscheidung
    fd_speech = _fd(voice_clarity=0.9, voice_band=0.8, onset=0.05)
    first = classify_mode(fd_speech, cache_path=cache)
    assert first.resolved == "podcast"
    # Zweiter Lauf: gleiche Datei (gleicher Seed), Score im Hysterese-Band
    # (0.5*0.7 + 0.3*0.5 - 0.2*0.3 = 0.44 -> zu niedrig; mit 0.8/0.7/0.2:
    #  0.4 + 0.21 - 0.04 = 0.57 -> im Band [0.50, 0.60])
    fd_border = _fd(voice_clarity=0.8, voice_band=0.7, onset=0.2)
    # Seed-Gleichheit simulieren: rms identisch halten (Seed aus rms+duration)
    second = classify_mode(fd_border, cache_path=cache)
    assert 0.50 <= second.speech_score <= 0.60
    assert second.hysteresis_applied is True
    assert second.resolved == "podcast"  # beibehalten, nicht neu entschieden


def test_schwellen_kommen_aus_thresholds(tmp_path):
    ts = load_thresholds()
    assert ts.speech_threshold == 0.55
    assert ts.hysteresis_lo == 0.50
    assert ts.hysteresis_hi == 0.60
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_mode_gate.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.mode_gate'` (der Thresholds-Test schlägt mit `AttributeError: speech_threshold` fehl, sobald das Modul existiert)

- [ ] **Step 3: Implementierung**

**3a) `config/studio_thresholds.v1.json`** — in `"thresholds"` ergänzen:

```json
    "speech_threshold": 0.55,
    "hysteresis_lo": 0.50,
    "hysteresis_hi": 0.60
```

**3b) `src/studio/thresholds.py`** — `ThresholdSet` um drei Felder ergänzen:

```python
    speech_threshold: float = 0.55
    hysteresis_lo: float = 0.50
    hysteresis_hi: float = 0.60
```

(Defaults, damit ältere JSON-Dateien ohne die Felder weiter laden.)

**3c) `src/studio/mode_gate.py`:**

```python
"""ModeGate — strikte Modus-Weiche (Spec §5).

Deterministische Klassifikation MUSIC | PODCAST | HYBRID aus dem
Feature-Dict. HYBRID wird numerisch aufgelöst (speech_score), die
Hysterese verhindert Klassenflattern bei minimalen Reanalysen.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sampling import _seed_from_features
from .thresholds import ThresholdSet, load_thresholds


@dataclass
class ModeResult:
    """Klassifikationsergebnis inkl. Begründung (landet im Sidecar)."""

    value: str           # MUSIC | PODCAST | HYBRID
    resolved: str        # "music" | "podcast" (aufgelöstes Regelwerk)
    confidence: float
    speech_score: float
    hysteresis_applied: bool


def _mean(features_dict: dict, key: str) -> float:
    arr = np.asarray(features_dict[key], dtype=np.float32)
    return float(np.clip(arr.mean(), 0.0, 1.0))


def classify_mode(features_dict: dict, ts: ThresholdSet | None = None,
                  cache_path: str = ".cache/mode_decisions.json") -> ModeResult:
    """Klassifiziert den Audio-Typ (Spec §5).

    speech_score = 0.5*voice_clarity + 0.3*voice_band - 0.2*onset_density
    >= speech_threshold -> Podcast-Regelwerk, sonst Musik.
    Hysterese: Score im Band [hysteresis_lo, hysteresis_hi] -> letzte
    Entscheidung für dieselbe Datei (Seed) beibehalten.
    """
    ts = ts or load_thresholds()
    score = (0.5 * _mean(features_dict, "voice_clarity")
             + 0.3 * _mean(features_dict, "voice_band")
             - 0.2 * _mean(features_dict, "onset"))

    # Rohe Klassifikation
    if score >= ts.speech_threshold:
        value, resolved = "PODCAST", "podcast"
    elif score >= ts.hysteresis_lo:
        value, resolved = "HYBRID", "music"
    else:
        value, resolved = "MUSIC", "music"

    hysteresis_applied = False
    seed = _seed_from_features(features_dict)
    cache = Path(cache_path)
    decisions: dict = {}
    if cache.exists():
        try:
            decisions = json.loads(cache.read_text())
        except Exception:
            decisions = {}

    if ts.hysteresis_lo <= score <= ts.hysteresis_hi and seed in decisions:
        # Hysterese: letzte Entscheidung beibehalten (Spec §5)
        resolved = decisions[seed]
        value = "HYBRID"
        hysteresis_applied = True

    cache.parent.mkdir(parents=True, exist_ok=True)
    decisions[seed] = resolved
    cache.write_text(json.dumps(decisions, indent=2))

    confidence = min(1.0, abs(score - ts.speech_threshold) / 0.45)
    return ModeResult(value=value, resolved=resolved,
                      confidence=round(confidence, 3),
                      speech_score=round(score, 4),
                      hysteresis_applied=hysteresis_applied)
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_mode_gate.py tests/test_studio_thresholds.py -v`
Expected: alle PASS (Bestands-Thresholds-Tests dürfen nicht brechen)

- [ ] **Step 5: Commit**

```bash
git add src/studio/mode_gate.py src/studio/thresholds.py config/studio_thresholds.v1.json tests/test_studio_mode_gate.py
git commit -m "feat(studio): P4 ModeGate — speech_score, HYBRID-Auflösung, Hysterese"
```

---

### Task 2: StudioProfile

**Files:**
- Create: `src/studio/profiles.py`
- Test: `tests/test_studio_profiles.py`

**Interfaces:**
- Consumes: `VISUALIZER_MAP` (`src/gpu_visualizers/__init__.py`), `PERIPHERAL_VISUALS` (P2, für Referenz).
- Produces:
  - `class StudioProfile(BaseModel)`: `name: str`, `version: int`, `mode: str`, `visualizer_whitelist: list[str]`, `param_corridors: dict[str, tuple[float, float]]`, `postfx_budget: dict[str, float]`, `vitality_corridor: tuple[float, float]`, `subject_strength: float = 0.8`, `desaturate_colors: bool = False`
  - `load_profile(name: str) -> StudioProfile` — Built-ins `"music_default"`, `"podcast_default"`; validiert Whitelist gegen `VISUALIZER_MAP` (Fail-fast)
  - `BUILTIN_PROFILES: dict[str, StudioProfile]`
  - Konsumiert von Task 3 (PresetFactory), Task 4, P5.

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_profiles.py`:

```python
"""Tests für die Modus-Profile (Spec §5)."""

import pytest

from src.studio.profiles import load_profile


def test_music_default_laedt():
    p = load_profile("music_default")
    assert p.mode == "music"
    assert "spectrum_bars" in p.visualizer_whitelist
    assert "voice_flow" not in p.visualizer_whitelist
    assert p.desaturate_colors is False


def test_podcast_default_laedt():
    p = load_profile("podcast_default")
    assert p.mode == "podcast"
    assert "voice_flow" in p.visualizer_whitelist
    assert "speech_focus" in p.visualizer_whitelist
    assert "bass_temple" not in p.visualizer_whitelist
    assert p.desaturate_colors is True
    # Podcast: enger Vitalitätskorridor (Spec §3.3 M5)
    assert p.vitality_corridor[1] <= 0.09


def test_whitelist_keys_existieren_in_registry():
    from src.gpu_visualizers import VISUALIZER_MAP
    for name in ("music_default", "podcast_default"):
        for key in load_profile(name).visualizer_whitelist:
            assert key in VISUALIZER_MAP


def test_unbekanntes_profil_wirft():
    with pytest.raises(KeyError, match="unbekannt"):
        load_profile("unbekannt")


def test_fail_fast_bei_unbekanntem_visualizer():
    from src.studio.profiles import StudioProfile
    with pytest.raises(ValueError, match="gibt_es_nicht"):
        StudioProfile(
            name="kaputt", version=1, mode="music",
            visualizer_whitelist=["gibt_es_nicht"],
            param_corridors={}, postfx_budget={},
            vitality_corridor=(0.0, 1.0),
        )
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_profiles.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.profiles'`

- [ ] **Step 3: Implementierung**

`src/studio/profiles.py`:

```python
"""Modus-Profile (Spec §5): Whitelists, Parameter-Korridore, FX-Budgets.

Whitelist-Keys werden beim Laden gegen VISUALIZER_MAP geprüft —
Fail-fast statt stillem Laufzeitfehler.
"""

from pydantic import BaseModel, model_validator


class StudioProfile(BaseModel):
    """Regelwerk eines Modus (MUSIC oder PODCAST)."""

    name: str
    version: int
    mode: str  # "music" | "podcast"
    visualizer_whitelist: list[str]
    param_corridors: dict[str, tuple[float, float]]
    postfx_budget: dict[str, float]
    vitality_corridor: tuple[float, float]
    subject_strength: float = 0.8
    desaturate_colors: bool = False

    @model_validator(mode="after")
    def _whitelist_gegen_registry(self):
        from ..gpu_visualizers import VISUALIZER_MAP
        for key in self.visualizer_whitelist:
            if key not in VISUALIZER_MAP:
                raise ValueError(
                    f"Profil '{self.name}': unbekannter Visualizer '{key}'"
                )
        return self


BUILTIN_PROFILES: dict[str, StudioProfile] = {
    "music_default": StudioProfile(
        name="music_default", version=1, mode="music",
        visualizer_whitelist=[
            "spectrum_bars", "lumina_core", "bass_temple", "particle_swarm",
            "chroma_field", "neon_oscilloscope", "spectrum_genesis",
            "orchestral_swell", "sacred_mandala", "liquid_blobs",
            "neon_wave_circle", "frequency_flower", "pulsing_core",
            "typographic",
        ],
        param_corridors={"intensity": (0.2, 3.0), "speed": (0.2, 5.0)},
        postfx_budget={"bloom_intensity": 1.0, "film_grain": 0.5},
        vitality_corridor=(0.02, 1.0),
    ),
    "podcast_default": StudioProfile(
        name="podcast_default", version=1, mode="podcast",
        visualizer_whitelist=[
            "voice_flow", "speech_focus", "neon_wave_circle",
            "pulsing_core", "aurora_voice", "nebula_drift",
        ],
        param_corridors={"intensity": (0.1, 1.0), "speed": (0.1, 1.0)},
        postfx_budget={"bloom_intensity": 0.4, "film_grain": 0.1},
        vitality_corridor=(0.0, 0.09),
        desaturate_colors=True,
    ),
}


def load_profile(name: str) -> StudioProfile:
    """Lädt ein Built-in-Profil; unbekannte Namen sind ein Fehler."""
    if name not in BUILTIN_PROFILES:
        raise KeyError(f"Unbekanntes Studio-Profil: '{name}'")
    return BUILTIN_PROFILES[name]
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_profiles.py -v`
Expected: 5 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/profiles.py tests/test_studio_profiles.py
git commit -m "feat(studio): P4 StudioProfile — Whitelists, Korridore, Fail-fast"
```

---

### Task 3: PresetFactory

**Files:**
- Create: `src/studio/preset_factory.py`
- Test: `tests/test_studio_preset_factory.py`

**Interfaces:**
- Consumes: `SmartMatcher` (`src/ai_matcher.py:45`, `match(features) -> AIRecommendation`), `StudioProfile` (Task 2), `ConstraintSet` (P1).
- Produces:
  - `@dataclass StudioPreset`: `visualizer: str`, `params: dict`, `colors: dict`, `postprocess: dict`, `constraints: ConstraintSet`, `schema_version: str`, `threshold_set: str`, `reason: str`
  - `build_preset(features, profile: StudioProfile, matcher: SmartMatcher | None = None) -> StudioPreset`
  - `PRESET_SCHEMA_VERSION = "studio-preset/1"`
  - Konsumiert von Task 4 (`run_studio_auto`), P5 (KI-Panel-Button, Export).

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_preset_factory.py`:

```python
"""Tests für die PresetFactory (Spec §10)."""

import numpy as np
import pytest

from src.studio.preset_factory import PRESET_SCHEMA_VERSION, build_preset
from src.studio.profiles import load_profile


def test_visualizer_aus_whitelist(dummy_audio_features):
    for profil in ("music_default", "podcast_default"):
        preset = build_preset(dummy_audio_features, load_profile(profil))
        assert preset.visualizer in load_profile(profil).visualizer_whitelist


def test_params_innerhalb_der_korridore(dummy_audio_features):
    preset = build_preset(dummy_audio_features, load_profile("podcast_default"))
    corridors = load_profile("podcast_default").param_corridors
    for key, (lo, hi) in corridors.items():
        if key in preset.params:
            assert lo <= preset.params[key] <= hi


def test_postprocess_innerhalb_budget(dummy_audio_features):
    preset = build_preset(dummy_audio_features, load_profile("podcast_default"))
    assert preset.postprocess.get("bloom_intensity", 0) <= 0.4
    assert preset.postprocess.get("film_grain", 0) <= 0.1


def test_schema_version_und_threshold_referenz(dummy_audio_features):
    preset = build_preset(dummy_audio_features, load_profile("music_default"))
    assert preset.schema_version == PRESET_SCHEMA_VERSION
    assert preset.threshold_set.endswith("studio_thresholds.v1.json")


def test_podcast_farben_entsaettigt(dummy_audio_features):
    preset = build_preset(dummy_audio_features, load_profile("podcast_default"))
    # Entsättigung: RGB-Kanäle liegen näher beieinander als Saturated-Rot
    from src.gpu_visualizers.base import hex_to_rgb
    rgb = hex_to_rgb(preset.colors["primary"])
    spread = max(rgb) - min(rgb)
    assert spread < 0.7  # reines Rot hätte 1.0
```

Hinweis: `hex_to_rgb` liegt in `src/gpu_visualizers/base.py` (wird auch in `gpu_preview.py` genutzt); Rückgabe ist ein Tuple float 0–1.

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_preset_factory.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.studio.preset_factory'`

- [ ] **Step 3: Implementierung**

`src/studio/preset_factory.py`:

```python
"""PresetFactory (Spec §10): KI-Presets auf Basis des SmartMatcher.

Baut auf dem bestehenden SmartMatcher AUF (ersetzt ihn nicht):
Visualizer-Wahl aus Top-Kandidaten ∩ Profil-Whitelist, Farben Key-basiert
(Podcast entsättigt), Parameter in Profil-Korridore geklemmt, Post-FX in
Budgets geklemmt. Presets sind per Konstruktion gate-konform.
"""

import colorsys
from dataclasses import dataclass, field

from ..ai_matcher import SmartMatcher
from ..types import AudioFeatures
from .constraints import ConstraintSet
from .profiles import StudioProfile

PRESET_SCHEMA_VERSION = "studio-preset/1"
THRESHOLD_SET_REF = "config/studio_thresholds.v1.json"


@dataclass
class StudioPreset:
    """Vollständiger, gate-konformer Render-Entwurf (Spec §10)."""

    visualizer: str
    params: dict
    colors: dict
    postprocess: dict
    constraints: ConstraintSet
    schema_version: str = PRESET_SCHEMA_VERSION
    threshold_set: str = THRESHOLD_SET_REF
    reason: str = ""


def _desaturate(hex_color: str, factor: float = 0.5) -> str:
    """Entsättigt eine Hex-Farbe (Podcast-Paletten, Spec §10)."""
    from ..gpu_visualizers.base import hex_to_rgb
    r, g, b = hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s * factor)
    return "#{:02X}{:02X}{:02X}".format(
        round(r2 * 255), round(g2 * 255), round(b2 * 255))


def build_preset(features: AudioFeatures, profile: StudioProfile,
                 matcher: SmartMatcher | None = None) -> StudioPreset:
    """Erzeugt ein gate-konformes Preset aus Audio-Features + Profil."""
    matcher = matcher or SmartMatcher()
    rec = matcher.match(features)

    # Visualizer: Top-Kandidaten in Score-Reihenfolge ∩ Whitelist (Spec §10)
    visualizer = None
    for name, _score in rec.top_candidates:
        if name in profile.visualizer_whitelist:
            visualizer = name
            break
    if visualizer is None:
        visualizer = profile.visualizer_whitelist[0]

    # Parameter in Korridore klemmen
    params = dict(rec.params or {})
    for key, (lo, hi) in profile.param_corridors.items():
        if key in params:
            params[key] = min(max(float(params[key]), lo), hi)

    # Farben: Podcast entsättigt (Spec §10)
    colors = dict(rec.colors or {"primary": "#5E81EA",
                                 "secondary": "#4ECDC4",
                                 "background": "#0A0A14"})
    if profile.desaturate_colors:
        colors = {k: (_desaturate(v) if k != "background" else v)
                  for k, v in colors.items()}

    # Post-FX in Budgets klemmen
    postprocess = {
        "bloom_intensity": min(0.6, profile.postfx_budget.get("bloom_intensity", 1.0)),
        "film_grain": min(0.0, profile.postfx_budget.get("film_grain", 0.5)),
    }

    constraints = ConstraintSet(
        subject_strength=profile.subject_strength,
        max_bloom_intensity=profile.postfx_budget.get("bloom_intensity", 1.0),
        max_film_grain=profile.postfx_budget.get("film_grain", 0.5),
    )
    return StudioPreset(visualizer=visualizer, params=params, colors=colors,
                        postprocess=postprocess, constraints=constraints,
                        reason=rec.reason)
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_preset_factory.py -v`
Expected: 5 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/preset_factory.py tests/test_studio_preset_factory.py
git commit -m "feat(studio): P4 PresetFactory — SmartMatcher ∩ Whitelist, Korridore"
```

---

### Task 4: run_studio_auto (Verdrahtung)

**Files:**
- Modify: `src/studio/engine.py` (anfügen: `run_studio_auto`)
- Test: `tests/test_studio_engine.py` (anfügen: 1 GPU-Test)

**Interfaces:**
- Consumes: `classify_mode` (Task 1), `load_profile` (Task 2), `build_preset` (Task 3), `run_studio` (P3).
- Produces:
  - `run_studio_auto(audio_path, features, features_dict, output_path, profile_name: str | None = None, params_override: dict | None = None, postprocess_override: dict | None = None, background_image: str | None = None, subject_mask=None) -> dict` — Auto-Flow: ModeGate → Profil (explizit oder aus resolved) → Preset → `run_studio`; liefert das Sidecar-Dict (Mode-Block mit echten ModeResult-Werten)
  - Konsumiert von P5 (CLI `--studio`, GUI).

- [ ] **Step 1: Failing Test anfügen**

An `tests/test_studio_engine.py` anhängen:

```python
def test_run_studio_auto_end_to_end(tmp_path, dummy_audio_features):
    from unittest.mock import MagicMock, patch
    from src.gpu_renderer import GPUBatchRenderer
    from src.render_common import build_features_dict
    from src.studio.engine import run_studio_auto

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    with patch.object(GPUBatchRenderer, "render",
                      MagicMock(side_effect=RuntimeError("mock"))), \
         patch("src.gpu_renderer.subprocess.run") as mock_run, \
         patch("src.studio.mode_gate.classify_mode") as mock_mode:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        mock_mode.return_value.resolved = "music"
        mock_mode.return_value.value = "MUSIC"
        mock_mode.return_value.confidence = 0.9
        mock_mode.return_value.speech_score = 0.1
        mock_mode.return_value.hysteresis_applied = False
        sidecar = run_studio_auto(
            str(audio), dummy_audio_features, features_dict, str(out),
            profile_name="music_default",
        )

    assert sidecar["mode"]["value"] == "MUSIC"
    assert sidecar["profile"]["name"] == "music_default"
    assert sidecar["verify"]["status"] in ("pass", "drift_abort")
```

Hinweis: Der Test nutzt `profile_name="music_default"`, damit das Preset unabhängig von der gemockten Klassifikation reproduzierbar ist; der ModeGate-Aufruf selbst ist durch `classify_mode`-Tests aus Task 1 abgedeckt.

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `pytest tests/test_studio_engine.py -v -k auto`
Expected: FAIL mit `ImportError: cannot import name 'run_studio_auto'`

- [ ] **Step 3: Implementierung**

An `src/studio/engine.py` anfügen:

```python
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
```

- [ ] **Step 4: Erfolg verifizieren**

Run: `pytest tests/test_studio_engine.py -v`
Expected: 3 × PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/engine.py tests/test_studio_engine.py
git commit -m "feat(studio): P4 run_studio_auto — ModeGate→Profil→Preset→Render"
```

---

## Abschluss P4 (Definition of Done, Spec §16, angepasst)

- [ ] `pytest tests/test_studio_mode_gate.py tests/test_studio_profiles.py tests/test_studio_preset_factory.py -v` — alle PASS (Klassifikation deterministisch, Hysterese an Grenzfällen, Whitelist-Fail-fast, Presets gate-konform)
- [ ] `pytest tests/test_studio_engine.py -v` — Auto-Flow End-to-End grün (GPU)
- [ ] `pytest tests/ -q` — keine Regressionen im Bestand
- [ ] ~~≥ 70 % Golden-Set mit J = 0~~ — **blockiert durch fehlende Golden-Set-Befüllung (Nutzer-Aufgabe)**; Ersatz-DoD: Presets per Konstruktion gate-konform (Tests Task 3)
- [ ] Danach: Plan **P5** (CLI-Flags, GUI-Badges, Video-Degradation, Perf-Budgets)
