# Spiral-Zoom-Effekt-Pass (Droste/Escher) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ein neuer, abschaltbarer Effekt „Spiral-Zoom“ verbiegt die **Visualizer-Ebene** zu einem endlosen Zoom ins Bild hinein (Droste) bzw. zu einer Escher-Spirale; ein Hintergrundfoto bleibt ruhig stehen. Das Zoom-Tempo folgt Lautstärke und Beats.

**Architecture:** Reine Mathematik (Einstellungen, Shader-Uniforms, CPU-Gegenstück des Shaders, Zoom-Phase aus Audio) in `src/spiral_zoom.py`, portiert aus dem Schwesterprojekt Fraktal-Zoom (`src/zoom-math/escher.ts`). Ein eigener GPU-Pass `src/gpu_spiral.py` (Aufbau wie `gpu_bloom.py`) kopiert eine HDR-Ebene in eine Mipmap-Textur und zeichnet sie verbogen zurück (inklusive Alpha). Renderer und Vorschau wenden ihn auf die Visualizer-FBO an — nach dem Visualizer-Render, **vor** dem Blit über den Hintergrund. Konfiguration läuft über das bestehende `postprocess`-Dict (`spiral_*`-Schlüssel).

**Tech Stack:** Python 3.11, ModernGL (GLSL `#version 330`), NumPy, Pydantic v2, PyQt6, pytest (+ `pytest-qt`).

**Spec:** Kein separates Spec-Dokument — der Abschnitt „Entscheidungen“ unten ist die Spezifikation. Quelle der Mathematik: `C:\Users\Buxe\Projects\Fraktal-Zoom\src\zoom-math\escher.ts` und `src\renderer\shaders.ts` (`ESCHER_FRAG`).

## Entscheidungen

1. **Ebene (Nutzer-Entscheidung 2026-09-29): nur der Visualizer.** Das Hintergrundfoto dreht **nicht** mit. Der Pass arbeitet in place auf der FBO, die `active_viz_tex` hält (`viz_fbo`, bei Timeline-Überblendung `viz_fbo_blend`), direkt vor `_blit_viz_to_fbo`. Alpha wird mit verbogen, damit Luma-/Occlusion-Alpha im Blit weiter stimmt. Offset/Skalierung des Visualizers wirken danach im Blit (die Spirale sitzt in der Mitte des Visualizers). Zitat-Overlays liegen danach (CPU) und bleiben gerade.
2. **Standard aus.** Ist der Effekt aus, wird nichts kopiert, nichts gezeichnet, nichts angelegt — die Ausgabe ist bitgleich zu heute (Golden-Set bleibt unberührt).
3. **Geometrie** relativ zu `min(Breite, Höhe)`, Zentrum = Bildmitte, Außenradius `R_out = 0.5 · min(B, H)`, Innenradius `R = R_out / K`. Alle Abtastpunkte liegen im Inkreis — der Pass liest nie außerhalb des Bildes. Die Ecken entstehen aus der Rekursion.
4. **Parameter** (Schlüssel im `postprocess`-Dict, Bereich, Standard):

   | Schlüssel | Bedeutung | Bereich | Standard |
   |---|---|---|---|
   | `spiral_enabled` | Effekt an/aus | bool | `false` |
   | `spiral_arms` | Spiralarme (0 = gerader Droste-Zoom), **ganzzahlig** | −3…3 | 0 |
   | `spiral_ratio` | Zoom-Faktor K pro Ebene | 1.5…6.0 | 2.5 |
   | `spiral_rotation` | Drehung pro Ebene in Grad | −90…90 | 0 |
   | `spiral_speed` | Grund-Tempo, Ebenen/Sekunde; Vorzeichen = Richtung (+ hinein) | −2…2 | 0.3 |
   | `spiral_energy` | Zusatz-Ebenen/Sekunde bei voller Lautstärke (`rms = 1`) | 0…2 | 0.5 |
   | `spiral_beat` | Zusatz-Ebenen pro Beat | 0…1 | 0.2 |
   | `spiral_feather` | Breite der Überblendung an der Ebenen-Naht (nicht in der GUI) | 0.02…0.5 | 0.15 |
   | `spiral_mix` | 0 = Original, 1 = voller Effekt | 0…1 | 1.0 |

5. **Zoom-Phase** (Ebenen, float64) wird **einmal vor der Frame-Schleife** aus `rms` und `beat_intensity` berechnet: pro Frame `(|speed| + energy·rms)/fps + beat·beat_intensity/Beat-Fläche`, aufsummiert, Richtung = Vorzeichen von `speed`. Jeder Beat bringt genau `beat` Ebenen, unabhängig von der Framerate. Die Phase hängt nur von früheren Frames ab → Vorschau bei Zeit t = Export-Frame bei t.
6. **Periodizität:** Die Abbildung ist in der Phase u exakt periodisch mit Periode 1 (eine Ebene). Der Shader bekommt `u mod 1` (auf der CPU in float64 gefaltet) — keine Präzisionsprobleme bei stundenlangen Songs.
7. **Nur ganzzahlige Arme** schließen nahtlos (Fraktal-Zoom-Erkenntnis). Schema (`int`), Einstellungen (runden + klemmen) und GUI-Regler (ganzzahliger Slider) erzwingen das.
8. **Der Kern wird ersetzt:** Alles innerhalb von `R_out / K` (beim Standard K = 2,5 ein Kreis mit 20 % der kurzen Bildseite als Radius) zeigt die nächste, kleinere Ebene statt des Originals — viele Visualizer haben dort ihren leuchtenden Mittelpunkt. Höherer Zoom-Faktor = kleinerer ersetzter Kern; „Stärke“ unter 100 % lässt das Original durchscheinen. So gewollt (das *ist* der Droste-Effekt), in der GUI-Tooltip erwähnt.
9. **Studio-Modus schaltet den Spiral-Zoom ab** (mit Warnung im Sidecar): Der Studio-Pfad misst den Beitrag des Visualizers Pixel für Pixel mit einer eigenen Render-Schleife ohne Spiral-Pass (`src/studio/probe.py`); Messung und Commit-Render liefen sonst auseinander.

## Global Constraints

- Keine neuen Abhängigkeiten. GLSL `#version 330`, ModernGL wie im Bestand.
- HDR bleibt erhalten: im Pass **kein** `clamp`, **kein** sRGB-Encode — Tonemapping macht zentral `_apply_postprocess`.
- `AudioAnalyzer.analyze()` nicht ändern (siehe `CLAUDE.md`).
- Zustandslos: jeder Frame nur aus Zeitpunkt + Features (kein Ping-Pong, vgl. Begründung in `src/gpu_visualizers/ink_bloom.py`).
- Effekt aus ⇒ Ausgabe bitgleich zu vorher.
- Kommentare, Docstrings, UI-Texte und Commit-Messages auf Deutsch; in Code-Kommentaren wie im Bestand ohne Umlaute (ae/oe/ue). UI-Texte dürfen Umlaute nur dort haben, wo der Bestand sie hat — hier ebenfalls ae/oe/ue.
- GPU-Tests mit `@pytest.mark.gpu` markieren (werden ohne OpenGL automatisch übersprungen, `tests/conftest.py`).
- Uniforms defensiv setzen: vom Treiber wegoptimierte Uniforms fehlen im Programm (`KeyError`). GLSL-Namen wie `sample`, `active` meiden.

## Review Focus

1. **Stille / Podcast ohne Beats:** Die Phase läuft nur mit dem Grund-Tempo weiter, bei `speed = 0` steht der Tunnel still — kein Crash, keine NaNs. → Tests in Task 2 (`test_silence_and_no_beats_gives_constant_speed`, `test_everything_zero_is_standstill`, `test_nan_and_length_mismatch`).
2. **Hochformat 9:16 und 4K:** Dasselbe Bild in jeder Auflösung, keine Abtastung außerhalb des Bildes. → Task 1 (`test_same_picture_at_every_resolution`, `test_samples_stay_inside_the_frame`).
3. **Einstündiger Song:** Phase wird groß (Tausende Ebenen); der Shader darf nur `u mod 1` sehen. → Task 1 (`test_period_is_one_level`) und Task 2 (`test_long_song_u_stays_in_unit_interval`).
4. **Hintergrundfoto vorhanden:** Effekt läuft mit Foto, das Foto bleibt unverändert, nichts wird schwarz. → Task 5 (`test_real_preview_off_is_bitidentical_on_differs[True]`, `test_real_preview_photo_stays_still`).
5. **Alte Configs / alte `.avproj` ohne Spiral-Schlüssel:** Effekt bleibt aus, Bild bitgleich. → Task 4 (`test_old_config_without_spiral_keys_stays_off`), Task 5 (`test_batch_render_without_spiral_never_calls_pass`, `test_real_preview_off_is_bitidentical_on_differs`), Task 7 (`test_old_project_without_spiral_keys_keeps_effect_off`).

---

## Dateiübersicht

| Datei | Aufgabe |
|---|---|
| `src/render_common.py` (ändern) | `beat_decay_frames(fps)` als gemeinsamer Helfer |
| `src/spiral_zoom.py` (neu) | `SpiralSettings`, `spiral_uniforms`, `spiral_map` (CPU-Spiegel), `compute_spiral_phase`, `phase_at_time` |
| `src/gpu_spiral.py` (neu) | `SpiralZoomPass`: Kopie + Mipmaps + Vollbild-Shader |
| `config/schemas.py` (ändern) | `spiral_*`-Felder in `PostProcessConfig` |
| `config/music_spiral_zoom.json` (neu) | Beispiel-Preset zum Ausprobieren |
| `src/gpu_renderer.py` (ändern) | `_apply_spiral`, Phase vor der Schleife, Aufruf vor Bloom, `release()` |
| `src/gpu_preview.py` (ändern) | derselbe Aufruf in der Live-Vorschau |
| `src/studio/engine.py` (ändern) | Studio-Modus schaltet den Effekt ab (Messungen brauchen unverbogenes Bild) |
| `tests/test_studio_spiral.py` (neu) | Studio-Abschaltung |
| `src/gui/state.py` (ändern) | `SPIRAL_STATE_KEYS`, Felder, `get_postprocess`, `to_dict` |
| `src/gui/main_window.py` (ändern) | Spiral-Schlüssel lösen Vorschau und „*“-Marker aus |
| `src/gui/params_panel.py` (ändern) | Gruppe „Spiral-Zoom“ mit Checkbox + 7 Reglern |
| `tests/test_spiral_zoom.py` (neu) | Mathematik + Phase (ohne GPU) |
| `tests/test_gpu_spiral.py` (neu) | GPU-Pass gegen CPU-Spiegel |
| `tests/test_spiral_config.py` (neu) | Schema + Preset |
| `tests/test_spiral_render_integration.py` (neu) | Renderer/Vorschau (gemockt + echte GPU) |
| `tests/test_gui_state.py`, `tests/test_gui_params_panel.py`, `tests/test_gui_main_window.py` (ergänzen) | GUI |
| `CHANGELOG.md`, `README.md` (ändern) | Doku |

---

### Task 1: Spiral-Mathematik (Einstellungen, Uniforms, CPU-Spiegel)

**Files:**
- Modify: `src/render_common.py:14-38`
- Create: `src/spiral_zoom.py`
- Test: `tests/test_spiral_zoom.py`

**Interfaces:**
- Consumes: nichts.
- Produces:
  - `render_common.beat_decay_frames(fps: int) -> int`
  - `spiral_zoom.SpiralSettings` (frozen dataclass: `enabled: bool, arms: int, ratio: float, rotation: float, speed: float, energy: float, beat: float, feather: float, mix: float`; Property `is_active -> bool`; Classmethod `from_postprocess(pp: dict | None) -> SpiralSettings`)
  - `spiral_zoom.spiral_uniforms(settings: SpiralSettings, width: int, height: int, u: float) -> dict` mit den Schlüsseln `u_size, u_r_out, u_m, u_per, u_ln_r, u_lf, u_u, u_mix` (Tupel bzw. float; Radien relativ zu `u_r_out`)
  - `spiral_zoom.spiral_map(uni: dict, px, py) -> dict` mit `q (N,2), jac (N,), q_in (N,2), jac_in (N,), w_in (N,)`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_spiral_zoom.py`:

```python
"""Tests fuer src/spiral_zoom.py: Einstellungen, Abbildung, Zoom-Phase."""

import numpy as np
import pytest

from src.spiral_zoom import SpiralSettings, spiral_map, spiral_uniforms

W, H = 1920, 1080


def _on(**kw):
    return SpiralSettings(enabled=True, **kw)


def _pixels(w, h, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, w, n), rng.uniform(0, h, n)


class TestSettings:
    def test_defaults_are_off(self):
        s = SpiralSettings.from_postprocess(None)
        assert s.enabled is False
        assert s.is_active is False

    def test_reads_and_clamps_postprocess(self):
        s = SpiralSettings.from_postprocess({
            "spiral_enabled": True,
            "spiral_arms": 7.4,
            "spiral_ratio": 99,
            "spiral_rotation": -500,
            "spiral_speed": "kaputt",
            "spiral_mix": float("nan"),
        })
        assert s.enabled is True
        assert s.arms == 3
        assert isinstance(s.arms, int)
        assert s.ratio == 6.0
        assert s.rotation == -90.0
        assert s.speed == SpiralSettings().speed  # Unsinn -> Default
        assert s.mix == SpiralSettings().mix

    def test_mix_zero_is_inactive(self):
        assert _on(mix=0.0).is_active is False
        assert _on(mix=0.5).is_active is True


class TestSpiralMap:
    def test_ring_is_identity_without_motion(self):
        """arms=0, keine Drehung, u=0: der Quell-Ring bildet sich auf sich selbst ab."""
        s = _on(arms=0, rotation=0.0, ratio=2.5, feather=0.15)
        uni = spiral_uniforms(s, W, H, 0.0)
        r_in = 0.5 * min(W, H) / 2.5
        rr = np.linspace(r_in * 1.02, r_in * 0.85 * 2.5 * 0.98, 50)
        ang = np.linspace(-3.0, 3.0, 50)
        px = W / 2 + rr * np.cos(ang)
        py = H / 2 + rr * np.sin(ang)
        m = spiral_map(uni, px, py)
        np.testing.assert_allclose(m["q"][:, 0], px, atol=1e-6)
        np.testing.assert_allclose(m["q"][:, 1], py, atol=1e-6)
        assert np.all(m["w_in"] == 0.0)

    def test_core_shows_scaled_copy(self):
        """Innerhalb des Innenradius erscheint das Bild um K vergroessert abgetastet."""
        s = _on(arms=0, rotation=0.0, ratio=3.0, feather=0.15)
        uni = spiral_uniforms(s, W, H, 0.0)
        r_in = 0.5 * H / 3.0
        zx, zy = 0.3 * r_in, -0.2 * r_in
        m = spiral_map(uni, [W / 2 + zx], [H / 2 + zy])
        np.testing.assert_allclose(m["q"][0], [W / 2 + 3 * zx, H / 2 + 3 * zy], atol=1e-6)
        assert m["jac"][0] == pytest.approx(3.0)

    @pytest.mark.parametrize("arms,rot", [(0, 0.0), (1, 0.0), (-2, 30.0), (3, -45.0)])
    def test_period_is_one_level(self, arms, rot):
        """map(u) == map(u + 1): deshalb reicht dem Shader u mod 1."""
        s = _on(arms=arms, rotation=rot, ratio=2.5)
        px, py = _pixels(W, H)
        uni_a = spiral_uniforms(s, W, H, 0.3)
        uni_b = dict(uni_a)
        uni_b["u_u"] = 1.3
        a = spiral_map(uni_a, px, py)
        b = spiral_map(uni_b, px, py)
        np.testing.assert_allclose(a["q"], b["q"], atol=1e-6)
        np.testing.assert_allclose(a["w_in"], b["w_in"], atol=1e-9)

    @pytest.mark.parametrize("arms", [-3, -1, 1, 2])
    def test_no_seam_on_the_left(self, arms):
        """Ganzzahlige Arme: am atan-Sprung links der Mitte bleibt die Abbildung stetig."""
        s = _on(arms=arms, rotation=20.0, ratio=2.5)
        uni = spiral_uniforms(s, W, H, 0.4)
        xs = W / 2 - np.linspace(50, 500, 40)
        above = spiral_map(uni, xs, np.full_like(xs, H / 2 + 1e-4))
        below = spiral_map(uni, xs, np.full_like(xs, H / 2 - 1e-4))
        np.testing.assert_allclose(above["q"], below["q"], atol=0.01)
        np.testing.assert_allclose(above["w_in"], below["w_in"], atol=1e-3)

    def test_samples_stay_inside_the_frame(self):
        """Alle benutzten Abtastpunkte liegen im Inkreis (nie ausserhalb des Bildes)."""
        rng = np.random.default_rng(1)
        c = np.array([W / 2, H / 2])
        r_out = 0.5 * min(W, H)
        for _ in range(20):
            s = _on(
                arms=int(rng.integers(-3, 4)),
                rotation=float(rng.uniform(-90, 90)),
                ratio=float(rng.uniform(1.5, 6.0)),
                feather=float(rng.uniform(0.02, 0.5)),
            )
            uni = spiral_uniforms(s, W, H, float(rng.uniform(0, 1)))
            px, py = _pixels(W, H, seed=int(rng.integers(1_000_000)))
            m = spiral_map(uni, px, py)
            assert np.all(np.linalg.norm(m["q"] - c, axis=1) <= r_out + 1e-6)
            used = m["w_in"] > 0
            assert np.all(np.linalg.norm(m["q_in"][used] - c, axis=1) <= r_out + 1e-6)

    @pytest.mark.parametrize("size", [(480, 270), (3840, 2160), (1080, 1920)])
    def test_same_picture_at_every_resolution(self, size):
        """Normierte Punkte (relativ zu min(B, H)) bilden in jeder Aufloesung gleich ab."""
        s = _on(arms=1, rotation=15.0, ratio=3.0)

        def normalized(w, h):
            m0 = min(w, h)
            zn = np.random.default_rng(7).uniform(-0.5, 0.5, (500, 2))
            m = spiral_map(
                spiral_uniforms(s, w, h, 0.25),
                w / 2 + zn[:, 0] * m0,
                h / 2 + zn[:, 1] * m0,
            )
            return (m["q"] - [w / 2, h / 2]) / m0, m["w_in"]

        q_ref, w_ref = normalized(1920, 1080)
        q, w = normalized(*size)
        np.testing.assert_allclose(q, q_ref, atol=1e-9)
        np.testing.assert_allclose(w, w_ref, atol=1e-9)
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `pytest tests/test_spiral_zoom.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.spiral_zoom'`

- [ ] **Step 3: `beat_decay_frames` in `src/render_common.py` herausziehen**

Über `compute_beat_intensity` einfügen:

```python
def beat_decay_frames(fps: int) -> int:
    """Laenge der Beat-Huellkurve in Frames (1.0 am Beat, linear auf 0)."""
    return max(3, int(fps * 0.1))
```

In `compute_beat_intensity` die Zeile `decay_frames = max(3, int(fps * 0.1))` ersetzen durch:

```python
    decay_frames = beat_decay_frames(fps)
```

- [ ] **Step 4: `src/spiral_zoom.py` anlegen (Einstellungen, Uniforms, Abbildung)**

```python
"""
Spiral-Zoom (Droste/Escher) — reine Mathematik, ohne GPU.

Portiert aus dem Schwesterprojekt Fraktal-Zoom (src/zoom-math/escher.ts).
Der Effekt-Pass verbiegt die Visualizer-Ebene (nicht den Hintergrund) in
logarithmisch-polaren Koordinaten: ein Ring um die Bildmitte wird endlos
ineinander gestapelt. Mit Spiralarmen != 0 wird daraus eine Escher-Spirale.

Alles hier ist zustandslos und deterministisch:
- spiral_uniforms() berechnet die Shader-Uniforms,
- spiral_map() ist das CPU-Gegenstueck zum Fragment-Shader in
  src/gpu_spiral.py (die Tests vergleichen beide),
- compute_spiral_phase() macht aus Audio-Features die Zoom-Phase pro Frame.

Geometrie (Pixel, Ursprung unten links wie v_uv * Groesse):
- Zentrum F = Bildmitte, Aussenradius R_out = 0.5 * min(B, H)
- Zoom-Faktor K pro Ebene, Innenradius R = R_out / K
- Gerechnet wird in Radien relativ zu R_out (z = (Pixel - F) / R_out).
  Wichtig bei Spiralarmen: M hat dann einen Imaginaerteil, und log(Pixel)
  wuerde ueber ihn den Winkel je nach Aufloesung verschieben — normiert
  sieht das Bild in jeder Aufloesung gleich aus.
- Alle Abtastpunkte liegen im Inkreis: der Pass liest nie ausserhalb
  des Bildes, die Ecken entstehen aus der Rekursion.
- Die Abbildung ist in der Phase u periodisch mit Periode 1 (eine Ebene),
  deshalb bekommt der Shader nur u mod 1.

Escher-Abbildung: Die Quelle (Droste-Stapel im Log-Raum) wiederholt sich
entlang 2*pi*i und P = lnK - i*theta. Eine Bildschirm-Umdrehung muss auf
2*pi*i + p*P landen, also M = 1 + p*P/(2*pi*i). Nur ganzzahlige p schliessen
nahtlos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .render_common import beat_decay_frames

ARMS_MIN, ARMS_MAX = -3, 3
RATIO_MIN, RATIO_MAX = 1.5, 6.0
ROTATION_MAX_DEG = 90.0
SPEED_MAX = 2.0
ENERGY_MAX = 2.0
BEAT_MAX = 1.0
FEATHER_MIN, FEATHER_MAX = 0.02, 0.5


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _num(value, default: float) -> float:
    """Wandelt in float um; None, NaN und Unsinn fallen auf den Default zurueck."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


@dataclass(frozen=True)
class SpiralSettings:
    """Einstellungen des Spiral-Zoom-Passes (Bereiche siehe Modul-Konstanten)."""

    enabled: bool = False
    arms: int = 0          # Spiralarme, ganzzahlig (0 = gerader Droste-Zoom)
    ratio: float = 2.5     # Zoom-Faktor K pro Ebene
    rotation: float = 0.0  # Drehung pro Ebene in Grad
    speed: float = 0.3     # Ebenen pro Sekunde, negativ = heraus zoomen
    energy: float = 0.5    # Zusatz-Ebenen pro Sekunde bei rms = 1
    beat: float = 0.2      # Zusatz-Ebenen pro Beat
    feather: float = 0.15  # Ueberblendung an der Ebenen-Naht
    mix: float = 1.0       # 0 = Original, 1 = voller Effekt

    @property
    def is_active(self) -> bool:
        return self.enabled and self.mix > 0.0

    @classmethod
    def from_postprocess(cls, pp: dict | None) -> "SpiralSettings":
        """Liest die spiral_*-Schluessel aus dem postprocess-Dict.

        Klemmt selbst auf die gueltigen Bereiche, weil GUI, KI-Vorschlaege
        und alte Projektdateien das Pydantic-Schema umgehen.
        """
        pp = pp or {}
        d = cls()
        arms = int(round(_num(pp.get("spiral_arms"), d.arms)))
        return cls(
            enabled=bool(pp.get("spiral_enabled", d.enabled)),
            arms=int(_clamp(arms, ARMS_MIN, ARMS_MAX)),
            ratio=_clamp(_num(pp.get("spiral_ratio"), d.ratio), RATIO_MIN, RATIO_MAX),
            rotation=_clamp(
                _num(pp.get("spiral_rotation"), d.rotation), -ROTATION_MAX_DEG, ROTATION_MAX_DEG
            ),
            speed=_clamp(_num(pp.get("spiral_speed"), d.speed), -SPEED_MAX, SPEED_MAX),
            energy=_clamp(_num(pp.get("spiral_energy"), d.energy), 0.0, ENERGY_MAX),
            beat=_clamp(_num(pp.get("spiral_beat"), d.beat), 0.0, BEAT_MAX),
            feather=_clamp(_num(pp.get("spiral_feather"), d.feather), FEATHER_MIN, FEATHER_MAX),
            mix=_clamp(_num(pp.get("spiral_mix"), d.mix), 0.0, 1.0),
        )


def spiral_uniforms(settings: SpiralSettings, width: int, height: int, u: float) -> dict:
    """Shader-Uniforms fuer eine Bildgroesse und die Zoom-Phase u.

    u wird hier (float64) auf [0, 1) gefaltet — die Abbildung ist periodisch
    mit Periode 1, und float32 im Shader bliebe so auch nach Stunden genau.
    """
    ln_k = math.log(settings.ratio)
    theta = math.radians(settings.rotation)
    p = settings.arms
    return {
        "u_size": (float(width), float(height)),
        # Radien werden relativ zu R_out gerechnet (aufloesungsunabhaengig)
        "u_r_out": 0.5 * min(width, height),
        # Bildschirm -> Quelle: M = 1 + p*P/(2*pi*i) mit P = lnK - i*theta
        "u_m": (1.0 - p * theta / (2.0 * math.pi), -p * ln_k / (2.0 * math.pi)),
        # Ebenen-Periode im Log-Raum (komplex): lnK - i*theta
        "u_per": (ln_k, -theta),
        # ln(R / R_out) = ln(1 / K)
        "u_ln_r": -ln_k,
        "u_lf": math.log(1.0 - settings.feather),
        "u_u": float(u % 1.0),
        "u_mix": float(settings.mix),
    }


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def spiral_map(uni: dict, px, py) -> dict:
    """CPU-Gegenstueck zum Fragment-Shader (vektorisiert, float64).

    Args:
        uni: Ergebnis von spiral_uniforms().
        px, py: Pixelpositionen im Bild (Pixelmitte = i + 0.5).

    Returns:
        q / q_in: Abtastpunkte der Hauptebene und der naechsten Ebene innen,
        jac / jac_in: Quell-Texel pro Bildpixel (bestimmt die Mip-Stufe),
        w_in: Mischgewicht von q_in (Ueberblendung an der Naht).
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    w, h = uni["u_size"]
    mx, my = uni["u_m"]
    per_re, per_im = uni["u_per"]
    cx, cy = 0.5 * w, 0.5 * h
    r_out = uni["u_r_out"]

    # Radien relativ zu R_out; jac bleibt so Quell-Texel pro Bildpixel
    zx = (px - cx) / r_out
    zy = (py - cy) / r_out
    zr = np.maximum(np.hypot(zx, zy), 1e-6)
    wr = np.log(zr)
    wi = np.arctan2(zy, zx)
    u = uni["u_u"]
    sre = wr * mx - wi * my - u * per_re
    sim = wr * my + wi * mx - u * per_im

    k = -np.floor((sre - uni["u_ln_r"] - uni["u_lf"]) / per_re)
    pre = sre + k * per_re
    pim = sim + k * per_im
    m_abs = math.hypot(mx, my)

    rad = np.exp(pre)
    ln_in = pre + per_re
    rad_in = np.exp(ln_in)
    im_in = pim + per_im
    w_in = 1.0 - _smoothstep(uni["u_lf"], 0.0, pre - uni["u_ln_r"])
    return {
        "q": np.stack(
            [cx + r_out * rad * np.cos(pim), cy + r_out * rad * np.sin(pim)], axis=-1
        ),
        "jac": rad * m_abs / zr,
        "q_in": np.stack(
            [cx + r_out * rad_in * np.cos(im_in), cy + r_out * rad_in * np.sin(im_in)], axis=-1
        ),
        "jac_in": rad_in * m_abs / zr,
        "w_in": w_in,
    }
```

Hinweis: `beat_decay_frames` wird erst in Task 2 benutzt; der Import steht schon hier, damit Task 2 nur Funktionen ergänzt. `flake8` meldet ihn bis dahin als ungenutzt (F401) — das ist nach Task 2 weg.

- [ ] **Step 5: Tests laufen lassen**

Run: `pytest tests/test_spiral_zoom.py tests/test_beat_sync.py tests/test_gpu_renderer.py -v`
Expected: PASS (die beiden bestehenden Dateien belegen, dass `compute_beat_intensity` unverändert rechnet)

- [ ] **Step 6: Commit**

```bash
git add src/render_common.py src/spiral_zoom.py tests/test_spiral_zoom.py
git commit -m "feat(spiral): Droste/Escher-Mathematik mit CPU-Spiegel des Shaders"
```

---

### Task 2: Zoom-Phase aus Audio

**Files:**
- Modify: `src/spiral_zoom.py` (Funktionen anhängen)
- Test: `tests/test_spiral_zoom.py` (Klasse `TestPhase` anhängen)

**Interfaces:**
- Consumes: `SpiralSettings`, `spiral_uniforms` (Task 1), `render_common.beat_decay_frames`, `render_common.compute_beat_intensity`.
- Produces:
  - `spiral_zoom.compute_spiral_phase(rms, beat_intensity, fps: int, settings: SpiralSettings) -> np.ndarray` (float64, Länge `min(len(rms), len(beat_intensity))`, `phase[0] == 0`)
  - `spiral_zoom.phase_at_time(phase: np.ndarray, t: float, fps: int) -> float`

- [ ] **Step 1: Failing Tests anhängen**

Oben in `tests/test_spiral_zoom.py` die Imports erweitern:

```python
from src.render_common import compute_beat_intensity
from src.spiral_zoom import (
    SpiralSettings,
    compute_spiral_phase,
    phase_at_time,
    spiral_map,
    spiral_uniforms,
)
```

(die alte Zeile `from src.spiral_zoom import SpiralSettings, spiral_map, spiral_uniforms` ersetzen). Am Dateiende:

```python
class TestPhase:
    def test_silence_and_no_beats_gives_constant_speed(self):
        s = _on(speed=0.5, energy=1.0, beat=1.0)
        ph = compute_spiral_phase(np.zeros(300), np.zeros(300), 30, s)
        assert ph[0] == 0.0
        assert ph[30] == pytest.approx(0.5)  # 1 s * 0.5 Ebenen/s

    def test_everything_zero_is_standstill(self):
        s = _on(speed=0.0, energy=0.0, beat=0.0)
        ph = compute_spiral_phase(np.zeros(100), np.zeros(100), 30, s)
        assert np.all(ph == 0.0)

    def test_monotonic_in_direction(self):
        rng = np.random.default_rng(3)
        rms = rng.random(600)
        beats = compute_beat_intensity(np.arange(0, 600, 15), 600, 30)
        fwd = compute_spiral_phase(rms, beats, 30, _on(speed=0.3))
        back = compute_spiral_phase(rms, beats, 30, _on(speed=-0.3))
        assert np.all(np.diff(fwd) > 0)
        assert np.all(np.diff(back) < 0)
        np.testing.assert_allclose(back, -fwd)

    @pytest.mark.parametrize("fps", [24, 30, 60])
    def test_each_beat_adds_exactly_beat_levels(self, fps):
        beats = compute_beat_intensity(np.array([fps]), 3 * fps, fps)
        s = _on(speed=0.0, energy=0.0, beat=0.7)
        ph = compute_spiral_phase(np.zeros(3 * fps), beats, fps, s)
        assert ph[-1] == pytest.approx(0.7)

    def test_fps_independent(self):
        s = _on(speed=0.4, energy=0.8)
        a = compute_spiral_phase(np.full(300, 0.5), np.zeros(300), 30, s)
        b = compute_spiral_phase(np.full(600, 0.5), np.zeros(600), 60, s)
        assert a[150] == pytest.approx(b[300])  # beide nach 5 s

    def test_prefix_consistent(self):
        """Eine gekuerzte Vorschau (preview_mode) liefert dieselben Werte."""
        rng = np.random.default_rng(4)
        rms = rng.random(300)
        bf = np.arange(0, 300, 20)
        s = _on(speed=0.3, energy=0.5, beat=0.2)
        full = compute_spiral_phase(rms, compute_beat_intensity(bf, 300, 30), 30, s)
        short = compute_spiral_phase(rms[:90], compute_beat_intensity(bf, 90, 30), 30, s)
        np.testing.assert_array_equal(short, full[:90])

    def test_nan_and_length_mismatch(self):
        rms = np.array([0.5, np.nan, 0.5, 0.5])
        beats = np.array([0.0, 1.0, 0.0])
        ph = compute_spiral_phase(rms, beats, 30, _on())
        assert len(ph) == 3
        assert np.all(np.isfinite(ph))
        assert len(compute_spiral_phase([], [], 30, _on())) == 0

    def test_phase_at_time_matches_render_frame(self):
        ph = np.arange(300, dtype=np.float64) * 0.01
        for i in (0, 45, 299):
            assert phase_at_time(ph, i / 30, 30) == ph[i]
        assert phase_at_time(ph, 99.0, 30) == ph[-1]
        assert phase_at_time(ph, -1.0, 30) == ph[0]
        assert phase_at_time(np.zeros(0), 1.0, 30) == 0.0

    def test_long_song_u_stays_in_unit_interval(self):
        """1 h bei 60 fps mit Hoechsttempo: der Shader sieht trotzdem nur [0, 1)."""
        n = 3600 * 60
        s = _on(speed=2.0, energy=2.0)
        ph = compute_spiral_phase(np.ones(n), np.zeros(n), 60, s)
        assert ph[-1] > 10_000
        u = spiral_uniforms(s, W, H, ph[-1])["u_u"]
        assert 0.0 <= u < 1.0
        assert u == pytest.approx(ph[-1] - np.floor(ph[-1]))
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `pytest tests/test_spiral_zoom.py -v`
Expected: FAIL mit `ImportError: cannot import name 'compute_spiral_phase'`

- [ ] **Step 3: Funktionen an `src/spiral_zoom.py` anhängen**

```python
def compute_spiral_phase(rms, beat_intensity, fps: int, settings: SpiralSettings) -> np.ndarray:
    """Zoom-Phase in Ebenen fuer jeden Frame (float64, beginnt bei 0).

    Pro Frame waechst die Phase um |speed|/fps + energy*rms/fps
    + beat*beat_intensity/Beat-Flaeche. Die Beat-Flaeche ist die Summe einer
    einzelnen Beat-Huellkurve (compute_beat_intensity) — so bringt jeder Beat
    genau `beat` Ebenen, unabhaengig von der Framerate. Das Vorzeichen von
    speed bestimmt die Richtung fuer alle drei Anteile.

    Die Phase haengt nur von frueheren Frames ab: eine gekuerzte Vorschau
    liefert dieselben Werte wie der volle Export.
    """
    rms = np.clip(np.nan_to_num(np.asarray(rms, dtype=np.float64)), 0.0, 1.0)
    beats = np.clip(np.nan_to_num(np.asarray(beat_intensity, dtype=np.float64)), 0.0, 1.0)
    n = min(len(rms), len(beats))
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    fps = max(int(fps), 1)
    beat_area = (beat_decay_frames(fps) + 1) / 2.0
    rate = (abs(settings.speed) + settings.energy * rms[:n]) / fps
    rate = rate + settings.beat * beats[:n] / beat_area
    direction = -1.0 if settings.speed < 0 else 1.0
    phase = np.zeros(n, dtype=np.float64)
    phase[1:] = np.cumsum(rate[:-1])
    return direction * phase


def phase_at_time(phase: np.ndarray, t: float, fps: int) -> float:
    """Phase zum Zeitpunkt t (Frame = round(t * fps), an die Grenzen geklemmt)."""
    if len(phase) == 0:
        return 0.0
    i = int(round(t * fps))
    return float(phase[min(max(i, 0), len(phase) - 1)])
```

- [ ] **Step 4: Tests laufen lassen**

Run: `pytest tests/test_spiral_zoom.py -v`
Expected: PASS (alle Tests aus Task 1 und 2)

- [ ] **Step 5: Commit**

```bash
git add src/spiral_zoom.py tests/test_spiral_zoom.py
git commit -m "feat(spiral): Zoom-Phase aus Lautstaerke und Beats"
```

---

### Task 3: GPU-Pass `SpiralZoomPass`

**Files:**
- Create: `src/gpu_spiral.py`
- Test: `tests/test_gpu_spiral.py`

**Interfaces:**
- Consumes: `SpiralSettings`, `spiral_uniforms`, `spiral_map` (Task 1); `TEXTURED_VERTEX_SHADER`, `create_textured_quad` aus `src/gpu_visualizers/base.py`.
- Produces: `gpu_spiral.SpiralZoomPass(ctx: moderngl.Context, width: int, height: int)` mit
  - `apply(scene_fbo, settings: SpiralSettings, u: float) -> None` — verbiegt `scene_fbo` in place; tut nichts, wenn `not settings.is_active`
  - `release() -> None`

**GL-Fallen (hier behandeln):**
- Aus der Textur, in die man zeichnet, darf man nicht lesen → Szene erst in eine eigene Textur kopieren.
- **Kopie immer FBO → FBO:** `ctx.copy_framebuffer(textur, fbo)` (Ziel = `Texture`) klemmt unter moderngl 5.12 HDR-Werte auf 1.0 (bei der Planung gemessen: 4.0 → 1.0). `ctx.copy_framebuffer(fbo_um_textur, fbo)` erhält sie (4.0 → 4.0). Deshalb bekommt `self._src` eine eigene FBO `self._src_fbo`. `test_hdr_values_survive` sichert das ab.
- `textureLod` mit Stufe > 0 auf einer Textur ohne Mipmaps liefert Schwarz → nach jeder Kopie `build_mipmaps()`. Der Test `test_matches_cpu_mirror` deckt Bereiche mit Mip-Stufe > 0 ab und schlägt ohne Mipmaps fehl.
- Der Luma-Alpha-Blit davor kann Blending eingeschaltet lassen → vor dem Zeichnen `ctx.disable(moderngl.BLEND)` (überschreiben, nicht addieren).
- Kein `clamp`, kein sRGB.

- [ ] **Step 1: Failing GPU-Tests schreiben**

`tests/test_gpu_spiral.py`:

```python
"""GPU-Tests fuer den Spiral-Zoom-Pass: Shader gegen CPU-Spiegel."""

import moderngl
import numpy as np
import pytest

from src.gpu_spiral import SpiralZoomPass
from src.spiral_zoom import SpiralSettings, spiral_map, spiral_uniforms

pytestmark = pytest.mark.gpu

# Zweierpotenzen: Mip-Stufen teilen sauber, der lineare Verlauf bleibt exakt linear
W, H = 256, 128


@pytest.fixture
def gl():
    ctx = moderngl.create_standalone_context()
    yield ctx
    ctx.release()


def _scene(ctx, img):
    tex = ctx.texture((W, H), 4, img.astype("f2").tobytes(), dtype="f2")
    return ctx.framebuffer(color_attachments=[tex])


def _gradient():
    """Farbe = (x/B, y/H, 0.5, 1): bleibt in jeder Mip-Stufe linear."""
    img = np.zeros((H, W, 4), dtype=np.float32)
    img[..., 0] = ((np.arange(W) + 0.5) / W)[None, :]
    img[..., 1] = ((np.arange(H) + 0.5) / H)[:, None]
    img[..., 2] = 0.5
    img[..., 3] = 1.0
    return img


def _read(fbo):
    data = np.frombuffer(fbo.read(components=4, dtype="f2"), dtype=np.float16)
    return data.reshape(H, W, 4).astype(np.float32)


def _pixel_grid():
    py, px = np.mgrid[0:H, 0:W]
    return px.ravel() + 0.5, py.ravel() + 0.5


def test_disabled_leaves_scene_untouched(gl):
    img = _gradient()
    fbo = _scene(gl, img)
    SpiralZoomPass(gl, W, H).apply(fbo, SpiralSettings(enabled=False, arms=2), 0.3)
    np.testing.assert_array_equal(_read(fbo), img.astype("f2").astype(np.float32))


def test_hdr_values_survive(gl):
    """Kein clamp im Pass: HDR-Werte > 1 bleiben erhalten."""
    img = np.full((H, W, 4), 4.0, dtype=np.float32)
    fbo = _scene(gl, img)
    SpiralZoomPass(gl, W, H).apply(fbo, SpiralSettings(enabled=True, arms=1), 0.4)
    np.testing.assert_allclose(_read(fbo)[..., :3], 4.0, atol=1e-2)


def test_ring_is_identity_even_with_blending_left_on(gl):
    img = _gradient()
    fbo = _scene(gl, img)
    s = SpiralSettings(enabled=True, arms=0, rotation=0.0, ratio=2.5, feather=0.15)
    gl.enable(moderngl.BLEND)
    gl.blend_func = moderngl.ONE, moderngl.ONE  # wie nach dem Luma-Alpha-Blit
    SpiralZoomPass(gl, W, H).apply(fbo, s, 0.0)
    out = _read(fbo)
    px, py = _pixel_grid()
    zr = np.hypot(px - W / 2, py - H / 2)
    r_in = 0.5 * H / 2.5
    ring = (zr > r_in * 1.05) & (zr < r_in * 0.85 * 2.5 * 0.95)
    assert ring.sum() > 500
    np.testing.assert_allclose(
        out.reshape(-1, 4)[ring, :3], img.reshape(-1, 4)[ring, :3], atol=2e-3
    )


@pytest.mark.parametrize("arms", [0, 1, -2])
@pytest.mark.parametrize("rotation", [0.0, 30.0])
@pytest.mark.parametrize("u", [0.0, 0.37])
def test_matches_cpu_mirror(gl, arms, rotation, u):
    img = _gradient()
    fbo = _scene(gl, img)
    s = SpiralSettings(enabled=True, arms=arms, rotation=rotation, ratio=2.5)
    SpiralZoomPass(gl, W, H).apply(fbo, s, u)
    out = _read(fbo).reshape(-1, 4)

    px, py = _pixel_grid()
    m = spiral_map(spiral_uniforms(s, W, H, u), px, py)
    color = lambda q: np.stack([q[:, 0] / W, q[:, 1] / H], axis=-1)  # noqa: E731
    w_in = m["w_in"][:, None]
    expected = color(m["q"]) * (1 - w_in) + color(m["q_in"]) * w_in

    # Nur Pixel vergleichen, deren Abtastung weit genug vom Bildrand liegt
    # (Randklemmung) und nicht aus sehr groben Mip-Stufen kommt.
    c = np.array([W / 2, H / 2])
    r_lim = 0.85 * 0.5 * min(W, H)
    uses_in = m["w_in"] > 0
    ok = (
        (m["jac"] <= 4.0)
        & (np.linalg.norm(m["q"] - c, axis=1) <= r_lim)
        & (~uses_in | ((m["jac_in"] <= 4.0) & (np.linalg.norm(m["q_in"] - c, axis=1) <= r_lim)))
        & (np.hypot(px - W / 2, py - H / 2) >= 4.0)
    )
    assert ok.sum() > 200, "Test prueft zu wenige Pixel"
    np.testing.assert_allclose(out[ok, :2], expected[ok], atol=0.01)
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `pytest tests/test_gpu_spiral.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'src.gpu_spiral'` (ohne GPU: alle SKIPPED — dann auf einem Rechner mit GPU ausführen)

- [ ] **Step 3: `src/gpu_spiral.py` anlegen**

```python
"""
Spiral-Zoom-Pass (Droste/Escher) fuer den GPU-Renderer.

Verbiegt eine HDR-Ebene (im Renderer: die Visualizer-Ebene vor dem Blit
ueber den Hintergrund) in
logarithmisch-polaren Koordinaten. Mathematik und CPU-Gegenstueck stehen in
src/spiral_zoom.py — dieser Shader muss spiral_map() exakt folgen
(tests/test_gpu_spiral.py vergleicht beide).

Ablauf pro Frame:
1. Ebene in eine eigene Textur kopieren (aus der Ziel-Textur kann man nicht
   gleichzeitig lesen) und Mipmaps bauen — tiefe Ebenen werden stark
   verkleinert und wuerden ohne Mipmaps flimmern.
2. Vollbild-Pass zurueck in die Ebene, Blending aus (ueberschreiben).
   Alpha wird mit verbogen (Luma-/Occlusion-Alpha im Blit bleibt gueltig).

HDR bleibt erhalten: kein clamp, kein sRGB — Tonemapping macht zentral der
Renderer.
"""

import moderngl

from .app_logging import get_logger
from .gpu_visualizers.base import TEXTURED_VERTEX_SHADER, create_textured_quad
from .spiral_zoom import SpiralSettings, spiral_uniforms

logger = get_logger(__name__)


_SPIRAL_FRAGMENT = """
#version 330
uniform sampler2D u_src;
uniform vec2 u_size;
uniform float u_r_out;
uniform vec2 u_m;
uniform vec2 u_per;
uniform float u_ln_r;
uniform float u_lf;
uniform float u_u;
uniform float u_mix;
in vec2 v_uv;
out vec4 f_color;

// Abtastung auf Radius R_out * exp(ln_rad) und Winkel ang um die Bildmitte.
// Die Mip-Stufe folgt dem Massstab der Abbildung (Quell-Texel pro Pixel).
vec4 sample_at(float ln_rad, float ang, float jac) {
    vec2 q = 0.5 * u_size + u_r_out * exp(ln_rad) * vec2(cos(ang), sin(ang));
    return textureLod(u_src, q / u_size, log2(max(jac, 1e-6)));
}

void main() {
    vec4 orig = textureLod(u_src, v_uv, 0.0);
    // Radien relativ zu R_out (aufloesungsunabhaengig, auch mit Spiralarmen)
    vec2 z = (v_uv * u_size - 0.5 * u_size) / u_r_out;
    float zr = max(length(z), 1e-6);
    vec2 w = vec2(log(zr), atan(z.y, z.x));
    // Bildschirm -> Quelle (komplexe Multiplikation mit M), dann Phase abziehen
    vec2 ws = vec2(w.x * u_m.x - w.y * u_m.y, w.x * u_m.y + w.y * u_m.x) - u_u * u_per;
    // Um ganze Ebenen verschieben, bis der Punkt im Quell-Ring liegt
    float k = -floor((ws.x - u_ln_r - u_lf) / u_per.x);
    vec2 p = ws + k * u_per;
    float m_abs = length(u_m);
    vec4 col = sample_at(p.x, p.y, exp(p.x) * m_abs / zr);
    // An der Innenkante des Rings weich in die naechste Ebene ueberblenden
    float w_in = 1.0 - smoothstep(u_lf, 0.0, p.x - u_ln_r);
    if (w_in > 0.0) {
        float ln_in = p.x + u_per.x;
        col = mix(col, sample_at(ln_in, p.y + u_per.y, exp(ln_in) * m_abs / zr), w_in);
    }
    f_color = mix(orig, col, u_mix);
}
"""


class SpiralZoomPass:
    """Droste/Escher-Verbiegung einer HDR-Ebene (in place).

    Der Aufrufer rendert die Ebene in ein f16-FBO und ruft danach
    apply(fbo, settings, u) auf — im Renderer vor dem Visualizer-Blit.
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        self._src = ctx.texture((width, height), 4, dtype="f2")
        self._src.repeat_x = False
        self._src.repeat_y = False
        # Kopie FBO -> FBO: eine Kopie direkt in die Textur klemmt HDR auf 1.0
        self._src_fbo = ctx.framebuffer(color_attachments=[self._src])
        self._prog = ctx.program(
            vertex_shader=TEXTURED_VERTEX_SHADER, fragment_shader=_SPIRAL_FRAGMENT
        )
        self._vao, self._vbo = create_textured_quad(ctx, self._prog)

    def apply(self, scene_fbo, settings: SpiralSettings, u: float):
        """Verbiegt scene_fbo in place; tut nichts, wenn der Effekt aus ist."""
        if not settings.is_active:
            return
        self.ctx.copy_framebuffer(self._src_fbo, scene_fbo)
        self._src.build_mipmaps()
        self._src.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        scene_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        for name, value in spiral_uniforms(settings, self.width, self.height, u).items():
            self._set(name, value)
        self._set("u_src", 0)
        self._src.use(location=0)
        self._vao.render(mode=moderngl.TRIANGLE_STRIP)

    def _set(self, name: str, value):
        # Vom Treiber wegoptimierte Uniforms fehlen im Programm (KeyError)
        try:
            self._prog[name].value = value
        except KeyError:
            pass

    def release(self):
        """Gibt alle GPU-Ressourcen des Passes frei."""
        for obj in (self._src_fbo, self._src, self._prog, self._vao, self._vbo):
            try:
                obj.release()
            except Exception:
                pass
```

- [ ] **Step 4: Tests laufen lassen**

Run: `pytest tests/test_gpu_spiral.py -v`
Expected: PASS (15 Tests: 3 einzelne + 12 Parameter-Kombinationen). Schlägt `test_matches_cpu_mirror` nur in Randpixeln knapp fehl, **nicht** die Toleranz aufweichen, sondern prüfen, ob Shader und `spiral_map` wirklich dieselbe Formel rechnen.

- [ ] **Step 5: Commit**

```bash
git add src/gpu_spiral.py tests/test_gpu_spiral.py
git commit -m "feat(spiral): GPU-Pass fuer Droste/Escher-Zoom mit Mipmap-Abtastung"
```

---

### Task 4: Schema, CLI-Durchreichung und Beispiel-Preset

**Files:**
- Modify: `config/schemas.py:182-201` (`PostProcessConfig`)
- Create: `config/music_spiral_zoom.json`
- Test: `tests/test_spiral_config.py`

**Interfaces:**
- Consumes: `SpiralSettings` (Task 1).
- Produces: `PostProcessConfig.spiral_*`-Felder (Namen und Standards exakt wie Tabelle „Entscheidungen“). `main.py` braucht keine Änderung: `cfg.postprocess.model_dump()` (main.py:143) reicht die neuen Felder automatisch durch — **ohne** Schemafelder würde Pydantic sie stillschweigend verwerfen.

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_spiral_config.py`:

```python
"""Spiral-Zoom in Config-Schema und Presets."""

import json
from dataclasses import fields

import pytest

from config.schemas import PostProcessConfig, load_and_validate_config
from src.spiral_zoom import SpiralSettings


def test_schema_defaults_match_renderer_defaults():
    cfg = PostProcessConfig()
    d = SpiralSettings()
    for f in fields(SpiralSettings):
        assert getattr(cfg, f"spiral_{f.name}") == getattr(d, f.name), f.name


def test_bounds_and_integer_arms():
    with pytest.raises(Exception):
        PostProcessConfig(spiral_arms=4)
    with pytest.raises(Exception):
        PostProcessConfig(spiral_arms=1.5)
    with pytest.raises(Exception):
        PostProcessConfig(spiral_ratio=1.0)
    with pytest.raises(Exception):
        PostProcessConfig(spiral_speed=3.0)
    assert PostProcessConfig(spiral_arms=-3).spiral_arms == -3


def test_config_file_reaches_render_postprocess(tmp_path):
    """Spiral-Schluessel ueberleben load_and_validate_config + model_dump (CLI-Pfad)."""
    cfg_path = tmp_path / "spiral.json"
    cfg_path.write_text(json.dumps({
        "audio_file": "input.mp3",
        "output_file": "out.mp4",
        "visual": {"type": "spectrum_bars", "resolution": [1280, 720], "fps": 30},
        "postprocess": {"spiral_enabled": True, "spiral_arms": 2, "spiral_speed": -0.5},
    }), encoding="utf-8")
    pp = load_and_validate_config(str(cfg_path)).postprocess.model_dump()
    s = SpiralSettings.from_postprocess(pp)
    assert s.is_active
    assert s.arms == 2
    assert s.speed == -0.5


def test_old_config_without_spiral_keys_stays_off():
    pp = load_and_validate_config("config/music_aggressive.json").postprocess.model_dump()
    assert SpiralSettings.from_postprocess(pp).is_active is False


def test_example_preset_is_valid_and_active():
    pp = load_and_validate_config("config/music_spiral_zoom.json").postprocess.model_dump()
    assert SpiralSettings.from_postprocess(pp).is_active
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `pytest tests/test_spiral_config.py -v`
Expected: FAIL (`AttributeError: 'PostProcessConfig' object has no attribute 'spiral_enabled'` bzw. fehlende Datei `config/music_spiral_zoom.json`)

- [ ] **Step 3: Felder in `PostProcessConfig` ergänzen**

In `config/schemas.py` nach `lut_strength: float = Field(default=1.0, ge=0.0, le=1.0)` einfügen:

```python
    # Spiral-Zoom (Droste/Escher) — Bedeutung siehe src/spiral_zoom.py
    spiral_enabled: bool = False
    spiral_arms: int = Field(default=0, ge=-3, le=3)
    spiral_ratio: float = Field(default=2.5, ge=1.5, le=6.0)
    spiral_rotation: float = Field(default=0.0, ge=-90.0, le=90.0)
    spiral_speed: float = Field(default=0.3, ge=-2.0, le=2.0)
    spiral_energy: float = Field(default=0.5, ge=0.0, le=2.0)
    spiral_beat: float = Field(default=0.2, ge=0.0, le=1.0)
    spiral_feather: float = Field(default=0.15, ge=0.02, le=0.5)
    spiral_mix: float = Field(default=1.0, ge=0.0, le=1.0)
```

- [ ] **Step 4: Beispiel-Preset `config/music_spiral_zoom.json` anlegen**

```json
{
  "audio_file": "input.mp3",
  "output_file": "spiral_zoom_output.mp4",
  "visual": {
    "type": "neon_wave_circle",
    "resolution": [1920, 1080],
    "fps": 30,
    "colors": {
      "primary": "#00E5FF",
      "secondary": "#FF3DA5",
      "background": "#05050F"
    },
    "params": {}
  },
  "postprocess": {
    "bloom_intensity": 0.8,
    "spiral_enabled": true,
    "spiral_arms": 1,
    "spiral_ratio": 2.5,
    "spiral_rotation": 0.0,
    "spiral_speed": 0.3,
    "spiral_energy": 0.6,
    "spiral_beat": 0.25,
    "spiral_mix": 1.0
  },
  "background_color": "#05050F"
}
```

- [ ] **Step 5: Tests laufen lassen**

Run: `pytest tests/test_spiral_config.py tests/test_gpu_bloom.py tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add config/schemas.py config/music_spiral_zoom.json tests/test_spiral_config.py
git commit -m "feat(spiral): Schema-Felder und Beispiel-Preset music_spiral_zoom"
```

---

### Task 5: Renderer und Live-Vorschau

**Files:**
- Modify: `src/gpu_renderer.py` (Imports ~Z. 19-28; `__init__` ~Z. 122-127; `render()` nach `features_dict` ~Z. 303 und in der Schleife nach dem Visualizer-Render ~Z. 507-510; neue Methoden neben `_apply_bloom` ~Z. 1205; `release()` ~Z. 1662)
- Modify: `src/gpu_preview.py` (Imports ~Z. 20; nach dem Visualizer-Render, vor dem Blit ~Z. 164)
- Test: `tests/test_spiral_render_integration.py`

**Interfaces:**
- Consumes: `SpiralZoomPass` (Task 3), `SpiralSettings`, `compute_spiral_phase`, `phase_at_time` (Task 1/2).
- Produces:
  - `GPUBatchRenderer._apply_spiral(settings: SpiralSettings, u: float, target_fbo) -> None` — legt den Pass beim ersten aktiven Aufruf an und verbiegt `target_fbo` in place.
  - `GPUBatchRenderer._viz_fbo_holding(tex) -> Framebuffer` — die FBO, deren Farbtextur `tex` ist (`viz_fbo` oder bei Timeline-Überblendung `viz_fbo_blend`).

Aufgerufen an zwei Stellen — Export-Schleife und Vorschau —, jeweils **nach dem Visualizer-Render und vor `_blit_viz_to_fbo`**. So wird nur die Visualizer-Ebene verbogen; das Hintergrundbild bleibt ruhig (Entscheidung 1). Der Timeline-Pfad liefert `active_viz_tex` aus `viz_fbo` oder `viz_fbo_blend` und bekommt den Effekt über `_viz_fbo_holding` automatisch. Die dritte Render-Schleife (`src/studio/probe.py`) bekommt ihn bewusst **nicht**; Task 6 schaltet ihn im Studio ab.

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_spiral_render_integration.py`:

```python
"""Spiral-Zoom im Render-Pfad: Export-Schleife und Live-Vorschau."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.gpu_preview as gpu_preview
from src.gpu_renderer import GPUBatchRenderer
from src.gpu_visualizers import VISUALIZER_MAP
from src.gpu_visualizers.base import BaseGPUVisualizer
from src.render_common import compute_beat_intensity
from src.spiral_zoom import SpiralSettings, compute_spiral_phase
from tests.test_gpu_preview import _make_mock_renderer, dummy_features  # noqa: F401
from tests.test_gpu_renderer import mock_features, mock_gl_context  # noqa: F401

SPIRAL_ON = {"spiral_enabled": True, "spiral_arms": 1, "spiral_speed": 0.5, "spiral_beat": 0.3}


def _render_mocked(mock_run, mock_popen, mock_create_ctx, ctx, features, postprocess, tmp_path):
    mock_create_ctx.return_value = ctx
    process = MagicMock()
    process.poll.return_value = None
    process.returncode = 0
    mock_popen.return_value = process
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"")
    renderer = GPUBatchRenderer(width=64, height=64, fps=30)
    with patch.object(GPUBatchRenderer, "_apply_spiral") as spy:
        renderer.render(
            audio_path=str(audio),
            visualizer_type="lumina_core",
            output_path=str(tmp_path / "o.mp4"),
            features=features,
            preview_mode=True,
            preview_duration=0.1,  # 3 Frames
            postprocess=postprocess,
        )
    return spy, renderer


@patch("src.gpu_renderer.moderngl.create_standalone_context")
@patch("src.gpu_renderer.subprocess.Popen")
@patch("src.gpu_renderer.subprocess.run")
def test_batch_render_applies_spiral_per_frame(
    mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features, tmp_path
):
    spy, renderer = _render_mocked(
        mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features, SPIRAL_ON, tmp_path
    )
    s = SpiralSettings.from_postprocess(SPIRAL_ON)
    expected = compute_spiral_phase(
        mock_features.rms[:3], compute_beat_intensity(mock_features.beat_frames, 3, 30), 30, s
    )
    assert spy.call_count == 3
    for call, u in zip(spy.call_args_list, expected):
        settings, got_u, target = call.args
        assert settings == s
        assert got_u == pytest.approx(u)
        assert target is renderer.viz_fbo  # Visualizer-Ebene, nicht die Szene


@patch("src.gpu_renderer.moderngl.create_standalone_context")
@patch("src.gpu_renderer.subprocess.Popen")
@patch("src.gpu_renderer.subprocess.run")
def test_batch_render_without_spiral_never_calls_pass(
    mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features, tmp_path
):
    spy, _ = _render_mocked(
        mock_run, mock_popen, mock_create_ctx, mock_gl_context, mock_features,
        {"bloom_intensity": 0.6}, tmp_path,
    )
    spy.assert_not_called()


@patch("src.gpu_preview.AudioAnalyzer")
@patch("src.gpu_preview.GPUPreviewRenderer")
def test_preview_uses_same_phase_as_export_frame(mock_renderer_cls, mock_analyzer_cls, dummy_features):
    mock_renderer = _make_mock_renderer()
    mock_renderer_cls.return_value = mock_renderer
    gpu_preview.render_gpu_preview(
        audio_path="dummy.mp3", visualizer_type="lumina_core", width=480, height=270,
        fps=30, features=dummy_features, preview_time_percent=0.5, postprocess=SPIRAL_ON,
    )
    s = SpiralSettings.from_postprocess(SPIRAL_ON)
    full = compute_spiral_phase(
        dummy_features.rms, compute_beat_intensity(dummy_features.beat_frames, 300, 30), 30, s
    )
    mock_renderer._apply_spiral.assert_called_once()
    settings, u, target = mock_renderer._apply_spiral.call_args.args
    assert settings == s
    assert u == pytest.approx(full[150])  # 10 s * 0.5 = 5 s = Frame 150
    assert target is mock_renderer.viz_fbo


@patch("src.gpu_preview.AudioAnalyzer")
@patch("src.gpu_preview.GPUPreviewRenderer")
def test_preview_without_spiral_never_calls_pass(mock_renderer_cls, mock_analyzer_cls, dummy_features):
    mock_renderer = _make_mock_renderer()
    mock_renderer_cls.return_value = mock_renderer
    gpu_preview.render_gpu_preview(
        audio_path="dummy.mp3", visualizer_type="lumina_core", width=480, height=270,
        fps=30, features=dummy_features, postprocess={"contrast": 1.1},
    )
    mock_renderer._apply_spiral.assert_not_called()


def _gradient_photo(tmp_path) -> str:
    from PIL import Image

    grad = np.linspace(0, 255, 160).astype(np.uint8)
    img = np.stack([np.tile(grad, (90, 1))] * 3, axis=-1)
    path = str(tmp_path / "bg.png")
    Image.fromarray(img).save(path)
    return path


def _preview(**kw) -> np.ndarray:
    img = gpu_preview.render_gpu_preview(
        audio_path="dummy.mp3", width=160, height=90, fps=30,
        preview_time_percent=0.5, background_opacity=1.0, **kw,
    )
    return np.asarray(img, dtype=np.int16)


@pytest.mark.gpu
@pytest.mark.parametrize("with_bg", [False, True])
def test_real_preview_off_is_bitidentical_on_differs(tmp_path, dummy_features, with_bg):
    """Echte GPU: aus = bitgleich zu ohne Schluessel; an = sichtbar anders (auch mit Foto)."""
    common = dict(
        visualizer_type="spectrum_bars", features=dummy_features,
        background_image=_gradient_photo(tmp_path) if with_bg else None,
    )
    base = _preview(**common)
    off = _preview(**common, postprocess={"spiral_enabled": False, "spiral_arms": 2})
    on = _preview(
        **common, postprocess={"spiral_enabled": True, "spiral_arms": 1, "spiral_rotation": 30}
    )
    assert np.array_equal(base, off)
    assert np.abs(on - base).mean() > 0.5


class _LeererViz(BaseGPUVisualizer):
    """Zeichnet nichts: die Visualizer-Ebene bleibt transparent."""

    def _setup(self):
        pass

    def render(self, features, time):
        pass


@pytest.mark.gpu
def test_real_preview_photo_stays_still(tmp_path, dummy_features, monkeypatch):
    """Nur der Visualizer wird verbogen: ohne Visualizer-Inhalt bleibt das Foto bitgleich."""
    monkeypatch.setitem(VISUALIZER_MAP, "_leer_test", _LeererViz)
    common = dict(
        visualizer_type="_leer_test", features=dummy_features,
        background_image=_gradient_photo(tmp_path),
    )
    off = _preview(**common)
    on = _preview(
        **common, postprocess={"spiral_enabled": True, "spiral_arms": 1, "spiral_rotation": 30}
    )
    assert np.array_equal(off, on)
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `pytest tests/test_spiral_render_integration.py -v`
Expected: FAIL — `AttributeError: <class 'GPUBatchRenderer'> does not have the attribute '_apply_spiral'` bzw. `AssertionError: Expected '_apply_spiral' to have been called once`. `test_real_preview_photo_stays_still` besteht schon vorher (ohne Pass ändert sich nichts) — er sichert ab, dass der Einbau das Foto nicht anfasst.

- [ ] **Step 3: Renderer anbinden (`src/gpu_renderer.py`)**

Imports ergänzen (nach `from .gpu_bloom import BloomPass, load_cube_lut`):

```python
from .gpu_spiral import SpiralZoomPass
from .spiral_zoom import SpiralSettings, compute_spiral_phase, phase_at_time
```

In `__init__` nach dem Bloom-`try/except`-Block:

```python
        # Spiral-Zoom-Pass (Droste/Escher): erst beim ersten aktiven Frame
        # angelegt — ist der Effekt aus, belegt er keinerlei GPU-Speicher.
        # False = Anlegen ist fehlgeschlagen, nicht jeden Frame neu versuchen.
        self._spiral = None
```

In `render()` direkt nach `features_dict = build_features_dict(features, frame_count, self.fps)`:

```python
        # Spiral-Zoom: Phase einmal vorab aus den Features (zustandslos,
        # Vorschau bei Zeit t == Export-Frame bei t)
        spiral = SpiralSettings.from_postprocess(postprocess)
        spiral_phase = None
        if spiral.is_active:
            spiral_phase = compute_spiral_phase(
                features_dict["rms"], features_dict["beat_intensity"], self.fps, spiral
            )
```

In der Frame-Schleife direkt nach dem `_DEBUG`-Block mit `debug_step3_after_viz.png` und **vor** dem folgenden `self.fbo.use()` (also vor dem Blit):

```python
                    # Spiral-Zoom verbiegt nur die Visualizer-Ebene (vor dem
                    # Blit) — ein Hintergrundbild bleibt ruhig stehen
                    if spiral_phase is not None:
                        self._apply_spiral(
                            spiral,
                            phase_at_time(spiral_phase, time, self.fps),
                            self._viz_fbo_holding(active_viz_tex),
                        )
```

Neue Methoden direkt vor `def _apply_bloom`:

```python
    def _viz_fbo_holding(self, tex):
        """FBO, deren Farbtextur tex ist (viz_fbo oder Timeline-Ueberblendung)."""
        blend = getattr(self, "viz_fbo_blend", None)
        if blend is not None and tex is blend.color_attachments[0]:
            return blend
        return self.viz_fbo

    def _apply_spiral(self, settings: SpiralSettings, u: float, target_fbo):
        """Verbiegt target_fbo (Visualizer-Ebene) per Spiral-Zoom.

        Legt den Pass beim ersten aktiven Aufruf an.
        """
        if not settings.is_active:
            return
        if self._spiral is None:
            try:
                self._spiral = SpiralZoomPass(self.ctx, self.width, self.height)
            except Exception as e:
                logger.warning(f"[GPU] Spiral-Zoom nicht verfuegbar: {e}")
                self._spiral = False
        if self._spiral:
            self._spiral.apply(target_fbo, settings, u)
```

In `release()` direkt nach dem Bloom-Block (`self._bloom = None`):

```python
            if getattr(self, "_spiral", None):
                self._spiral.release()
            self._spiral = None
```

- [ ] **Step 4: Vorschau anbinden (`src/gpu_preview.py`)**

Import ergänzen (nach `from .render_common import build_features_dict`):

```python
from .spiral_zoom import SpiralSettings, compute_spiral_phase, phase_at_time
```

Nach dem Visualizer-Render (`if getattr(renderer, "viz_ms_fbo", None) is not None: ... else: ...`) und **vor** dem Kommentar `# Visualizer von viz_fbo auf main fbo blitten (mit Offset/Scale)`:

```python
        # Spiral-Zoom wie im Haupt-Renderer: nur die Visualizer-Ebene, vor
        # dem Blit; gleiche Phase wie der Export-Frame zum Vorschau-Zeitpunkt
        spiral = SpiralSettings.from_postprocess(postprocess)
        if spiral.is_active:
            spiral_phase = compute_spiral_phase(
                features_dict["rms"], features_dict["beat_intensity"], fps, spiral
            )
            renderer._apply_spiral(
                spiral, phase_at_time(spiral_phase, preview_time, fps), renderer.viz_fbo
            )
```

- [ ] **Step 5: Tests laufen lassen**

Run: `pytest tests/test_spiral_render_integration.py tests/test_gpu_renderer.py tests/test_gpu_renderer_extended.py tests/test_gpu_renderer_timeline.py tests/test_gpu_preview.py tests/test_golden_corpus.py -v`
Expected: PASS (Golden-Corpus unverändert = Effekt aus ist bitgleich)

- [ ] **Step 6: Commit**

```bash
git add src/gpu_renderer.py src/gpu_preview.py tests/test_spiral_render_integration.py
git commit -m "feat(spiral): Spiral-Zoom auf die Visualizer-Ebene in Export und Vorschau"
```

---

### Task 6: Studio-Modus schaltet den Spiral-Zoom ab

**Files:**
- Modify: `src/studio/engine.py` (neue Funktion vor `def run_studio` ~Z. 155; Aufruf nach `mask_warnings: list[str] = []` ~Z. 173)
- Test: `tests/test_studio_spiral.py`

**Interfaces:**
- Consumes: nichts aus früheren Tasks außer dem Schlüssel `spiral_enabled`.
- Produces: `src.studio.engine.disable_spiral_for_studio(postprocess: dict | None) -> tuple[dict, list[str]]` (Kopie mit `spiral_enabled = False` + Warnungen fürs Sidecar).

Warum: `run_studio` misst mit `ProbeRenderer` (eigene Render-Schleife ohne Spiral-Pass) und rendert danach über `GPUBatchRenderer.render()` (mit Spiral-Pass). Ohne Abschalten würden Messung und Ergebnis auseinanderlaufen.

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_studio_spiral.py`:

```python
"""Studio-Modus: Spiral-Zoom wird abgeschaltet (Messungen setzen ein unverbogenes Bild voraus)."""

from unittest.mock import MagicMock, patch

import pytest

from src.studio.engine import disable_spiral_for_studio, run_studio


def test_helper_switches_spiral_off_with_warning():
    pp, warnings = disable_spiral_for_studio({"spiral_enabled": True, "bloom_intensity": 0.5})
    assert pp["spiral_enabled"] is False
    assert pp["bloom_intensity"] == 0.5
    assert len(warnings) == 1
    assert "Spiral-Zoom" in warnings[0]


def test_helper_leaves_other_configs_alone():
    original = {"contrast": 1.2}
    pp, warnings = disable_spiral_for_studio(original)
    assert pp == original
    assert pp is not original  # Kopie, Aufrufer-Dict bleibt unberuehrt
    assert warnings == []
    assert disable_spiral_for_studio(None) == ({}, [])


class _Stop(Exception):
    pass


def test_run_studio_measures_without_spiral():
    """Schon der Probe-Solve bekommt spiral_enabled = False."""
    seen = {}

    def fake_solve(probe, viz_factory, features_dict, plan, postprocess, *args, **kwargs):
        seen.update(postprocess)
        raise _Stop

    with patch("src.studio.engine.check_feasibility", return_value=MagicMock(should_render=True)), \
         patch("src.studio.engine.build_sample_plan", return_value=MagicMock()), \
         patch("src.studio.engine.solve_constraints", side_effect=fake_solve), \
         patch("src.studio.probe.ProbeRenderer"):
        with pytest.raises(_Stop):
            run_studio(
                audio_path="song.mp3", visualizer="lumina_core", features=None,
                features_dict={}, output_path="out.mp4",
                postprocess={"spiral_enabled": True, "spiral_arms": 1},
            )
    assert seen["spiral_enabled"] is False
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `pytest tests/test_studio_spiral.py -v`
Expected: FAIL mit `ImportError: cannot import name 'disable_spiral_for_studio'`

- [ ] **Step 3: Funktion und Aufruf in `src/studio/engine.py`**

Direkt vor `def run_studio(`:

```python
def disable_spiral_for_studio(postprocess: dict | None) -> tuple[dict, list[str]]:
    """Schaltet den Spiral-Zoom fuer den Studio-Modus ab.

    Die Studio-Messungen (Beitrag des Visualizers, Abstand zum Motiv) laufen
    im ProbeRenderer, der keinen Spiral-Pass hat — Messung und Commit-Render
    liefen sonst auseinander. Gibt eine Kopie und ggf. eine Warnung zurueck.
    """
    pp = dict(postprocess or {})
    if not pp.get("spiral_enabled"):
        return pp, []
    pp["spiral_enabled"] = False
    return pp, [
        "Spiral-Zoom im Studio-Modus deaktiviert: die Studio-Messungen "
        "laufen ohne Spiral-Pass und passten sonst nicht zum Ergebnis."
    ]
```

In `run_studio` direkt nach `mask_warnings: list[str] = []`:

```python
    postprocess, spiral_warnings = disable_spiral_for_studio(postprocess)
    mask_warnings.extend(spiral_warnings)
```

(`mask_warnings` landet bereits im Sidecar unter `"warnings"`.)

- [ ] **Step 4: Tests laufen lassen**

Run: `pytest tests/test_studio_spiral.py tests/test_studio_engine.py tests/test_studio_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/studio/engine.py tests/test_studio_spiral.py
git commit -m "feat(studio): Spiral-Zoom im Studio-Modus abschalten (Messungen unverbogen)"
```

---

### Task 7: GUI-Zustand und Hauptfenster

**Files:**
- Modify: `src/gui/state.py` (Modulkopf, `_STATE_KEYS` Z. 11-26, `__init__` Z. 56-65, `get_postprocess` Z. 115-127, `to_dict` Z. 145-175)
- Modify: `src/gui/main_window.py` (Import Z. 25, `_PROJECT_KEYS` Z. 429-439, `_on_state_changed` Z. 441-453)
- Test: `tests/test_gui_state.py`, `tests/test_gui_main_window.py` (Tests anhängen)

**Interfaces:**
- Consumes: `SpiralSettings` (Task 1).
- Produces:
  - `src.gui.state.SPIRAL_STATE_KEYS: tuple[str, ...]` = `("pp_spiral_enabled", "pp_spiral_arms", "pp_spiral_ratio", "pp_spiral_rotation", "pp_spiral_speed", "pp_spiral_energy", "pp_spiral_beat", "pp_spiral_mix")`
  - `AppState.pp_spiral_*`-Attribute (Standards aus `SpiralSettings()`); `get_postprocess()` liefert zusätzlich `spiral_enabled … spiral_mix` (Schlüssel ohne `pp_`)
  - `MainWindow._PREVIEW_KEYS: frozenset[str]` (neu, ersetzt das Inline-Set in `_on_state_changed`)

- [ ] **Step 1: Failing Tests anhängen**

An `tests/test_gui_state.py` anhängen (Imports oben ergänzen):

```python
from src.gui.state import SPIRAL_STATE_KEYS
from src.spiral_zoom import SpiralSettings


def test_spiral_settings_roundtrip_and_reach_postprocess():
    s = AppState()
    s.pp_spiral_enabled = True
    s.pp_spiral_arms = -2
    s.pp_spiral_speed = -0.75
    restored = AppState.from_dict(s.to_dict())
    assert restored.pp_spiral_enabled is True
    assert restored.pp_spiral_arms == -2
    assert restored.pp_spiral_speed == -0.75
    pp = restored.get_postprocess()
    assert pp["spiral_enabled"] is True
    assert pp["spiral_arms"] == -2
    assert SpiralSettings.from_postprocess(pp).is_active


def test_old_project_without_spiral_keys_keeps_effect_off():
    data = AppState().to_dict()
    for key in SPIRAL_STATE_KEYS:
        data.pop(key)
    restored = AppState.from_dict(data)
    assert restored.pp_spiral_enabled is False
    assert not SpiralSettings.from_postprocess(restored.get_postprocess()).is_active


def test_spiral_state_defaults_match_renderer_defaults():
    s = AppState()
    d = SpiralSettings()
    for key in SPIRAL_STATE_KEYS:
        assert getattr(s, key) == getattr(d, key[len("pp_spiral_"):]), key


def test_spiral_keys_emit_changed(qtbot):
    s = AppState()
    received = []
    s.changed.connect(received.append)
    s.pp_spiral_ratio = 4.0
    assert "pp_spiral_ratio" in received
```

An `tests/test_gui_main_window.py` anhängen:

```python
def test_spiral_keys_trigger_preview_and_dirty_marker():
    from src.gui.main_window import MainWindow
    from src.gui.state import SPIRAL_STATE_KEYS

    assert set(SPIRAL_STATE_KEYS) <= MainWindow._PREVIEW_KEYS
    assert set(SPIRAL_STATE_KEYS) <= MainWindow._PROJECT_KEYS
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `pytest tests/test_gui_state.py tests/test_gui_main_window.py -v`
Expected: FAIL mit `ImportError: cannot import name 'SPIRAL_STATE_KEYS'`

- [ ] **Step 3: `src/gui/state.py` erweitern**

Import ergänzen und Konstante vor `class AppState` anlegen:

```python
from src.spiral_zoom import SpiralSettings

# Spiral-Zoom-Regler; im postprocess-Dict heissen sie ohne "pp_"-Praefix
SPIRAL_STATE_KEYS = (
    "pp_spiral_enabled", "pp_spiral_arms", "pp_spiral_ratio", "pp_spiral_rotation",
    "pp_spiral_speed", "pp_spiral_energy", "pp_spiral_beat", "pp_spiral_mix",
)
```

`_STATE_KEYS` erweitern — die schließende Zeile `})` des `frozenset({...})` wird zu:

```python
    }) | frozenset(SPIRAL_STATE_KEYS)
```

In `__init__` nach `self.pp_chromatic: float = 0.0`:

```python
        _spiral = SpiralSettings()
        self.pp_spiral_enabled: bool = _spiral.enabled
        self.pp_spiral_arms: int = _spiral.arms
        self.pp_spiral_ratio: float = _spiral.ratio
        self.pp_spiral_rotation: float = _spiral.rotation
        self.pp_spiral_speed: float = _spiral.speed
        self.pp_spiral_energy: float = _spiral.energy
        self.pp_spiral_beat: float = _spiral.beat
        self.pp_spiral_mix: float = _spiral.mix
```

In `get_postprocess()` nach `"chromatic_aberration": self.pp_chromatic,`:

```python
            **{key[len("pp_"):]: getattr(self, key) for key in SPIRAL_STATE_KEYS},
```

In `to_dict()` nach `"pp_chromatic": self.pp_chromatic,`:

```python
            **{key: getattr(self, key) for key in SPIRAL_STATE_KEYS},
```

(`apply_dict` und `from_dict` brauchen nichts: sie übernehmen alle Schlüssel aus `_STATE_KEYS`.)

- [ ] **Step 4: `src/gui/main_window.py` anpassen**

Import Z. 25 ändern zu:

```python
from src.gui.state import AppState, SPIRAL_STATE_KEYS
```

`_PROJECT_KEYS`: die schließende Zeile `})` wird zu `}) | frozenset(SPIRAL_STATE_KEYS)`.

Direkt danach eine neue Klassenkonstante mit dem bisherigen Inline-Set anlegen:

```python
    # State-Keys, die eine neue Vorschau ausloesen
    _PREVIEW_KEYS = frozenset({
        "visualizer_type", "viz_params", "viz_offset_x", "viz_offset_y", "viz_scale",
        "bg_blur", "bg_vignette", "bg_opacity",
        "pp_contrast", "pp_saturation", "pp_brightness", "pp_warmth", "pp_grain",
        "pp_exposure", "pp_bloom", "pp_bloom_threshold", "pp_vignette", "pp_chromatic",
        "background_path", "preview_time_percent",
        "quotes", "quotes_enabled", "quote_config", "ki_suggested_colors",
        "color_mode", "base_hue", "color_saturation", "brightness",
    }) | frozenset(SPIRAL_STATE_KEYS)
```

In `_on_state_changed` das Inline-Set ersetzen:

```python
        if key in self._PREVIEW_KEYS:
            self._preview_timer.start(150)
```

- [ ] **Step 5: Tests laufen lassen**

Run: `pytest tests/test_gui_state.py tests/test_gui_main_window.py tests/test_app_state.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/gui/state.py src/gui/main_window.py tests/test_gui_state.py tests/test_gui_main_window.py
git commit -m "feat(gui): Spiral-Zoom im App-Zustand, Projektdatei und Vorschau-Ausloeser"
```

---

### Task 8: GUI-Regler „Spiral-Zoom“

**Files:**
- Modify: `src/gui/params_panel.py` (nach `layout.addWidget(pp_box)` ~Z. 168; `_on_state_changed` vor `elif key == "resolution":` ~Z. 403; neue Hilfsmethoden)
- Test: `tests/test_gui_params_panel.py` (Tests anhängen)

**Interfaces:**
- Consumes: `AppState.pp_spiral_*`, `SPIRAL_STATE_KEYS` (Task 7); `self._make_labeled_slider`, `self._set`, `self._updating` (Bestand).
- Produces: `ParamsPanel.chk_spiral: QCheckBox`, `ParamsPanel.spiral_sliders: dict[str, tuple[QSlider, QLabel, int, str]]` (State-Schlüssel → Slider, Label, Faktor, Format).

- [ ] **Step 1: Failing Tests anhängen**

An `tests/test_gui_params_panel.py` anhängen:

```python
def test_spiral_controls_write_state(qtbot):
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    panel.chk_spiral.setChecked(True)
    assert state.pp_spiral_enabled is True

    panel.spiral_sliders["pp_spiral_ratio"][0].setValue(400)
    assert state.pp_spiral_ratio == 4.0

    panel.spiral_sliders["pp_spiral_arms"][0].setValue(-2)
    assert state.pp_spiral_arms == -2
    assert isinstance(state.pp_spiral_arms, int)

    panel.spiral_sliders["pp_spiral_mix"][0].setValue(40)
    assert state.pp_spiral_mix == 0.4


def test_spiral_controls_follow_state(qtbot):
    """Projekt laden / KI setzt Werte: Regler und Labels ziehen nach."""
    state = AppState()
    panel = ParamsPanel(state)
    qtbot.addWidget(panel)

    state.pp_spiral_speed = -1.25
    slider, label, _, _ = panel.spiral_sliders["pp_spiral_speed"]
    assert slider.value() == -125
    assert label.text() == "-1.25"

    assert not slider.isEnabled()  # Effekt aus -> Regler grau
    state.pp_spiral_enabled = True
    assert panel.chk_spiral.isChecked()
    assert slider.isEnabled()
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `pytest tests/test_gui_params_panel.py -k spiral -v`
Expected: FAIL mit `AttributeError: 'ParamsPanel' object has no attribute 'chk_spiral'`

- [ ] **Step 3: Gruppe anlegen**

In `src/gui/params_panel.py` direkt nach `layout.addWidget(pp_box)`:

```python
        # Spiral-Zoom (Droste/Escher) — verbiegt das fertige Bild
        spiral_box = QGroupBox("Spiral-Zoom")
        spiral_layout = QGridLayout(spiral_box)
        self.chk_spiral = QCheckBox("Endloser Zoom ins Bild")
        self.chk_spiral.setChecked(bool(self.state.pp_spiral_enabled))
        self.chk_spiral.setToolTip(
            "Stapelt den Visualizer endlos in sich selbst und zoomt im Takt hinein. "
            "Ein Hintergrundbild bleibt ruhig stehen."
        )
        self.chk_spiral.toggled.connect(self._on_spiral_toggled)
        spiral_layout.addWidget(self.chk_spiral, 0, 0, 1, 3)

        # (State-Schluessel, Anzeigename, Min, Max, Faktor Slider->Wert, Format, Tooltip)
        spiral_rows = [
            ("pp_spiral_arms", "Spiralarme", -3, 3, 1, "{:+d}",
             "0 = gerader Zoom, sonst Anzahl der Spiralarme (Vorzeichen = Drehsinn)."),
            ("pp_spiral_ratio", "Zoom-Faktor", 150, 600, 100, "{:.2f}x",
             "Wie viel kleiner jede Ebene gegenueber der vorigen ist. Hoeher = "
             "kleinerer Kreis in der Bildmitte, der durch die naechste Ebene ersetzt wird."),
            ("pp_spiral_rotation", "Drehung", -90, 90, 1, "{:+d}°",
             "Drehung pro Ebene."),
            ("pp_spiral_speed", "Tempo", -200, 200, 100, "{:+.2f}",
             "Ebenen pro Sekunde; negativ = heraus zoomen."),
            ("pp_spiral_energy", "Energie-Schub", 0, 200, 100, "{:.2f}",
             "Laute Stellen zoomen schneller."),
            ("pp_spiral_beat", "Beat-Schub", 0, 100, 100, "{:.2f}",
             "Jeder Beat schiebt den Zoom um diesen Teil einer Ebene weiter."),
            ("pp_spiral_mix", "Staerke", 0, 100, 100, "{:.0%}",
             "Mischung mit dem unveraenderten Bild."),
        ]
        self.spiral_sliders = {}
        for row, (key, name, lo, hi, factor, fmt, tip) in enumerate(spiral_rows, start=1):
            slider, label = self._make_labeled_slider(
                lo, hi, int(round(getattr(self.state, key) * factor))
            )
            slider.setToolTip(tip)
            self.spiral_sliders[key] = (slider, label, factor, fmt)
            slider.valueChanged.connect(
                lambda v, k=key, f=factor: self._set(k, self._spiral_value(v, f))
            )
            slider.valueChanged.connect(
                lambda v, lbl=label, f=factor, fm=fmt: lbl.setText(
                    fm.format(self._spiral_value(v, f))
                )
            )
            label.setText(fmt.format(self._spiral_value(slider.value(), factor)))
            spiral_layout.addWidget(QLabel(name), row, 0)
            spiral_layout.addWidget(slider, row, 1)
            spiral_layout.addWidget(label, row, 2)
        self._update_spiral_enabled_ui()
        layout.addWidget(spiral_box)
```

Neue Hilfsmethoden (z. B. direkt vor `def _set`):

```python
    @staticmethod
    def _spiral_value(slider_value: int, factor: int):
        """Slider-Wert -> State-Wert (Faktor 1 bleibt ganzzahlig, z.B. Spiralarme)."""
        return int(slider_value) if factor == 1 else slider_value / factor

    def _on_spiral_toggled(self, checked: bool):
        self._update_spiral_enabled_ui()
        self._set("pp_spiral_enabled", bool(checked))

    def _update_spiral_enabled_ui(self):
        """Regler nur aktiv, wenn der Effekt eingeschaltet ist."""
        on = self.chk_spiral.isChecked()
        for slider, label, _, _ in self.spiral_sliders.values():
            slider.setEnabled(on)
            label.setEnabled(on)
```

In `_on_state_changed` vor `elif key == "resolution":`:

```python
            elif key == "pp_spiral_enabled":
                self.chk_spiral.setChecked(bool(self.state.pp_spiral_enabled))
                self._update_spiral_enabled_ui()
            elif key in self.spiral_sliders:
                slider, _, factor, _ = self.spiral_sliders[key]
                slider.setValue(int(round(getattr(self.state, key) * factor)))
```

Hinweis: Das Label folgt über das `valueChanged`-Signal automatisch; `_set` ignoriert Rückschreiben während `_updating`. Prüfen, dass `QCheckBox`, `QGroupBox`, `QGridLayout`, `QLabel` im bestehenden `from PyQt6.QtWidgets import (...)` stehen (sie werden im Panel bereits benutzt).

- [ ] **Step 4: Tests laufen lassen**

Run: `pytest tests/test_gui_params_panel.py tests/test_gui_smoke.py tests/test_gui_main_window.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gui/params_panel.py tests/test_gui_params_panel.py
git commit -m "feat(gui): Regler-Gruppe Spiral-Zoom im Parameter-Panel"
```

---

### Task 9: Doku, Gesamtlauf und Sichtprüfung

**Files:**
- Modify: `CHANGELOG.md` (neuer Abschnitt oberhalb `## [3.2.0]`)
- Modify: `README.md:25` (Feature-Zeile HDR-Pipeline) und Z. 207 (Preset-Liste)

**Interfaces:**
- Consumes: alles aus Task 1–8.
- Produces: Doku; keine Code-Schnittstellen.

- [ ] **Step 1: CHANGELOG ergänzen**

Oberhalb von `## [3.2.0] — 2026-08-14` einfügen:

```markdown
## [Unreleased]

### Added
- **Spiral-Zoom (Droste/Escher)** als neuer Effekt: der Visualizer steckt
  endlos in sich selbst und zoomt im Takt hinein; mit Spiralarmen wird daraus
  eine Escher-Spirale. Ein Hintergrundbild bleibt ruhig stehen.
  Tempo aus Grundgeschwindigkeit, Lautstaerke und Beats, vorab berechnet,
  daher Vorschau == Export. Standardmaessig aus; aus = bitgleich zu vorher.
  Mathematik portiert aus dem Schwesterprojekt Fraktal-Zoom, mit
  CPU-Gegenstueck und GPU-Vergleichstest (`src/spiral_zoom.py`,
  `src/gpu_spiral.py`). Neue GUI-Gruppe „Spiral-Zoom“, neue
  `spiral_*`-Schluessel im `postprocess`-Block, Beispiel-Preset
  `config/music_spiral_zoom.json`.
```

- [ ] **Step 2: README ergänzen**

Zeile 25 (`- **HDR-Render-Pipeline**: …`) am Ende um diesen Satz erweitern:

```markdown
 Optional **Spiral-Zoom** (Droste/Escher): der Visualizer steckt endlos in sich selbst und zoomt im Takt, das Hintergrundbild bleibt ruhig.
```

Zeile 207 (Musik-Presets): `` `music_spiral_zoom` `` an die Liste anhängen.

- [ ] **Step 3: Gesamte Testsuite + Stil**

Run: `pytest tests/ -v`
Expected: PASS (bisherige Testzahl + neue Tests; GPU-Tests laufen nur mit GPU)

Run: `black src/spiral_zoom.py src/gpu_spiral.py tests/test_spiral_zoom.py tests/test_gpu_spiral.py tests/test_spiral_config.py tests/test_spiral_render_integration.py && flake8 src/spiral_zoom.py src/gpu_spiral.py`
Expected: keine Fehler

- [ ] **Step 4: Sichtprüfung für den Nutzer (kopierbare Befehle)**

Mit einem echten Song (Pfad anpassen):

```bash
# 1) 5-Sekunden-Vorschau mit dem Beispiel-Preset
python main.py render song.mp3 --config config/music_spiral_zoom.json --preview -o spiral_test.mp4

# 2) Dasselbe mit Hintergrundfoto (Foto bleibt ruhig, nur der Visualizer dreht)
python main.py render song.mp3 --config config/music_spiral_zoom.json --preview -bg bild.jpg --background-opacity 1.0 -o spiral_test_foto.mp4

# 3) GUI: Gruppe "Spiral-Zoom" -> Haken setzen, Regler bewegen, Vorschau muss sich aendern
python gui.py
```

Worauf achten: Zoom läuft ruckfrei; an den Ebenen-Übergängen keine harte Kante; links der Bildmitte keine Naht; bei Beats ein spürbarer Schub; mit Foto bleibt das Foto ruhig und nichts wird schwarz; Effekt aus sieht exakt aus wie vorher.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md README.md
git commit -m "docs: Spiral-Zoom in Changelog und README"
```

---

## Später (bewusst nicht in diesem Plan)

- **Ganzes-Bild-Variante** (Foto dreht mit): derselbe `SpiralZoomPass`, aufgerufen auf `self.fbo` nach dem Blit, vor dem Bloom — per Schalter wählbar.
- **Kaleidoskop-Tunnel** aus Fraktal-Zoom (`kaleido.ts`): Winkel-Faltung im selben Log-Polar-Raum, als zusätzlicher Modus des Passes.
- **Eigenes Zentrum** (Zoom nicht in die Bildmitte), KI-Vorschläge für Spiral-Parameter, Szenen-spezifische Spiral-Werte in der Timeline.
