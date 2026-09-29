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
