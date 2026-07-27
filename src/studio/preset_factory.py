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
