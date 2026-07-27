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
