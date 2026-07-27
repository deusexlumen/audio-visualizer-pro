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
