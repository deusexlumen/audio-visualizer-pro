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
