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
