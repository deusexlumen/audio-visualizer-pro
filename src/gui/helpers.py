"""Hilfsfunktionen fuer die PyQt6 GUI."""

import numpy as np


def _mean(arr) -> float:
    arr = np.asarray(arr)
    return float(arr.mean()) if arr.size else 0.0


def _std(arr) -> float:
    arr = np.asarray(arr)
    return float(arr.std()) if arr.size else 0.0


def _features_to_dict(features) -> dict:
    """Wandelt AudioFeatures in ein statistisches Dict fuer GUI und KI um."""
    return {
        "duration": float(getattr(features, "duration", 0)),
        "tempo": float(getattr(features, "tempo", 120)),
        "mode": str(getattr(features, "mode", "music")),
        "rms_mean": _mean(getattr(features, "rms", [])),
        "rms_std": _std(getattr(features, "rms", [])),
        "onset_mean": _mean(getattr(features, "onset", [])),
        "onset_std": _std(getattr(features, "onset", [])),
        "spectral_mean": _mean(getattr(features, "spectral_centroid", [])),
        "brightness": _mean(getattr(features, "spectral_centroid", [])),
        "noisiness": _mean(getattr(features, "zero_crossing_rate", [])),
        "transient_mean": _mean(getattr(features, "transient", [])),
        "voice_clarity_mean": _mean(getattr(features, "voice_clarity", [])),
    }
