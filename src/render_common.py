"""
Gemeinsame Hilfsfunktionen fuer GPUBatchRenderer und GPU-Preview.

Buendelt Logik, die vorher dupliziert in gpu_renderer.py und
gpu_preview.py lag: Beat-Intensitaet und den Aufbau des
Feature-Dictionaries fuer die Visualizer.
"""

import numpy as np

from .types import AudioFeatures


def compute_beat_intensity(beat_frames, frame_count: int, fps: int) -> np.ndarray:
    """Berechnet die Beat-Decay-Envelope (1.0 am Beat, linear abfallend).

    Args:
        beat_frames: Frame-Indizes der erkannten Beats.
        frame_count: Anzahl der Video-Frames.
        fps: Frames pro Sekunde.

    Returns:
        Float32-Array der Laenge frame_count mit Werten 0.0-1.0.
    """
    beat_intensity = np.zeros(frame_count, dtype=np.float32)
    if beat_frames is None or len(beat_frames) == 0:
        return beat_intensity

    decay_frames = max(3, int(fps * 0.1))
    for bf in beat_frames:
        if bf >= frame_count:
            continue
        end = min(bf + decay_frames + 1, frame_count)
        if end > bf:
            dists = np.arange(end - bf, dtype=np.float32)
            vals = np.clip(1.0 - dists / decay_frames, 0.0, 1.0)
            beat_intensity[bf:end] = np.maximum(beat_intensity[bf:end], vals)
    return beat_intensity


def _slice_or_zeros(arr, frame_count: int) -> np.ndarray:
    """Schneidet ein Feature-Array auf frame_count zu (oder liefert Nullen)."""
    if arr is None or len(arr) == 0:
        return np.zeros(frame_count, dtype=np.float32)
    return arr[:frame_count]


def _slice_2d(arr, frame_count: int) -> np.ndarray:
    """Schneidet ein 2D-Feature-Array (z.B. Chroma) auf frame_count Spalten zu."""
    if arr.ndim > 1 and arr.shape[1] >= frame_count:
        return arr[:, :frame_count]
    return arr


def build_features_dict(features: AudioFeatures, frame_count: int, fps: int) -> dict:
    """Baut das vollstaendige Feature-Dictionary fuer die Visualizer.

    Args:
        features: Analysierte AudioFeatures.
        frame_count: Anzahl der zu rendernden Frames.
        fps: Video-Framerate.

    Returns:
        Dictionary mit allen Feature-Arrays (auf frame_count zugeschnitten),
        beat_intensity und Metadaten (tempo, mode, duration, fps, frame_count).
    """
    return {
        "rms": _slice_or_zeros(features.rms, frame_count),
        "onset": _slice_or_zeros(features.onset, frame_count),
        "beat_intensity": compute_beat_intensity(features.beat_frames, frame_count, fps),
        "chroma": _slice_2d(features.chroma, frame_count),
        "spectral_centroid": _slice_or_zeros(features.spectral_centroid, frame_count),
        "spectral_rolloff": _slice_or_zeros(features.spectral_rolloff, frame_count),
        "zero_crossing_rate": _slice_or_zeros(features.zero_crossing_rate, frame_count),
        "transient": _slice_or_zeros(features.transient, frame_count),
        "voice_clarity": _slice_or_zeros(features.voice_clarity, frame_count),
        "voice_band": _slice_or_zeros(features.voice_band, frame_count),
        "mfcc": _slice_2d(features.mfcc, frame_count),
        "tempogram": _slice_2d(features.tempogram, frame_count),
        "beat_frames": features.beat_frames,
        "tempo": float(features.tempo),
        "mode": features.mode,
        "duration": float(features.duration),
        "fps": fps,
        "frame_count": frame_count,
    }
