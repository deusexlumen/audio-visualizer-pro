"""
Lokale Audio-Segmentierung fuer Szenen-Timelines.

Zerlegt einen Track in strukturelle Abschnitte (Intro, Aufbau, Refrain, ...)
allein aus den bereits berechneten Audio-Features — ohne LLM, ohne Kosten.
Grundlage sind chroma-, MFCC- und RMS-Verlaeufe, aus denen per
agglomerativer Clusterung (librosa) Segmentgrenzen entstehen. Fuer Sprache
greift ein pausen-/energiebasierter Fallback.

Die Segment-Statistiken (mittlere Lautheit, Tempo, Helligkeit, Onset-Dichte,
dominante Tonklasse) sind kompakt genug, um sie spaeter an ein LLM zu geben,
das daraus Szenen-Labels und Visualizer-Wechsel vorschlaegt.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .app_logging import get_logger
from .types import AudioFeatures

logger = get_logger(__name__)

_CHROMA_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class Segment:
    """Ein zusammenhaengender Audio-Abschnitt mit aggregierten Kennwerten."""

    start: float          # Sekunden
    end: float            # Sekunden
    stats: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Zeilenweise Min-Max-Normalisierung, robust gegen konstante Zeilen."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    lo = mat.min(axis=1, keepdims=True)
    hi = mat.max(axis=1, keepdims=True)
    span = np.maximum(hi - lo, 1e-6)
    return (mat - lo) / span


def _target_segment_count(duration: float) -> int:
    """Waehlt eine sinnvolle Segmentanzahl aus der Dauer (2..8)."""
    if duration <= 20:
        return 2
    return int(np.clip(round(duration / 30.0), 2, 8))


def _feature_matrix(features: AudioFeatures) -> np.ndarray:
    """Baut eine (features x frames)-Matrix aus chroma, MFCC und RMS."""
    n = features.frame_count
    parts = []

    chroma = np.asarray(features.chroma, dtype=np.float32)
    if chroma.ndim == 2:
        # Auf (12, frames) bringen
        if chroma.shape[0] != 12 and chroma.shape[1] == 12:
            chroma = chroma.T
        chroma = chroma[:, :n] if chroma.shape[1] >= n else np.pad(
            chroma, ((0, 0), (0, n - chroma.shape[1])), mode="edge"
        )
        parts.append(_normalize_rows(chroma))

    mfcc = np.asarray(features.mfcc, dtype=np.float32)
    if mfcc.ndim == 2 and mfcc.size > 0:
        if mfcc.shape[0] > mfcc.shape[1]:
            mfcc = mfcc.T
        mfcc = mfcc[:, :n] if mfcc.shape[1] >= n else np.pad(
            mfcc, ((0, 0), (0, n - mfcc.shape[1])), mode="edge"
        )
        parts.append(_normalize_rows(mfcc))

    rms = np.asarray(features.rms, dtype=np.float32).ravel()
    rms = rms[:n] if rms.size >= n else np.pad(rms, (0, n - rms.size), mode="edge")
    parts.append(_normalize_rows(rms))

    return np.vstack(parts) if parts else _normalize_rows(rms)


def _segment_stats(features: AudioFeatures, f0: int, f1: int) -> dict:
    """Aggregiert Kennwerte fuer den Frame-Bereich [f0, f1)."""
    f0 = max(0, f0)
    f1 = min(features.frame_count, max(f1, f0 + 1))

    def _mean(arr, default=0.0):
        a = np.asarray(arr, dtype=np.float32).ravel()
        if a.size == 0:
            return default
        seg = a[f0:min(f1, a.size)]
        return float(seg.mean()) if seg.size else default

    # Dominante Tonklasse ueber den Abschnitt
    chroma = np.asarray(features.chroma, dtype=np.float32)
    dominant = None
    if chroma.ndim == 2 and chroma.size:
        if chroma.shape[0] != 12 and chroma.shape[1] == 12:
            chroma = chroma.T
        seg = chroma[:, f0:min(f1, chroma.shape[1])]
        if seg.size:
            idx = int(np.argmax(seg.mean(axis=1)))
            dominant = _CHROMA_NAMES[idx % 12]

    onset = np.asarray(features.onset, dtype=np.float32).ravel()
    onset_seg = onset[f0:min(f1, onset.size)] if onset.size else np.array([])
    onset_density = float((onset_seg > 0.3).mean()) if onset_seg.size else 0.0

    return {
        "rms": round(_mean(features.rms), 4),
        "centroid": round(_mean(features.spectral_centroid), 4),
        "rolloff": round(_mean(features.spectral_rolloff), 4),
        "voice_band": round(_mean(features.voice_band), 4),
        "onset_density": round(onset_density, 4),
        "dominant_chroma": dominant,
        "tempo": round(float(features.tempo), 1),
    }


def _boundaries_music(features: AudioFeatures, k: int) -> List[int]:
    """Strukturelle Grenzen via agglomerativer Clusterung (librosa)."""
    import librosa

    mat = _feature_matrix(features)
    # librosa erwartet (features, frames); k = Anzahl Segmente
    bounds = librosa.segment.agglomerative(mat, k)
    bounds = sorted(set(int(b) for b in bounds))
    if bounds and bounds[0] != 0:
        bounds = [0] + bounds
    if not bounds or bounds[-1] != features.frame_count:
        bounds.append(features.frame_count)
    return bounds


def _boundaries_speech(features: AudioFeatures, k: int) -> List[int]:
    """Pausen-/Energiebasierte Grenzen fuer Sprache.

    Teilt an ruhigen Stellen (niedrige RMS) moeglichst gleichmaessig in k Abschnitte.
    """
    n = features.frame_count
    rms = np.asarray(features.rms, dtype=np.float32).ravel()
    if rms.size < n:
        rms = np.pad(rms, (0, n - rms.size), mode="edge")
    rms = rms[:n]

    # Kandidaten fuer Grenzen: lokale RMS-Minima (Sprechpausen)
    quiet = rms < (rms.mean() * 0.5)
    # Ziel-Grenzpositionen gleichmaessig verteilt
    targets = np.linspace(0, n, k + 1)[1:-1]
    bounds = [0]
    for t in targets:
        lo = int(max(0, t - n * 0.1))
        hi = int(min(n, t + n * 0.1))
        window = np.where(quiet[lo:hi])[0]
        if window.size:
            # Ruhigste Stelle im Fenster
            local = window[np.argmin(rms[lo:hi][window])]
            bounds.append(lo + int(local))
        else:
            bounds.append(int(t))
    bounds.append(n)
    return sorted(set(bounds))


def segment_audio(features: AudioFeatures, min_segment_seconds: float = 5.0) -> List[Segment]:
    """Zerlegt den Track in strukturelle Segmente mit Kennwerten.

    Args:
        features: Analysierte Audio-Features.
        min_segment_seconds: Zu kurze Segmente werden mit dem Nachbarn verschmolzen.

    Returns:
        Liste von Segmenten (nach Startzeit sortiert, lueckenlos).
    """
    fps = features.fps or 30
    n = features.frame_count
    if n <= 1 or features.duration <= 0:
        return [Segment(0.0, float(features.duration), _segment_stats(features, 0, max(1, n)))]

    k = _target_segment_count(features.duration)

    try:
        if features.mode == "speech":
            bounds = _boundaries_speech(features, k)
        else:
            bounds = _boundaries_music(features, k)
    except Exception as e:
        logger.warning(f"[Segmentierung] Clusterung fehlgeschlagen ({e}), nutze gleichmaessige Teilung.")
        bounds = [int(round(x)) for x in np.linspace(0, n, k + 1)]

    # Frame-Grenzen -> Segmente
    segments: List[Segment] = []
    for i in range(len(bounds) - 1):
        f0, f1 = bounds[i], bounds[i + 1]
        if f1 <= f0:
            continue
        segments.append(Segment(
            start=f0 / fps,
            end=f1 / fps,
            stats=_segment_stats(features, f0, f1),
        ))

    segments = _merge_short(segments, min_segment_seconds)
    # Endzeit des letzten Segments exakt auf die Dauer setzen
    if segments:
        segments[-1].end = float(features.duration)
    return segments


def _merge_short(segments: List[Segment], min_seconds: float) -> List[Segment]:
    """Verschmilzt zu kurze Segmente mit dem vorherigen Abschnitt."""
    if not segments:
        return segments
    merged: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        if seg.duration < min_seconds and merged:
            # In den Vorgaenger einschmelzen (Grenze verschieben)
            merged[-1].end = seg.end
        else:
            merged.append(seg)
    return merged
