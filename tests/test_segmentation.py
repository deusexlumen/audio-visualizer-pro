"""Tests fuer die lokale Audio-Segmentierung."""

import numpy as np

from src.segmentation import segment_audio, Segment
from src.types import AudioFeatures


def _features_two_halves(fps=30, seconds=20) -> AudioFeatures:
    """Baut Features mit zwei klar verschiedenen Haelften (leise/laut, Tonwechsel)."""
    n = fps * seconds
    half = n // 2

    rms = np.concatenate([np.full(half, 0.15), np.full(n - half, 0.75)]).astype(np.float32)
    onset = np.concatenate([np.full(half, 0.05), np.full(n - half, 0.6)]).astype(np.float32)

    chroma = np.zeros((12, n), dtype=np.float32)
    chroma[0, :half] = 1.0   # erste Haelfte: Tonklasse C
    chroma[7, half:] = 1.0   # zweite Haelfte: Tonklasse G

    mfcc = np.zeros((13, n), dtype=np.float32)
    mfcc[1, :half] = -2.0
    mfcc[1, half:] = 3.0

    return AudioFeatures(
        duration=seconds,
        sample_rate=22050,
        fps=fps,
        frame_count=n,
        rms=rms,
        onset=onset,
        spectral_centroid=np.linspace(0.2, 0.8, n).astype(np.float32),
        spectral_rolloff=np.linspace(0.3, 0.7, n).astype(np.float32),
        zero_crossing_rate=np.full(n, 0.1, dtype=np.float32),
        transient=onset.copy(),
        voice_clarity=rms.copy(),
        voice_band=rms.copy(),
        chroma=chroma,
        mfcc=mfcc,
        tempogram=np.zeros((384, n), dtype=np.float32),
        tempo=120.0,
        key="C",
        mode="music",
        beat_frames=np.arange(0, n, fps).astype(np.int64),
    )


def test_findet_mindestens_zwei_segmente():
    feats = _features_two_halves()
    segments = segment_audio(feats)
    assert len(segments) >= 2
    assert all(isinstance(s, Segment) for s in segments)


def test_segmente_sind_lueckenlos_und_sortiert():
    feats = _features_two_halves()
    segments = segment_audio(feats)
    assert segments[0].start == 0.0
    assert abs(segments[-1].end - feats.duration) < 1e-3
    for a, b in zip(segments, segments[1:]):
        assert abs(a.end - b.start) < 1e-6
        assert a.start < a.end


def test_grenze_liegt_nahe_der_mitte():
    feats = _features_two_halves(seconds=20)
    segments = segment_audio(feats)
    # Irgendeine Segmentgrenze sollte nahe der Mitte (10s) liegen
    inner_bounds = [s.end for s in segments[:-1]]
    assert any(abs(b - 10.0) < 3.0 for b in inner_bounds)


def test_stats_enthalten_kennwerte():
    feats = _features_two_halves()
    seg = segment_audio(feats)[0]
    assert "rms" in seg.stats
    assert "dominant_chroma" in seg.stats
    assert "tempo" in seg.stats


def test_speech_modus_nutzt_fallback():
    feats = _features_two_halves()
    feats.mode = "speech"
    segments = segment_audio(feats)
    assert len(segments) >= 2
    assert abs(segments[-1].end - feats.duration) < 1e-3


def test_sehr_kurzes_audio_ein_segment():
    feats = _features_two_halves(seconds=3)
    segments = segment_audio(feats)
    assert len(segments) >= 1
    assert segments[0].start == 0.0
