"""Tests für das stratifizierte Sampling (Spec §4)."""

import numpy as np
import pytest

from src.studio.sampling import build_sample_plan, verification_extras


def _features(duration=60.0, fps=30, bpm=120.0):
    """Synthetisches Feature-Dict: Beat-Grid bei bpm, Peaks auf den Beats."""
    frame_count = int(duration * fps)
    beat_period = 60.0 / bpm  # 0.5 s bei 120 BPM
    beat_frames = np.arange(0, frame_count, int(beat_period * fps))
    rms = np.full(frame_count, 0.2, dtype=np.float32)
    onset = np.full(frame_count, 0.05, dtype=np.float32)
    rms[beat_frames] = 0.9
    onset[beat_frames] = 1.0
    return {
        "rms": rms,
        "onset": onset,
        "beat_frames": beat_frames,
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "tempo": bpm,
    }


def test_determinismus():
    fd = _features()
    a = build_sample_plan(fd)
    b = build_sample_plan(fd)
    assert a.timestamps == b.timestamps
    assert a.seed == b.seed


def test_standard_18_samples_stratifiziert():
    plan = build_sample_plan(_features())
    assert plan.n == 18
    assert len(plan.categories["uniform"]) == 6
    assert len(plan.categories["peaks"]) == 6
    assert len(plan.categories["quiet"]) == 3
    assert len(plan.categories["quotes"]) == 3  # ohne Quotes: auf Peaks aufgefüllt
    assert plan.timestamps == sorted(plan.timestamps)


def test_kein_beat_aliasing():
    """Regression Spec §4: uniform-Samples dürfen nicht beat-phasenverriegelt sein."""
    fd = _features(duration=60.0, bpm=120.0)
    plan = build_sample_plan(fd)
    beat_period = 0.5
    phases = [t % beat_period for t in plan.categories["uniform"]]
    # Bei Verriegelung wären alle Phasen identisch
    assert len({round(p, 2) for p in phases}) >= 3


def test_peaks_treffen_onset_peaks():
    fd = _features()
    plan = build_sample_plan(fd)
    fps = fd["fps"]
    for t in plan.categories["peaks"]:
        idx = min(int(t * fps), len(fd["onset"]) - 1)
        assert fd["onset"][idx] >= 0.5  # Peak-Zeitpunkt, kein Tal


def test_quotes_stratum_nutzt_quote_zeiten():
    fd = _features()
    quotes = [(10.0, 12.0), (30.0, 32.0), (50.0, 52.0)]
    plan = build_sample_plan(fd, quote_times=quotes)
    for t in plan.categories["quotes"]:
        assert any(start <= t <= end for start, end in quotes)


def test_kurzes_audio_reduziert_adaptiv():
    fd = _features(duration=3.0)
    plan = build_sample_plan(fd)
    assert plan.n < 18
    assert plan.timestamps  # aber nicht leer


def test_verification_extras_disjunkt_und_deterministisch():
    fd = _features()
    plan = build_sample_plan(fd)
    extras_a = verification_extras(plan, fd["duration"])
    extras_b = verification_extras(plan, fd["duration"])
    assert extras_a == extras_b
    assert len(extras_a) == 6
    assert not set(extras_a) & set(plan.timestamps)
