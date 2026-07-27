"""Tests für die Modus-Weiche (Spec §5)."""

import numpy as np
import pytest

from src.studio.mode_gate import classify_mode
from src.studio.thresholds import load_thresholds


def _fd(voice_clarity, voice_band, onset, duration=30.0, fps=30):
    n = int(duration * fps)
    return {
        "voice_clarity": np.full(n, voice_clarity, dtype=np.float32),
        "voice_band": np.full(n, voice_band, dtype=np.float32),
        "onset": np.full(n, onset, dtype=np.float32),
        "rms": np.full(n, 0.5, dtype=np.float32),
        "duration": duration, "fps": fps, "frame_count": n,
    }


def test_klar_sprache_ist_podcast(tmp_path):
    fd = _fd(voice_clarity=0.9, voice_band=0.8, onset=0.05)
    result = classify_mode(fd, cache_path=str(tmp_path / "m.json"))
    assert result.resolved == "podcast"
    assert result.speech_score >= 0.55


def test_klar_musik_ist_music(tmp_path):
    fd = _fd(voice_clarity=0.05, voice_band=0.1, onset=0.7)
    result = classify_mode(fd, cache_path=str(tmp_path / "m.json"))
    assert result.resolved == "music"
    assert result.value in ("MUSIC", "HYBRID")


def test_determinismus(tmp_path):
    fd = _fd(voice_clarity=0.6, voice_band=0.5, onset=0.3)
    a = classify_mode(fd, cache_path=str(tmp_path / "m.json"))
    b = classify_mode(fd, cache_path=str(tmp_path / "m.json"))
    assert (a.value, a.resolved, a.speech_score) == (b.value, b.resolved, b.speech_score)


def test_hysterese_haelt_letzte_entscheidung(tmp_path):
    cache = str(tmp_path / "m.json")
    # Erster Lauf: klare Podcast-Entscheidung
    fd_speech = _fd(voice_clarity=0.9, voice_band=0.8, onset=0.05)
    first = classify_mode(fd_speech, cache_path=cache)
    assert first.resolved == "podcast"
    # Zweiter Lauf: gleiche Datei (gleicher Seed), Score im Hysterese-Band
    # (0.5*0.7 + 0.3*0.5 - 0.2*0.3 = 0.44 -> zu niedrig; mit 0.8/0.7/0.2:
    #  0.4 + 0.21 - 0.04 = 0.57 -> im Band [0.50, 0.60])
    fd_border = _fd(voice_clarity=0.8, voice_band=0.7, onset=0.2)
    # Seed-Gleichheit simulieren: rms identisch halten (Seed aus rms+duration)
    second = classify_mode(fd_border, cache_path=cache)
    assert 0.50 <= second.speech_score <= 0.60
    assert second.hysteresis_applied is True
    assert second.resolved == "podcast"  # beibehalten, nicht neu entschieden


def test_schwellen_kommen_aus_thresholds(tmp_path):
    ts = load_thresholds()
    assert ts.speech_threshold == 0.55
    assert ts.hysteresis_lo == 0.50
    assert ts.hysteresis_hi == 0.60
