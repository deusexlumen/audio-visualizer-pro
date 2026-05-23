"""Tests fuer Zitat-Zeitstempel-Verfeinerung."""

import numpy as np
import pytest

from src.quote_refiner import refine_quote_timestamps
from src.types import Quote


class TestRefineQuoteTimestamps:
    def test_no_quotes_returns_empty(self):
        features = {"fps": 30, "duration": 60.0, "onset": np.zeros(1800)}
        result = refine_quote_timestamps([], features)
        assert result == []

    def test_max_duration_limit(self):
        """Zitate duerfen nicht laenger als max_duration sein."""
        quotes = [Quote(text="Lang", start_time=10.0, end_time=30.0, confidence=0.8)]
        features = {"fps": 30, "duration": 60.0, "onset": np.zeros(1800)}
        result = refine_quote_timestamps(quotes, features, max_duration=5.0)
        assert result[0].end_time - result[0].start_time <= 5.0 + 0.01

    def test_snap_to_onset(self):
        """Startzeit wird an Onset gesnappt, wenn nahe genug."""
        onset = np.zeros(300)
        onset[100] = 0.8  # Peak bei ~3.33s
        quotes = [Quote(text="Test", start_time=3.5, end_time=6.0, confidence=0.8)]
        features = {"fps": 30, "duration": 10.0, "onset": onset}
        result = refine_quote_timestamps(quotes, features, snap_threshold=1.0)
        # Sollte an den Peak bei 3.33s gesnappt werden
        assert abs(result[0].start_time - 3.33) < 0.1

    def test_start_not_before_zero(self):
        """Startzeit wird nicht negativ."""
        quotes = [Quote(text="Test", start_time=-5.0, end_time=2.0, confidence=0.8)]
        features = {"fps": 30, "duration": 10.0, "onset": np.zeros(300)}
        result = refine_quote_timestamps(quotes, features)
        assert result[0].start_time >= 0.0

    def test_end_not_after_duration(self):
        """Endzeit wird auf Duration begrenzt."""
        quotes = [Quote(text="Test", start_time=8.0, end_time=20.0, confidence=0.8)]
        features = {"fps": 30, "duration": 10.0, "onset": np.zeros(300)}
        result = refine_quote_timestamps(quotes, features)
        assert result[0].end_time <= 10.0

    def test_min_end_after_start(self):
        """Endzeit muss mindestens 0.5s nach Startzeit liegen."""
        quotes = [Quote(text="Test", start_time=5.0, end_time=5.1, confidence=0.8)]
        features = {"fps": 30, "duration": 10.0, "onset": np.zeros(300)}
        result = refine_quote_timestamps(quotes, features)
        assert result[0].end_time >= result[0].start_time + 0.5

    def test_text_and_confidence_preserved(self):
        """Text und Confidence bleiben erhalten."""
        quotes = [Quote(text="Original", start_time=1.0, end_time=3.0, confidence=0.75)]
        features = {"fps": 30, "duration": 10.0, "onset": np.zeros(300)}
        result = refine_quote_timestamps(quotes, features)
        assert result[0].text == "Original"
        assert result[0].confidence == 0.75

    def test_beats_used_for_snap(self):
        """Beat-Frames werden fuer Snap verwendet."""
        beat_frames = np.array([90, 180, 270])  # 3s, 6s, 9s
        quotes = [Quote(text="Test", start_time=6.2, end_time=8.0, confidence=0.8)]
        features = {
            "fps": 30,
            "duration": 10.0,
            "onset": np.zeros(300),
            "beat_frames": beat_frames,
        }
        result = refine_quote_timestamps(quotes, features, snap_threshold=1.0)
        # Sollte an 6.0s gesnappt werden
        assert abs(result[0].start_time - 6.0) < 0.1
