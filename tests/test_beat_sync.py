"""
Tests fuer Beat-Synchronisation.
"""

import numpy as np
import pytest
from src.beat_sync import (
    get_nearest_beat_time,
    sync_quotes_to_beats,
    is_on_beat,
    get_beat_intensity,
    get_next_beat_time,
    create_beat_grid_overlay,
)
from src.types import Quote


class TestBeatSync:
    """Tests fuer Beat-Sync Utilities."""
    
    def test_get_nearest_beat_time_basic(self):
        """Sollte naechsten Beat finden."""
        beat_frames = np.array([0, 30, 60, 90], dtype=np.int32)
        fps = 30
        
        # Bei 0.5s (Frame 15) -> naechster Beat bei Frame 30 (1.0s)
        result = get_nearest_beat_time(0.5, beat_frames, fps)
        assert result == 1.0
        
        # Bei 1.1s (Frame 33) -> naechster Beat bei Frame 60 (2.0s)
        result = get_nearest_beat_time(1.1, beat_frames, fps)
        assert result == 2.0
    
    def test_get_nearest_beat_time_empty(self):
        """Ohne Beats sollte Originalzeit zurueckgegeben werden."""
        result = get_nearest_beat_time(1.0, np.array([]), 30)
        assert result == 1.0
    
    def test_sync_quotes_to_beats(self):
        """Quotes sollten auf Beats verschoben werden."""
        quotes = [
            Quote(text="Test", start_time=0.45, end_time=2.0, confidence=1.0),
            Quote(text="Test2", start_time=2.1, end_time=4.0, confidence=1.0),
        ]
        beat_frames = np.array([0, 30, 60, 90], dtype=np.int32)  # 0, 1, 2, 3s @ 30fps
        fps = 30
        
        synced = sync_quotes_to_beats(quotes, beat_frames, fps, shift_threshold=0.3)
        
        # Quote 1: 0.45s -> naechster Beat bei 1.0s (Differenz 0.55 > 0.3, nicht verschoben)
        assert synced[0].start_time == 0.45
        
        # Quote 2: 2.1s -> naechster Beat bei 3.0s (Differenz 0.9 > 0.3, nicht verschoben)
        assert synced[1].start_time == 2.1
    
    def test_sync_quotes_within_threshold(self):
        """Quotes nahe bei Beats sollten verschoben werden."""
        quotes = [
            Quote(text="Test", start_time=0.95, end_time=2.0, confidence=1.0),
        ]
        beat_frames = np.array([0, 30, 60], dtype=np.int32)  # 0, 1, 2s @ 30fps
        fps = 30
        
        synced = sync_quotes_to_beats(quotes, beat_frames, fps, shift_threshold=0.3)
        
        # 0.95s -> naechster Beat bei 1.0s (Differenz 0.05 <= 0.3, verschoben)
        assert synced[0].start_time == 1.0
    
    def test_is_on_beat(self):
        """Sollte korrekt erkennen ob Frame auf Beat liegt."""
        beat_frames = np.array([30, 60, 90], dtype=np.int32)
        
        assert is_on_beat(30, beat_frames, tolerance=1) == True
        assert is_on_beat(31, beat_frames, tolerance=1) == True
        assert is_on_beat(35, beat_frames, tolerance=1) == False
        assert is_on_beat(60, beat_frames, tolerance=0) == True
    
    def test_get_beat_intensity(self):
        """Intensitaet sollte auf Beat maximal sein und abklingen."""
        beat_frames = np.array([30], dtype=np.int32)
        
        assert get_beat_intensity(30, beat_frames, decay_frames=6) == pytest.approx(1.0)
        assert get_beat_intensity(31, beat_frames, decay_frames=6) == pytest.approx(5.0 / 6.0)
        assert get_beat_intensity(35, beat_frames, decay_frames=6) == pytest.approx(1.0 / 6.0)
        assert get_beat_intensity(36, beat_frames, decay_frames=6) == pytest.approx(0.0)
    
    def test_get_next_beat_time(self):
        """Sollte naechsten Beat zurueckgeben."""
        beat_frames = np.array([30, 60, 90], dtype=np.int32)
        fps = 30
        
        assert get_next_beat_time(0.5, beat_frames, fps) == 1.0
        assert get_next_beat_time(1.5, beat_frames, fps) == 2.0
        assert get_next_beat_time(3.5, beat_frames, fps) is None
    
    def test_create_beat_grid_overlay(self):
        """Sollte Alpha-Wert fuer Beat-Marker berechnen."""
        beat_frames = np.array([30], dtype=np.int32)
        fps = 30
        
        assert create_beat_grid_overlay(1920, 1080, 30, beat_frames, fps) == 1.0
        # Frame 31 ist nahe genug fuer is_on_beat -> 1.0
        assert create_beat_grid_overlay(1920, 1080, 32, beat_frames, fps) == pytest.approx(0.1)
        assert create_beat_grid_overlay(1920, 1080, 40, beat_frames, fps) == 0.0
