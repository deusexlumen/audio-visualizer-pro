"""
Tests für den SmartMatcher (AI Matcher).
"""

import pytest
import numpy as np
from src.types import AudioFeatures
from src.ai_matcher import SmartMatcher, AIRecommendation


def create_dummy_features(
    mode="speech",
    tempo=90.0,
    rms_mean=0.3,
    onset_density=0.05,
    duration=10.0,
    key="C major",
    fps=30,
) -> AudioFeatures:
    """Erzeugt Dummy-Features mit kontrollierbaren Werten."""
    frame_count = int(duration * fps)
    
    # RMS: konstant um rms_mean mit kleiner Variation
    rms = np.full(frame_count, rms_mean)
    
    # Onset: sparse, basierend auf onset_density
    onset = np.zeros(frame_count)
    beat_interval = int(1.0 / (onset_density + 0.001))
    for i in range(0, frame_count, max(beat_interval, 1)):
        onset[i] = 0.8
    
    # Chroma: C major-ish (C und G laut)
    chroma = np.zeros((12, frame_count))
    chroma[0, :] = 0.5   # C
    chroma[7, :] = 0.3   # G
    
    return AudioFeatures(
        duration=duration,
        sample_rate=44100,
        fps=fps,
        rms=rms,
        onset=onset,
        spectral_centroid=np.full(frame_count, 0.4),
        spectral_rolloff=np.full(frame_count, 0.5),
        zero_crossing_rate=np.full(frame_count, 0.05),
        chroma=chroma,
        mfcc=np.zeros((13, frame_count)),
        tempogram=np.zeros((384, frame_count)),
        tempo=tempo,
        key=key,
        mode=mode,
    )


class TestSmartMatcher:
    """Test-Suite für den SmartMatcher."""
    
    def test_returns_ai_recommendation(self):
        matcher = SmartMatcher()
        features = create_dummy_features()
        result = matcher.match(features)
        
        assert isinstance(result, AIRecommendation)
        assert isinstance(result.visualizer, str)
        assert len(result.visualizer) > 0
        assert isinstance(result.reason, str)
        assert 0.0 <= result.confidence <= 1.0
        assert 'primary' in result.colors
        assert 'background' in result.colors
    
    def test_calm_podcast_recommends_typographic(self):
        matcher = SmartMatcher()
        features = create_dummy_features(
            mode="speech",
            rms_mean=0.15,
            onset_density=0.02,
            tempo=80.0,
        )
        result = matcher.match(features)
        
        assert result.visualizer == 'typographic'
        assert result.confidence > 0.7
    
    def test_energetic_music_recommends_spectrum_or_particles(self):
        matcher = SmartMatcher()
        features = create_dummy_features(
            mode="music",
            rms_mean=0.7,
            onset_density=0.25,
            tempo=140.0,
        )
        result = matcher.match(features)
        
        assert result.visualizer in ['spectrum_bars', 'neon_oscilloscope', 'particle_swarm']
        assert result.confidence > 0.7
    
    def test_slow_music_recommends_mandala(self):
        matcher = SmartMatcher()
        features = create_dummy_features(
            mode="music",
            rms_mean=0.2,
            onset_density=0.03,
            tempo=60.0,
        )
        result = matcher.match(features)
        
        assert result.visualizer == 'sacred_mandala'
    
    def test_hybrid_recommends_wave_or_pulsing(self):
        matcher = SmartMatcher()
        features = create_dummy_features(
            mode="hybrid",
            rms_mean=0.4,
            onset_density=0.1,
            tempo=100.0,
        )
        result = matcher.match(features)
        
        assert result.visualizer in ['neon_wave_circle', 'pulsing_core']
    
    def test_key_colors_are_valid_hex(self):
        matcher = SmartMatcher()
        features = create_dummy_features(key="G major")
        result = matcher.match(features)
        
        for color_name, hex_val in result.colors.items():
            assert hex_val.startswith('#')
            assert len(hex_val) == 7
            # Prüfe, ob gültige Hex-Zeichen
            int(hex_val[1:], 16)
    
    def test_minor_key_darkens_colors(self):
        matcher = SmartMatcher()
        major = create_dummy_features(key="C major")
        minor = create_dummy_features(key="C minor")
        
        result_major = matcher.match(major)
        result_minor = matcher.match(minor)
        
        # Moll-Hintergrund sollte dunkler sein
        assert result_minor.colors['background'] <= result_major.colors['background']
    
    def test_visual_config_conversion(self):
        matcher = SmartMatcher()
        features = create_dummy_features()
        result = matcher.match(features)
        
        config = result.to_visual_config(resolution=(1280, 720), fps=30)
        
        assert config.type == result.visualizer
        assert config.resolution == (1280, 720)
        assert config.fps == 30
        assert config.params == result.params
    
    def test_extract_features_computes_aggregates(self):
        matcher = SmartMatcher()
        features = create_dummy_features(rms_mean=0.5, onset_density=0.1)
        
        extracted = matcher._extract_features(features)
        
        assert extracted['rms_mean'] == pytest.approx(0.5, abs=0.01)
        assert extracted['mode'] == 'speech'
        assert extracted['tempo'] == 90.0
    
    def test_none_key_fallback_colors(self):
        matcher = SmartMatcher()
        features = create_dummy_features(key=None)
        result = matcher.match(features)
        
        assert result.colors['primary'] == '#667EEA'
        assert result.colors['background'] == '#1A1A2E'
