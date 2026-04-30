"""
test_analyzer.py - Tests für den AudioAnalyzer

Testet die Audio-Feature-Extraktion.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import wave
import struct

# Füge src zum Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer import AudioAnalyzer


def create_test_audio(duration=1.0, sample_rate=44100, freq=440):
    """Erstellt eine Test-Audio-Datei (Sinus-Welle)."""
    num_samples = int(duration * sample_rate)
    
    # Generiere Sinus-Welle
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        sample = np.sin(2 * np.pi * freq * t) * 0.5
        samples.append(sample)
    
    # Konvertiere zu 16-bit PCM
    samples = np.array(samples)
    samples = (samples * 32767).astype(np.int16)
    
    # Speichere als WAV
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    
    with wave.open(temp_file.name, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    
    return temp_file.name


@pytest.fixture
def test_audio_file():
    """Erstellt eine temporäre Test-Audio-Datei."""
    path = create_test_audio(duration=2.0, freq=440)
    yield path
    # Cleanup
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def analyzer():
    """Erstellt einen AudioAnalyzer mit temporärem Cache."""
    import tempfile
    cache_dir = tempfile.mkdtemp()
    return AudioAnalyzer(cache_dir=cache_dir)


def test_analyze_basic(analyzer, test_audio_file):
    """Testet grundlegende Analyse-Funktionalität."""
    features = analyzer.analyze(test_audio_file, fps=30)
    
    # Überprüfe grundlegende Attribute
    assert features.duration > 0
    assert features.sample_rate == 44100  # Analyse nutzt 44.1kHz für hohe Zeitauflösung
    assert features.fps == 30
    assert features.tempo >= 0  # Reine Sinus-Welle hat möglicherweise kein erkennbares Tempo
    assert features.mode in ['music', 'speech', 'hybrid']
    
    print(f"\nDauer: {features.duration:.2f}s")
    print(f"Tempo: {features.tempo:.1f} BPM")
    print(f"Mode: {features.mode}")


def test_feature_shapes(analyzer, test_audio_file):
    """Testet die Shapes der extrahierten Features."""
    fps = 30
    features = analyzer.analyze(test_audio_file, fps=fps)
    
    expected_frames = int(features.duration * fps)
    
    # Zeitliche Features sollten die richtige Länge haben
    assert len(features.rms) == expected_frames, f"RMS: {len(features.rms)} != {expected_frames}"
    assert len(features.onset) == expected_frames
    assert len(features.spectral_centroid) == expected_frames
    assert len(features.spectral_rolloff) == expected_frames
    assert len(features.zero_crossing_rate) == expected_frames
    
    # Chroma sollte Shape (12, frames) haben
    assert features.chroma.shape[0] == 12
    
    # MFCC sollte Shape (13, frames) haben
    assert features.mfcc.shape[0] == 13


def test_feature_ranges(analyzer, test_audio_file):
    """Testet dass alle Features im gültigen Bereich liegen."""
    features = analyzer.analyze(test_audio_file, fps=30)
    
    # Alle Features sollten zwischen 0 und 1 liegen (normalisiert)
    assert 0 <= features.rms.min() <= features.rms.max() <= 1
    assert 0 <= features.onset.min() <= features.onset.max() <= 1
    assert 0 <= features.spectral_centroid.min() <= features.spectral_centroid.max() <= 1
    assert 0 <= features.spectral_rolloff.min() <= features.spectral_rolloff.max() <= 1
    assert 0 <= features.zero_crossing_rate.min() <= features.zero_crossing_rate.max() <= 1


def test_caching(analyzer, test_audio_file):
    """Testet dass Caching funktioniert."""
    # Erste Analyse
    features1 = analyzer.analyze(test_audio_file, fps=30)
    
    # Zweite Analyse (sollte aus Cache kommen)
    features2 = analyzer.analyze(test_audio_file, fps=30)
    
    # Sollten identisch sein
    assert np.allclose(features1.rms, features2.rms)
    assert np.allclose(features1.onset, features2.onset)
    assert features1.duration == features2.duration


def test_force_reanalyze(analyzer, test_audio_file):
    """Testet force_reanalyze Option."""
    # Erste Analyse
    features1 = analyzer.analyze(test_audio_file, fps=30)
    
    # Zweite Analyse mit force_reanalyze
    features2 = analyzer.analyze(test_audio_file, fps=30, force_reanalyze=True)
    
    # Sollten immer noch identisch sein (gleiche Quelldaten)
    assert np.allclose(features1.rms, features2.rms)


def test_normalize_method(analyzer):
    """Testet die _normalize Hilfsmethode."""
    # Test mit bekannten Werten
    data = np.array([0, 50, 100])
    normalized = analyzer._normalize(data)
    
    assert normalized.min() == 0
    assert abs(normalized.max() - 1) < 1e-10  # Floating-Point Toleranz
    assert abs(normalized[1] - 0.5) < 1e-10  # Floating-Point Toleranz


def test_interpolate_method(analyzer):
    """Testet die _interpolate_to_length Hilfsmethode."""
    data = np.array([0, 50, 100])
    
    # Interpoliere zu 5 Werten
    interpolated = analyzer._interpolate_to_length(data, 5)
    
    assert len(interpolated) == 5
    assert interpolated[0] == 0
    assert interpolated[-1] == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
