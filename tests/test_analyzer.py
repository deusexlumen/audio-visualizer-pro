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


def _stats_arrays(cent_mean, voice_mean, rms_std):
    """Baut Feature-Arrays mit exakt den gewuenschten Kennwerten."""
    spec_cent = np.full(100, cent_mean, dtype=np.float32)
    voice = np.full(100, voice_mean, dtype=np.float32)
    # Abwechselnd +/- Abweichung um 0.5 -> Standardabweichung = rms_std
    rms = np.tile([0.5 - rms_std, 0.5 + rms_std], 50).astype(np.float32)
    return spec_cent, voice, rms


class TestModeDetection:
    """Modus-Erkennung: Kennwerte stammen aus dem Golden-Korpus."""

    def test_podcast_kennwerte_ergeben_speech(self, analyzer):
        """Podcast-Werte (podcast_macy) muessen 'speech' liefern.

        Regression: die alte UND-Kette verlangte cent_mean < 2000 und
        voice_mean > 0.45 — beides ist nach dem Pre-Emphasis-Filter
        unerreichbar, jeder Podcast landete auf 'music'.
        """
        assert analyzer._detect_mode_advanced(*_stats_arrays(5201, 0.372, 0.107)) == "speech"

    def test_musik_kennwerte_ergeben_music(self, analyzer):
        """Musik-Werte (music_severance) muessen 'music' liefern."""
        assert analyzer._detect_mode_advanced(*_stats_arrays(6055, 0.288, 0.154)) == "music"

    def test_werte_dazwischen_ergeben_hybrid(self, analyzer):
        """Unklare Faelle landen bewusst auf 'hybrid' statt zu raten."""
        assert analyzer._detect_mode_advanced(*_stats_arrays(5900, 0.315, 0.135)) == "hybrid"

    def test_einzelner_ausreisser_kippt_nicht(self, analyzer):
        """Zwei klare Sprach-Merkmale schlagen ein gegenlaeufiges drittes."""
        assert analyzer._detect_mode_advanced(*_stats_arrays(7000, 0.380, 0.100)) == "speech"

    def test_feature_score_bereich(self, analyzer):
        """Der Merkmals-Score bleibt in [-1, 1] und ist an den Kanten gesaettigt."""
        assert analyzer._mode_feature_score(0.05, 0.10, 0.17) == 1.0
        assert analyzer._mode_feature_score(0.30, 0.10, 0.17) == -1.0
        assert abs(analyzer._mode_feature_score(0.135, 0.10, 0.17)) < 1e-6


class TestTempoEstimation:
    """Tempo-Schaetzung."""

    def test_plausibilitaet(self, analyzer):
        """Nur endliche Werte im BPM-Fenster gelten als plausibel."""
        assert analyzer._tempo_plausible(120.0)
        assert analyzer._tempo_plausible(analyzer.TEMPO_MIN_BPM)
        assert analyzer._tempo_plausible(analyzer.TEMPO_MAX_BPM)
        assert not analyzer._tempo_plausible(float('inf'))
        assert not analyzer._tempo_plausible(39.0)
        assert not analyzer._tempo_plausible(251.0)

    def test_tempogram_fallback_ignoriert_lag_null(self, analyzer, monkeypatch):
        """Der Tempogram-Zweig darf nicht am Lag 0 haengenbleiben.

        Regression: Lag 0 hat immer die groesste Energie und entspricht
        inf BPM — ein blankes argmax lieferte deshalb ausnahmslos den
        Fallback-Wert von 120 BPM.
        """
        import src.analyzer as analyzer_module

        def _fail(*args, **kwargs):
            raise RuntimeError("librosa-Schaetzer nicht verfuegbar")

        monkeypatch.setattr(analyzer_module, "_librosa_tempo", _fail)

        sr, hop = 22050, 512
        # Klick alle 17 Frames -> rund 152 BPM (Oktaven: 76 / 304)
        onset_env = np.zeros(1200, dtype=np.float32)
        onset_env[::17] = 1.0

        tempo = analyzer._estimate_tempo_simple(onset_env, sr, hop)

        assert np.isfinite(tempo)
        assert analyzer.TEMPO_MIN_BPM <= tempo <= analyzer.TEMPO_MAX_BPM
        assert tempo != analyzer.TEMPO_FALLBACK_BPM


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
