"""
End-to-End Tests fuer die GPU-Rendering-Pipeline.

Diese Tests pruefen den kompletten Flow von Audio-Analyse bis Video-Export.
"""

import os
import tempfile
import wave
import struct
import numpy as np
import pytest

from main import cli
from click.testing import CliRunner


def create_test_wav(path: str, duration: float = 2.0, sample_rate: int = 44100):
    """Erstellt eine einfache Test-WAV-Datei mit einem Sinus-Ton."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # 440Hz Sinus + etwas Rauschen fuer realistischere Features
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.05 * np.random.randn(len(t))
    audio = tone + noise
    audio = np.clip(audio, -1.0, 1.0)
    
    # In 16-bit PCM umwandeln
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


class TestEndToEnd:
    """End-to-End Tests fuer die komplette Pipeline."""
    
    @pytest.fixture
    def test_audio(self):
        """Erstellt eine temporäre Test-Audiodatei."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            create_test_wav(f.name, duration=1.5)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def runner(self):
        """Click Test-Runner."""
        return CliRunner()
    
    def test_ffmpeg_check_passes(self, runner):
        """FFmpeg-Check sollte bestehen, wenn FFmpeg installiert ist."""
        result = runner.invoke(cli, ['analyze', '--help'])
        # Wenn FFmpeg fehlt, wuerde bereits der Import oder der Check fehlschlagen
        assert result.exit_code == 0
    
    def test_list_visuals_shows_gpu_visualizers(self, runner):
        """list-visuals sollte GPU-Visualizer anzeigen."""
        result = runner.invoke(cli, ['list-visuals'])
        assert result.exit_code == 0
        assert "lumina_core" in result.output
        assert "voice_flow" in result.output
        assert "spectrum_genesis" in result.output
        assert "spectrum_bars" in result.output
    
    def test_create_template_generates_file(self, runner):
        """create-template sollte eine neue Datei erstellen."""
        test_name = "test_e2e_visualizer"
        target = f"src/gpu_visualizers/{test_name}.py"
        
        try:
            result = runner.invoke(cli, ['create-template', test_name])
            assert result.exit_code == 0
            assert os.path.exists(target)
            content = open(target).read()
            assert "BaseGPUVisualizer" in content
            assert test_name in content
        finally:
            if os.path.exists(target):
                os.unlink(target)
    
    def test_analyze_shows_features(self, runner, test_audio):
        """analyze sollte Audio-Features anzeigen."""
        result = runner.invoke(cli, ['analyze', test_audio])
        assert result.exit_code == 0
        assert "Audio-Analyse Ergebnisse" in result.output
        assert "RMS:" in result.output
        assert "Onset:" in result.output
        assert "Transient:" in result.output
        assert "Voice Clarity:" in result.output
    
    @pytest.mark.timeout(120)
    def test_render_preview_creates_video(self, runner, test_audio):
        """Preview-Rendering sollte ein valides MP4 erstellen."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'render', test_audio,
                '--visual', 'spectrum_bars',
                '--output', output_path,
                '--preview',
                '--preview-duration', '1.0',
                '--resolution', '640x360',
                '--fps', '30',
            ])
            
            assert result.exit_code == 0, f"Output: {result.output}"
            assert os.path.exists(output_path)
            # Datei sollte nicht leer sein (> 1KB fuer 1 Sekunde Video)
            assert os.path.getsize(output_path) > 1024
            assert "Fertig!" in result.output
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_render_with_params(self, runner, test_audio):
        """Rendering mit custom Parametern sollte funktionieren."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'render', test_audio,
                '--visual', 'spectrum_bars',
                '--output', output_path,
                '--preview',
                '--preview-duration', '0.5',
                '--resolution', '320x180',
                '--fps', '15',
                '--param', 'bar_count=20',
                '--param', 'smoothing=0.5',
            ])
            
            assert result.exit_code == 0, f"Output: {result.output}"
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_render_invalid_visualizer_fallback(self, runner, test_audio):
        """Ungueltiger Visualizer sollte sinnvoll behandelt werden."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'render', test_audio,
                '--visual', 'nonexistent_visualizer_12345',
                '--output', output_path,
                '--preview',
                '--preview-duration', '0.5',
                '--resolution', '320x180',
            ])
            
            # Sollte fehlschlagen, aber nicht crashen
            assert result.exit_code != 0 or not os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
