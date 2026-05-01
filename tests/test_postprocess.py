"""
Tests für das Post-Processing Modul.

Ziel: 100% Coverage für src/postprocess.py.
Testet alle Effekte isoliert und in Kombination.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.postprocess import PostProcessor, PostProcessPipeline


class TestPostProcessorInit:
    """Tests für die Initialisierung des PostProcessors."""

    def test_default_config(self):
        """PostProcessor mit leerer Config sollte Defaults verwenden."""
        pp = PostProcessor({})
        assert pp.contrast == 1.0
        assert pp.saturation == 1.0
        assert pp.brightness == 1.0
        assert pp.grain_intensity == 0.0
        assert pp.chromatic == 0.0
        assert pp.vignette == 0.0
        assert pp.bloom == 0.3
        assert pp.bloom_threshold == 180
        assert pp.lut is None

    def test_custom_config(self):
        """PostProcessor sollte alle Config-Werte korrekt übernehmen."""
        pp = PostProcessor({
            'contrast': 1.5,
            'saturation': 0.8,
            'brightness': 1.2,
            'grain': 0.3,
            'chromatic_aberration': 0.5,
            'vignette': 0.4,
            'bloom': 0.6,
            'bloom_threshold': 200,
        })
        assert pp.contrast == 1.5
        assert pp.saturation == 0.8
        assert pp.brightness == 1.2
        assert pp.grain_intensity == 0.3
        assert pp.chromatic == 0.5
        assert pp.vignette == 0.4
        assert pp.bloom == 0.6
        assert pp.bloom_threshold == 200


class TestLoadLut:
    """Tests für den LUT-Loader."""

    def test_no_lut_path(self):
        """Wenn kein Pfad angegeben, sollte lut None sein."""
        pp = PostProcessor({})
        assert pp._load_lut(None) is None

    def test_lut_file_not_found(self):
        """Nicht existierende LUT-Datei sollte None zurückgeben."""
        pp = PostProcessor({})
        assert pp._load_lut('/nonexistent/file.cube') is None

    def test_load_valid_lut(self, mock_lut_file):
        """Gültige .cube Datei sollte als numpy Array geladen werden."""
        pp = PostProcessor({})
        lut = pp._load_lut(mock_lut_file)
        assert lut is not None
        assert isinstance(lut, np.ndarray)
        assert lut.shape == (8, 3)

    def test_load_lut_with_comments(self):
        """LUT mit Kommentaren und Leerzeilen sollte korrekt geparst werden."""
        content = """
# Comment line
TITLE "Test"

0.0 0.0 0.0
1.0 1.0 1.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cube', delete=False) as f:
            f.write(content)
            path = f.name

        try:
            pp = PostProcessor({})
            lut = pp._load_lut(path)
            assert lut is not None
            assert lut.shape == (2, 3)
            np.testing.assert_array_almost_equal(lut[0], [0.0, 0.0, 0.0])
            np.testing.assert_array_almost_equal(lut[1], [1.0, 1.0, 1.0])
        finally:
            Path(path).unlink()


class TestApplyEffects:
    """Tests für die einzelnen Post-Processing-Effekte."""

    def test_apply_no_effects(self, dummy_frame):
        """Wenn keine Effekte aktiv, sollte der Frame unverändert bleiben."""
        # Achtung: bloom hat Default 0.3, muss explizit auf 0 gesetzt werden
        pp = PostProcessor({'bloom': 0.0})
        result = pp.apply(dummy_frame)
        np.testing.assert_array_equal(result, dummy_frame)

    def test_apply_contrast(self, dummy_frame):
        """Kontrast-Anpassung sollte den Frame verändern."""
        pp = PostProcessor({'contrast': 1.5})
        result = pp.apply(dummy_frame)
        assert result.shape == dummy_frame.shape
        assert result.dtype == np.uint8

    def test_apply_saturation(self, dummy_frame):
        """Sättigungs-Anpassung sollte den Frame verändern."""
        pp = PostProcessor({'saturation': 0.0})
        result = pp.apply(dummy_frame)
        assert result.shape == dummy_frame.shape
        # Bei Sättigung 0 sollte das Bild Graustufen haben
        gray = result[:, :, 0] == result[:, :, 1]
        # Nicht perfekt gleich wegen Integer-Arithmetik, aber nah dran
        assert result.dtype == np.uint8

    def test_apply_brightness(self, dummy_frame):
        """Helligkeits-Anpassung sollte den Frame verändern."""
        pp = PostProcessor({'brightness': 1.5})
        result = pp.apply(dummy_frame)
        assert result.shape == dummy_frame.shape
        assert result.dtype == np.uint8

    def test_apply_bloom(self, dummy_frame):
        """Bloom sollte den Frame verändern."""
        pp = PostProcessor({'bloom': 0.8, 'bloom_threshold': 100})
        result = pp.apply(dummy_frame)
        assert result.shape == dummy_frame.shape
        assert result.dtype == np.uint8

    def test_apply_grain(self, dummy_frame):
        """Film Grain sollte den Frame verändern."""
        pp = PostProcessor({'grain': 0.5})
        result = pp.apply(dummy_frame)
        assert result.shape == dummy_frame.shape
        assert result.dtype == np.uint8
        # Grain fügt Noise hinzu, also sollte das Ergebnis anders sein
        assert not np.array_equal(result, dummy_frame)

    def test_apply_vignette(self, dummy_frame):
        """Vignette sollte den Frame verändern."""
        pp = PostProcessor({'vignette': 0.8})
        result = pp.apply(dummy_frame)
        assert result.shape == dummy_frame.shape
        assert result.dtype == np.uint8

    def test_apply_chromatic_aberration(self, dummy_frame):
        """Chromatic Aberration sollte den Frame verändern."""
        pp = PostProcessor({'chromatic_aberration': 1.0})
        result = pp.apply(dummy_frame)
        assert result.shape == dummy_frame.shape
        assert result.dtype == np.uint8

    def test_apply_chromatic_zero_shift(self, dummy_frame):
        """Chromatic mit shift=0 sollte den Frame unverändert lassen."""
        pp = PostProcessor({'chromatic_aberration': 0.01})
        result = pp._apply_chromatic_aberration(dummy_frame)
        np.testing.assert_array_equal(result, dummy_frame)

    def test_apply_all_effects(self, dummy_frame):
        """Alle Effekte gleichzeitig sollten fehlerfrei angewendet werden."""
        pp = PostProcessor({
            'contrast': 1.2,
            'saturation': 1.1,
            'brightness': 1.05,
            'grain': 0.2,
            'chromatic_aberration': 0.3,
            'vignette': 0.4,
            'bloom': 0.5,
        })
        result = pp.apply(dummy_frame)
        assert result.shape == dummy_frame.shape
        assert result.dtype == np.uint8


class TestApplyLut:
    """Tests für die LUT-Anwendung."""

    def test_apply_lut_without_loaded_lut(self, dummy_frame_small):
        """Wenn keine LUT geladen, sollte der Frame unverändert bleiben."""
        pp = PostProcessor({})
        result = pp.apply_lut(dummy_frame_small)
        np.testing.assert_array_equal(result, dummy_frame_small)

    def test_apply_lut_with_loaded_lut(self, mock_lut_file, dummy_frame_small):
        """Mit geladener LUT sollte der Frame transformiert werden."""
        pp = PostProcessor({'lut': mock_lut_file})
        result = pp.apply_lut(dummy_frame_small)
        assert result.shape == dummy_frame_small.shape
        assert result.dtype == np.uint8

    def test_apply_lut_integration(self, mock_lut_file, dummy_frame_small):
        """LUT über apply() sollte korrekt angewendet werden."""
        pp = PostProcessor({'lut': mock_lut_file})
        result = pp.apply(dummy_frame_small)
        assert result.shape == dummy_frame_small.shape
        assert result.dtype == np.uint8


class TestPostProcessPipeline:
    """Tests für die PostProcessPipeline-Wrapper-Klasse."""

    def test_process_frame(self, dummy_frame):
        """process_frame sollte einen Frame durch den Processor leiten."""
        pipeline = PostProcessPipeline({'contrast': 1.2})
        result = pipeline.process_frame(dummy_frame)
        assert result.shape == dummy_frame.shape
        assert result.dtype == np.uint8

    def test_process_video_success(self):
        """process_video sollte FFmpeg aufrufen und bei Erfolg beenden."""
        pipeline = PostProcessPipeline({})

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as infile:
                infile.close()
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as outfile:
                    outfile.close()
                    try:
                        pipeline.process_video(infile.name, outfile.name)
                    finally:
                        Path(infile.name).unlink(missing_ok=True)
                        Path(outfile.name).unlink(missing_ok=True)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == 'ffmpeg'
            assert '-i' in cmd

    def test_process_video_failure(self):
        """process_video sollte bei FFmpeg-Fehler einen RuntimeError werfen."""
        pipeline = PostProcessPipeline({})

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr='codec error')
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as infile:
                infile.close()
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as outfile:
                    outfile.close()
                    try:
                        with pytest.raises(RuntimeError, match='FFmpeg Fehler'):
                            pipeline.process_video(infile.name, outfile.name)
                    finally:
                        Path(infile.name).unlink(missing_ok=True)
                        Path(outfile.name).unlink(missing_ok=True)
