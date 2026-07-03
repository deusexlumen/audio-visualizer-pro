"""
Shared pytest fixtures für das Audio Visualizer Pro Test-Suite.

Bietet wiederverwendbare Dummy-Daten und Mocks für alle Tests.
"""

import tempfile
import numpy as np
import pytest

from src.types import AudioFeatures, Quote

# Ergebnis des GPU-Checks wird gecached, damit nicht jeder Test
# einen eigenen OpenGL-Kontext erzeugen muss.
_GPU_AVAILABLE = None


def gpu_available() -> bool:
    """Prueft einmalig, ob ein OpenGL-Kontext erzeugt werden kann."""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            import moderngl
            ctx = moderngl.create_standalone_context()
            ctx.release()
            _GPU_AVAILABLE = True
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: Test benoetigt eine echte GPU mit OpenGL 3.3+"
    )


@pytest.fixture
def require_gpu():
    """Ueberspringt den Test, wenn keine GPU/OpenGL verfuegbar ist (z.B. headless CI)."""
    if not gpu_available():
        pytest.skip("Keine GPU/OpenGL verfuegbar")


@pytest.fixture
def dummy_audio_features() -> AudioFeatures:
    """Erzeugt valide AudioFeatures für schnelle Tests.

    Returns:
        AudioFeatures mit deterministischen Dummy-Werten.
    """
    frame_count = 30
    fps = 30
    return AudioFeatures(
        duration=frame_count / fps,
        sample_rate=44100,
        fps=fps,
        frame_count=frame_count,
        rms=np.random.rand(frame_count).astype(np.float32),
        onset=np.random.rand(frame_count).astype(np.float32),
        spectral_centroid=np.random.rand(frame_count).astype(np.float32),
        spectral_rolloff=np.random.rand(frame_count).astype(np.float32),
        zero_crossing_rate=np.random.rand(frame_count).astype(np.float32),
        transient=np.random.rand(frame_count).astype(np.float32),
        voice_clarity=np.random.rand(frame_count).astype(np.float32),
        voice_band=np.random.rand(frame_count).astype(np.float32),
        chroma=np.random.rand(12, frame_count).astype(np.float32),
        mfcc=np.random.rand(13, frame_count).astype(np.float32),
        tempogram=np.random.rand(384, frame_count).astype(np.float32),
        tempo=120.0,
        key="C",
        mode="music",
        beat_frames=np.array([0, 15, 29]),
    )


@pytest.fixture
def dummy_frame() -> np.ndarray:
    """Erzeugt einen einfachen RGB-Testframe.

    Returns:
        NumPy-Array der Shape (H, W, 3), dtype uint8.
    """
    return np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)


@pytest.fixture
def dummy_frame_small() -> np.ndarray:
    """Erzeugt einen kleinen RGB-Testframe für schnelle LUT-Tests.

    Returns:
        NumPy-Array der Shape (10, 10, 3), dtype uint8.
    """
    return np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)


@pytest.fixture
def sample_quotes() -> list:
    """Erzeugt eine Liste von Test-Zitaten.

    Returns:
        Liste mit 2 Quote-Objekten.
    """
    return [
        Quote(text="Test quote one", start_time=1.0, end_time=3.0, confidence=0.9),
        Quote(text="Test quote two", start_time=5.0, end_time=7.0, confidence=0.8),
    ]


@pytest.fixture
def mock_lut_file() -> str:
    """Erzeugt eine temporäre .cube LUT-Datei.

    Yields:
        Pfad zur temporären LUT-Datei.
    """
    lut_content = """# Simple 2x2x2 test LUT
TITLE "Test LUT"
LUT_3D_SIZE 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cube', delete=False) as f:
        f.write(lut_content)
        path = f.name
    yield path
    import os
    os.unlink(path)
