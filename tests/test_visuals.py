"""
test_visuals.py - Tests fuer alle GPU-Visualizer.

Sicherstellt, dass alle GPU-Visualizer erfolgreich rendern.
"""

import pytest
import numpy as np
import moderngl

from src.gpu_visualizers import list_visualizers, get_visualizer
from src.types import AudioFeatures


@pytest.fixture
def dummy_features():
    """Minimal Features fuer schnelle GPU-Tests."""
    return {
        "rms": np.random.rand(30).astype(np.float32),
        "onset": np.random.rand(30).astype(np.float32),
        "beat_intensity": np.random.rand(30).astype(np.float32),
        "spectral_centroid": np.random.rand(30).astype(np.float32),
        "chroma": np.random.rand(12, 30).astype(np.float32),
        "transient": np.random.rand(30).astype(np.float32),
        "voice_clarity": np.random.rand(30).astype(np.float32),
        "fps": 30,
        "frame_count": 30,
        "mode": "music",
        "tempo": 120.0,
    }


@pytest.fixture(scope="module")
def gl_context():
    """Erzeugt einen ModernGL Standalone-Context fuer alle Tests."""
    ctx = moderngl.create_standalone_context()
    yield ctx
    ctx.release()


@pytest.fixture
def fbo(gl_context):
    """Erzeugt ein Framebuffer fuer Test-Rendering."""
    texture = gl_context.texture((640, 480), 3)
    fbo = gl_context.framebuffer(color_attachments=[texture])
    yield fbo
    fbo.release()
    texture.release()


def test_all_visualizers(gl_context, fbo, dummy_features):
    """Testet dass alle GPU-Visualizer erfolgreich rendern."""
    available = list_visualizers()
    print(f"\nGefundene GPU-Visualizer: {available}")
    assert len(available) > 0, "Keine Visualizer gefunden!"

    for name in available:
        print(f"\nTesting {name}...")
        viz_cls = get_visualizer(name)
        viz = viz_cls(gl_context, 640, 480)

        fbo.use()
        gl_context.clear(0.05, 0.05, 0.05)
        viz.render(dummy_features, 0.5)

        # Framebuffer auslesen zur Validierung
        pixels = fbo.read(components=3)
        assert len(pixels) == 640 * 480 * 3, f"{name}: Falsche Pixel-Anzahl"

        print(f"  OK {name}")


def test_visualizer_registry():
    """Testet das Registry-System."""
    available = list_visualizers()
    print(f"\nVerfuegbare Visualizer: {available}")

    # Sollte mindestens die Signature-Visualizer haben
    expected = ['lumina_core', 'voice_flow', 'spectrum_genesis']
    for vis in expected:
        assert vis in available, f"Visualizer '{vis}' nicht gefunden!"


def test_get_feature_at_frame(gl_context, dummy_features):
    """Testet die _get_feature_at_frame Hilfsmethode der Base-Klasse."""
    viz_cls = get_visualizer('lumina_core')
    viz = viz_cls(gl_context, 640, 480)

    f = viz._get_feature_at_frame(dummy_features, 15)

    assert 'rms' in f
    assert 'onset' in f
    assert 'chroma' in f
    assert 'transient' in f
    assert 'voice_clarity' in f

    assert 0 <= f['rms'] <= 1
    assert 0 <= f['onset'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
