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
    """Erzeugt einen ModernGL Standalone-Context fuer alle Tests.

    Ueberspringt die Tests, wenn keine GPU/OpenGL verfuegbar ist
    (z.B. headless CI), statt mit einem kryptischen Fehler abzubrechen.
    """
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"Keine GPU/OpenGL verfuegbar: {e}")
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


@pytest.mark.gpu
def test_all_visualizers_sichtbar_und_ohne_nan(gl_context, dummy_features):
    """Visueller Smoke-Test: jeder Visualizer liefert sichtbares Bild ohne NaNs.

    Regressionsnetz fuer Shader-Umbauten: ein komplett schwarzes/eingefrorenes
    Bild oder NaN-Pixel fallen sofort auf, ohne Referenzbilder zu pflegen.
    """
    # GL-Fehlerflag frueherer Tests leeren, sonst schlaegt die Textur-Erzeugung fehl
    _ = gl_context.error

    # Nicht kleiner waehlen: einige Geometrie-Visualizer erwarten >=100px Kantenlaenge
    width, height = 256, 144
    texture = gl_context.texture((width, height), 3, dtype="f2")
    small_fbo = gl_context.framebuffer(color_attachments=[texture])
    try:
        for name in list_visualizers():
            viz_cls = get_visualizer(name)
            viz = viz_cls(gl_context, width, height)

            small_fbo.use()
            gl_context.clear(0.0, 0.0, 0.0)
            # Mittlerer Zeitpunkt, damit zeitabhaengige Effekte aktiv sind
            viz.render(dummy_features, 0.5)

            raw = np.frombuffer(
                small_fbo.read(components=3, dtype="f2"), dtype=np.float16
            ).astype(np.float32)

            assert not np.isnan(raw).any(), f"{name}: NaN-Pixel im Output"
            assert not np.isinf(raw).any(), f"{name}: Inf-Pixel im Output"
            assert float(np.ptp(raw)) > 0.01, (
                f"{name}: Output ist (nahezu) uniform — Visualizer rendert nichts"
            )
    finally:
        small_fbo.release()
        texture.release()
        # Etwaige eigene GL-Fehler nicht an nachfolgende Tests weiterreichen
        _ = gl_context.error


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


def test_auto_discovery_keeps_manual_registry():
    """Auto-Discovery darf die manuelle Registry nicht ueberschreiben."""
    from src.gpu_visualizers import _MANUAL_VISUALIZER_MAP, VISUALIZER_MAP

    for name in _MANUAL_VISUALIZER_MAP:
        assert name in VISUALIZER_MAP, f"Manueller Eintrag '{name}' fehlt in gemischter Registry"
        assert VISUALIZER_MAP[name] is _MANUAL_VISUALIZER_MAP[name]


def test_validate_visualizer_class_passes_for_builtin():
    """Validator sollte alle eingebauten Visualizer als valide einstufen."""
    from src.gpu_visualizers import validate_visualizer_class, list_visualizers

    for name in list_visualizers():
        cls = get_visualizer(name)
        errors = validate_visualizer_class(cls)
        assert errors == [], f"Visualizer '{name}' ist nicht valide: {errors}"


def test_validate_visualizer_class_catches_bad_params():
    """Validator sollte fehlerhafte PARAMS erkennen."""
    from src.gpu_visualizers import validate_visualizer_class
    from src.gpu_visualizers.base import BaseGPUVisualizer

    class BadParamsVisualizer(BaseGPUVisualizer):
        PARAMS = {
            "good": (1.0, 0.0, 2.0, 0.1),
            "bad_tuple": (1.0, 0.0, 2.0),  # zu kurz
            "bad_type": ("a", 0, 1, 0.1),  # default nicht numerisch
        }

        def _setup(self):
            pass

        def render(self, features: dict, time: float):
            pass

    errors = validate_visualizer_class(BadParamsVisualizer)
    assert any("bad_tuple" in e for e in errors)
    assert any("bad_type" in e for e in errors)


def test_validate_visualizer_class_catches_missing_render():
    """Validator sollte fehlende render()-Methode erkennen."""
    from src.gpu_visualizers import validate_visualizer_class
    from src.gpu_visualizers.base import BaseGPUVisualizer

    class NoRenderVisualizer(BaseGPUVisualizer):
        PARAMS = {"intensity": (1.0, 0.0, 2.0, 0.1)}

        def _setup(self):
            pass

        # render() absichtlich nicht implementiert

    errors = validate_visualizer_class(NoRenderVisualizer)
    assert any("render()" in e for e in errors)


def test_wizard_templates_render(gl_context, fbo, dummy_features):
    """Generierte Wizard-Templates muessen einen Frame rendern koennen."""
    import sys
    import tempfile
    import shutil
    from pathlib import Path
    from src.visualizer_wizard import VisualizerWizard
    from src.gpu_visualizers import refresh_registry, get_visualizer

    tmpdir = tempfile.mkdtemp()
    original_path = Path("src/gpu_visualizers")
    generated = []

    try:
        for viz_type in VisualizerWizard.list_types():
            wizard = VisualizerWizard(f"test_wizard_{viz_type}", viz_type=viz_type)
            target = wizard.write(original_path)
            generated.append((wizard.module_name, target))

        refresh_registry()

        for module_name, _ in generated:
            cls = get_visualizer(module_name)
            viz = cls(gl_context, 640, 480)
            fbo.use()
            gl_context.clear(0.0, 0.0, 0.0)
            viz.render(dummy_features, 0.5)
            pixels = fbo.read(components=3)
            assert len(pixels) == 640 * 480 * 3, f"{module_name}: Falsche Pixel-Anzahl"
    finally:
        # Dateien und Import-Cache aufraeumen
        for module_name, target in generated:
            if target.exists():
                target.unlink()
            sys.modules.pop(f"src.gpu_visualizers.{module_name}", None)
        refresh_registry()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_add_create_visualizer_button():
    """GUI-Hilfsfunktion sollte einen QPushButton in das Layout einfuegen."""
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget, QPushButton
    from src.gui.state import AppState
    from src.visualizer_wizard import add_create_visualizer_button

    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    layout = QVBoxLayout(parent)
    state = AppState()

    btn = add_create_visualizer_button(layout, state, parent_window=parent)

    assert isinstance(btn, QPushButton)
    assert btn.text() == "Neuen Visualizer erstellen..."
    assert layout.indexOf(btn) >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
