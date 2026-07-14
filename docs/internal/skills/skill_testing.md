# 🧪 SKILL SPECIFICATION: Testing & Coverage

## 1. ARCHITEKTUR-PRINZIPIEN (Unverrückbar)
- **Isolation:** Jeder Test MUSS unabhängig von externer Hardware laufen können.
- **Mocking:** OpenGL/ModernGL-Contexts werden mit `unittest.mock` oder `pytest.mock` gemockt.
- **Fixtures:** Wiederverwendbare Test-Daten (Audio-Features, Dummy-Frames) als pytest-Fixtures.
- **Parametrisierung:** Mehrere Visualizer/Configs mit `@pytest.mark.parametrize` testen.

## 2. TEST-KATEGORIEN

| Kategorie | Ziel | Beispiel |
|---|---|---|
| Unit-Tests | Einzelne Funktionen isoliert | `PostProcessor.apply()` mit Dummy-Frame |
| Integration-Tests | Modul-Interaktionen | `GPUBatchRenderer` mit gemocktem Context |
| E2E-Tests | Kompletter Flow | CLI-Befehle mit `CliRunner` |

## 3. MODERNGl-MOCKING-PATTERN

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_moderngl():
    """Mocked ModernGL Context für GPU-Tests ohne Grafikkarte."""
    mock_ctx = MagicMock()
    mock_fbo = MagicMock()
    mock_texture = MagicMock()
    
    mock_ctx.framebuffer.return_value = mock_fbo
    mock_ctx.texture.return_value = mock_texture
    mock_ctx.program.return_value = MagicMock()
    
    with patch('moderngl.create_standalone_context', return_value=mock_ctx):
        with patch('src.gpu_renderer.moderngl', mock_ctx):
            yield mock_ctx
```

## 4. DUMMY-DATA-FACTORY

```python
def make_dummy_features(frame_count=30, fps=30) -> AudioFeatures:
    """Erzeugt valide AudioFeatures für schnelle Tests."""
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
        beat_frames=np.array([0, 15, 30]),
    )
```

## 5. ANTI-PATTERNS (Verbotene Syntax)
- VERBOTEN: Tests die auf echte GPU/FFmpeg angewiesen sind ohne `@pytest.mark.skipif`
- VERBOTEN: `time.sleep()` in Tests
- VERBOTEN: File-System-Lecks (temporäre Dateien nicht aufräumen)
