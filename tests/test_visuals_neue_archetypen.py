"""Tests fuer die neu gebauten Visualizer (Wellen 3 und 4).

Prueft die drei Design-Prinzipien, die sich automatisch pruefen lassen:

- Prinzip 3 (Modus = Empfindlichkeit): derselbe Visualizer laeuft in
  beiden Modi und wird bei Sprache ruhiger, statt eine andere Optik zu
  zeigen.
- Prinzip 5 (Hintergrundbild-Harmonie): genug dunkle Flaeche, damit ein
  untergelegtes Bild sichtbar bleibt. Die Regel gilt bewusst nur fuer die
  neuen Visualizer — bei den aelteren waere sie eine Nachruestung.
- Deterministisches Verhalten bei Zeitspruengen (Vorschau-Scrubbing):
  die zeitlich integrierten Zustaende (Drehung, Puls) duerfen nicht vom
  Aufrufverlauf abhaengen, sonst flackert die Vorschau.
"""

import numpy as np
import moderngl
import pytest

from src.gpu_visualizers import get_visualizer

NEUE_VISUALIZER = [
    # Welle 3
    "retro_sun", "dna_helix", "kaleidoscope",
    # Welle 4
    "spirograph", "voronoi_cells",
    # Welle 5
    "ink_bloom", "silk_ribbons",
    # Welle 6
    "scissor_lattice",
]

WIDTH, HEIGHT = 256, 144


def _features(mode: str, frames: int = 120):
    rng = np.random.default_rng(42)
    return {
        "rms": rng.uniform(0.3, 0.7, frames).astype(np.float32),
        "onset": rng.uniform(0.0, 0.8, frames).astype(np.float32),
        "beat_intensity": rng.uniform(0.0, 0.9, frames).astype(np.float32),
        "spectral_centroid": rng.uniform(0.3, 0.7, frames).astype(np.float32),
        "spectral_rolloff": rng.uniform(0.3, 0.7, frames).astype(np.float32),
        "zero_crossing_rate": rng.uniform(0.1, 0.5, frames).astype(np.float32),
        "transient": rng.uniform(0.0, 0.9, frames).astype(np.float32),
        "voice_clarity": rng.uniform(0.2, 0.6, frames).astype(np.float32),
        "voice_band": rng.uniform(0.2, 0.6, frames).astype(np.float32),
        "chroma": rng.uniform(0.0, 1.0, (12, frames)).astype(np.float32),
        "fps": 30,
        "frame_count": frames,
        "mode": mode,
        "tempo": 120.0,
    }


@pytest.fixture(scope="module")
def gl_context():
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"Keine GPU/OpenGL verfuegbar: {e}")
    try:
        ctx.__enter__()
    except Exception:
        pass
    yield ctx
    ctx.release()


def _render(ctx, name, mode, times=(0.0, 0.5, 1.0), viz=None):
    """Rendert eine Folge von Zeitpunkten und liefert das letzte Bild."""
    texture = ctx.texture((WIDTH, HEIGHT), 3, dtype="f2")
    fbo = ctx.framebuffer(color_attachments=[texture])
    try:
        instance = viz if viz is not None else get_visualizer(name)(ctx, WIDTH, HEIGHT)
        features = _features(mode)
        for t in times:
            fbo.use()
            ctx.clear(0.0, 0.0, 0.0)
            instance.render(features, t)
        raw = np.frombuffer(fbo.read(components=3, dtype="f2"),
                            dtype=np.float16).astype(np.float32)
        return raw.reshape((HEIGHT, WIDTH, 3))
    finally:
        fbo.release()
        texture.release()
        _ = ctx.error


@pytest.mark.gpu
@pytest.mark.parametrize("name", NEUE_VISUALIZER)
def test_rendert_in_beiden_modi(gl_context, name):
    for mode in ("music", "speech", "hybrid"):
        img = _render(gl_context, name, mode)
        assert not np.isnan(img).any(), f"{name}/{mode}: NaN im Bild"
        assert not np.isinf(img).any(), f"{name}/{mode}: Inf im Bild"
        assert float(np.ptp(img)) > 0.01, f"{name}/{mode}: Bild ist uniform"


@pytest.mark.gpu
@pytest.mark.parametrize("name", NEUE_VISUALIZER)
def test_sprachmodus_ist_ruhiger(gl_context, name):
    """Prinzip 3: gleiche Optik, aber gedaempfte Reaktion bei Sprache."""
    music = _render(gl_context, name, "music")
    speech = _render(gl_context, name, "speech")
    assert float(speech.mean()) <= float(music.mean()) * 1.02, (
        f"{name}: Sprach-Modus ist nicht ruhiger als der Musik-Modus"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("name", NEUE_VISUALIZER)
def test_laesst_hintergrund_durch(gl_context, name):
    """Prinzip 5: genug dunkle Flaeche fuer ein untergelegtes Bild.

    Gemessen wird der Anteil praktisch schwarzer Pixel — dort blendet der
    Blit-Shader die Visualizer-Ebene aus und das Bild bleibt sichtbar.
    """
    img = _render(gl_context, name, "music")
    luma = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    dark_share = float((luma < 0.02).mean())
    assert dark_share > 0.35, (
        f"{name}: nur {dark_share:.0%} der Flaeche ist dunkel — "
        f"ein Hintergrundbild waere weitgehend uebermalt"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("name", NEUE_VISUALIZER)
def test_zeitsprung_ist_reproduzierbar(gl_context, name):
    """Nach einem Sprung in der Zeitachse zaehlt nur die absolute Zeit.

    Sonst haengt die Vorschau davon ab, welche Frames vorher gerendert
    wurden — dasselbe Bild saehe bei jedem Klick anders aus.
    """
    viz_cls = get_visualizer(name)
    a = _render(gl_context, name, "music", times=(3.0,),
                viz=viz_cls(gl_context, WIDTH, HEIGHT))
    # Zweite Instanz, vorher an ganz anderer Stelle gerendert
    other = viz_cls(gl_context, WIDTH, HEIGHT)
    _render(gl_context, name, "music", times=(0.2, 0.4), viz=other)
    b = _render(gl_context, name, "music", times=(3.0,), viz=other)
    assert np.allclose(a, b, atol=1e-3), (
        f"{name}: Bild bei t=3.0 haengt vom vorherigen Aufrufverlauf ab"
    )
