"""GPU-Tests fuer den Compositor des Folgenbilds."""

import moderngl
import numpy as np
import pytest
from PIL import Image

from src.episode.compositor import EpisodeCompositor, cover_crop, image_too_small
from src.episode.timeline import CameraPose, EpisodeState

W, H = 256, 144


def _state(stage=0.0, card=0.0, scale=1.0):
    return EpisodeState(camera=CameraPose(scale, 0.0, 0.0), stage_amount=stage,
                        card_amount=card, chapter_index=-1, text_index=-1, text_alpha=0.0)


def test_cover_crop_square_to_16_9():
    assert cover_crop(Image.new("RGB", (1024, 1024)), 1920, 1080).size == (1920, 1080)


def test_cover_crop_keeps_center():
    img = Image.new("RGB", (1024, 1024), (0, 0, 0))
    img.putpixel((512, 512), (255, 0, 0))
    out = np.asarray(cover_crop(img, 1024, 576))
    y, x = np.unravel_index(out[..., 0].argmax(), out.shape[:2])
    assert abs(x - 512) <= 2 and abs(y - 288) <= 2


def test_cover_crop_wide_no_distortion():
    out = cover_crop(Image.new("RGB", (1536, 1024)), 1920, 1080)
    assert out.size == (1920, 1080)


def test_image_too_small():
    assert image_too_small((1536, 1024), 1920, 1080)
    assert image_too_small((1024, 1024), 1920, 1080)
    assert not image_too_small((2560, 1440), 1920, 1080)




@pytest.fixture
def gl():
    ctx = moderngl.create_standalone_context()
    yield ctx
    ctx.release()


@pytest.fixture
def image(tmp_path):
    rng = np.random.default_rng(3)
    # Glattes Zufallsbild: grobe Bloecke, damit Filterung nicht dominiert
    small = rng.integers(0, 256, (18, 32, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((W, H), Image.NEAREST)
    path = tmp_path / "bild.png"
    img.save(path)
    return path, np.asarray(img)


def _run(gl, path, state, overlay_value=0):
    comp = EpisodeCompositor(gl, W, H, str(path))
    overlay = gl.texture((W, H), 4, np.full((H, W, 4), overlay_value, np.uint8).tobytes())
    target = gl.framebuffer(color_attachments=[gl.texture((W, H), 4)])
    comp.render(overlay, state, target)
    out = np.frombuffer(target.read(components=3), np.uint8).reshape(H, W, 3)
    comp.release()
    return out


@pytest.mark.gpu
def test_colour_fidelity_neutral_state(gl, image):
    path, ref = image
    out = _run(gl, path, _state())
    diff = np.abs(out[:72].astype(int) - ref[:72].astype(int)).mean()
    assert diff <= 3.0


@pytest.mark.gpu
def test_black_overlay_changes_nothing_and_white_saturates(gl, image):
    path, ref = image
    assert np.abs(_run(gl, path, _state(), 0)[:72].astype(int) - ref[:72]).mean() <= 3.0
    assert _run(gl, path, _state(), 255)[:72].min() >= 250


@pytest.mark.gpu
def test_stage_darkens(gl, image):
    path, _ = image
    neutral = _run(gl, path, _state())
    stage = _run(gl, path, _state(stage=1.0))
    assert stage[:72].mean() < 0.5 * neutral[:72].mean()


@pytest.mark.gpu
def test_light_line_dimmed_on_stage(gl, tmp_path):
    path = tmp_path / "schwarz.png"
    Image.new("RGB", (W, H)).save(path)
    normal = _run(gl, path, _state(), 128)[:72].mean()
    dimmed = _run(gl, path, _state(stage=1.0), 128)[:72].mean()
    assert dimmed == pytest.approx(normal / 2, abs=3)


@pytest.mark.gpu
def test_card_shifts_image_right(gl, tmp_path):
    img = np.zeros((H, W, 3), np.uint8)
    img[:, : W // 2] = 255  # linke Haelfte weiss
    path = tmp_path / "halb.png"
    Image.fromarray(img).save(path)
    out = _run(gl, path, _state(card=1.0))
    # Weisse Kante wandert um 34 % der Breite nach rechts
    # (Suche ab 40 %: links der verschobenen Kante ist die Luecke schwarz)
    start = int(W * 0.40)
    edge = start + int(np.argmax(out[10, start:, 0] < 128))
    assert abs(edge - int(W * 0.84)) <= 3


@pytest.mark.gpu
def test_card_gap_is_black_not_smeared(gl, tmp_path):
    # Review: die Verschiebung zog die Bildkante als Streifen in die Luecke
    path = tmp_path / "weiss.png"
    Image.new("RGB", (W, H), (255, 255, 255)).save(path)
    out = _run(gl, path, _state(card=1.0))
    assert out[:72, : int(0.30 * W)].max() <= 10
    assert out[:72, int(0.40 * W):].min() >= 245
