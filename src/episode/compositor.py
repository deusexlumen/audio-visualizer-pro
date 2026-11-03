"""GPU-Pass, der das Folgenbild zusammensetzt (Spec §5, §8.1-8.3).

Laeuft NACH Bloom und Tonemapping: das Folgenbild wird nie durch Bloom,
ACES, Aberration oder Grain gezogen und behaelt seine Originalfarben.
Die getonemappte Visualizer-Ebene (Stimm-Lichtstrich) kommt per
Screen-Blend darueber; Schwarz in dieser Ebene aendert das Bild nicht.
"""

import moderngl
from PIL import Image, ImageFilter

from ..gpu_visualizers.base import TEXTURED_VERTEX_SHADER, create_textured_quad
from .timeline import EpisodeState

# Verschiebung des Bildes nach rechts, wenn die Kapitel-Karte voll steht
CARD_SHIFT = 0.34
# Ab dieser Bildhoehe (0 = oben) beginnt der Lesbarkeits-Verlauf nach unten
SHADE_START = 0.55
SHADE_STRENGTH = 0.55
# Helligkeit des Bildes auf der Zitat-Buehne
STAGE_BRIGHTNESS = 0.35
# Helligkeit des Lichtstrichs auf Buehne und Karte
LINE_DIM = 0.5

_FRAGMENT = """
#version 330
uniform sampler2D u_sharp;
uniform sampler2D u_blur;
uniform sampler2D u_overlay;
uniform float u_scale;
uniform vec2 u_offset;
uniform float u_stage;
uniform float u_card;
in vec2 v_uv;
out vec4 f_color;

void main() {
    // v_uv.y = 0 ist (wie beim bestehenden Hintergrund) die Bildoberkante
    vec2 uv = vec2(v_uv.x - %(shift)f * u_card, v_uv.y);
    uv = 0.5 + (uv - 0.5) / u_scale + u_offset;
    vec3 sharp = texture(u_sharp, uv).rgb;
    vec3 soft = texture(u_blur, uv).rgb;
    vec3 img = mix(sharp, soft, u_stage) * mix(1.0, %(stage_b)f, u_stage);
    // Luecke links der verschobenen Bildkante schwarz statt gestreckter Randpixel
    img *= smoothstep(0.0, 0.005, uv.x);
    // Weicher Verlauf unten fuer lesbare Zitate
    img *= 1.0 - %(shade_s)f * smoothstep(%(shade_y)f, 1.0, v_uv.y);
    vec3 line = texture(u_overlay, v_uv).rgb * mix(1.0, %(line_dim)f, max(u_stage, u_card));
    f_color = vec4(1.0 - (1.0 - img) * (1.0 - line), 1.0);
}
""" % {"shift": CARD_SHIFT, "stage_b": STAGE_BRIGHTNESS, "shade_s": SHADE_STRENGTH,
       "shade_y": SHADE_START, "line_dim": LINE_DIM}


def cover_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    """Schneidet mittig auf das Zielformat zu und skaliert hochwertig (LANCZOS)."""
    src_w, src_h = img.size
    target = width / height
    if src_w / src_h > target:
        crop_w = round(src_h * target)
        left = (src_w - crop_w) // 2
        box = (left, 0, left + crop_w, src_h)
    else:
        crop_h = round(src_w / target)
        top = (src_h - crop_h) // 2
        box = (0, top, src_w, top + crop_h)
    return img.crop(box).resize((width, height), Image.LANCZOS)


def image_too_small(image_size, width: int, height: int) -> bool:
    """True, wenn der Zuschnitt auf das Zielformat schmaler als width ist."""
    src_w, src_h = image_size
    crop_w = min(src_w, src_h * width / height)
    return crop_w < width


class EpisodeCompositor:
    """Zeichnet Folgenbild + Lichtstrich fuer einen EpisodeState in ein FBO."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int,
                 image_path: str, stage_blur_px: float = 18.0):
        self.ctx = ctx
        img = cover_crop(Image.open(image_path).convert("RGB"), width, height)
        # Blur einmal vorab; die Buehne blendet nur zwischen beiden Texturen
        radius = stage_blur_px * width / 1920.0
        soft = img.filter(ImageFilter.GaussianBlur(radius=radius))
        self._sharp = self._texture(img)
        self._blur = self._texture(soft)
        self._prog = ctx.program(vertex_shader=TEXTURED_VERTEX_SHADER,
                                 fragment_shader=_FRAGMENT)
        self._vao, self._vbo = create_textured_quad(ctx, self._prog)

    def _texture(self, img: Image.Image):
        tex = self.ctx.texture(img.size, 3, img.tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def render(self, overlay_tex, state: EpisodeState, target_fbo) -> None:
        target_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self._sharp.use(location=0)
        self._blur.use(location=1)
        overlay_tex.use(location=2)
        p = self._prog
        p["u_sharp"].value = 0
        p["u_blur"].value = 1
        p["u_overlay"].value = 2
        p["u_scale"].value = float(state.camera.scale)
        p["u_offset"].value = (float(state.camera.offset_x), float(state.camera.offset_y))
        p["u_stage"].value = float(state.stage_amount)
        p["u_card"].value = float(state.card_amount)
        self._vao.render(mode=moderngl.TRIANGLE_STRIP)

    def release(self) -> None:
        for obj in (self._vao, self._vbo, self._prog, self._sharp, self._blur):
            obj.release()
