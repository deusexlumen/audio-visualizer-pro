"""ProbeRenderer — Differenz-Render für Probe/Preview/Verify (Spec §3.2).

Rendert A(t) (vollständig) und B(t) (Visualizer-Beitrag 0) mit identischem
u_time; Rauschaufhebung ist dadurch gegeben (Seeding via fract(u_time),
gpu_renderer.py:1132-1142). B ist bei statischem Hintergrund zeitinvariant
und kann vom Aufrufer gecacht werden (Spec §3.2.2). Spiegelt den Batch-
Loop (gpu_renderer.py:462-534) exakt: Clear -> Hintergrund -> Viz-Blit ->
Bloom -> Post-Process.
"""

import numpy as np

from ..gpu_renderer import GPUPreviewRenderer
from .metrics import contribution, to_measure_raster
from .types import MeasureConstraints

MIN_PROBE = (854, 480)


def probe_resolution(target_w: int, target_h: int) -> tuple[int, int]:
    """probe_res = max(480p, Ziel/4), Seitenverhältnis identisch (Spec §3.4)."""
    scale = max(0.25, MIN_PROBE[0] / target_w, MIN_PROBE[1] / target_h)
    return max(1, round(target_w * scale)), max(1, round(target_h * scale))


class ProbeRenderer:
    """Einzel-Frame-Renderer für Messzwecke (kein Encode, kein FFmpeg)."""

    def __init__(self, width: int, height: int, fps: int = 30):
        self._r = GPUPreviewRenderer(width=width, height=height, fps=fps)

    @property
    def ctx(self):
        return self._r.ctx

    def release(self):
        self._r.release()

    def render_frame(
        self, viz, features_dict, time_s, bg_texture,
        postprocess: dict, constraints: MeasureConstraints,
    ) -> np.ndarray:
        """Rendert ein Frame; bei alpha_cap=0 wird der Visualizer-Pass
        übersprungen (Blit-Alpha 0 — reine Ersparnis, Spec §3.2.2)."""
        r = self._r
        r.fbo.use()
        r.ctx.clear(0.0, 0.0, 0.0)
        if bg_texture is not None:
            r._render_background(bg_texture, 1.0, 0.0)
        if constraints.alpha_cap > 0.0:
            r._render_viz_into(viz, r.viz_fbo, features_dict, time_s)
            r.fbo.use()
            r._blit_viz_to_fbo(
                r.viz_fbo.color_attachments[0],
                alpha_cap=constraints.alpha_cap,
                alpha_from_luma=constraints.alpha_from_luma,
                luma_knee_lo=constraints.luma_knee_lo,
                luma_knee_hi=constraints.luma_knee_hi,
                subject_strength=constraints.subject_strength,
            )
        pp = dict(postprocess or {})
        if constraints.grain_free:
            pp["film_grain"] = 0.0  # C15 Regel 3: M5 nur grain-frei
        bloom_intensity = pp.get("bloom_intensity", 0.6)
        if r._bloom is not None and bloom_intensity > 0.0:
            r._apply_bloom(
                intensity=bloom_intensity,
                threshold=pp.get("bloom_threshold", 1.0),
                radius=pp.get("bloom_radius", 1.0),
            )
        r._apply_postprocess(
            r.fbo.color_attachments[0],
            contrast=pp.get("contrast", 1.0),
            saturation=pp.get("saturation", 1.0),
            brightness=pp.get("brightness", 0.0),
            warmth=pp.get("warmth", 0.0),
            film_grain=pp.get("film_grain", 0.0),
            time=time_s,
            exposure=pp.get("exposure", 1.0),
            vignette=pp.get("vignette", 0.0),
            chromatic_aberration=pp.get("chromatic_aberration", 0.0),
            lut_path=pp.get("lut"),
            lut_strength=pp.get("lut_strength", 1.0),
        )
        raw = r.post_fbo.read(components=3)
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(r.height, r.width, 3)
            .copy()
        )

    def render_pair(self, viz, features_dict, time_s, bg_texture,
                    postprocess, constraints) -> tuple[np.ndarray, np.ndarray]:
        """(A, B): B mit alpha_cap=0, identisches u_time für beide."""
        a = self.render_frame(viz, features_dict, time_s, bg_texture,
                              postprocess, constraints)
        b_constraints = MeasureConstraints(
            alpha_cap=0.0,
            alpha_from_luma=constraints.alpha_from_luma,
            grain_free=constraints.grain_free,
        )
        b = self.render_frame(viz, features_dict, time_s, bg_texture,
                              postprocess, b_constraints)
        return a, b

    def contribution_map(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """contrib-Map auf dem normalisierten Messraster."""
        return contribution(to_measure_raster(a), to_measure_raster(b))
