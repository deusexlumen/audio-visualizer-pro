"""Feasibility-Precheck (Spec §7).

Läuft VOR dem ersten Probe-Render, rein analytisch auf Masken-Statistik
(Budget ≤ 200 ms, kein GL). Unlösbare Fälle brechen hier ab — v1 hätte
bis zu 4 Full-Renders gerendert, die nie bestehen konnten.
"""

from dataclasses import dataclass, field

import numpy as np

# Peripher-geometrische Visualizer (Rahmen/Rand/Ecken) für den
# Layout-Fallback — Keys aus VISUALIZER_MAP (Spec §7).
PERIPHERAL_VISUALS: tuple[str, ...] = (
    "spectrum_bars",
    "neon_wave_circle",
    "neon_oscilloscope",
    "typographic",
)


@dataclass
class FeasibilityConfig:
    """Grenzen des Prechecks (Defaults aus Spec §7)."""

    subject_area_limit: float = 0.75
    text_zone_min_area: float = 0.04
    grid: int = 32  # Raster für die Freiflächen-Geometrie


@dataclass
class FeasibilityResult:
    """Befund des Prechecks."""

    status: str  # "ok" | "layout_fallback" | "infeasible"
    should_render: bool
    visualizer_whitelist: list[str] | None = None
    actions: list[str] = field(default_factory=list)
    reason: str = ""


def _max_rect_in_histogram(heights: np.ndarray) -> int:
    """Größtes Rechteck im Histogramm (Standard-Stack-Algorithmus)."""
    stack: list[int] = []
    best = 0
    n = len(heights)
    for i in range(n + 1):
        cur = heights[i] if i < n else 0
        while stack and heights[stack[-1]] > cur:
            h = heights[stack.pop()]
            left = stack[-1] + 1 if stack else 0
            best = max(best, int(h) * (i - left))
        stack.append(i)
    return best


def _largest_free_rect_area(mask: np.ndarray, grid: int) -> float:
    """Relativer Flächenanteil des größten freien Rechtecks.

    Raster-Approximation (grid x grid): Zellen mit Masken-Mittel > 0.5
    gelten als belegt; größtes leeres Rechteck via Histogramm-Methode.
    """
    h, w = mask.shape
    cell_h, cell_w = max(1, h // grid), max(1, w // grid)
    free = np.ones((grid, grid), dtype=bool)
    for gy in range(grid):
        for gx in range(grid):
            cell = mask[gy * cell_h:(gy + 1) * cell_h,
                        gx * cell_w:(gx + 1) * cell_w]
            if cell.size and float(cell.mean()) > 0.5:
                free[gy, gx] = False
    # Größtes freies Rechteck über zeilenweise Histogramme
    heights = np.zeros(grid, dtype=int)
    best = 0
    for row in range(grid):
        heights = np.where(free[row], heights + 1, 0)
        best = max(best, _max_rect_in_histogram(heights))
    return best / float(grid * grid)


def check_feasibility(
    mask: np.ndarray | None,
    requires_text_zone: bool = False,
    m3_active: bool = True,
    config: FeasibilityConfig | None = None,
) -> FeasibilityResult:
    """Analytischer Precheck vor jedem Render (Spec §7).

    - Subjektfläche > Limit: Layout-Fallback (periphere Whitelist)
    - 100 % Subjekt + Textpflicht: Zielkonflikt -> infeasible (0 Renders)
    - Keine Textzone >= Mindestfläche: Scrim erzwingen, text_zone_alpha 0.05
    """
    cfg = config or FeasibilityConfig()
    if mask is None:
        return FeasibilityResult("ok", should_render=True)

    mask = np.asarray(mask, dtype=np.float32)
    subject_area = float((mask > 0.5).mean())
    actions: list[str] = []

    # Textzonen-Prüfung (geometrisch, auf der Subjekt-Maske)
    if requires_text_zone:
        free_rect = _largest_free_rect_area(mask, cfg.grid)
        if free_rect < cfg.text_zone_min_area:
            actions.append(
                "scrim erzwingen: keine Textzone >= "
                f"{cfg.text_zone_min_area:.0%} (größte freie Zone "
                f"{free_rect:.1%}); text_zone_alpha=0.05"
            )

    if subject_area > cfg.subject_area_limit and m3_active:
        if subject_area >= 0.999 and requires_text_zone:
            # Zielkonflikt: Subjekt überall, aber Textpflicht —
            # geometrisch unvereinbar, Abbruch VOR jedem Render (Spec §7)
            return FeasibilityResult(
                "infeasible",
                should_render=False,
                actions=actions,
                reason=(
                    f"Subjektfläche {subject_area:.0%} bei aktivem M3 und "
                    "Textpflicht: keine freie Geometrie für Visualizer "
                    "und Textzone."
                ),
            )
        return FeasibilityResult(
            "layout_fallback",
            should_render=True,
            visualizer_whitelist=list(PERIPHERAL_VISUALS),
            actions=actions + [
                f"Subjektfläche {subject_area:.0%} > {cfg.subject_area_limit:.0%}: "
                "Visualizer auf periphere Whitelist eingeschränkt"
            ],
        )

    return FeasibilityResult("ok", should_render=True, actions=actions)
