"""Tests für den Penalty-Solver (Spec §8)."""

import pytest

from src.studio.solver import LEVER_LADDER, compute_j, solve
from src.studio.thresholds import load_thresholds


@pytest.fixture
def ts():
    return load_thresholds()


def _metrics_fn_factory(model):
    """Baut eine metrics_fn aus einem linearen Modell:
    model = {"M1": {"alpha_cap": 0.8, "bloom_intensity": 0.1}, ...}"""
    def metrics_fn(params):
        out = {}
        for key, coeffs in model.items():
            out[key] = sum(c * params.get(p, 0.0) for p, c in coeffs.items())
        return out
    return metrics_fn


def test_compute_j_ober_untergrenzen(ts):
    # M1 = 2x Schwelle => Beitrag 1.0; M5 music unter min => 0.4 * Anteil
    tau1 = ts.m1_overlay_energy_max
    j = compute_j({"M1": 2 * tau1, "M5": 0.01}, ts, mode="music")
    expected = 1.0 + 0.4 * (0.02 - 0.01) / 0.02
    assert j == pytest.approx(expected)


def test_compute_j_konform_ist_null(ts):
    m1_ok = ts.m1_overlay_energy_max * 0.5
    assert compute_j({"M1": m1_ok, "M5": 0.05}, ts, mode="music") == 0.0


def test_compute_j_m2_gewicht_null(ts):
    m1_ok = ts.m1_overlay_energy_max * 0.5
    j1 = compute_j({"M1": m1_ok, "M5": 0.05, "M2": 0.99}, ts, mode="music")
    assert j1 == 0.0  # M2 nur Report


def test_j_faellt_streng_monoton(ts):
    # Lösbares Modell: M1 linear in alpha_cap
    fn = _metrics_fn_factory({"M1": {"alpha_cap": 0.8}, "M5": {"intensity": 0.0}})
    fn2 = lambda p: {**fn(p), "M5": 0.05}
    result = solve(fn2, {"alpha_cap": 1.0}, ts)
    assert result.status == "solved"
    trace = result.j_trace
    assert all(b < a - 0.01 for a, b in zip(trace, trace[1:]))


def test_plateau_bei_unverbesserbarer_metrik(ts):
    # Konstante Metrik: kein Hebel verbessert J
    fn = lambda p: {"M1": 0.9, "M5": 0.05}
    result = solve(fn, {"alpha_cap": 1.0}, ts)
    assert result.status == "plateau"
    assert "M1" in result.infeasible_metrics


def test_zyklusschutz_bei_geklemmten_werten(ts):
    # alpha_cap wird bei 0.0 geklemmt => Kandidaten wiederholen sich
    fn = lambda p: {"M1": 0.5 + p.get("glow", 0.0), "M5": 0.05}
    result = solve(fn, {"alpha_cap": 0.05}, ts)
    assert result.status == "plateau"  # kein Endlos-Loop


def test_m1_m5_konflikt_loest_ueber_chroma_hebel(ts):
    """Spec §15: Chroma-Modulation vor Intensity bei M5 zu niedrig."""
    def fn(p):
        return {
            "M1": 0.8 * p.get("alpha_cap", 0.0),
            "M5": 0.1 * p.get("intensity", 0.0) + 0.5 * p.get("chroma_modulation", 0.0),
        }
    result = solve(fn, {"alpha_cap": 1.0}, ts, mode="music")
    assert result.status == "solved"
    levers = [s["lever"] for s in result.steps]
    # M5-Verletzung wurde über chroma_modulation behoben, nicht intensity
    assert "chroma_modulation" in levers
    if "intensity" in levers:
        assert levers.index("chroma_modulation") < levers.index("intensity")


def test_iterationslimit(ts):
    # Viele kleine Verletzungen: Limit greift
    fn = _metrics_fn_factory({"M1": {"alpha_cap": 0.01}, "M5": {"intensity": 0.0}})
    fn2 = lambda p: {**fn(p), "M5": 0.05}
    result = solve(fn2, {"alpha_cap": 1.0}, ts, max_iterations=3)
    assert result.iterations <= 3
