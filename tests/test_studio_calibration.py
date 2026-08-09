"""Tests für den Schwellen-Kalibrier-Harness (Spec §3.5)."""

import pytest

from tools.calibrate_thresholds import set_hash, sweep_threshold


def test_sweep_perfectly_separable():
    # Perfekt trennbar: schlechte Renders haben hohe M1, gute niedrige.
    values = [0.30, 0.28, 0.35, 0.10, 0.08, 0.12]
    labels = [False, False, False, True, True, True]  # True = gut
    best = sweep_threshold(values, labels, higher_is_bad=True)
    assert best["sensitivity"] == pytest.approx(1.0)
    assert best["specificity"] == pytest.approx(1.0)
    assert 0.12 < best["threshold"] < 0.28


def test_set_hash_stable(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text('{"renders": []}')
    h1 = set_hash(str(p))
    assert len(h1) == 64
    assert h1 == set_hash(str(p))


def test_effective_label_prefers_human():
    from tools.calibrate_thresholds import effective_label
    assert effective_label({"good": False, "human_label": "good"}) == (True, "human")
    assert effective_label({"good": True, "human_label": "bad"}) == (False, "human")
    assert effective_label({"good": True, "human_label": None}) == (True, "construction")
    assert effective_label({"good": False}) == (False, "construction")
