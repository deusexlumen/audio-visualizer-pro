"""Tests für das Human-Label-Tool des Golden Sets."""

import pytest

from tools.label_golden import label_stats, set_labels


def _data():
    return {"renders": [
        {"id": "music_severance__pulsing_core_cap03", "human_label": None},
        {"id": "music_severance__pulsing_core_cap10", "human_label": None},
        {"id": "podcast_macy__voice_flow_cap03", "human_label": "good"},
    ]}


def test_set_labels_exact_match():
    data = _data()
    changed = set_labels(data, "podcast_macy__voice_flow_cap03", "bad")
    assert changed == ["podcast_macy__voice_flow_cap03"]
    assert data["renders"][2]["human_label"] == "bad"


def test_set_labels_glob():
    data = _data()
    changed = set_labels(data, "music_severance__*", "good")
    assert len(changed) == 2


def test_set_labels_rejects_invalid_label():
    with pytest.raises(ValueError, match="Label"):
        set_labels(_data(), "*", "meh")


def test_label_stats():
    data = _data()
    assert label_stats(data) == (1, 0, 2)
    set_labels(data, "music_severance__pulsing_core_cap03", "bad")
    assert label_stats(data) == (1, 1, 1)
