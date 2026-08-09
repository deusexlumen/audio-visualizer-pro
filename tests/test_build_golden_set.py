"""Tests für die Multi-Audio-Helfer des Golden-Set-Builders."""

from pathlib import Path

from PIL import Image

from tools.build_golden_set import (build_contact_sheet, make_label_entry,
                                    variant_id)


def test_variant_id():
    assert (variant_id("music_severance", "pulsing_core", 0.3, False)
            == "music_severance__pulsing_core_cap03")
    assert (variant_id("podcast_macy", "voice_flow", 1.0, True)
            == "podcast_macy__voice_flow_cap10_mask")


def test_make_label_entry_schema():
    audio = {"id": "a1", "mode": "music"}
    metrics = {"M1": 0.01, "M3": None, "M4": None, "M5": 0.05}
    e = make_label_entry("a1__viz_cap03", audio, "viz", 0.3, False,
                         metrics, good=True)
    assert e["id"] == "a1__viz_cap03"
    assert e["good"] is True
    assert e["human_label"] is None
    assert e["audio"] == "a1"
    assert e["mode"] == "music"
    assert e["metrics"]["M3"] is None
    assert "construction_note" in e


def test_contact_sheet_uses_frame_paths_and_human_tag(tmp_path):
    frame_dir = tmp_path / "frames" / "a1"
    frame_dir.mkdir(parents=True)
    entries = []
    for i in range(4):
        name = f"v{i}.png"
        Image.new("RGB", (854, 480), (i * 60, 0, 0)).save(frame_dir / name)
        entries.append({
            "id": f"a1__v{i}",
            "frame": f"frames/a1/{name}",
            "good": i % 2 == 0,
            "human_label": "good" if i == 0 else None,
        })
    out = tmp_path / "sheet.png"
    build_contact_sheet(tmp_path, entries, out, cols=2)
    assert out.is_file()
    assert Image.open(out).size[0] == 2 * 320
    assert Image.open(out).size[1] == 2 * (int(320 * 480 / 854) + 34)
