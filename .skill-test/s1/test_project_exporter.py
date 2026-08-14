"""Tests fuer den Projekt-Export-Kern (project_exporter.py)."""

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from project_exporter import (
    MissingAssetsError, build_export_plan, export_project, find_asset_references,
    sha256_file, verify_export,
)


@pytest.fixture
def project(tmp_path):
    """Mini-Projekt: JSON + Audio + Hintergrund + Config."""
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake-audio-data")
    bg = tmp_path / "bg.png"
    bg.write_bytes(b"fake-image-data")
    cfg = tmp_path / "preset.json"
    cfg.write_text("{}", encoding="utf-8")

    project_path = tmp_path / "demo.avproj"
    project_path.write_text(json.dumps({
        "version": 1,
        "audio_path": str(audio),
        "background_path": str(bg),
        "extra_config_path": str(cfg),
        "visualizer_type": "spectrum_genesis",
    }), encoding="utf-8")
    return project_path


def test_find_asset_references(project):
    data = json.loads(project.read_text(encoding="utf-8"))
    refs = find_asset_references(data)
    assert set(refs) == {"audio_path", "background_path", "extra_config_path"}
    # Nicht-Pfad-Felder werden ignoriert
    assert "visualizer_type" not in refs


def test_relative_paths_resolve_against_project(tmp_path):
    (tmp_path / "song.mp3").write_bytes(b"x")
    p = tmp_path / "rel.avproj"
    p.write_text(json.dumps({"audio_path": "song.mp3"}), encoding="utf-8")
    plan = build_export_plan(p)
    audio = next(e for e in plan.entries if e.key == "audio_path")
    assert audio.source == (tmp_path / "song.mp3").resolve()
    assert audio.exists


def test_missing_asset_detected(tmp_path):
    p = tmp_path / "broken.avproj"
    p.write_text(json.dumps({"audio_path": str(tmp_path / "weg.mp3")}), encoding="utf-8")
    plan = build_export_plan(p)
    assert len(plan.missing) == 1
    assert plan.missing[0].key == "audio_path"


def test_export_raises_on_missing(tmp_path):
    p = tmp_path / "broken.avproj"
    p.write_text(json.dumps({"audio_path": str(tmp_path / "weg.mp3")}), encoding="utf-8")
    with pytest.raises(MissingAssetsError) as exc:
        export_project(p, tmp_path / "out.zip")
    assert len(exc.value.missing) == 1


def test_export_allow_missing_skips_file(tmp_path):
    p = tmp_path / "broken.avproj"
    p.write_text(json.dumps({"audio_path": str(tmp_path / "weg.mp3")}), encoding="utf-8")
    manifest = export_project(p, tmp_path / "out.zip", allow_missing=True)
    assert manifest["file_count"] == 1  # nur project.json
    assert len(manifest["missing"]) == 1


def test_export_zip_layout_and_manifest(project, tmp_path):
    out = tmp_path / "export.zip"
    manifest = export_project(project, out)

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "demo/project.json" in names
    assert "demo/audio/song.mp3" in names
    assert "demo/backgrounds/bg.png" in names
    assert "demo/configs/preset.json" in names
    assert "demo/manifest.json" in names

    assert manifest["file_count"] == 4
    by_path = {f["path"]: f for f in manifest["files"]}
    assert by_path["demo/audio/song.mp3"]["sha256"] == sha256_file(
        tmp_path / "song.mp3"
    )

    # Verifikation gegen Manifest: keine Probleme
    assert verify_export(out) == []


def test_export_without_manifest(project, tmp_path):
    out = tmp_path / "export.zip"
    export_project(project, out, include_manifest=False)
    with zipfile.ZipFile(out) as zf:
        assert not any(n.endswith("manifest.json") for n in zf.namelist())
    with pytest.raises(ValueError):
        verify_export(out)


def test_verify_detects_corruption(project, tmp_path):
    out = tmp_path / "export.zip"
    export_project(project, out)
    # ZIP neu schreiben mit manipuliertem Inhalt
    with zipfile.ZipFile(out) as zf:
        contents = {n: zf.read(n) for n in zf.namelist()}
    contents["demo/audio/song.mp3"] = b"manipulated"
    with zipfile.ZipFile(out, "w") as zf:
        for n, data in contents.items():
            zf.writestr(n, data)
    problems = verify_export(out)
    assert any("song.mp3" in p for p in problems)


def test_filename_collision_deduped(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "bg.png").write_bytes(b"a")
    (dir_b / "bg.png").write_bytes(b"b")
    p = tmp_path / "coll.avproj"
    p.write_text(json.dumps({
        "background_path": str(dir_a / "bg.png"),
        "intro_path": str(dir_b / "bg.png"),
    }), encoding="utf-8")
    plan = build_export_plan(p)
    arcnames = [e.arcname for e in plan.entries]
    assert len(arcnames) == len(set(arcnames))
