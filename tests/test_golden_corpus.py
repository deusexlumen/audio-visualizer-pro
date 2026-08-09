"""Tests für das Golden-Set-Korpus-Manifest (golden-corpus/1)."""

import json
from pathlib import Path

import pytest

from tools.golden_corpus import CorpusError, load_corpus, missing_audio_files


def _write_manifest(tmp_path, audios):
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps({"version": "golden-corpus/1", "audios": audios}),
                 encoding="utf-8")
    return p


def _entry(**kw):
    base = {"id": "a1", "path": "a.m4a", "mode": "music",
            "description": "d", "source": "s"}
    base.update(kw)
    return base


def test_load_corpus_resolves_paths(tmp_path):
    (tmp_path / "a.m4a").write_bytes(b"x")
    p = _write_manifest(tmp_path, [_entry()])
    audios = load_corpus(p)
    assert audios[0]["id"] == "a1"
    assert audios[0]["path"] == str((tmp_path / "a.m4a").resolve())


def test_load_corpus_rejects_bad_mode(tmp_path):
    p = _write_manifest(tmp_path, [_entry(mode="jazz")])
    with pytest.raises(CorpusError, match="mode"):
        load_corpus(p)


def test_load_corpus_rejects_duplicate_ids(tmp_path):
    p = _write_manifest(tmp_path, [_entry(), _entry()])
    with pytest.raises(CorpusError, match="oppel"):
        load_corpus(p)


def test_load_corpus_rejects_missing_field(tmp_path):
    p = _write_manifest(tmp_path, [_entry(description="")])
    with pytest.raises(CorpusError, match="description"):
        load_corpus(p)


def test_missing_audio_files(tmp_path):
    (tmp_path / "da.m4a").write_bytes(b"x")
    p = _write_manifest(tmp_path, [_entry(id="da", path="da.m4a"),
                                   _entry(id="fehlt", path="fehlt.m4a")])
    audios = load_corpus(p)
    assert missing_audio_files(audios) == ["fehlt"]


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_repo_manifest_complete():
    """Das eingecheckte Manifest ist valide und alle Audiodateien existieren."""
    audios = load_corpus(REPO_ROOT / "tests" / "golden" / "corpus.json")
    assert len(audios) == 6
    modes = [a["mode"] for a in audios]
    assert modes.count("music") == 3
    assert modes.count("podcast") == 3
    assert missing_audio_files(audios) == []
