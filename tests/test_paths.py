"""Tests fuer src/paths.py (Ressourcen- vs. Nutzerdaten-Pfade)."""

import sys

from src import paths


def test_is_frozen_normal_lauf():
    assert paths.is_frozen() is False


def test_is_frozen_erkennt_pyinstaller(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert paths.is_frozen() is True


def test_resource_path_normal_zeigt_auf_repo_root():
    p = paths.resource_path("config", "default.json")
    assert p.name == "default.json"
    assert p.exists()


def test_resource_path_frozen_nutzt_meipass(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    p = paths.resource_path("config", "default.json")
    assert p == tmp_path / "config" / "default.json"


def test_user_data_dir_nutzt_localappdata(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    p = paths.user_data_dir("cache", "quotes")
    assert p == tmp_path / paths.APP_NAME / "cache" / "quotes"


def test_user_data_dir_ohne_localappdata_fallback_home(monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    p = paths.user_data_dir("logs")
    assert str(p).startswith(str(paths.Path.home()))


def test_user_config_dir_nutzt_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    p = paths.user_config_dir("recipes")
    assert p == tmp_path / paths.APP_NAME / "recipes"


def test_user_data_dir_ohne_parts_gibt_wurzel_zurueck(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    p = paths.user_data_dir()
    assert p == tmp_path / paths.APP_NAME
