"""Tests fuer src/ffmpeg_locator.py (Suche + gemockter Download)."""

import io
import os
import zipfile
from unittest.mock import patch

import pytest

from src import ffmpeg_locator as loc


@pytest.fixture(autouse=True)
def _clear_cache():
    """Verhindert, dass ein Test-Ergebnis den naechsten Test kontaminiert."""
    loc._cache.clear()
    yield
    loc._cache.clear()


@pytest.fixture
def local_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    return tmp_path


def test_find_ffmpeg_via_path(monkeypatch, local_appdata):
    monkeypatch.setattr(loc.shutil, "which", lambda name: r"C:\Tools\ffmpeg.exe" if name == "ffmpeg" else None)
    assert loc.find_ffmpeg() == r"C:\Tools\ffmpeg.exe"


def _exe_name(binary: str) -> str:
    """Wie `_find_local_exe` es erwartet: nur unter Windows mit .exe."""
    return f"{binary}.exe" if os.name == "nt" else binary


def test_find_ffmpeg_lokal_wenn_nicht_im_path(monkeypatch, local_appdata):
    monkeypatch.setattr(loc.shutil, "which", lambda name: None)
    install_dir = loc._local_install_dir()
    nested = install_dir / "ffmpeg-essentials" / "bin"
    nested.mkdir(parents=True)
    exe = nested / _exe_name("ffmpeg")
    exe.write_text("dummy")

    found = loc.find_ffmpeg()
    assert found == str(exe)


def test_find_ffmpeg_nichts_gefunden(monkeypatch, local_appdata):
    monkeypatch.setattr(loc.shutil, "which", lambda name: None)
    assert loc.find_ffmpeg() is None


def test_get_ffmpeg_path_wirft_klare_fehlermeldung(monkeypatch, local_appdata):
    monkeypatch.setattr(loc.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="FFmpeg wurde nicht gefunden"):
        loc.get_ffmpeg_path()


def test_get_ffprobe_path_wirft_klare_fehlermeldung(monkeypatch, local_appdata):
    monkeypatch.setattr(loc.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="ffprobe wurde nicht gefunden"):
        loc.get_ffprobe_path()


def _fake_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"ffmpeg-essentials/bin/{_exe_name('ffmpeg')}", "dummy-binary")
        zf.writestr(f"ffmpeg-essentials/bin/{_exe_name('ffprobe')}", "dummy-binary")
    return buf.getvalue()


def test_download_ffmpeg_erfolgreich_ohne_pruefsumme(monkeypatch, local_appdata):
    zip_bytes = _fake_zip_bytes()

    def fake_download(url, dest, progress_callback):
        dest.write_bytes(zip_bytes)
        if progress_callback:
            progress_callback(len(zip_bytes), len(zip_bytes))

    monkeypatch.setattr(loc, "_download_with_progress", fake_download)
    monkeypatch.setattr(loc, "_fetch_expected_sha256", lambda: None)

    progress_calls = []
    path = loc.download_ffmpeg(progress_callback=lambda d, t: progress_calls.append((d, t)))

    assert path.endswith(_exe_name("ffmpeg"))
    assert progress_calls


def test_download_ffmpeg_pruefsumme_mismatch_wirft_fehler(monkeypatch, local_appdata):
    zip_bytes = _fake_zip_bytes()

    def fake_download(url, dest, progress_callback):
        dest.write_bytes(zip_bytes)

    monkeypatch.setattr(loc, "_download_with_progress", fake_download)
    monkeypatch.setattr(loc, "_fetch_expected_sha256", lambda: "0" * 64)

    with pytest.raises(ValueError, match="Pruefsumme"):
        loc.download_ffmpeg()

    # Zip-Datei darf nach Pruefsummen-Fehler nicht liegen bleiben
    assert not (loc._local_install_dir() / "ffmpeg_download.zip").exists()


def test_download_ffmpeg_korruptes_archiv_wirft_fehler(monkeypatch, local_appdata):
    def fake_download(url, dest, progress_callback):
        dest.write_bytes(b"das ist kein zip")

    monkeypatch.setattr(loc, "_download_with_progress", fake_download)
    monkeypatch.setattr(loc, "_fetch_expected_sha256", lambda: None)

    with pytest.raises(Exception):
        loc.download_ffmpeg()
