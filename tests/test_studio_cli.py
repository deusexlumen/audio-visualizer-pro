"""Tests für die Studio-CLI-Flags (Spec §11.3)."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from main import cli


def _fake_sidecar(out):
    return {
        "schema_version": "studio-decision/2.1",
        "mode": {"value": "MUSIC"},
        "verify": {"status": "pass"},
    }


def test_studio_flag_ruft_engine(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    runner = CliRunner()
    with patch("main._run_studio_pipeline") as mock_pipe:
        mock_pipe.return_value = _fake_sidecar(str(tmp_path / "out.mp4"))
        result = runner.invoke(cli, ["render", str(audio), "--studio",
                                     "-o", str(tmp_path / "out.mp4")])
    assert result.exit_code == 0, result.output
    mock_pipe.assert_called_once()
    assert mock_pipe.call_args.kwargs.get("dry_run") is False


def test_studio_dry_ohne_commit(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    runner = CliRunner()
    with patch("main._run_studio_pipeline") as mock_pipe:
        mock_pipe.return_value = _fake_sidecar(str(tmp_path / "out.mp4"))
        result = runner.invoke(cli, ["render", str(audio), "--studio",
                                     "--studio-dry",
                                     "-o", str(tmp_path / "out.mp4")])
    assert result.exit_code == 0, result.output
    assert mock_pipe.call_args.kwargs.get("dry_run") is True


def test_studio_strict_wird_durchgereicht(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    runner = CliRunner()
    with patch("main._run_studio_pipeline") as mock_pipe:
        mock_pipe.return_value = _fake_sidecar(str(tmp_path / "out.mp4"))
        result = runner.invoke(cli, ["render", str(audio), "--studio",
                                     "--studio-strict",
                                     "-o", str(tmp_path / "out.mp4")])
    assert result.exit_code == 0, result.output
    assert mock_pipe.call_args.kwargs.get("strict") is True


def test_ohne_studio_flag_bleibt_klassisch(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    runner = CliRunner()
    with patch("main._run_studio_pipeline") as mock_pipe, \
         patch("main.GPUBatchRenderer") as mock_renderer:
        mock_renderer.return_value.render = MagicMock()
        result = runner.invoke(cli, ["render", str(audio),
                                     "-o", str(tmp_path / "out.mp4")])
    mock_pipe.assert_not_called()
