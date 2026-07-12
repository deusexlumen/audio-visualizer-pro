"""CLI-Tests fuer main.py (Click)."""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from main import cli


def test_render_help_includes_background_color():
    runner = CliRunner()
    result = runner.invoke(cli, ["render", "--help"])
    assert result.exit_code == 0
    assert "--background-color" in result.output


def test_create_config_includes_background_color(tmp_path):
    runner = CliRunner()
    output = tmp_path / "template.json"
    result = runner.invoke(cli, ["create-config", "--output", str(output)])
    assert result.exit_code == 0
    data = json.loads(output.read_text())
    assert data["background_color"] == "#0A0A0A"
    assert "colors" in data["visual"]
    assert data["visual"]["colors"]["background"] == "#0A0A0A"


def test_list_visuals_shows_signature_marker():
    runner = CliRunner()
    result = runner.invoke(cli, ["list-visuals"])
    assert result.exit_code == 0
    assert "lumina_core" in result.output
    assert "Signature" in result.output


def test_batch_ueberspringt_ungueltige_jobs(tmp_path):
    """Jobs ohne 'audio'-Key oder mit fehlender Datei werden uebersprungen,
    der Batch-Lauf bricht nicht ab."""
    jobs = [
        {"visual": "lumina_core", "output": "out1.mp4"},  # kein 'audio'
        {"audio": str(tmp_path / "gibt_es_nicht.mp3"), "output": "out2.mp4"},
    ]
    batch_file = tmp_path / "jobs.json"
    batch_file.write_text(json.dumps(jobs))

    runner = CliRunner()
    with patch("main._check_ffmpeg"):
        result = runner.invoke(cli, ["batch", str(batch_file)])

    assert result.exit_code == 0
    assert "2 Jobs gefunden" in result.output
    assert "hat keinen 'audio'-Key" in result.output
    assert "Audio-Datei nicht gefunden" in result.output


def test_batch_fehlende_datei_bricht_ab(tmp_path):
    """Nicht existierende Batch-Datei liefert Click-Fehler."""
    runner = CliRunner()
    result = runner.invoke(cli, ["batch", str(tmp_path / "fehlt.json")])
    assert result.exit_code != 0
