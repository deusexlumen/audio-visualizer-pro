"""CLI-Tests fuer main.py (Click)."""

import json
from pathlib import Path
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
