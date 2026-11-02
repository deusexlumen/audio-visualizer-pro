"""CLI-Zweig fuer den Folgen-Look und das Preset podcast_folge.json."""

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from config.schemas import load_and_validate_config
from main import cli
from src.episode.config import EpisodeRenderSpec
from tests.test_gpu_renderer_timeline import _write_wav


def _config(tmp_path, **extra):
    data = json.loads(open("config/podcast_folge.json", encoding="utf-8").read())
    data.update(extra)
    path = tmp_path / "folge.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_preset_is_valid_episode_look():
    cfg = load_and_validate_config("config/podcast_folge.json")
    assert cfg.look == "episode" and cfg.visual.type == "voice_line"
    assert tuple(cfg.visual.resolution) == (1920, 1080) and cfg.visual.fps == 30
    assert cfg.visual.colors.primary.upper() == "#F2B866"
    pp = cfg.postprocess
    assert pp.chromatic_aberration == 0 and pp.film_grain == 0 and pp.vignette == 0


def test_cli_episode_requires_image(tmp_path):
    wav = tmp_path / "folge_1.wav"
    _write_wav(wav, seconds=1)
    with patch("main._check_ffmpeg"), patch("main.GPUBatchRenderer") as renderer_cls:
        result = CliRunner().invoke(cli, ["render", str(wav), "-c", str(_config(tmp_path)),
                                          "-o", str(tmp_path / "o.mp4")])
    assert result.exit_code != 0
    assert "Bild" in result.output
    renderer_cls.assert_not_called()


def test_cli_episode_passes_spec_and_writes_chapters(tmp_path):
    wav = tmp_path / "Das_tote_Internet.wav"
    _write_wav(wav, seconds=1)
    img = tmp_path / "bild.png"
    img.write_bytes(b"")
    cfg = _config(tmp_path, chapters=[{"start": 0, "title": "Einstieg"}],
                  quotes=[{"text": "Der Mensch stört", "start_time": 0.2, "end_time": 0.8,
                           "kind": "stage", "accent_word": "stört"}],
                  episode_number="12")
    out = tmp_path / "folge.mp4"
    renderer = MagicMock()
    with patch("main._check_ffmpeg"), patch("main.GPUBatchRenderer", return_value=renderer):
        result = CliRunner().invoke(cli, ["render", str(wav), "-c", str(cfg),
                                          "-bg", str(img), "-o", str(out)])
    assert result.exit_code == 0, result.output
    spec = renderer.render.call_args.kwargs["episode"]
    assert isinstance(spec, EpisodeRenderSpec)
    assert spec.image_path == str(img)
    assert spec.title == "Das tote Internet" and spec.number == "12"
    assert [c.title for c in spec.chapters] == ["Einstieg"]
    assert spec.quotes[0].kind == "stage" and spec.quotes[0].accent_word == "stört"
    chapter_file = tmp_path / "folge.kapitel.txt"
    assert chapter_file.read_text(encoding="utf-8").startswith("0:00 Einstieg")
    # Weniger als 3 Kapitel: YouTube-Warnung wird ausgegeben
    assert "3" in result.output
