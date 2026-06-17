import subprocess
from pathlib import Path
from threading import Event

import pytest

from src.intro_renderer import get_media_info, render_with_intro, IntroRendererError


def _generate_test_video(path: Path, duration: float, width: int = 1280, height: int = 720):
    """Erzeugt ein kurzes Test-Video mit Ton via FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={width}x{height}:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration}",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


@pytest.fixture
def intro_and_main(tmp_path):
    intro = tmp_path / "intro.mp4"
    main = tmp_path / "main.mp4"
    _generate_test_video(intro, 1.0)
    _generate_test_video(main, 3.0)
    return str(intro), str(main)


def test_get_media_info(intro_and_main):
    intro_path, main_path = intro_and_main
    info = get_media_info(main_path)
    assert info["width"] == 1280
    assert info["height"] == 720
    assert info["fps"] == 30.0
    assert info["duration"] >= 2.9
    assert info["audio_sample_rate"] > 0


def test_render_with_intro(intro_and_main, tmp_path):
    intro_path, main_path = intro_and_main
    output = tmp_path / "result.mp4"

    progress_values = []

    def _progress(p):
        progress_values.append(p)

    result = render_with_intro(
        intro_path=intro_path,
        main_video_path=main_path,
        output_path=str(output),
        fade_duration=0.5,
        progress_callback=_progress,
    )

    assert Path(result).exists()
    info = get_media_info(result)
    # Ergebnis sollte ca. intro_dur + main_dur - fade sein (1 + 3 - 0.5 = 3.5)
    assert info["duration"] >= 3.4
    assert info["width"] == 1280
    assert info["height"] == 720
    assert progress_values
    assert progress_values[-1] == pytest.approx(1.0, abs=0.01)


def test_render_with_intro_missing_file(tmp_path):
    with pytest.raises(IntroRendererError):
        render_with_intro(
            intro_path=str(tmp_path / "missing.mp4"),
            main_video_path=str(tmp_path / "main.mp4"),
            output_path=str(tmp_path / "out.mp4"),
        )


def test_render_with_intro_cancel(intro_and_main, tmp_path):
    intro_path, main_path = intro_and_main
    output = tmp_path / "cancel.mp4"
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(IntroRendererError):
        render_with_intro(
            intro_path=intro_path,
            main_video_path=main_path,
            output_path=str(output),
            cancel_event=cancel_event,
        )
