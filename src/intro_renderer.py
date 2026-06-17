"""
Intro Renderer

Baut ein Intro-Video mit Crossfade vor ein Haupt-Video.
Portiert die Kern-Logik aus IntroTool (FFmpeg xfade + audio-fade).
"""

import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional


class IntroRendererError(Exception):
    """Fehler beim Intro-Rendering."""
    pass


def get_media_info(path: str) -> dict:
    """
    Liest Metadaten eines Videos/Audio-Files mit ffprobe.

    Returns:
        dict mit duration, width, height, fps, audio_sample_rate
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,pix_fmt",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.CalledProcessError as e:
        raise IntroRendererError(f"ffprobe fehlgeschlagen fuer {path}: {e.stderr}") from e

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    stream = streams[0] if streams else {}
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))

    fps_str = stream.get("r_frame_rate", "30/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    except Exception:
        fps = 30.0

    duration = float(fmt.get("duration", 0.0))

    # Audio sample rate separat ermitteln (erster Audiostream)
    audio_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "json",
        path,
    ]
    sample_rate = 48000
    try:
        audio_result = subprocess.run(
            audio_cmd,
            capture_output=True,
            text=True,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        audio_data = json.loads(audio_result.stdout)
        audio_streams = audio_data.get("streams", [])
        if audio_streams:
            sample_rate = int(audio_streams[0].get("sample_rate", 48000))
    except Exception:
        pass

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "audio_sample_rate": sample_rate,
    }


def _build_filter_complex(
    intro_info: dict,
    main_info: dict,
    fade_duration: float,
) -> str:
    """Baut den FFmpeg filter_complex fuer Intro + Crossfade."""
    w = main_info["width"]
    h = main_info["height"]
    fps = main_info["fps"]
    sr = main_info["audio_sample_rate"]

    intro_dur = intro_info["duration"]
    if fade_duration >= intro_dur:
        fade_duration = max(0.1, intro_dur - 0.1)

    offset = intro_dur - fade_duration
    delay_ms = int(offset * 1000)

    # Sicherstellen, dass offset nicht 0 wird (verhindert leere xfade)
    if offset <= 0:
        offset = 0.01
        delay_ms = 10

    return (
        # Intro-Video auf Hauptvideo-Format anpassen
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"format=yuv420p,fps={fps}[v0];"
        # Haupt-Video auf gleiches Format anpassen
        f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"format=yuv420p,fps={fps}[v1];"
        # Video-Crossfade
        f"[v0][v1]xfade=transition=fade:duration={fade_duration}:offset={offset}[outv];"
        # Intro-Audio fade-out
        f"[0:a]aformat=sample_fmts=fltp:sample_rates={sr}:channel_layouts=stereo,"
        f"afade=t=out:st={offset}:d={fade_duration}[a0];"
        # Haupt-Audio delayed + fade-in
        f"[1:a]aformat=sample_fmts=fltp:sample_rates={sr}:channel_layouts=stereo,"
        f"afade=t=in:st=0:d={fade_duration},adelay={delay_ms}|{delay_ms}[a1];"
        # Audio mixen
        f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0[outa]"
    )


def render_with_intro(
    intro_path: str,
    main_video_path: str,
    output_path: str,
    fade_duration: float = 1.0,
    progress_callback: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """
    Rendert ein Intro vor ein Haupt-Video mit Crossfade.

    Args:
        intro_path: Pfad zum Intro-Video.
        main_video_path: Pfad zum Haupt-Video.
        output_path: Zielpfad fuer das Ergebnis.
        fade_duration: Dauer des Crossfades in Sekunden.
        progress_callback: Optionaler Callback(progress_0_to_1).
        cancel_event: Optionales Event zum Abbruch.

    Returns:
        output_path
    """
    intro_path = Path(intro_path)
    main_video_path = Path(main_video_path)
    output_path = Path(output_path)

    if not intro_path.exists():
        raise IntroRendererError(f"Intro-Datei nicht gefunden: {intro_path}")
    if not main_video_path.exists():
        raise IntroRendererError(f"Haupt-Video nicht gefunden: {main_video_path}")

    intro_info = get_media_info(str(intro_path))
    main_info = get_media_info(str(main_video_path))

    if intro_info["duration"] <= 0:
        raise IntroRendererError("Intro-Video hat keine gueltige Dauer")
    if main_info["duration"] <= 0:
        raise IntroRendererError("Haupt-Video hat keine gueltige Dauer")

    filter_complex = _build_filter_complex(intro_info, main_info, fade_duration)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(intro_path),
        "-i", str(main_video_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ]

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )

    total_duration = intro_info["duration"] + main_info["duration"]
    time_regex = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

    try:
        for line in process.stdout:
            if cancel_event is not None and cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise IntroRendererError("Intro-Rendering abgebrochen")

            match = time_regex.search(line)
            if match and progress_callback is not None:
                hours, minutes, seconds = match.groups()
                current = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                progress_callback(min(1.0, current / total_duration))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    if process.returncode != 0:
        raise IntroRendererError(f"FFmpeg Intro-Rendering fehlgeschlagen (Code {process.returncode})")

    if progress_callback is not None:
        progress_callback(1.0)

    return str(output_path)
