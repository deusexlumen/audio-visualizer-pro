#!/usr/bin/env python3
"""Prueft MP4-Export mit FFprobe und optional kompletter FFmpeg-Dekodierung."""
from __future__ import annotations
import argparse
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

def probe(path: Path) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe fehlt")
    cmd = [exe, "-v", "error", "-show_entries",
           "format=duration,format_name:stream=index,codec_type,codec_name,width,height,avg_frame_rate,duration",
           "-of", "json", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ffprobe Fehler")
    return json.loads(result.stdout)

def duration(meta: dict) -> float | None:
    values = [meta.get("format", {}).get("duration")]
    values.extend(x.get("duration") for x in meta.get("streams", []))
    for value in values:
        try:
            v = float(value)
            if math.isfinite(v) and v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return None

def fps_value(stream: dict) -> float | None:
    try:
        value = float(Fraction(stream.get("avg_frame_rate", "0/1")))
        return value if value > 0 and math.isfinite(value) else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None

def verify(video: Path, original: Path | None, width: int | None,
           height: int | None, fps: float | None, tolerance: float,
           full_decode: bool) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict = {}
    result = {"ok": False, "video": str(video.resolve()),
              "source_audio": str(original.resolve()) if original else None,
              "errors": errors, "warnings": warnings, "measurements": metrics}
    if not video.is_file() or video.stat().st_size == 0:
        errors.append("Video fehlt oder ist leer")
        return result
    if video.suffix.lower() != ".mp4":
        errors.append("Ausgabe ist nicht .mp4")
    try:
        meta = probe(video)
    except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as ex:
        errors.append(f"Videoanalyse fehlgeschlagen: {ex}")
        return result
    videos = [x for x in meta.get("streams", []) if x.get("codec_type") == "video"]
    audios = [x for x in meta.get("streams", []) if x.get("codec_type") == "audio"]
    if not videos:
        errors.append("Kein Videostream")
    if not audios:
        errors.append("Kein Audiostream")
    v = videos[0] if videos else {}
    runtime = duration(meta)
    actual_fps = fps_value(v)
    metrics.update({"size_bytes": video.stat().st_size,
                    "duration_s": runtime, "width": v.get("width"),
                    "height": v.get("height"), "fps": actual_fps,
                    "video_codec": v.get("codec_name"),
                    "audio_codec": audios[0].get("codec_name") if audios else None})
    if runtime is None:
        errors.append("Dauer unlesbar")
    if width is not None and v.get("width") != width:
        errors.append(f"Breite abweichend: {v.get('width')} != {width}")
    if height is not None and v.get("height") != height:
        errors.append(f"Hoehe abweichend: {v.get('height')} != {height}")
    if fps is not None and (actual_fps is None or abs(actual_fps - fps) > 0.15):
        errors.append(f"FPS abweichend: {actual_fps} != {fps}")
    if original:
        if not original.is_file():
            errors.append("Originalaudio fehlt")
        else:
            try:
                src_duration = duration(probe(original))
                metrics["source_duration_s"] = src_duration
                if src_duration is None or runtime is None:
                    errors.append("Dauervergleich nicht moeglich")
                else:
                    delta = abs(src_duration - runtime)
                    metrics["duration_difference_s"] = round(delta, 3)
                    if delta > tolerance:
                        errors.append(f"Dauerabweichung {delta:.2f}s > {tolerance:.2f}s")
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as ex:
                errors.append(f"Originalaudio nicht pruefbar: {ex}")
    if full_decode and videos and audios:
        exe = shutil.which("ffmpeg")
        if not exe:
            errors.append("FFmpeg fehlt fuer Voll-Dekodierung")
        else:
            try:
                completed = subprocess.run([exe, "-hide_banner", "-nostdin", "-v", "error",
                    "-xerror", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0",
                    "-f", "null", "-"], capture_output=True, text=True, timeout=1200)
                metrics["full_decode_passed"] = completed.returncode == 0
                if completed.returncode:
                    errors.append("Dekodierung fehlgeschlagen: " + completed.stderr[-2000:])
            except (OSError, subprocess.TimeoutExpired) as ex:
                errors.append(f"Dekodierung fehlgeschlagen: {ex}")
    elif not full_decode:
        warnings.append("Voll-Dekodierung nicht angefordert")
    warnings.append("Visuelle und akustische Sichtpruefung zusaetzlich erforderlich")
    result["ok"] = not errors
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--duration-tolerance", type=float, default=1.5)
    parser.add_argument("--full-decode", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.duration_tolerance < 0:
        parser.error("Toleranz muss >= 0 sein")
    report = verify(args.video, args.source_audio, args.width, args.height,
                    args.fps, args.duration_tolerance, args.full_decode)
    body = json.dumps(report, indent=2, ensure_ascii=False)
    print(body)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(body + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
