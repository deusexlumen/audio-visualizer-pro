"""Vollstaendiger GPU-Render-Test mit FFmpeg-Output-Analyse."""
import numpy as np
import tempfile
import os
import subprocess
import wave

sr = 44100
duration = 1.0
t = np.linspace(0, duration, int(sr * duration), False)
tone = 0.5 * np.sin(2 * np.pi * 440 * t)
audio = np.clip(tone, -1.0, 1.0)
audio_int16 = (audio * 32767).astype(np.int16)

test_wav = tempfile.mktemp(suffix='.wav')
with wave.open(test_wav, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(audio_int16.tobytes())

output_mp4 = tempfile.mktemp(suffix='.mp4')

from src.gpu_renderer import GPUBatchRenderer

renderer = GPUBatchRenderer(width=320, height=180, fps=30)
renderer.render(
    audio_path=test_wav,
    visualizer_type="spectrum_bars",
    output_path=output_mp4,
    preview_mode=True,
    preview_duration=1.0,
)

# Extrahiere ersten Frame mit FFmpeg
frame_png = tempfile.mktemp(suffix='.png')
subprocess.run([
    'ffmpeg', '-y', '-i', output_mp4, '-vf', 'select=eq(n\,0)', '-vframes', '1', frame_png
], capture_output=True)

from PIL import Image
if os.path.exists(frame_png):
    img = Image.open(frame_png).convert('RGB')
    arr = np.array(img)
    print(f"Video frame shape: {arr.shape}")
    print(f"Video frame min/max: {arr.min()} / {arr.max()}")
    print(f"Video frame mean: {arr.mean():.1f}")
    print(f"Video frame std: {arr.std():.1f}")
    # Check if uniform
    if arr.std() < 5:
        print("WARNUNG: Video-Frame ist fast einfarbig!")
    else:
        print("OK: Video-Frame hat Variation.")
    os.unlink(frame_png)
else:
    print("Konnte Frame nicht extrahieren")

os.unlink(test_wav)
os.unlink(output_mp4)
