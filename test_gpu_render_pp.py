"""GPU-Render-Test mit Post-Processing."""
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

# Test mit extremem Post-Processing das braun machen koennte
pp = {"contrast": 0.5, "saturation": 2.0, "brightness": 0.5, "warmth": 1.0, "film_grain": 0.0}

renderer = GPUBatchRenderer(width=320, height=180, fps=30)
renderer.render(
    audio_path=test_wav,
    visualizer_type="spectrum_bars",
    output_path=output_mp4,
    preview_mode=True,
    preview_duration=1.0,
    postprocess=pp,
)

frame_png = tempfile.mktemp(suffix='.png')
subprocess.run([
    'ffmpeg', '-y', '-i', output_mp4, '-vf', 'select=eq(n\,0)', '-vframes', '1', frame_png
], capture_output=True)

from PIL import Image
if os.path.exists(frame_png):
    img = Image.open(frame_png).convert('RGB')
    arr = np.array(img)
    print(f"With PP: shape={arr.shape}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, std={arr.std():.1f}")
    os.unlink(frame_png)

os.unlink(test_wav)
os.unlink(output_mp4)
