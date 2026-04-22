"""Schneller GPU-Frame-Test um den braunen Screen zu diagnostizieren."""
import numpy as np
import tempfile
import os

# Test-Audio erzeugen
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

from src.analyzer import AudioAnalyzer
analyzer = AudioAnalyzer()
features = analyzer.analyze(test_wav, fps=30)
print(f"Features: duration={features.duration}, frame_count={features.frame_count}")
print(f"rms range: {features.rms.min():.3f} - {features.rms.max():.3f}")
print(f"beat_frames type: {type(features.beat_frames)}, value: {features.beat_frames}")
print(f"chroma shape: {features.chroma.shape}")

# Teste mehrere Visualizer
viz_names = ["lumina_core", "spectrum_bars", "pulsing_core", "chroma_field", "voice_flow"]

for viz_name in viz_names:
    try:
        import moderngl
        from src.gpu_visualizers import get_visualizer
        
        ctx = moderngl.create_standalone_context()
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((640, 360), 3)])
        
        viz_cls = get_visualizer(viz_name)
        viz = viz_cls(ctx, 640, 360)
        
        features_dict = {
            "rms": features.rms[:features.frame_count],
            "onset": features.onset[:features.frame_count],
            "beat_intensity": np.zeros(features.frame_count, dtype=np.float32),
            "chroma": features.chroma,
            "spectral_centroid": features.spectral_centroid[:features.frame_count],
            "fps": 30,
            "frame_count": features.frame_count,
        }
        
        fbo.use()
        ctx.clear(0.05, 0.05, 0.05)
        viz.render(features_dict, 0.5)
        
        pixels = fbo.read(components=3)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape((360, 640, 3))
        
        variation = img.max() - img.min()
        is_uniform = variation < 10
        print(f"[{viz_name}] variation={variation}, mean={img.mean():.1f}, uniform={is_uniform}")
        
        ctx.release()
        fbo.release()
        
    except Exception as e:
        print(f"[{viz_name}] FEHLER: {e}")

os.unlink(test_wav)
