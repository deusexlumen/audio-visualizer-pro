"""
GPU Visualizer Performance Benchmark.

Misst die Render-Performance aller GPU-Visualizer bei 1920x1080.
Rendert 900 Frames (30 Sekunden @ 30fps) pro Visualizer und zeigt
Statistiken: Gesamtzeit, FPS, ms/Frame.
"""

import time
import json
import numpy as np
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from src.gpu_renderer import GPUPreviewRenderer
from src.gpu_visualizers import list_visualizers, get_visualizer


def generate_dummy_features(frame_count: int = 900, fps: int = 30):
    """Erzeugt realistische Dummy-Features fuer den Benchmark."""
    np.random.seed(42)
    rms = np.clip(np.random.rand(frame_count) * 0.7 + np.sin(np.linspace(0, frame_count * 0.1, frame_count)) * 0.3, 0, 1)
    onset = np.zeros(frame_count)
    beat_frames = np.arange(0, frame_count, 15)
    for bf in beat_frames:
        if bf < frame_count:
            onset[bf] = np.random.uniform(0.5, 1.0)
    spectral = np.clip(np.random.rand(frame_count) * 0.5 + 0.3, 0, 1)
    chroma = np.random.rand(12, frame_count)
    for i in range(frame_count):
        s = chroma[:, i].sum()
        if s > 0:
            chroma[:, i] /= s

    # Neue Features fuer v2.0
    transient = onset * 1.2 + np.random.rand(frame_count) * 0.2
    transient = np.clip(transient, 0, 1)
    voice_clarity = np.clip(np.random.rand(frame_count) * 0.4 + 0.3, 0, 1)
    
    # Beat-Intensity fuer Audio-Sync Features
    beat_intensity = np.zeros(frame_count)
    for bf in beat_frames:
        if bf < frame_count:
            for j in range(min(6, frame_count - bf)):
                beat_intensity[bf + j] = 1.0 - j / 6.0

    return {
        "rms": rms,
        "onset": onset,
        "beat_intensity": beat_intensity,
        "chroma": chroma,
        "spectral_centroid": spectral,
        "transient": transient,
        "voice_clarity": voice_clarity,
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps,
        "progress": np.linspace(0, 1, frame_count),
        "mode": "hybrid",
        "tempo": 128.0,
    }


def benchmark_visualizer(renderer, viz_name: str, features: dict, num_frames: int = 900):
    """Benchmarkt einen einzelnen Visualizer."""
    viz_cls = get_visualizer(viz_name)
    viz = viz_cls(renderer.ctx, renderer.width, renderer.height)

    # Warmup: 10 Frames
    renderer.fbo.use()
    for i in range(10):
        renderer.ctx.clear(0.05, 0.05, 0.05)
        viz.render(features, i / features["fps"])
        renderer.fbo.read(components=3)

    # Benchmark (nur GPU-Render, KEIN fbo.read() um reine Shader-Performance zu messen)
    renderer.fbo.use()
    start = time.perf_counter()
    for i in range(num_frames):
        renderer.ctx.clear(0.05, 0.05, 0.05)
        viz.render(features, i / features["fps"])
    end = time.perf_counter()

    total = end - start
    fps = num_frames / total
    ms_per_frame = (total / num_frames) * 1000
    return total, fps, ms_per_frame


def main():
    WIDTH, HEIGHT, FPS = 1920, 1080, 30
    NUM_FRAMES = 900  # 30 Sekunden

    print("=" * 60)
    print("GPU Visualizer Performance Benchmark")
    print("=" * 60)
    print(f"Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps")
    print(f"Frames per visualizer: {NUM_FRAMES} ({NUM_FRAMES / FPS:.1f}s)")
    print()

    features = generate_dummy_features(NUM_FRAMES, FPS)
    renderer = GPUPreviewRenderer(width=WIDTH, height=HEIGHT, fps=FPS)

    results = []
    visualizers = list_visualizers()

    for name in visualizers:
        print(f"Benchmarking {name}...", end=" ", flush=True)
        try:
            total, fps, ms = benchmark_visualizer(renderer, name, features, NUM_FRAMES)
            results.append({
                "Visualizer": name,
                "Total (s)": f"{total:.2f}",
                "FPS": f"{fps:.1f}",
                "ms/Frame": f"{ms:.2f}",
                "Real-time": "YES" if fps >= FPS else "NO",
            })
            print(f"{fps:.1f} FPS")
        except Exception as e:
            results.append({
                "Visualizer": name,
                "Total (s)": "ERROR",
                "FPS": "ERROR",
                "ms/Frame": "ERROR",
                "Real-time": f"FAIL: {e}",
            })
            print(f"ERROR: {e}")

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    if HAS_TABULATE:
        print(tabulate(results, headers="keys", tablefmt="github"))
    else:
        print(f"{'Visualizer':<20} {'Total (s)':>10} {'FPS':>10} {'ms/Frame':>10} {'Real-time':>10}")
        print("-" * 65)
        for r in results:
            print(f"{r['Visualizer']:<20} {r['Total (s)']:>10} {r['FPS']:>10} {r['ms/Frame']:>10} {r['Real-time']:>10}")

    # Zusammenfassung
    fps_values = []
    for r in results:
        try:
            fps_values.append(float(r["FPS"]))
        except:
            pass

    if fps_values:
        print()
        print(f"Fastest: {max(fps_values):.1f} FPS")
        print(f"Slowest: {min(fps_values):.1f} FPS")
        print(f"Average: {sum(fps_values) / len(fps_values):.1f} FPS")
        print(f"All real-time capable: {'YES' if min(fps_values) >= FPS else 'NO'}")

    # JSON speichern
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
