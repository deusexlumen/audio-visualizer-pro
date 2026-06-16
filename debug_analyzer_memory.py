"""
Diagnose-Skript fuer Analyzer-RAM-Verbrauch.

Berechnet die erwarteten Array-Groessen fuer eine Audio-Datei gegebener Laenge
und zeigt, warum der Analyzer bei langen Dateien (z.B. 20min Podcast) so viel
RAM verbraucht.
"""

import numpy as np


def estimate_analyzer_memory(duration_min: float, fps: int = 60):
    """Schaetzt den RAM-Verbrauch des Analyzers fuer eine Datei der Laenge duration_min."""
    sr = 44100
    n_fft = 2048

    duration_sec = duration_min * 60
    # Heuristik aus analyzer.py: >10min -> hop_length=512
    hop_length = 512 if duration_sec > 600 else 256
    samples = int(duration_sec * sr)
    stft_frames = int(np.ceil(samples / hop_length))
    video_frames = int(np.ceil(duration_sec * fps))

    # y: float32
    y_mb = samples * 4 / (1024 ** 2)
    # stft_complex: complex64 = 8 Byte/Element
    stft_complex_mb = (n_fft // 2 + 1) * stft_frames * 8 / (1024 ** 2)
    # stft (np.abs): float32
    stft_mb = (n_fft // 2 + 1) * stft_frames * 4 / (1024 ** 2)
    # tempogram: default win_length=384, float32
    tempogram_mb = 384 * stft_frames * 4 / (1024 ** 2)
    # chroma_raw: 12 x stft_frames, float32
    chroma_mb = 12 * stft_frames * 4 / (1024 ** 2)
    # mfcc_raw: 13 x stft_frames, float32
    mfcc_mb = 13 * stft_frames * 4 / (1024 ** 2)

    print(f"Dauer:            {duration_min:.1f} min ({duration_sec:.0f}s)")
    print(f"hop_length:       {hop_length}  (>10min -> 512)")
    print(f"FPS:              {fps}")
    print(f"STFT-Frames:      {stft_frames:,}")
    print(f"Video-Frames:     {video_frames:,}")
    print()
    print(f"Audio-Buffer y:   {y_mb:8.0f} MB")
    print(f"stft_complex:     {stft_complex_mb:8.0f} MB  <-- wird NACH np.abs freigegeben")
    print(f"stft (abs):       {stft_mb:8.0f} MB")
    print(f"stft**2 temp:     {stft_mb:8.0f} MB  <-- temporaer fuer onset_strength")
    print(f"tempogram:        {tempogram_mb:8.0f} MB")
    print(f"chroma_raw:       {chroma_mb:8.0f} MB")
    print(f"mfcc_raw:         {mfcc_mb:8.0f} MB")
    print()
    print(f"Peak (alt):       {y_mb + stft_complex_mb + stft_mb + stft_mb:8.0f} MB")
    print(f"Peak nach Fix:    {y_mb + stft_mb + stft_mb:8.0f} MB")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Audio Visualizer Pro - Analyzer RAM-Diagnose")
    print("=" * 60)
    print()
    for minutes in [1, 5, 10, 20]:
        estimate_analyzer_memory(minutes, fps=60)
        print("-" * 60)
