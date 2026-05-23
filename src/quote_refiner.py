"""
Zitat-Zeitstempel-Verfeinerung mit Audio-Analyse.

Nutzt Onset-Detection und Beat-Frames, um Gemini-Schaetzungen
zu korrigieren und praeziser an tatsaechliche Audio-Ereignisse
anzugleichen.
"""

import numpy as np
from typing import List, Optional

from .types import Quote


def refine_quote_timestamps(
    quotes: List[Quote],
    features: dict,
    snap_threshold: float = 1.0,
    max_duration: float = 10.0,
) -> List[Quote]:
    """
    Verfeinert Zitat-Zeitstempel anhand von Audio-Features.

    Logik:
    - Startzeit wird an den naechsten starken Onset/Beat gesnappt,
      falls dieser innerhalb von snap_threshold Sekunden liegt.
    - Endzeit wird begrenzt, falls sie ueber das naechste starke
      Audio-Ereignis hinausgeht (verhindert, dass Zitate ueber
      Pausen oder Beat-Drops hinweg laufen).
    - Maximale Dauer wird auf max_duration begrenzt.

    Args:
        quotes: Liste von Quote-Objekten mit rohen Gemini-Zeitstempeln
        features: AudioFeatures-Dictionary (fps, onset, beat_frames, voice_clarity)
        snap_threshold: Max. Abstand in Sekunden fuer Snap (default 1.0)
        max_duration: Max. Zitat-Laenge in Sekunden (default 10.0)

    Returns:
        Neue Liste von Quote-Objekten mit verfeinerten Zeitstempeln
    """
    if not quotes:
        return quotes

    fps = features.get("fps", 30)
    onset = features.get("onset")
    beat_frames = features.get("beat_frames")
    duration = features.get("duration", 0.0)

    if duration <= 0:
        return quotes

    # Beat-Frames als Sekunden-Array aufbereiten
    beat_times = []
    if beat_frames is not None and len(beat_frames) > 0:
        beat_times = [float(f) / fps for f in np.asarray(beat_frames).flatten()]

    # Onset-Peaks finden (lokale Maxima ueber Threshold)
    onset_peaks = []
    if onset is not None and len(onset) > 0:
        onset_arr = np.asarray(onset).flatten()
        threshold = 0.3  # Mindest-Onset-Staerke fuer Peak
        for i in range(1, len(onset_arr) - 1):
            if onset_arr[i] > threshold and onset_arr[i] > onset_arr[i - 1] and onset_arr[i] > onset_arr[i + 1]:
                onset_peaks.append(float(i) / fps)

    # Kombiniere Beat-Times und Onset-Peaks, sortiere
    event_times = sorted(set(beat_times + onset_peaks))

    refined = []
    for q in quotes:
        new_start = q.start_time
        new_end = q.end_time

        # --- STARTZEIT VERFEINERN ---
        # Finde naechstes Audio-Event innerhalb snap_threshold
        best_event = None
        best_dist = snap_threshold
        for t in event_times:
            dist = abs(t - q.start_time)
            if dist < best_dist:
                best_dist = dist
                best_event = t
        if best_event is not None:
            new_start = best_event

        # --- ENDZEIT VERFEINERN ---
        # Wenn Endzeit ueber naechstes starkes Event liegt, kuerze
        next_event = None
        for t in event_times:
            if t > new_start + 1.0:  # Mindestens 1s nach Start
                next_event = t
                break
        if next_event is not None and new_end > next_event + 1.0:
            # Nur kuerzen, wenn die Differenz gross ist (mehr als 1s)
            new_end = min(new_end, next_event + 0.5)

        # --- MAXIMALE DAUER ---
        if new_end - new_start > max_duration:
            new_end = new_start + max_duration

        # --- GRENZEN ---
        new_start = max(0.0, min(new_start, duration - 1.0))
        new_end = max(new_start + 0.5, min(new_end, duration))

        refined.append(Quote(
            text=q.text,
            start_time=round(new_start, 2),
            end_time=round(new_end, 2),
            confidence=q.confidence,
        ))

    return refined
