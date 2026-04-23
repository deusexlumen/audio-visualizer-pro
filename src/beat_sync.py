"""
Beat-Synchronisation Utilities.

Ermoeglicht praecises Timing von Quotes, Uebergaengen und Effekten
anhand der erkannten Beats im Audio.
"""

import numpy as np
from typing import List, Optional
from .types import Quote


def get_nearest_beat_time(time_seconds: float, beat_frames: np.ndarray, fps: int) -> float:
    """Findet die Zeit des naechsten Beats ab der gegebenen Zeit.
    
    Args:
        time_seconds: Startzeit in Sekunden
        beat_frames: Array von Beat-Frame-Indizes
        fps: Frames pro Sekunde
        
    Returns:
        Zeit des naechsten Beats in Sekunden (oder Originalzeit wenn keine Beats)
    """
    if len(beat_frames) == 0:
        return time_seconds
    
    target_frame = int(time_seconds * fps)
    
    # Finde den naechsten Beat >= target_frame
    future_beats = beat_frames[beat_frames >= target_frame]
    if len(future_beats) > 0:
        return future_beats[0] / fps
    
    # Wenn kein zukuenftiger Beat, nimm den letzten
    return beat_frames[-1] / fps


def sync_quotes_to_beats(quotes: List[Quote], beat_frames: np.ndarray, fps: int,
                         shift_threshold: float = 0.3) -> List[Quote]:
    """Verschiebt Quote-Startzeiten zum naechsten Beat.
    
    Nur wenn der Quote innerhalb von shift_threshold Sekunden eines Beats liegt,
    wird er verschoben. Das verhindert extreme Verschiebungen.
    
    Args:
        quotes: Liste von Quotes
        beat_frames: Array von Beat-Frame-Indizes
        fps: Frames pro Sekunde
        shift_threshold: Max. Verschiebung in Sekunden
        
    Returns:
        Neue Liste von Quotes mit synchronisierten Startzeiten
    """
    if len(beat_frames) == 0 or not quotes:
        return quotes
    
    synced = []
    for quote in quotes:
        target_frame = int(quote.start_time * fps)
        
        # Naechsten Beat finden
        future_beats = beat_frames[beat_frames >= target_frame]
        if len(future_beats) == 0:
            synced.append(quote)
            continue
        
        nearest_beat_frame = future_beats[0]
        nearest_beat_time = nearest_beat_frame / fps
        
        # Nur verschieben wenn nahe genug
        if abs(nearest_beat_time - quote.start_time) <= shift_threshold:
            new_quote = Quote(
                text=quote.text,
                start_time=nearest_beat_time,
                end_time=max(nearest_beat_time + 1.0, quote.end_time),
                confidence=quote.confidence
            )
            synced.append(new_quote)
        else:
            synced.append(quote)
    
    return synced


def is_on_beat(frame_idx: int, beat_frames: np.ndarray, tolerance: int = 2) -> bool:
    """Prueft ob ein Frame-Index auf oder nahe eines Beats liegt.
    
    Args:
        frame_idx: Aktueller Frame-Index
        beat_frames: Array von Beat-Frame-Indizes
        tolerance: Toleranz in Frames (+/-)
        
    Returns:
        True wenn Frame auf Beat liegt
    """
    if len(beat_frames) == 0:
        return False
    return np.any(np.abs(beat_frames - frame_idx) <= tolerance)


def get_beat_intensity(frame_idx: int, beat_frames: np.ndarray, 
                       decay_frames: int = 6) -> float:
    """Berechnet die Beat-Intensitaet fuer einen Frame (1.0 auf Beat, abnehmend).
    
    Args:
        frame_idx: Aktueller Frame-Index
        beat_frames: Array von Beat-Frame-Indizes
        decay_frames: Anzahl Frames ueber die der Beat ausklingt
        
    Returns:
        Intensitaet von 0.0 bis 1.0
    """
    if len(beat_frames) == 0:
        return 0.0
    
    distances = beat_frames - frame_idx
    # Nur vergangene oder aktuelle Beats
    past_distances = distances[distances <= 0]
    if len(past_distances) == 0:
        return 0.0
    
    nearest_distance = abs(past_distances[-1])
    if nearest_distance > decay_frames:
        return 0.0
    
    return 1.0 - (nearest_distance / decay_frames)


def get_next_beat_time(current_time: float, beat_frames: np.ndarray, fps: int) -> Optional[float]:
    """Gibt die Zeit des naechsten Beats zurueck.
    
    Args:
        current_time: Aktuelle Zeit in Sekunden
        beat_frames: Array von Beat-Frame-Indizes
        fps: Frames pro Sekunde
        
    Returns:
        Zeit des naechsten Beats oder None
    """
    if len(beat_frames) == 0:
        return None
    
    current_frame = int(current_time * fps)
    future_beats = beat_frames[beat_frames > current_frame]
    
    if len(future_beats) == 0:
        return None
    
    return future_beats[0] / fps


def create_beat_grid_overlay(width: int, height: int, frame_idx: int, 
                             beat_frames: np.ndarray, fps: int) -> float:
    """Berechnet einen Beat-Grid Alpha-Wert fuer Debug-Overlays.
    
    Args:
        width, height: Bildaufloesung
        frame_idx: Aktueller Frame
        beat_frames: Beat-Frame-Indizes
        fps: FPS
        
    Returns:
        Alpha-Wert fuer Beat-Marker (0.0 - 1.0)
    """
    if len(beat_frames) == 0:
        return 0.0
    
    if is_on_beat(frame_idx, beat_frames, tolerance=1):
        return 1.0
    
    # Kurzer Glow nach dem Beat
    intensity = get_beat_intensity(frame_idx, beat_frames, decay_frames=3)
    return intensity * 0.3
