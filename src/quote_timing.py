"""Lokale Zeitkorrektur fuer Zitate.

Die Zeitstempel der KI treffen den richtigen Satz, liegen an den Raendern
aber gern ein paar Zehntel daneben: das Overlay startet mitten im ersten
Wort oder haengt eine Sprechpause lang nach. Hier wird ausschliesslich aus
den bereits berechneten Audio-Features (RMS) bestimmt, wo wirklich
gesprochen wird — und die Zitatgrenzen auf diese Kanten eingerastet.

Kein Netzwerk, keine neue Abhaengigkeit, deterministisch und damit testbar.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Anteil des lauten Bereichs, ab dem ein Frame als Sprache zaehlt.
# Relativ zum 90%-Perzentil, damit leise und laut gemasterte Dateien
# gleich behandelt werden.
SPEECH_THRESHOLD_RATIO = 0.22
# Kuerzere Lautinseln sind Klicks/Atmer, keine Sprache.
MIN_SPEECH_SECONDS = 0.18
# Kuerzere Luecken sind Wortpausen und trennen kein Segment.
MIN_GAP_SECONDS = 0.30
# Weiter als das wird eine Zitatgrenze nie verschoben.
MAX_SNAP_SECONDS = 1.20
# Kleiner Vorlauf, damit das erste Wort nicht angeschnitten wird.
LEAD_IN_SECONDS = 0.12


def speech_segments(
    rms: Sequence[float],
    fps: float,
    threshold_ratio: float = SPEECH_THRESHOLD_RATIO,
    min_speech: float = MIN_SPEECH_SECONDS,
    min_gap: float = MIN_GAP_SECONDS,
) -> List[Tuple[float, float]]:
    """Findet zusammenhaengende Sprechabschnitte als (start, end) in Sekunden."""
    arr = np.asarray(rms, dtype=np.float32).ravel()
    if arr.size == 0 or fps <= 0:
        return []

    # Schwelle zwischen Grundrauschen und lautem Bereich, damit leise und
    # laut gemasterte Dateien gleich behandelt werden. Liegen beide dicht
    # beieinander (Stille oder Dauerton), gibt es keine Sprachkontur.
    floor = float(np.percentile(arr, 10))
    loud = float(np.percentile(arr, 90))
    if loud <= 0.0 or (loud - floor) < 1e-3:
        return []
    active = arr > (floor + (loud - floor) * threshold_ratio)
    if not active.any():
        return []

    # Wechsel 0->1 und 1->0 ueber die Differenz der gepolsterten Maske
    padded = np.concatenate(([False], active, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    segments = [(int(s), int(e)) for s, e in zip(starts, ends)]

    # Kurze Luecken schliessen (Wortpausen trennen keinen Sprechabschnitt)
    gap_frames = max(int(round(min_gap * fps)), 1)
    merged: List[List[int]] = []
    for s, e in segments:
        if merged and (s - merged[-1][1]) <= gap_frames:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Zu kurze Inseln verwerfen
    min_frames = max(int(round(min_speech * fps)), 1)
    return [
        (s / fps, e / fps) for s, e in merged if (e - s) >= min_frames
    ]


def _nearest(value: float, candidates: Sequence[float], max_shift: float):
    """Naechstgelegener Kandidat innerhalb von max_shift, sonst None."""
    best = None
    best_d = max_shift
    for c in candidates:
        d = abs(c - value)
        if d <= best_d:
            best, best_d = c, d
    return best


def snap_to_speech(
    start: float,
    end: float,
    segments: Sequence[Tuple[float, float]],
    max_shift: float = MAX_SNAP_SECONDS,
    lead_in: float = LEAD_IN_SECONDS,
) -> Tuple[float, float]:
    """Rastet ein Zeitfenster auf die naechsten Sprech-Kanten ein.

    Der Start wandert auf den Beginn eines Sprechabschnitts (minus einem
    kleinen Vorlauf), das Ende auf dessen Ende. Findet sich in Reichweite
    keine Kante, bleibt der Wert unveraendert. Die Reihenfolge
    start < end bleibt garantiert erhalten.
    """
    if not segments:
        return start, end

    snapped_start = _nearest(start, [s for s, _ in segments], max_shift)
    snapped_end = _nearest(end, [e for _, e in segments], max_shift)

    new_start = start if snapped_start is None else max(0.0, snapped_start - lead_in)
    new_end = end if snapped_end is None else snapped_end

    if new_end <= new_start:
        # Einrasten wuerde das Fenster umdrehen — dann lieber nichts tun.
        return start, end
    return new_start, new_end


def snap_quotes(quotes, features, max_shift: float = MAX_SNAP_SECONDS):
    """Rastet alle Zitate eines Clips auf die Sprech-Kanten ein.

    Erwartet Objekte mit start_time/end_time (Quote) und liefert neue
    Instanzen desselben Typs zurueck. Fehlen Features, bleibt alles wie es ist.
    """
    rms = getattr(features, "rms", None)
    fps = float(getattr(features, "fps", 0) or 0)
    if rms is None or fps <= 0:
        return list(quotes)

    segments = speech_segments(rms, fps)
    if not segments:
        return list(quotes)

    result = []
    for q in quotes:
        s, e = snap_to_speech(q.start_time, q.end_time, segments, max_shift)
        result.append(q.model_copy(update={"start_time": s, "end_time": e}))
    return result


# ---------------------------------------------------------------------------
# Zuordnung Zitat-Text -> Zeitfenster innerhalb transkribierter Segmente
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Kleinschreibung ohne Satzzeichen — fuer den Textvergleich."""
    return " ".join(
        "".join(c if (c.isalnum() or c.isspace()) else " " for c in (text or "").lower()).split()
    )


def locate_in_segments(text: str, segments: Sequence[dict]):
    """Findet das Zeitfenster eines Zitat-Textes in transkribierten Segmenten.

    `segments` sind Dicts mit start/end/text. Der Text wird im
    zusammengesetzten Transkript gesucht; die Grenzen werden innerhalb der
    beteiligten Segmente ueber die Zeichenposition interpoliert. Das ist
    genauer als das ganze Segment zu nehmen, ohne dass die KI Sekunden
    schaetzen muesste.

    Gibt (start, end) zurueck oder None, wenn der Text nicht vorkommt.
    """
    if not text or not segments:
        return None

    spans = []   # (zeichen_start, zeichen_ende, t_start, t_ende)
    joined = []
    cursor = 0
    for seg in segments:
        seg_text = _normalize(seg.get("text", ""))
        if not seg_text:
            continue
        try:
            t0 = float(seg.get("start", 0.0))
            t1 = float(seg.get("end", t0))
        except (TypeError, ValueError):
            continue
        if t1 <= t0:
            t1 = t0 + 0.1
        spans.append((cursor, cursor + len(seg_text), t0, t1))
        joined.append(seg_text)
        cursor += len(seg_text) + 1

    if not spans:
        return None

    haystack = " ".join(joined)
    needle = _normalize(text)
    pos = haystack.find(needle)
    if pos < 0:
        # Fallback: laengste gemeinsame Wortfolge ueber den Anfang des Zitats
        words = needle.split()
        while len(words) > 3:
            words = words[:-1]
            pos = haystack.find(" ".join(words))
            if pos >= 0:
                needle = " ".join(words)
                break
        if pos < 0:
            return None

    return _interpolate(pos, pos + len(needle), spans)


def _interpolate(char_start: int, char_end: int, spans):
    """Rechnet Zeichenpositionen im Transkript in Sekunden um."""
    def to_time(char_pos, is_end):
        for c0, c1, t0, t1 in spans:
            if c0 <= char_pos <= c1:
                frac = (char_pos - c0) / max(c1 - c0, 1)
                return t0 + frac * (t1 - t0)
        # Ausserhalb aller Segmente: an den Rand klemmen
        return spans[-1][3] if is_end else spans[0][2]

    start = to_time(char_start, False)
    end = to_time(char_end, True)
    if end <= start:
        end = start + 0.5
    return start, end
