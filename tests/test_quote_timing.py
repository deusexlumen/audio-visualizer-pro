"""Tests fuer die lokale Zeitkorrektur von Zitaten (src/quote_timing.py)."""

import numpy as np
import pytest

from src.quote_timing import (
    locate_in_segments,
    snap_quotes,
    snap_to_speech,
    speech_segments,
)
from src.types import AudioFeatures, Quote


FPS = 30


def rms_with_speech(spans, duration=20.0, fps=FPS, level=0.8):
    """Baut eine RMS-Spur, die nur in den angegebenen Spannen laut ist."""
    arr = np.full(int(duration * fps), 0.02, dtype=np.float32)
    for start, end in spans:
        arr[int(start * fps):int(end * fps)] = level
    return arr


class TestSpeechSegments:
    def test_findet_einzelnen_abschnitt(self):
        segs = speech_segments(rms_with_speech([(4.0, 7.0)]), FPS)
        assert len(segs) == 1
        assert segs[0][0] == pytest.approx(4.0, abs=0.05)
        assert segs[0][1] == pytest.approx(7.0, abs=0.05)

    def test_trennt_abschnitte_bei_langer_pause(self):
        segs = speech_segments(rms_with_speech([(2.0, 4.0), (8.0, 10.0)]), FPS)
        assert len(segs) == 2

    def test_schliesst_kurze_wortpausen(self):
        # 0.1 s Luecke ist eine Wortpause, kein neuer Abschnitt
        segs = speech_segments(rms_with_speech([(2.0, 4.0), (4.1, 6.0)]), FPS)
        assert len(segs) == 1
        assert segs[0][1] == pytest.approx(6.0, abs=0.05)

    def test_verwirft_zu_kurze_lautinseln(self):
        # 0.05 s Klick ist keine Sprache
        segs = speech_segments(rms_with_speech([(3.0, 3.05)]), FPS)
        assert segs == []

    def test_leere_eingabe(self):
        assert speech_segments(np.array([]), FPS) == []
        assert speech_segments(np.zeros(100, dtype=np.float32), FPS) == []

    def test_fps_null_liefert_leer(self):
        assert speech_segments(rms_with_speech([(1.0, 3.0)]), 0) == []


class TestSnapToSpeech:
    segments = [(4.0, 7.0), (10.0, 13.0)]

    def test_rastet_auf_naechste_kanten_ein(self):
        start, end = snap_to_speech(4.4, 7.3, self.segments)
        # Vorlauf von 0.12 s, damit das erste Wort nicht angeschnitten wird
        assert start == pytest.approx(4.0 - 0.12, abs=0.01)
        assert end == pytest.approx(7.0, abs=0.01)

    def test_laesst_weit_entfernte_werte_stehen(self):
        start, end = snap_to_speech(0.5, 2.0, self.segments)
        assert (start, end) == (0.5, 2.0)

    def test_ohne_segmente_unveraendert(self):
        assert snap_to_speech(1.0, 2.0, []) == (1.0, 2.0)

    def test_dreht_das_fenster_nicht_um(self):
        # Kante liegt so, dass Einrasten start hinter end schieben wuerde
        start, end = snap_to_speech(6.9, 7.05, [(7.0, 12.0), (6.9, 7.0)])
        assert start < end

    def test_max_shift_wird_beachtet(self):
        start, end = snap_to_speech(4.9, 7.9, self.segments, max_shift=0.5)
        assert (start, end) == (4.9, 7.9)


class TestSnapQuotes:
    def _features(self, rms):
        n = rms.size
        return AudioFeatures(
            duration=n / FPS, sample_rate=22050, fps=FPS, frame_count=n,
            rms=rms, onset=np.zeros(n), spectral_centroid=np.zeros(n),
            spectral_rolloff=np.zeros(n), zero_crossing_rate=np.zeros(n),
            chroma=np.zeros((12, n)), mfcc=np.zeros((13, n)),
            tempogram=np.zeros((10, n)), tempo=120.0, mode="speech",
        )

    def test_korrigiert_zitatgrenzen(self):
        rms = rms_with_speech([(4.0, 7.0)])
        quotes = [Quote(text="Test", start_time=4.35, end_time=7.4)]
        result = snap_quotes(quotes, self._features(rms))
        assert result[0].start_time == pytest.approx(3.88, abs=0.05)
        assert result[0].end_time == pytest.approx(7.0, abs=0.05)
        assert result[0].text == "Test"

    def test_gibt_neue_objekte_zurueck(self):
        rms = rms_with_speech([(4.0, 7.0)])
        quotes = [Quote(text="Test", start_time=4.35, end_time=7.4)]
        result = snap_quotes(quotes, self._features(rms))
        assert quotes[0].start_time == 4.35  # Original unangetastet
        assert result[0] is not quotes[0]

    def test_ohne_features_unveraendert(self):
        quotes = [Quote(text="Test", start_time=1.0, end_time=2.0)]
        assert snap_quotes(quotes, None) == quotes


class TestLocateInSegments:
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Hallo und willkommen zu dieser Folge."},
        {"start": 4.0, "end": 9.0,
         "text": "Heute reden wir ueber die Werbeoekonomie im Netz."},
        {"start": 9.0, "end": 13.0, "text": "Das wird spannend, versprochen."},
    ]

    def test_findet_woertliches_zitat(self):
        window = locate_in_segments("Heute reden wir ueber die Werbeoekonomie",
                                    self.segments)
        assert window is not None
        start, end = window
        assert start == pytest.approx(4.0, abs=0.3)
        assert 4.0 < end <= 9.0

    def test_ignoriert_satzzeichen_und_gross_klein(self):
        assert locate_in_segments("HALLO UND WILLKOMMEN!", self.segments) is not None

    def test_zitat_ueber_segmentgrenze(self):
        text = "dieser Folge Heute reden wir"
        window = locate_in_segments(text, self.segments)
        assert window is not None
        start, end = window
        assert start < 4.0 < end

    def test_erfundenes_zitat_liefert_none(self):
        assert locate_in_segments("Das hat niemand gesagt", self.segments) is None

    def test_leere_eingaben(self):
        assert locate_in_segments("", self.segments) is None
        assert locate_in_segments("Hallo", []) is None

    def test_teiltreffer_ueber_den_anfang(self):
        # Die KI haengt hinten etwas an, das so nicht gesagt wurde
        window = locate_in_segments(
            "Das wird spannend, versprochen und noch viel mehr dazu", self.segments)
        assert window is not None
        assert window[0] == pytest.approx(9.0, abs=0.4)

    def test_segmente_ohne_zeiten_werden_uebersprungen(self):
        segs = [{"text": "Ohne Zeitangabe"}, {"start": 2.0, "end": 4.0, "text": "Mit Zeit"}]
        window = locate_in_segments("Mit Zeit", segs)
        assert window is not None
