"""Tests fuer den Zitat-Editor (Text + Zeitfenster + Sprech-Einrasten)."""

import numpy as np
import pytest

from src.gui.quote_editor import QuoteEditorDialog, QuoteWaveform
from src.types import AudioFeatures, Quote

FPS = 30


def make_features(spans=((4.0, 7.0),), duration=20.0):
    n = int(duration * FPS)
    rms = np.full(n, 0.02, dtype=np.float32)
    for start, end in spans:
        rms[int(start * FPS):int(end * FPS)] = 0.8
    return AudioFeatures(
        duration=duration, sample_rate=22050, fps=FPS, frame_count=n,
        rms=rms, onset=np.zeros(n), spectral_centroid=np.zeros(n),
        spectral_rolloff=np.zeros(n), zero_crossing_rate=np.zeros(n),
        chroma=np.zeros((12, n)), mfcc=np.zeros((13, n)),
        tempogram=np.zeros((10, n)), tempo=120.0, mode="speech",
    )


class TestQuoteEditorDialog:
    def test_uebernimmt_werte_aus_dem_zitat(self, qtbot):
        quote = Quote(text="Hallo Welt", start_time=4.4, end_time=6.8, confidence=0.9)
        dialog = QuoteEditorDialog(quote, features=make_features())
        qtbot.addWidget(dialog)

        assert dialog.txt.toPlainText() == "Hallo Welt"
        assert dialog.spin_start.value() == pytest.approx(4.4)
        assert dialog.spin_end.value() == pytest.approx(6.8)

    def test_result_quote_liefert_bearbeitete_werte(self, qtbot):
        quote = Quote(text="Alt", start_time=4.4, end_time=6.8, confidence=0.75)
        dialog = QuoteEditorDialog(quote, features=make_features())
        qtbot.addWidget(dialog)

        dialog.txt.setPlainText("Neu")
        dialog.spin_start.setValue(5.0)
        dialog.spin_end.setValue(6.0)

        result = dialog.result_quote()
        assert result.text == "Neu"
        assert result.start_time == pytest.approx(5.0)
        assert result.end_time == pytest.approx(6.0)
        # Confidence bleibt erhalten
        assert result.confidence == pytest.approx(0.75)

    def test_ende_bleibt_hinter_dem_start(self, qtbot):
        quote = Quote(text="Test", start_time=4.0, end_time=6.0)
        dialog = QuoteEditorDialog(quote, features=make_features())
        qtbot.addWidget(dialog)

        dialog.spin_start.setValue(8.0)
        assert dialog.spin_end.value() > dialog.spin_start.value()

    def test_einrasten_zieht_auf_sprechgrenzen(self, qtbot):
        quote = Quote(text="Test", start_time=4.45, end_time=7.35)
        dialog = QuoteEditorDialog(quote, features=make_features([(4.0, 7.0)]))
        qtbot.addWidget(dialog)

        assert dialog.btn_snap.isEnabled()
        dialog._snap()
        assert dialog.spin_start.value() == pytest.approx(3.88, abs=0.06)
        assert dialog.spin_end.value() == pytest.approx(7.0, abs=0.06)

    def test_ohne_features_kein_einrasten(self, qtbot):
        quote = Quote(text="Test", start_time=1.0, end_time=2.0)
        dialog = QuoteEditorDialog(quote)
        qtbot.addWidget(dialog)

        assert not dialog.btn_snap.isEnabled()
        assert dialog.result_quote().start_time == pytest.approx(1.0)

    def test_ohne_audio_pfad_keine_wiedergabe(self, qtbot):
        dialog = QuoteEditorDialog(Quote(text="T", start_time=1.0, end_time=2.0))
        qtbot.addWidget(dialog)
        assert not dialog.btn_play.isEnabled()

    def test_leerer_text_faellt_auf_das_original_zurueck(self, qtbot):
        quote = Quote(text="Original", start_time=1.0, end_time=2.0)
        dialog = QuoteEditorDialog(quote, features=make_features())
        qtbot.addWidget(dialog)

        dialog.txt.setPlainText("   ")
        assert dialog.result_quote().text == "Original"


class TestQuoteWaveform:
    def test_zeitachse_und_pixel_sind_invers(self, qtbot):
        wave = QuoteWaveform()
        qtbot.addWidget(wave)
        wave.resize(400, 100)
        wave.set_view(10.0, 20.0)

        x = wave._t_to_x(15.0)
        assert x == pytest.approx(200.0, abs=1.0)
        assert wave._x_to_t(x) == pytest.approx(15.0, abs=0.1)

    def test_ungueltiges_fenster_wird_korrigiert(self, qtbot):
        wave = QuoteWaveform()
        qtbot.addWidget(wave)
        wave.set_view(5.0, 5.0)
        assert wave._view[1] > wave._view[0]

    def test_zeichnen_ohne_audio_stuerzt_nicht_ab(self, qtbot):
        wave = QuoteWaveform()
        qtbot.addWidget(wave)
        wave.resize(300, 100)
        wave.set_range(1.0, 2.0)
        wave.show()
        qtbot.waitExposed(wave)
