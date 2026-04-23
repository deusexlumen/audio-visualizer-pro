"""
Tests fuer Quote Overlay Renderer.
"""

import numpy as np
import pytest
from src.quote_overlay import QuoteOverlayRenderer, QuoteOverlayConfig
from src.gemini_integration import Quote


def create_test_frame(width=1920, height=1080, color=(30, 30, 40)):
    """Erstellt einen Test-Frame mit einheitlicher Farbe."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def create_test_quotes():
    """Erstellt Test-Zitate."""
    return [
        Quote(
            text="Das ist ein wichtiges Key-Zitat fuer den Test.",
            start_time=1.0,
            end_time=5.0,
            confidence=0.95
        ),
        Quote(
            text="Zweites Zitat.",
            start_time=10.0,
            end_time=14.0,
            confidence=0.85
        )
    ]


class TestQuoteOverlayRenderer:
    
    def test_init_without_quotes(self):
        """Renderer ohne Quotes sollte funktionieren."""
        renderer = QuoteOverlayRenderer()
        assert renderer.quotes == []
        assert renderer.config.enabled is True
    
    def test_init_with_quotes(self):
        """Renderer mit Quotes initialisieren."""
        quotes = create_test_quotes()
        renderer = QuoteOverlayRenderer(quotes)
        assert len(renderer.quotes) == 2
        assert renderer.quotes[0].text == "Das ist ein wichtiges Key-Zitat fuer den Test."
    
    def test_apply_no_quotes_returns_original(self):
        """Ohne Quotes sollte der Frame unveraendert bleiben."""
        frame = create_test_frame()
        renderer = QuoteOverlayRenderer()
        result = renderer.apply(frame, time_seconds=2.0)
        np.testing.assert_array_equal(frame, result)
    
    def test_apply_disabled_returns_original(self):
        """Wenn Overlays deaktiviert, Frame unveraendert."""
        frame = create_test_frame()
        quotes = create_test_quotes()
        config = QuoteOverlayConfig(enabled=False)
        renderer = QuoteOverlayRenderer(quotes, config)
        result = renderer.apply(frame, time_seconds=2.0)
        np.testing.assert_array_equal(frame, result)
    
    def test_apply_no_active_quote_returns_original(self):
        """Wenn kein Zitat zur Zeit aktiv, Frame unveraendert."""
        frame = create_test_frame()
        quotes = create_test_quotes()
        renderer = QuoteOverlayRenderer(quotes)
        # Zeit vor dem ersten Zitat
        result = renderer.apply(frame, time_seconds=0.5)
        np.testing.assert_array_equal(frame, result)
        # Zeit zwischen den Zitaten
        result = renderer.apply(frame, time_seconds=7.0)
        np.testing.assert_array_equal(frame, result)
        # Zeit nach allen Zitaten
        result = renderer.apply(frame, time_seconds=20.0)
        np.testing.assert_array_equal(frame, result)
    
    def test_apply_active_quote_modifies_frame(self):
        """Wenn ein Zitat aktiv, sollte der Frame veraendert sein."""
        frame = create_test_frame()
        quotes = create_test_quotes()
        renderer = QuoteOverlayRenderer(quotes)
        # Zeit im ersten Zitat
        result = renderer.apply(frame, time_seconds=3.0)
        # Frame sollte veraendert sein
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
        # Es sollte mindestens einige helle Pixel geben (Text)
        assert np.any(result != frame)
    
    def test_apply_returns_correct_shape(self):
        """Ergebnis sollte immer (H, W, 3) uint8 sein."""
        frame = create_test_frame()
        quotes = create_test_quotes()
        renderer = QuoteOverlayRenderer(quotes)
        result = renderer.apply(frame, time_seconds=3.0)
        assert result.shape == (1080, 1920, 3)
        assert result.dtype == np.uint8
    
    def test_fade_in_at_start(self):
        """Fade-In am Anfang des Zitats."""
        frame = create_test_frame()
        quotes = [Quote(text="Test", start_time=1.0, end_time=5.0, confidence=1.0)]
        renderer = QuoteOverlayRenderer(quotes)
        
        # Direkt am Start sollte es noch leicht transparent sein
        result_early = renderer.apply(frame, time_seconds=1.1)
        # Spaeter sollte es sichtbarer sein
        result_later = renderer.apply(frame, time_seconds=2.0)
        
        # Beide sollten vom Original abweichen
        assert np.any(result_early != frame)
        assert np.any(result_later != frame)
    
    def test_fade_out_at_end(self):
        """Fade-Out am Ende des Zitats."""
        frame = create_test_frame()
        quotes = [Quote(text="Test", start_time=1.0, end_time=5.0, confidence=1.0)]
        renderer = QuoteOverlayRenderer(quotes)
        
        # Kurz vor Ende sollte es noch sichtbar sein
        result_before = renderer.apply(frame, time_seconds=4.6)
        # Nach Ende sollte es nicht mehr sichtbar sein
        result_after = renderer.apply(frame, time_seconds=5.5)
        
        assert np.any(result_before != frame)
        np.testing.assert_array_equal(result_after, frame)
    
    def test_text_wrapping(self):
        """Langer Text sollte umgebrochen werden."""
        frame = create_test_frame()
        long_text = "Dies ist ein sehr langer Text der unbedingt umgebrochen werden muss damit er auf den Bildschirm passt."
        quotes = [Quote(text=long_text, start_time=1.0, end_time=5.0, confidence=1.0)]
        renderer = QuoteOverlayRenderer(quotes)
        result = renderer.apply(frame, time_seconds=3.0)
        assert result.shape == frame.shape
        assert np.any(result != frame)
    
    def test_multiple_quotes_sequential(self):
        """Mehrere Zitate nacheinander."""
        frame = create_test_frame()
        quotes = [
            Quote(text="Erstes Zitat", start_time=1.0, end_time=3.0, confidence=1.0),
            Quote(text="Zweites Zitat", start_time=5.0, end_time=7.0, confidence=1.0),
        ]
        renderer = QuoteOverlayRenderer(quotes)
        
        # Erstes Zitat aktiv
        r1 = renderer.apply(frame.copy(), time_seconds=2.0)
        # Kein Zitat aktiv
        r2 = renderer.apply(frame.copy(), time_seconds=4.0)
        # Zweites Zitat aktiv
        r3 = renderer.apply(frame.copy(), time_seconds=6.0)
        
        assert np.any(r1 != frame)
        np.testing.assert_array_equal(r2, frame)
        assert np.any(r3 != frame)
    
    def test_custom_config(self):
        """Eigene Konfiguration anwenden."""
        frame = create_test_frame()
        quotes = create_test_quotes()
        config = QuoteOverlayConfig(
            font_size=24,
            font_color=(255, 0, 0),
            box_color=(0, 0, 255, 100),
            box_margin_bottom=200
        )
        renderer = QuoteOverlayRenderer(quotes, config)
        result = renderer.apply(frame, time_seconds=3.0)
        assert result.shape == frame.shape
        assert np.any(result != frame)
    
    def test_empty_quote_text(self):
        """Leerer Text sollte keinen Fehler werfen."""
        frame = create_test_frame()
        quotes = [Quote(text="", start_time=1.0, end_time=5.0, confidence=1.0)]
        renderer = QuoteOverlayRenderer(quotes)
        result = renderer.apply(frame, time_seconds=3.0)
        np.testing.assert_array_equal(result, frame)


class TestQuoteOverlayConfig:
    
    def test_default_values(self):
        """Standard-Werte pruefen."""
        config = QuoteOverlayConfig()
        assert config.enabled is True
        assert config.font_size == 36
        assert config.font_color == (255, 255, 255)
        assert config.box_color == (0, 0, 0, 160)
        assert config.box_padding == 24
        assert config.fade_duration == 0.5
        assert config.auto_scale_font is True
        assert config.min_font_size == 16
        assert config.max_font_size == 72
        assert config.slide_animation == "none"
        assert config.slide_distance == 100.0
        assert config.slide_out_animation == "none"
        assert config.slide_out_distance == 100.0
        assert config.scale_in is False
        assert config.typewriter is False
        assert config.typewriter_speed == 15.0
        assert config.typewriter_mode == "char"
        assert config.glow_pulse is False
        assert config.glow_pulse_intensity == 0.5
    
    def test_custom_values(self):
        """Eigene Werte setzen."""
        config = QuoteOverlayConfig(
            font_size=48,
            font_color=(0, 255, 0),
            fade_duration=1.0,
            auto_scale_font=False,
            min_font_size=12,
            slide_animation="up",
            slide_distance=150.0,
            scale_in=True,
            typewriter=True,
            typewriter_speed=20.0,
            typewriter_mode="word",
            glow_pulse=True,
            glow_pulse_intensity=0.8,
        )
        assert config.font_size == 48
        assert config.font_color == (0, 255, 0)
        assert config.fade_duration == 1.0
        assert config.auto_scale_font is False
        assert config.min_font_size == 12
        assert config.slide_animation == "up"
        assert config.slide_distance == 150.0
        assert config.scale_in is True
        assert config.typewriter is True
        assert config.typewriter_speed == 20.0
        assert config.typewriter_mode == "word"
        assert config.glow_pulse is True
        assert config.glow_pulse_intensity == 0.8
