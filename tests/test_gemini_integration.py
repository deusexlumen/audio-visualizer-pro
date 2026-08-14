"""
Tests für Gemini Integration.

Alle Tests nutzen Mocks – keine echten API-Calls!
"""

import os
import json
import tempfile
import pytest
from unittest.mock import Mock, patch


class TestGeminiIntegration:
    """Test-Suite für GeminiIntegration."""

    def test_init_without_api_key_raises(self):
        """Sollte fehlschlagen, wenn kein API Key vorhanden ist."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('src.gemini_integration.genai'):
                from src.gemini_integration import GeminiIntegration
                with pytest.raises(ValueError, match="API Key"):
                    GeminiIntegration()

    def test_init_with_api_key(self):
        """Sollte funktionieren, wenn API Key als Parameter übergeben wird."""
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            mock_genai.Client.return_value = Mock()
            
            gemini = GeminiIntegration(api_key="test-key-123")
            assert gemini.api_key == "test-key-123"
            # Standard-Modell kommt jetzt aus config/settings.json (konfigurierbar)
            assert gemini.model == "gemini-flash-lite-latest"

    def test_transcribe_audio(self):
        """Sollte Audio transkribieren und Text zurückgeben."""
        import tempfile
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(b'dummy audio data')
                tmp_path = tmp.name
            
            try:
                # Mock FileState fuer ACTIVE-Check
                mock_active_state = Mock()
                mock_active_state.name = "ACTIVE"
                mock_processing_state = Mock()
                mock_processing_state.name = "PROCESSING"
                mock_types = Mock()
                mock_types.FileState.ACTIVE = mock_active_state
                mock_types.FileState.PROCESSING = mock_processing_state
                mock_genai.types = mock_types
                
                mock_client = Mock()
                mock_file = Mock()
                mock_file.state = mock_active_state
                mock_file.name = "files/test-audio"
                mock_client.files.upload.return_value = mock_file
                mock_client.files.get.return_value = mock_file
                
                mock_response = Mock()
                mock_response.text = "  Das ist ein Test-Transkript.  "
                mock_client.models.generate_content.return_value = mock_response
                
                mock_genai.Client.return_value = mock_client
                
                gemini = GeminiIntegration(api_key="test-key")
                result = gemini.transcribe_audio(tmp_path)
                
                assert result == "Das ist ein Test-Transkript."
                mock_client.files.upload.assert_called_once()
                mock_client.models.generate_content.assert_called_once()
            finally:
                os.unlink(tmp_path)

    def test_extract_quotes(self):
        """Sollte Zitate aus Audio extrahieren."""
        import tempfile
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(b'dummy audio data')
                tmp_path = tmp.name
            
            try:
                # Mock FileState fuer ACTIVE-Check
                mock_active_state = Mock()
                mock_active_state.name = "ACTIVE"
                mock_processing_state = Mock()
                mock_processing_state.name = "PROCESSING"
                mock_types = Mock()
                mock_types.FileState.ACTIVE = mock_active_state
                mock_types.FileState.PROCESSING = mock_processing_state
                mock_genai.types = mock_types
                
                mock_client = Mock()
                mock_file = Mock()
                mock_file.state = mock_active_state
                mock_file.name = "files/test-audio"
                mock_client.files.upload.return_value = mock_file
                mock_client.files.get.return_value = mock_file
                
                mock_response = Mock()
                mock_response.text = '[{"text": "Hello world today", "start_time": 10.5, "end_time": 15.2, "confidence": 0.9}]'
                mock_client.models.generate_content.return_value = mock_response
                
                mock_genai.Client.return_value = mock_client
                
                gemini = GeminiIntegration(api_key="test-key")
                quotes = gemini.extract_quotes(tmp_path, max_quotes=1)
                
                assert len(quotes) == 1
                assert quotes[0].text == "Hello world today"
                assert quotes[0].start_time == 10.5
                assert quotes[0].end_time == 15.2
                assert quotes[0].confidence == 0.9
            finally:
                os.unlink(tmp_path)

    def test_extract_quotes_sorted_by_time(self):
        """Sollte Zitate nach Startzeit sortieren."""
        import tempfile
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(b'dummy audio data')
                tmp_path = tmp.name
            
            try:
                # Mock FileState fuer ACTIVE-Check
                mock_active_state = Mock()
                mock_active_state.name = "ACTIVE"
                mock_processing_state = Mock()
                mock_processing_state.name = "PROCESSING"
                mock_types = Mock()
                mock_types.FileState.ACTIVE = mock_active_state
                mock_types.FileState.PROCESSING = mock_processing_state
                mock_genai.types = mock_types
                
                mock_client = Mock()
                mock_file = Mock()
                mock_file.state = mock_active_state
                mock_file.name = "files/test-audio"
                mock_client.files.upload.return_value = mock_file
                mock_client.files.get.return_value = mock_file
                
                mock_response = Mock()
                mock_response.text = '[{"text": "Zweites Zitat hier", "start_time": 20.0, "end_time": 25.0, "confidence": 0.8}, {"text": "Erstes Zitat hier", "start_time": 5.0, "end_time": 10.0, "confidence": 0.9}]'
                mock_client.models.generate_content.return_value = mock_response
                
                mock_genai.Client.return_value = mock_client
                
                gemini = GeminiIntegration(api_key="test-key")
                quotes = gemini.extract_quotes(tmp_path)
                
                assert quotes[0].text == "Erstes Zitat hier"
                assert quotes[1].text == "Zweites Zitat hier"
            finally:
                os.unlink(tmp_path)

    def test_parse_json_response_with_markdown(self):
        """Sollte JSON aus Markdown-Code-Blöcken extrahieren."""
        with patch('src.gemini_integration.genai'):
            from src.gemini_integration import GeminiIntegration
            
            markdown_text = '```json\n[{"text": "Test", "start_time": 1.0}]\n```'
            result = GeminiIntegration._parse_json_response(markdown_text)
            
            assert result[0]["text"] == "Test"
            assert result[0]["start_time"] == 1.0

    def test_file_not_found(self):
        """Sollte FileNotFoundError werfen, wenn Audio nicht existiert."""
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            mock_genai.Client.return_value = Mock()
            
            gemini = GeminiIntegration(api_key="test-key")
            
            with pytest.raises(FileNotFoundError):
                gemini.transcribe_audio("nicht_existent.mp3")

    def test_transcribe_audio_async_returns_future(self):
        """Asynchrone Methode sollte ein Future zurueckgeben, das spaeter das Ergebnis liefert."""
        import tempfile
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(b'dummy audio data')
                tmp_path = tmp.name
            
            try:
                mock_active_state = Mock()
                mock_active_state.name = "ACTIVE"
                mock_types = Mock()
                mock_types.FileState.ACTIVE = mock_active_state
                mock_genai.types = mock_types
                
                mock_client = Mock()
                mock_file = Mock()
                mock_file.state = mock_active_state
                mock_file.name = "files/test-audio"
                mock_client.files.upload.return_value = mock_file
                mock_client.files.get.return_value = mock_file
                
                mock_response = Mock()
                mock_response.text = "Async Transkript"
                mock_client.models.generate_content.return_value = mock_response
                
                mock_genai.Client.return_value = mock_client
                
                gemini = GeminiIntegration(api_key="test-key")
                future = gemini.transcribe_audio_async(tmp_path)
                
                # Muss ein concurrent.futures.Future sein
                import concurrent.futures
                assert isinstance(future, concurrent.futures.Future)
                assert future.result() == "Async Transkript"
            finally:
                os.unlink(tmp_path)

    def test_extract_quotes_async_returns_future(self):
        """Asynchrone Zitat-Extraktion sollte ein Future zurueckgeben."""
        import tempfile
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(b'dummy audio data')
                tmp_path = tmp.name
            
            try:
                mock_active_state = Mock()
                mock_active_state.name = "ACTIVE"
                mock_types = Mock()
                mock_types.FileState.ACTIVE = mock_active_state
                mock_genai.types = mock_types
                
                mock_client = Mock()
                mock_file = Mock()
                mock_file.state = mock_active_state
                mock_file.name = "files/test-audio"
                mock_client.files.upload.return_value = mock_file
                mock_client.files.get.return_value = mock_file
                
                mock_response = Mock()
                mock_response.text = '[{"text": "Async Zitat hier", "start_time": 1.0, "end_time": 5.0, "confidence": 0.9}]'
                mock_client.models.generate_content.return_value = mock_response
                
                mock_genai.Client.return_value = mock_client
                
                gemini = GeminiIntegration(api_key="test-key")
                future = gemini.extract_quotes_async(tmp_path, max_quotes=1)
                
                import concurrent.futures
                assert isinstance(future, concurrent.futures.Future)
                quotes = future.result()
                assert len(quotes) == 1
                assert quotes[0].text == "Async Zitat hier"
            finally:
                os.unlink(tmp_path)

    def test_shutdown_executor(self):
        """Shutdown sollte den ThreadPool sauber beenden."""
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            mock_genai.Client.return_value = Mock()
            
            gemini = GeminiIntegration(api_key="test-key")
            # Sollte ohne Exception durchlaufen
            gemini.shutdown()

    def test_model_name(self):
        """Das Standard-Modell kommt aus settings.json (konfigurierbar)."""
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            mock_genai.Client.return_value = Mock()

            gemini = GeminiIntegration(api_key="test-key")
            assert gemini.model == "gemini-flash-lite-latest"

    def test_model_env_override(self):
        """GEMINI_MODEL in der Umgebung ueberschreibt das Standard-Modell."""
        from src import app_settings
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-custom-xyz"}):
            app_settings.load_settings(force_reload=True)
            with patch('src.gemini_integration.genai') as mock_genai:
                from src.gemini_integration import GeminiIntegration
                mock_genai.Client.return_value = Mock()
                gemini = GeminiIntegration(api_key="test-key")
                assert gemini.model == "gemini-custom-xyz"
        app_settings.load_settings(force_reload=True)

    def test_model_fallback_bei_ungueltiger_id(self):
        """Ungueltige Modell-ID -> Ersatz aus models.list() nach Praeferenz."""
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            mock_client = Mock()
            # models.get schlaegt fehl (404), models.list liefert Kandidaten
            mock_client.models.get.side_effect = Exception("404 not found")
            m1 = Mock(); m1.name = "models/gemini-2.5-flash-lite"
            m1.supported_actions = ["generateContent"]
            m2 = Mock(); m2.name = "models/gemini-2.5-flash"
            m2.supported_actions = ["generateContent"]
            mock_client.models.list.return_value = [m2, m1]
            mock_genai.Client.return_value = mock_client

            gemini = GeminiIntegration(api_key="test-key")
            resolved = gemini._ensure_model()
            # Praeferenz 'flash-lite' vor 'flash'
            assert "flash-lite" in resolved

    def test_validate_optimized_result_clamps_and_filters(self):
        """Das Ergebnis soll validiert, gecuttet und auf gueltige Werte reduziert werden."""
        with patch('src.gemini_integration.genai'):
            from src.gemini_integration import GeminiIntegration

            gemini = GeminiIntegration(api_key="test-key")
            current_params = {"pulse_intensity": 0.5, "bar_count": 40, "color_mode": "chroma"}
            param_specs = {
                "pulse_intensity": (0.5, 0.0, 1.0, 0.05),
                "bar_count": (40, 10, 100, 5),
            }
            colors = {"primary": "#111111", "secondary": "#222222", "background": "#333333"}

            optimized = {
                "params": {
                    "pulse_intensity": 99.0,  # muss gecuttet werden
                    "bar_count": 42,          # muss auf Step 5 gerundet werden
                    "unknown_param": 123,     # darf nicht im Ergebnis bleiben
                    "color_mode": "fixed",    # String-Parameter erlaubt
                },
                "colors": {
                    "primary": "#GGGGGG",   # ungueltig -> Fallback
                    "secondary": "#00FF00", # gueltig
                },
                "postprocess": {
                    "contrast": 5.0,        # muss gecuttet werden
                    "saturation": -1.0,     # muss gecuttet werden
                    "brightness": "bad",    # muss Fallback werden
                },
                "background": {
                    "opacity": 1.5,         # muss gecuttet werden
                },
            }

            result = gemini._validate_optimized_result(
                optimized, current_params, colors, param_specs
            )

            assert result["params"]["pulse_intensity"] == 1.0
            assert result["params"]["bar_count"] == 40
            assert "unknown_param" not in result["params"]
            assert result["params"]["color_mode"] == "fixed"
            assert result["colors"]["primary"] == colors["primary"]
            assert result["colors"]["secondary"] == "#00FF00"
            assert result["postprocess"]["contrast"] == 2.5
            assert result["postprocess"]["saturation"] == 0.3
            assert result["postprocess"]["brightness"] == 0.0
            assert result["background"]["opacity"] == 1.0

    def test_optimize_all_settings_ohne_response_schema(self):
        """Kein response_schema — die Gemini-API lehnt offene Maps ab.

        Frueher wurde hier ein Schema mit additionalProperties mitgeschickt.
        Die API antwortete darauf mit einem Fehler, jeder Aufruf landete im
        Fallback und der Nutzer bekam stillschweigend Standardwerte.
        """
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration

            mock_client = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps({
                "params": {"pulse_intensity": 0.6},
                "colors": {"primary": "#FF0055", "secondary": "#00CCFF", "background": "#0A0A0A"},
                "postprocess": {"contrast": 1.0, "saturation": 1.0, "brightness": 0.0,
                                "warmth": 0.0, "film_grain": 0.0},
                "background": {"opacity": 0.3, "blur": 0.0, "vignette": 0.0},
                "quotes": {},
            })
            mock_client.models.generate_content.return_value = mock_response
            mock_genai.Client.return_value = mock_client

            gemini = GeminiIntegration(api_key="test-key")
            result = gemini.optimize_all_settings(
                visualizer_type="pulsing_core",
                current_params={"pulse_intensity": 0.5},
                audio_features={"tempo": 120, "mode": "music", "rms_mean": 0.5},
                colors={"primary": "#FF0055", "secondary": "#00CCFF", "background": "#0A0A0A"},
                param_specs={"pulse_intensity": (0.5, 0.0, 1.0, 0.05)},
            )

            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            config = call_kwargs.get("config", {})
            assert config.get("temperature") == 0.2
            assert config.get("response_mime_type") == "application/json"
            assert "response_schema" not in config
            assert "system_instruction" in config
            assert result["params"]["pulse_intensity"] == pytest.approx(0.6, abs=0.001)

    def test_optimize_all_settings_category_fallback(self):
        """Fallback-Algorithmus soll kategorienbasierte Werte liefern."""
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration

            mock_client = Mock()
            mock_client.models.generate_content.side_effect = Exception("API error")
            mock_genai.Client.return_value = mock_client

            gemini = GeminiIntegration(api_key="test-key")
            # default.json umgehen, damit der interne Fallback getestet wird
            gemini._load_default_config = Mock(return_value={})

            audio_features = {
                "tempo": 140,
                "mode": "music",
                "rms_mean": 0.7,
                "rms_std": 0.15,
                "onset_mean": 0.4,
                "dynamic_range": 0.5,
                "brightness": 0.6,
                "voice_clarity_mean": 0.1,
            }
            param_specs = {
                "pulse_intensity": (0.5, 0.0, 1.0, 0.05),  # intensity
                "bar_count": (40, 10, 100, 5),              # count
                "flow_speed": (0.5, 0.1, 1.0, 0.05),        # speed
                "smoothing": (0.3, 0.0, 0.8, 0.05),         # reactivity
            }

            result = gemini.optimize_all_settings(
                visualizer_type="spectrum_bars",
                current_params={},
                audio_features=audio_features,
                colors={"primary": "#FF0055", "secondary": "#00CCFF", "background": "#0A0A0A"},
                param_specs=param_specs,
            )

            # Intensity sollte bei hoher Energie steigen
            assert result["params"]["pulse_intensity"] > 0.5
            # Count sollte bei hoher Energie steigen
            assert result["params"]["bar_count"] >= 40
            # Speed sollte bei hohem Tempo steigen
            assert result["params"]["flow_speed"] > 0.5
            # Werte muessen in den gueltigen Bereichen liegen
            assert 0.0 <= result["params"]["pulse_intensity"] <= 1.0
            assert 10 <= result["params"]["bar_count"] <= 100
            assert 0.1 <= result["params"]["flow_speed"] <= 1.0
            assert 0.0 <= result["params"]["smoothing"] <= 0.8

    def test_param_categorization_covers_known_params(self):
        """Bekannte Parameter muessen einer Kategorie zugeordnet sein."""
        with patch('src.gemini_integration.genai'):
            from src.gemini_integration import _get_param_category

            assert _get_param_category("pulse_intensity") == "intensity"
            assert _get_param_category("bar_count") == "count"
            assert _get_param_category("flow_speed") == "speed"
            assert _get_param_category("bar_width") == "size"
            assert _get_param_category("color_shift") == "color"
            assert _get_param_category("smoothing") == "reactivity"
            assert _get_param_category("viz_offset_x") == "transform"
            assert _get_param_category("background_color") == "special"
            assert _get_param_category("unknown_param") == "other"


class TestSegmentPipeline:
    """Zwei-Stufen-Extraktion: Segment-Transkript -> Auswahl auf Text."""

    def _gemini(self):
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            mock_genai.Client.return_value = Mock()
            return GeminiIntegration(api_key="test-key")

    def test_clean_segments_verwirft_eintraege_ohne_zeiten(self):
        from src.gemini_integration import GeminiIntegration
        raw = [
            {"start": 0.0, "end": 2.0, "text": "Erstes Segment"},
            {"text": "Ohne Zeiten"},
            {"start": 5.0, "text": "Nur Start"},
            {"start": 2.0, "end": 4.0, "text": "  "},
        ]
        cleaned = GeminiIntegration._clean_segments(raw)
        assert len(cleaned) == 1
        assert cleaned[0]["text"] == "Erstes Segment"

    def test_clean_segments_sortiert_und_begrenzt_auf_laenge(self):
        from src.gemini_integration import GeminiIntegration
        raw = [
            {"start": 8.0, "end": 12.0, "text": "Spaet"},
            {"start": 1.0, "end": 3.0, "text": "Frueh"},
            {"start": 20.0, "end": 22.0, "text": "Nach Ende des Audios"},
        ]
        cleaned = GeminiIntegration._clean_segments(raw, audio_duration=10.0)
        assert [c["text"] for c in cleaned] == ["Frueh", "Spaet"]
        assert cleaned[1]["end"] == 10.0

    def test_clean_segments_korrigiert_ungueltige_reihenfolge(self):
        from src.gemini_integration import GeminiIntegration
        cleaned = GeminiIntegration._clean_segments(
            [{"start": 5.0, "end": 3.0, "text": "Verdreht"}]
        )
        assert cleaned[0]["end"] > cleaned[0]["start"]

    def test_quotes_from_segments_rechnet_zeiten_lokal(self):
        gemini = self._gemini()
        segments = [
            {"start": 0.0, "end": 4.0, "text": "Hallo und willkommen zu dieser Folge."},
            {"start": 4.0, "end": 9.0,
             "text": "Das Fundament von diesem Schlamassel ist die Werbeoekonomie."},
        ]
        with patch.object(gemini, "_select_quotes_from_text", return_value=[
            {"text": "Das Fundament von diesem Schlamassel", "confidence": 0.9},
        ]):
            quotes = gemini._quotes_from_segments(segments, max_quotes=3)

        assert len(quotes) == 1
        # Zeit stammt aus dem zweiten Segment, nicht aus einer KI-Schaetzung
        assert 4.0 <= quotes[0].start_time < 6.0
        assert quotes[0].end_time <= 9.0

    def test_quotes_from_segments_verwirft_erfundene_zitate(self):
        gemini = self._gemini()
        segments = [{"start": 0.0, "end": 4.0, "text": "Hallo und willkommen."}]
        with patch.object(gemini, "_select_quotes_from_text", return_value=[
            {"text": "Diesen Satz hat niemand gesagt", "confidence": 0.95},
        ]):
            quotes = gemini._quotes_from_segments(segments, max_quotes=3)
        assert quotes == []

    def test_extract_quotes_faellt_ohne_segmente_auf_alten_pfad_zurueck(self):
        gemini = self._gemini()
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(b'dummy')
            tmp_path = tmp.name
        try:
            with patch.object(gemini, "transcribe_segments", return_value=[]), \
                 patch.object(gemini, "_upload_audio_with_retry", return_value=Mock()):
                response = Mock()
                response.text = ('[{"text": "Ein ganzer Satz hier", '
                                 '"start_time": 3.0, "end_time": 6.0, '
                                 '"confidence": 0.9}]')
                gemini.client.models.generate_content.return_value = response
                quotes = gemini.extract_quotes(tmp_path, max_quotes=2, use_cache=False)
            assert len(quotes) == 1
            assert quotes[0].start_time == 3.0
        finally:
            os.unlink(tmp_path)

    def test_finalize_quotes_rastet_mit_features_ein(self):
        import numpy as np
        from src.types import AudioFeatures, Quote

        gemini = self._gemini()
        fps = 30
        n = 20 * fps
        rms = np.full(n, 0.02, dtype=np.float32)
        rms[int(4.0 * fps):int(7.0 * fps)] = 0.8
        features = AudioFeatures(
            duration=20.0, sample_rate=22050, fps=fps, frame_count=n,
            rms=rms, onset=np.zeros(n), spectral_centroid=np.zeros(n),
            spectral_rolloff=np.zeros(n), zero_crossing_rate=np.zeros(n),
            chroma=np.zeros((12, n)), mfcc=np.zeros((13, n)),
            tempogram=np.zeros((10, n)), tempo=120.0, mode="speech",
        )
        quotes = [Quote(text="Test", start_time=4.4, end_time=7.4, confidence=0.9)]
        result = gemini._finalize_quotes(quotes, max_quotes=5, features=features)
        assert result[0].start_time == pytest.approx(3.88, abs=0.06)
        assert result[0].end_time == pytest.approx(7.0, abs=0.06)

    def test_window_is_plausible(self):
        from src.gemini_integration import GeminiIntegration as G
        text = "Das Netz basiert aktuell auf Masse statt Klasse"  # 8 Woerter
        assert G._window_is_plausible(text, 10.0, 13.0)
        # Entartetes Fenster (Segment-Zeiten am Audio-Ende zusammengefallen)
        assert not G._window_is_plausible(text, 90.01, 90.03)
        # Viel zu lang fuer die Textmenge
        assert not G._window_is_plausible(text, 10.0, 40.0)
        # Zu schnell gesprochen, um wahr zu sein
        assert not G._window_is_plausible(text, 10.0, 10.9)

    def test_quotes_from_segments_verwirft_entartete_fenster(self):
        gemini = self._gemini()
        segments = [
            {"start": 89.9, "end": 90.0, "text": "Du holst dir genau den Liter Milch."},
        ]
        with patch.object(gemini, "_select_quotes_from_text", return_value=[
            {"text": "Du holst dir genau den Liter Milch", "confidence": 0.9},
        ]):
            quotes = gemini._quotes_from_segments(segments, max_quotes=3,
                                                  audio_duration=90.0)
        assert quotes == []


class TestOptimizeResultValidation:
    """Der Optimierer muss die Antwort auch dann verstehen, wenn das Modell
    die Bloecke anders benennt — sonst landet das Ergebnis lautlos im Nichts."""

    def _gemini(self):
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            mock_genai.Client.return_value = Mock()
            return GeminiIntegration(api_key="test-key")

    def test_akzeptiert_alternative_schluesselnamen(self):
        gemini = self._gemini()
        specs = {"num_petals": (8, 3, 16, 1), "brightness": (1.0, 0.5, 2.0, 0.05)}
        raw = {
            "parameters": {"num_petals": 6, "brightness": 1.2},
            "post_process": {"contrast": 1.2},
            "colour": {"primary": "#112233"},
        }
        result = gemini._validate_optimized_result(raw, {}, {}, specs)
        assert result["params"]["num_petals"] == 6
        assert result["postprocess"]["contrast"] == pytest.approx(1.2)
        assert result["colors"]["primary"] == "#112233"

    def test_clamped_auf_die_spec_grenzen(self):
        gemini = self._gemini()
        specs = {"num_petals": (8, 3, 16, 1)}
        result = gemini._validate_optimized_result({"params": {"num_petals": 99}},
                                                   {}, {}, specs)
        assert result["params"]["num_petals"] == 16

    def test_blur_darf_ueber_eins_gehen(self):
        gemini = self._gemini()
        raw = {"background": {"blur": 12.0, "opacity": 0.8, "vignette": 0.4}}
        result = gemini._validate_optimized_result(raw, {}, {}, {})
        assert result["background"]["blur"] == pytest.approx(12.0)
        assert result["background"]["opacity"] == pytest.approx(0.8)

    def test_unbekannte_parameter_werden_verworfen(self):
        gemini = self._gemini()
        specs = {"num_petals": (8, 3, 16, 1)}
        result = gemini._validate_optimized_result(
            {"params": {"gibt_es_nicht": 5}}, {}, {}, specs)
        assert result["params"] == {}
