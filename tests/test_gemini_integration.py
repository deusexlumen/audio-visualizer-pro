"""
Tests für Gemini Integration.

Alle Tests nutzen Mocks – keine echten API-Calls!
"""

import os
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
            assert gemini.model == "gemini-3.1-flash-lite"

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
        """Das Standard-Modell soll gemini-3.1-flash-lite sein."""
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            mock_genai.Client.return_value = Mock()

            gemini = GeminiIntegration(api_key="test-key")
            assert gemini.model == "gemini-3.1-flash-lite"

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
