"""
Tests für Gemini Integration.

Alle Tests nutzen Mocks – keine echten API-Calls!
"""

import os
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
            assert gemini.model == "gemini-3.1-flash-lite-preview"

    def test_transcribe_audio(self):
        """Sollte Audio transkribieren und Text zurückgeben."""
        import tempfile
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(b'dummy audio data')
                tmp_path = tmp.name
            
            try:
                mock_client = Mock()
                mock_file = Mock()
                mock_client.files.upload.return_value = mock_file
                
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
                mock_client = Mock()
                mock_file = Mock()
                mock_client.files.upload.return_value = mock_file
                
                mock_response = Mock()
                mock_response.text = '[{"text": "Hello world", "start_time": 10.5, "end_time": 15.2, "confidence": 0.9}]'
                mock_client.models.generate_content.return_value = mock_response
                
                mock_genai.Client.return_value = mock_client
                
                gemini = GeminiIntegration(api_key="test-key")
                quotes = gemini.extract_quotes(tmp_path, max_quotes=1)
                
                assert len(quotes) == 1
                assert quotes[0].text == "Hello world"
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
                mock_client = Mock()
                mock_client.files.upload.return_value = Mock()
                
                mock_response = Mock()
                mock_response.text = '[{"text": "Zweites", "start_time": 20.0, "end_time": 25.0, "confidence": 0.8}, {"text": "Erstes", "start_time": 5.0, "end_time": 10.0, "confidence": 0.9}]'
                mock_client.models.generate_content.return_value = mock_response
                
                mock_genai.Client.return_value = mock_client
                
                gemini = GeminiIntegration(api_key="test-key")
                quotes = gemini.extract_quotes(tmp_path)
                
                assert quotes[0].text == "Erstes"
                assert quotes[1].text == "Zweites"
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
