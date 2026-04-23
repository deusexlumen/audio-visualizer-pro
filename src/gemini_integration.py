"""
Gemini Integration für Audio Visualizer Pro.

Nutzt Gemini 3.1 Flash-Lite für:
- Audio-Transkription
- Key-Zitat-Extraktion direkt aus Audio (mit Zeitstempeln)
"""

import os
import json
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None


@dataclass
class Quote:
    """Ein Key-Zitat mit Zeitstempel und Konfidenz."""
    text: str
    start_time: float  # Sekunden
    end_time: float    # Sekunden
    confidence: float  # 0.0-1.0


class GeminiIntegration:
    """
    Wrapper für Gemini 3.1 Flash-Lite API.
    """

    def __init__(self, api_key: Optional[str] = None):
        if genai is None:
            raise ImportError(
                "google-genai ist nicht installiert. "
                "Bitte 'pip install google-genai' ausführen."
            )

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API Key fehlt. "
                "Bitte als Parameter übergeben oder als GEMINI_API_KEY "
                "Umgebungsvariable setzen."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-3.1-flash-lite-preview"

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transkribiert eine Audio-Datei mit Gemini.

        Args:
            audio_path: Pfad zur Audio-Datei

        Returns:
            Der transkribierte Text
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio nicht gefunden: {audio_path}")

            try:
                uploaded_file = self.client.files.upload(file=str(audio_path))
            except Exception as e:
                raise RuntimeError(f"Audio-Upload zu Gemini fehlgeschlagen: {e}") from e

            prompt = (
                "Erstelle ein genaues Transkript dieses Audios. "
                "Gib nur den gesprochenen Text aus, keine zusätzlichen Kommentare."
            )

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt, uploaded_file]
                )
            except Exception as e:
                raise RuntimeError(f"Transkription durch Gemini fehlgeschlagen: {e}") from e

            return response.text.strip()
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Unerwarteter Fehler bei der Transkription: {e}") from e

    def extract_quotes(self, audio_path: str, max_quotes: int = 5) -> List[Quote]:
        """
        Extrahiert Key-Zitate direkt aus einer Audio-Datei.

        Nutzt Gemini's Audio-Verarbeitung für Zeitstempel-Schätzung.
        Die Zeitstempel sind Schätzungen (±5 Sekunden).

        Args:
            audio_path: Pfad zur Audio-Datei
            max_quotes: Maximale Anzahl an Zitaten (default: 5)

        Returns:
            Liste von Quote-Objekten, sortiert nach Startzeit
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio nicht gefunden: {audio_path}")

            try:
                uploaded_file = self.client.files.upload(file=str(audio_path))
            except Exception as e:
                raise RuntimeError(f"Audio-Upload zu Gemini fehlgeschlagen: {e}") from e

            prompt = f"""
            Analysiere dieses Podcast-Audio und extrahiere die {max_quotes} besten Key-Zitate.

            Ein "Key-Zitat" ist:
            - Ein besonders prägnanter, emotionaler oder witziger Satz
            - Eine Aussage, die den Kern einer Idee zusammenfasst
            - Etwas, das man sich merken möchte

            Gib das Ergebnis als JSON-Array zurück. Jedes Element hat diese Felder:
            - "text": Der Zitat-Text (max. 15 Wörter)
            - "start_time": Geschätzte Startzeit in Sekunden (float)
            - "end_time": Geschätzte Endzeit in Sekunden (float)
            - "confidence": Wie gut das Zitat ist, von 0.0 bis 1.0 (float)

            Wichtig: Die Zeitangaben müssen realistisch sein. Ein typisches Zitat dauert 3-8 Sekunden.
            """

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt, uploaded_file],
                    config={
                        "response_mime_type": "application/json",
                    }
                )
            except Exception as e:
                raise RuntimeError(f"Zitat-Extraktion durch Gemini fehlgeschlagen: {e}") from e

            quotes_data = self._parse_json_response(response.text)

            quotes = []
            for q in quotes_data:
                quotes.append(Quote(
                    text=str(q.get("text", "")).strip(),
                    start_time=float(q.get("start_time", 0.0)),
                    end_time=float(q.get("end_time", 0.0)),
                    confidence=float(q.get("confidence", 0.5))
                ))

            return sorted(quotes, key=lambda x: x.start_time)
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Unerwarteter Fehler bei der Zitat-Extraktion: {e}") from e

    def optimize_visualizer_params(self, visualizer_type: str, current_params: dict,
                                   audio_features: dict, user_prompt: str = None) -> dict:
        """
        Nutzt Gemini, um Visualizer-Parameter basierend auf Audio-Analyse zu optimieren.
        
        Args:
            visualizer_type: Name des Visualizers (z.B. 'pulsing_core')
            current_params: Aktuelle Parameter des Users
            audio_features: Dictionary mit Audio-Features (tempo, mode, rms_mean, etc.)
        
        Returns:
            Dictionary mit optimierten Parametern
        """
        try:
            prompt = f"""
            Du bist ein professioneller Motion-Graphics-Designer fuer Musikvideos.
            
            AUDIO-ANALYSE:
            - Dauer: {audio_features.get('duration', 0):.1f}s
            - Tempo: {audio_features.get('tempo', 120):.0f} BPM
            - Modus: {audio_features.get('mode', 'music')}
            - Durchschnittliche Lautstaerke (RMS): {audio_features.get('rms_mean', 0.5):.2f}
            - Beat-Staerke (Onset): {audio_features.get('onset_mean', 0.3):.2f}
            - Dominante Frequenz: {audio_features.get('spectral_mean', 0.5):.2f}
            
            VISUALIZER: {visualizer_type}
            AKTUELLE PARAMETER: {json.dumps(current_params, indent=2)}
            
            GIB DIE OPTIMALEN PARAMETER ZURUECK als JSON-Objekt.
            
            Regeln:
            - Aggressives Musik (hohes Tempo, hoher RMS) -> Hohe Intensitaet, schnelles Easing, viele Partikel
            - Ruhiger Podcast (niedriges Tempo, niedriger RMS) -> Sanfte Werte, langsames Easing, weniger Partikel
            - Hybrid -> Ausgewogene mittlere Werte
            - Halte dich an die Wertebereiche der aktuellen Parameter
            - Antworte NUR mit JSON, keine Erklaerungen
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                }
            )
            
            optimized = self._parse_json_response(response.text)
            
            # Validiere und filtere nur bekannte Parameter
            if isinstance(optimized, dict):
                return {k: v for k, v in optimized.items() if k in current_params}
            return current_params
            
        except Exception as e:
            print(f"[Gemini] Parameter-Optimierung fehlgeschlagen: {e}")
            return current_params

    def generate_background_prompt(self, audio_features: dict) -> str:
        """
        Generiert einen Bildgenerierungs-Prompt basierend auf Audio-Analyse.
        
        Args:
            audio_features: Dictionary mit Audio-Features (tempo, mode, rms_mean, etc.)
            
        Returns:
            Englischer Prompt fuer Midjourney/DALL-E/Stable Diffusion
        """
        try:
            mood = "energetic and intense" if audio_features.get('tempo', 120) > 120 else "calm and atmospheric"
            if audio_features.get('mode') == 'speech':
                mood = "minimal and focused"
            elif audio_features.get('mode') == 'hybrid':
                mood = "dynamic and balanced"
                
            rms = audio_features.get('rms_mean', 0.5)
            if rms > 0.6:
                intensity = "high intensity, bold colors"
            elif rms > 0.3:
                intensity = "medium intensity, balanced colors"
            else:
                intensity = "soft, muted tones"
            
            prompt = f"""
            Du bist ein Prompt-Engineer fuer KI-Bildgenerierung (Midjourney, DALL-E, Stable Diffusion).
            
            AUDIO-ANALYSE:
            - Dauer: {audio_features.get('duration', 0):.1f}s
            - Tempo: {audio_features.get('tempo', 120):.0f} BPM
            - Modus: {audio_features.get('mode', 'music')}
            - Durchschnittliche Lautstaerke (RMS): {rms:.2f}
            - Beat-Staerke (Onset): {audio_features.get('onset_mean', 0.3):.2f}
            - Dominante Frequenz: {audio_features.get('spectral_mean', 0.5):.2f}
            
            STIMMUNG: {mood}
            INTENSITAET: {intensity}
            
            Aufgabe: Erstelle einen detaillierten, englischen Prompt fuer ein Hintergrundbild,
            das perfekt zu diesem Audio passt. Beschreibe:
            - Farbpalette (konkrete Farben)
            - Stimmung/Atmosphaere
            - Stil (z.B. cinematic, abstract, minimal, photorealistic)
            - Wichtige visuelle Elemente
            - Licht und Schatten
            
            Antworte NUR mit dem Prompt-Text (auf Englisch), keine Erklaerungen.
            Maximal 80 Woerter. Keine Anfuehrungszeichen am Anfang/Ende.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt]
            )
            
            return response.text.strip().strip('"').strip("'")
            
        except Exception as e:
            print(f"[Gemini] Bild-Prompt-Generierung fehlgeschlagen: {e}")
            return "abstract ambient background with soft gradients and atmospheric lighting, cinematic color grading, minimal composition, 8k quality"

    @staticmethod
    def _parse_json_response(text: str):
        """Hilfsmethode: Extrahiert JSON aus der API-Antwort."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Fallback: Versuche JSON aus Markdown-Code-Blöcken zu extrahieren
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    json_str = text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = text
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError, ValueError):
                return []
