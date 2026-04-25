"""
Gemini Integration für Audio Visualizer Pro.

Nutzt Gemini 3.1 Flash-Lite für:
- Audio-Transkription
- Key-Zitat-Extraktion direkt aus Audio (mit Zeitstempeln)
"""

import os
import json
import time
import subprocess
import tempfile
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None


def _compress_audio_for_upload(input_path: str, output_path: str) -> bool:
    """
    Komprimiert Audio fuer Gemini-Upload.
    Mono, 16kHz, niedrige Bitrate = deutlich kleinere Datei.
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000",      # 16kHz Sample-Rate (genug fuer Sprache)
            "-ac", "1",          # Mono
            "-b:a", "32k",       # 32 kbps Bitrate
            "-f", "mp3",         # MP3 Format
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


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

    def _upload_audio_with_retry(self, audio_path: str, max_retries: int = 3):
        """
        Laedt Audio zu Gemini hoch mit Retry-Logik, Komprimierung und Warten auf ACTIVE.
        
        Args:
            audio_path: Pfad zur Audio-Datei
            max_retries: Maximale Anzahl Upload-Versuche
            
        Returns:
            Das hochgeladene File-Objekt (Status ACTIVE)
        """
        audio_path = Path(audio_path)
        original_size = audio_path.stat().st_size / (1024 * 1024)  # MB
        
        # Wenn Datei > 5MB, vorher komprimieren
        upload_path = str(audio_path)
        temp_compressed = None
        
        if original_size > 5:
            temp_compressed = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_compressed.close()
            if _compress_audio_for_upload(str(audio_path), temp_compressed.name):
                compressed_size = os.path.getsize(temp_compressed.name) / (1024 * 1024)
                print(f"[Gemini] Audio komprimiert: {original_size:.1f}MB -> {compressed_size:.1f}MB")
                upload_path = temp_compressed.name
            else:
                print(f"[Gemini] Komprimierung fehlgeschlagen, verwende Original ({original_size:.1f}MB)")
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                uploaded_file = self.client.files.upload(file=upload_path)
                
                # WICHTIG: Auf ACTIVE warten! Sonst bricht Gemini die Verbindung ab.
                max_wait = 60  # Max 60 Sekunden warten
                waited = 0
                while getattr(uploaded_file.state, 'name', str(uploaded_file.state)) == "PROCESSING" and waited < max_wait:
                    time.sleep(2)
                    waited += 2
                    uploaded_file = self.client.files.get(name=uploaded_file.name)
                    state_name = getattr(uploaded_file.state, 'name', str(uploaded_file.state))
                    print(f"[Gemini] Datei-Status: {state_name} ({waited}s)")
                
                state_name = getattr(uploaded_file.state, 'name', str(uploaded_file.state))
                if state_name != "ACTIVE":
                    raise RuntimeError(f"Datei nicht ACTIVE nach Upload: {state_name}")
                
                # Cleanup temp file
                if temp_compressed and os.path.exists(temp_compressed.name):
                    os.unlink(temp_compressed.name)
                return uploaded_file
            except Exception as e:
                last_error = e
                print(f"[Gemini] Upload Versuch {attempt}/{max_retries} fehlgeschlagen: {e}")
                if attempt < max_retries:
                    wait = attempt * 2  # Exponential Backoff: 2s, 4s, 6s
                    print(f"[Gemini] Warte {wait}s vor naechstem Versuch...")
                    time.sleep(wait)
        
        # Cleanup temp file bei Fehler
        if temp_compressed and os.path.exists(temp_compressed.name):
            os.unlink(temp_compressed.name)
        
        raise RuntimeError(f"Audio-Upload zu Gemini fehlgeschlagen nach {max_retries} Versuchen: {last_error}")

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

            uploaded_file = self._upload_audio_with_retry(str(audio_path))

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

    def optimize_all_settings(self, visualizer_type: str, current_params: dict,
                              audio_features: dict, colors: dict,
                              user_prompt: str = None) -> dict:
        """
        Nutzt Gemini, um ALLE Einstellungen basierend auf Audio-Analyse zu optimieren.
        
        Gibt ein umfassendes Dictionary zurueck mit:
        - params: Visualizer-Parameter
        - colors: {primary, secondary, background}
        - postprocess: {contrast, saturation, brightness, warmth, film_grain}
        - background: {opacity, blur, vignette}
        - quotes: {font_size, box_color, font_color, position, display_duration,
                   auto_scale_font, slide_animation, slide_out_animation,
                   scale_in, typewriter, glow_pulse}
        
        Args:
            visualizer_type: Name des Visualizers
            current_params: Aktuelle Visualizer-Parameter
            audio_features: Dictionary mit Audio-Features
            colors: Aktuelle Farbpalette
            user_prompt: Optionaler User-Wunsch
        
        Returns:
            Dictionary mit ALLEN optimierten Einstellungen (Params, Colors, Post-Process,
            Background UND Quote-Overlay)
        """
        try:
            prompt = f"""
            Du bist ein professioneller Motion-Graphics-Designer und Color-Grading-Experte fuer Musikvideos.
            
            AUDIO-ANALYSE:
            - Dauer: {audio_features.get('duration', 0):.1f}s
            - Tempo: {audio_features.get('tempo', 120):.0f} BPM
            - Modus: {audio_features.get('mode', 'music')}
            - Durchschnittliche Lautstaerke (RMS): {audio_features.get('rms_mean', 0.5):.2f}
            - Beat-Staerke (Onset): {audio_features.get('onset_mean', 0.3):.2f}
            - Dominante Frequenz: {audio_features.get('spectral_mean', 0.5):.2f}
            
            VISUALIZER: {visualizer_type}
            AKTUELLE PARAMETER: {json.dumps(current_params, indent=2)}
            AKTUELLE FARBEN: {json.dumps(colors, indent=2)}
            
            GIB ALLE EINSTELLUNGEN ZURUECK als JSON-Objekt mit dieser Struktur:
            {{
              "params": {{...}},
              "colors": {{"primary": "#...", "secondary": "#...", "background": "#..."}},
              "postprocess": {{"contrast": 1.0, "saturation": 1.0, "brightness": 0.0, "warmth": 0.0, "film_grain": 0.0}},
              "background": {{"opacity": 0.3, "blur": 0.0, "vignette": 0.0}},
              "quotes": {{
                "font_size": 52,
                "box_color": "#1a1a2e",
                "font_color": "#FFFFFF",
                "position": "bottom",
                "display_duration": 8.0,
                "auto_scale_font": true,
                "slide_animation": "none",
                "slide_out_animation": "none",
                "scale_in": false,
                "typewriter": false,
                "glow_pulse": false,
                "box_padding": 32,
                "box_radius": 16,
                "box_margin_bottom": 100,
                "max_width_ratio": 0.75,
                "fade_duration": 0.6,
                "line_spacing": 1.35,
                "max_font_size": 72,
                "max_chars_per_line": 40
              }}
            }}
            
            Regeln fuer Parameter:
            - Aggressives Musik (hohes Tempo, hoher RMS) -> Hohe Intensitaet, schnelles Easing, viele Partikel
            - Ruhiger Podcast (niedriges Tempo, niedriger RMS) -> Sanfte Werte, langsames Easing, weniger Partikel
            - Halte dich an die Wertebereiche der aktuellen Parameter
            
            Regeln fuer Farben:
            - Musik: Klassisches Color-Grading (warmes Orange/Teal, oder kuehles Cyan/Magenta)
            - Podcast: Gedämpft, professionell (Blau/Grau-Töne)
            - Aggressiv: Sättigte, kontrastreiche Farben
            - Ruhig: Pastell, muted tones
            
            Regeln fuer Post-Process:
            - Cinematic Warm: contrast 1.15, saturation 1.05, brightness -0.02, warmth 0.25, film_grain 0.15
            - Cyberpunk Cold: contrast 1.25, saturation 1.3, brightness -0.05, warmth -0.35, film_grain 0.1
            - Vintage Film: contrast 0.9, saturation 0.75, brightness 0.02, warmth 0.4, film_grain 0.35
            - Noir: contrast 1.4, saturation 0.0, brightness -0.08, warmth 0.0, film_grain 0.4
            - Golden Hour: contrast 1.1, saturation 1.1, brightness 0.03, warmth 0.5, film_grain 0.1
            - Concert Neon: contrast 1.2, saturation 1.4, brightness -0.03, warmth 0.0, film_grain 0.05
            - Neutral: contrast 1.0, saturation 1.0, brightness 0.0, warmth 0.0, film_grain 0.0
            
            Regeln fuer Hintergrund:
            - opacity: 0.0-1.0 (0 = kein Bild, 1 = nur Bild)
            - blur: 0.0-20.0 (hoher Blur fuer sanften Hintergrund)
            - vignette: 0.0-1.0 (hoher Wert fuer mehr Randabdunkelung)
            
            Regeln fuer Quote-Overlays (Text-Einblendungen):
            - Podcast/News: Grosse Schrift (52-64px), dezente Box (#1a1a2e), Position bottom,
              kein Slide/Scale/Typewriter (false), keine Glow-Pulse (false), Auto-Skalierung an
            - Musik/Energie: Mittlere Schrift (40-48px), dunkle Box, Position center,
              Slide-In up erlaubt, Scale-In erlaubt, Glow-Pulse erlaubt
            - Ruhig/Meditation: Sehr grosse Schrift (56-72px), transparente Box, Position center,
              kein Slide, kein Scale, Fade only
            - Storytelling: Mittlere Schrift (48-52px), dunkle Box, Position bottom,
              Typewriter-Effekt erlaubt fuer Dramatik
            - Wertebereiche:
              font_size: 12-96
              box_color: Hex-Farbe (z.B. #1a1a2e)
              font_color: Hex-Farbe (z.B. #FFFFFF)
              position: "bottom", "center", "top"
              display_duration: 2.0-20.0
              auto_scale_font: true/false
              slide_animation: "none", "up", "down", "left", "right"
              slide_out_animation: "none", "up", "down", "left", "right"
              scale_in: true/false
              typewriter: true/false
              glow_pulse: true/false
              box_padding: 0-80 (Innenabstand der Box)
              box_radius: 0-50 (Eckenradius in Pixeln, 0 = scharfe Ecken)
              box_margin_bottom: 20-300 (Abstand vom Rand bei Position "bottom")
              max_width_ratio: 0.3-1.0 (Max. Breite als Anteil der Bildbreite)
              fade_duration: 0.1-2.0 (Sekunden fuer Ein-/Ausblendung)
              line_spacing: 0.8-2.0 (Zeilenabstand, 1.0 = normal, 1.5 = weit)
              max_font_size: 20-96 (Obergrenze bei Auto-Skalierung)
              max_chars_per_line: 20-80 (Max. Zeichen pro Zeile vor Umbruch)
            
            Werte-Bereiche Post-Process:
            - contrast: 0.5 - 2.0
            - saturation: 0.0 - 2.0
            - brightness: -0.5 - 0.5
            - warmth: -1.0 - 1.0
            - film_grain: 0.0 - 1.0
            - opacity: 0.0 - 1.0
            - blur: 0.0 - 20.0
            - vignette: 0.0 - 1.0
            
            Antworte NUR mit JSON, keine Erklaerungen.
            """
            
            if user_prompt:
                prompt += f"\nZusaetzlicher User-Wunsch: {user_prompt}\n"
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                }
            )
            
            optimized = self._parse_json_response(response.text)
            
            default_quotes = {
                "font_size": 52,
                "box_color": "#1a1a2e",
                "font_color": "#FFFFFF",
                "position": "bottom",
                "display_duration": 8.0,
                "auto_scale_font": True,
                "slide_animation": "none",
                "slide_out_animation": "none",
                "scale_in": False,
                "typewriter": False,
                "glow_pulse": False,
                "box_padding": 32,
                "box_radius": 16,
                "box_margin_bottom": 100,
                "max_width_ratio": 0.75,
                "fade_duration": 0.6,
                "line_spacing": 1.35,
                "max_font_size": 72,
                "max_chars_per_line": 40,
            }
            
            if not isinstance(optimized, dict):
                return {"params": current_params, "colors": colors,
                        "postprocess": {"contrast": 1.0, "saturation": 1.0, "brightness": 0.0,
                                        "warmth": 0.0, "film_grain": 0.0},
                        "background": {"opacity": 0.3, "blur": 0.0, "vignette": 0.0},
                        "quotes": default_quotes}
            
            # Validiere und filtere Parameter
            result = {
                "params": {k: v for k, v in optimized.get("params", {}).items() if k in current_params} if optimized.get("params") else current_params,
                "colors": optimized.get("colors", colors),
                "postprocess": optimized.get("postprocess", {"contrast": 1.0, "saturation": 1.0, "brightness": 0.0,
                                                               "warmth": 0.0, "film_grain": 0.0}),
                "background": optimized.get("background", {"opacity": 0.3, "blur": 0.0, "vignette": 0.0}),
                "quotes": {**default_quotes, **(optimized.get("quotes", {}) or {})},
            }
            
            # Stelle sicher, dass alle erwarteten Keys existieren
            for k in ["contrast", "saturation", "brightness", "warmth", "film_grain"]:
                if k not in result["postprocess"]:
                    result["postprocess"][k] = {"contrast": 1.0, "saturation": 1.0, "brightness": 0.0,
                                                "warmth": 0.0, "film_grain": 0.0}[k]
            for k in ["opacity", "blur", "vignette"]:
                if k not in result["background"]:
                    result["background"][k] = {"opacity": 0.3, "blur": 0.0, "vignette": 0.0}[k]
            
            return result
            
        except Exception as e:
            print(f"[Gemini] All-Settings-Optimierung fehlgeschlagen: {e}")
            return {"params": current_params, "colors": colors,
                    "postprocess": {"contrast": 1.0, "saturation": 1.0, "brightness": 0.0,
                                    "warmth": 0.0, "film_grain": 0.0},
                    "background": {"opacity": 0.3, "blur": 0.0, "vignette": 0.0},
                    "quotes": {
                        "font_size": 52, "box_color": "#1a1a2e", "font_color": "#FFFFFF",
                        "position": "bottom", "display_duration": 8.0,
                        "auto_scale_font": True, "slide_animation": "none",
                        "slide_out_animation": "none", "scale_in": False,
                        "typewriter": False, "glow_pulse": False,
                        "box_padding": 32, "box_radius": 16, "box_margin_bottom": 100,
                        "max_width_ratio": 0.75, "fade_duration": 0.6,
                        "line_spacing": 1.35, "max_font_size": 72,
                        "max_chars_per_line": 40,
                    }}

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
