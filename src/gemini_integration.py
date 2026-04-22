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

    def extract_quotes(self, audio_path: str, audio_duration: float = None,
                        max_quotes: int = None) -> List[Quote]:
        """
        Extrahiert Key-Zitate direkt aus einer Audio-Datei.

        Smartes Verhalten:
        - Anzahl Zitate passt sich der Audio-Dauer an (nicht starr 5)
        - Confidence-Filter: nur Zitate mit confidence >= 0.6
        - Mindestlaenge: mindestens 3 Woerter

        Args:
            audio_path: Pfad zur Audio-Datei
            audio_duration: Dauer des Audios in Sekunden (fuer dynamische Anzahl)
            max_quotes: Maximale Anzahl an Zitaten (None = automatisch aus Dauer)

        Returns:
            Liste von Quote-Objekten, sortiert nach Startzeit
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio nicht gefunden: {audio_path}")

            # Dynamische Anzahl basierend auf Dauer
            if max_quotes is None and audio_duration is not None:
                if audio_duration < 60:
                    max_quotes = 2
                elif audio_duration < 180:
                    max_quotes = 3
                elif audio_duration < 300:
                    max_quotes = 4
                elif audio_duration < 600:
                    max_quotes = 6
                else:
                    max_quotes = min(10, max(5, int(audio_duration / 90)))
            elif max_quotes is None:
                max_quotes = 5

            try:
                uploaded_file = self._upload_audio_with_retry(str(audio_path))
            except Exception as e:
                raise RuntimeError(f"Audio-Upload zu Gemini fehlgeschlagen: {e}") from e

            prompt = f"""
            Analysiere dieses Audio und extrahiere ALLE wichtigen Key-Zitate.

            Ein "Key-Zitat" ist:
            - Ein besonders praegnanter, emotionaler oder witziger Satz
            - Eine Aussage, die den Kern einer Idee zusammenfasst
            - Etwas, das man sich merken moechte
            - KEINE banalen Floskeln wie "Also", "Ja genau", "Stimmt"

            Filtere STRENG:
            - Extrahiere nur wirklich starke Zitate (confidence > 0.6)
            - Wenn das Audio nur 1-2 gute Zitate hat, gib nur die zurueck
            - Wenn es 8 gute Zitate hat, gib alle 8 zurueck
            - Qualitaet > Quantitaet

            Gib das Ergebnis als JSON-Array zurueck. Jedes Element hat diese Felder:
            - "text": Der Zitat-Text (max. 15 Woerter, konzentriert und praegnant)
            - "start_time": Geschaezte Startzeit in Sekunden (float)
            - "end_time": Geschaezte Endzeit in Sekunden (float)
            - "confidence": Wie gut das Zitat ist, von 0.0 bis 1.0 (float)

            Wichtig: Die Zeitangaben muessen realistisch sein. Ein typisches Zitat dauert 3-8 Sekunden.
            """

            # Retry mit Exponential Backoff bei 503/UEBERLASTET
            last_error = None
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[prompt, uploaded_file],
                        config={
                            "response_mime_type": "application/json",
                        }
                    )
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    # 503 oder "high demand" oder "unavailable" -> retry
                    if "503" in str(e) or "unavailable" in error_str or "high demand" in error_str:
                        wait_time = 2 * (attempt + 1)
                        print(f"[Gemini] 503 Ueberlastet, Versuch {attempt + 1}/3. Warte {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    # Andere Fehler sofort weitergeben
                    raise RuntimeError(f"Zitat-Extraktion durch Gemini fehlgeschlagen: {e}") from e
            else:
                # Alle 3 Versuche fehlgeschlagen
                raise RuntimeError(
                    f"Gemini ist nach 3 Versuchen weiterhin nicht erreichbar (503 UNAVAILABLE). "
                    f"Bitte versuche es in ein paar Minuten erneut oder fuege Zitate manuell hinzu."
                )

            quotes_data = self._parse_json_response(response.text)

            quotes = []
            for q in quotes_data:
                text = str(q.get("text", "")).strip()
                # Mindestlaenge: 3 Woerter
                if len(text.split()) < 3:
                    continue
                quotes.append(Quote(
                    text=text,
                    start_time=float(q.get("start_time", 0.0)),
                    end_time=float(q.get("end_time", 0.0)),
                    confidence=float(q.get("confidence", 0.5))
                ))

            # Confidence-Filter: nur Zitate >= 0.6
            quotes = [q for q in quotes if q.confidence >= 0.6]

            # Nach Confidence sortieren (beste zuerst)
            quotes = sorted(quotes, key=lambda x: x.confidence, reverse=True)

            # Auf max_quotes begrenzen
            quotes = quotes[:max_quotes]

            # Nach Startzeit sortieren fuer finale Ausgabe
            quotes = sorted(quotes, key=lambda x: x.start_time)

            return quotes
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
                              param_specs: dict = None,
                              user_prompt: str = None) -> dict:
        """
        Nutzt Gemini, um ALLE Einstellungen basierend auf Audio-Analyse zu optimieren.
        
        NEU: Param-Spezifikationen (min/max/step) werden mitgegeben, damit die KI
        die gueltigen Bereiche kennt. Falls die KI nicht antwortet, gibt es einen
        deterministischen Fallback-Algorithmus.
        
        Gibt ein umfassendes Dictionary zurueck mit:
        - params: Visualizer-Parameter (geclamped auf min/max)
        - colors: {primary, secondary, background}
        - postprocess: {contrast, saturation, brightness, warmth, film_grain}
        - background: {opacity, blur, vignette}
        - quotes: {...}
        
        Args:
            visualizer_type: Name des Visualizers
            current_params: Aktuelle Visualizer-Parameter
            audio_features: Dictionary mit Audio-Features
            colors: Aktuelle Farbpalette
            param_specs: {name: (default, min, max, step)} fuer gueltige Bereiche
            user_prompt: Optionaler User-Wunsch
        
        Returns:
            Dictionary mit ALLEN optimierten Einstellungen
        """
        # Fallback: deterministische Parameter-Berechnung
        def _fallback_params():
            """Berechnet Parameter deterministisch aus Audio-Features."""
            tempo = audio_features.get('tempo', 120)
            mode = audio_features.get('mode', 'music')
            rms_mean = audio_features.get('rms_mean', 0.5)
            rms_std = audio_features.get('rms_std', 0.1)
            onset_mean = audio_features.get('onset_mean', 0.3)
            
            result = {}
            if param_specs:
                for name, (default, min_val, max_val, step) in param_specs.items():
                    if 'intensity' in name or 'strength' in name or 'scale' in name or 'impact' in name:
                        # Dynamik-basiert: hohe RMS-Std = hohe Intensitaet
                        val = min_val + (max_val - min_val) * min(1.0, rms_mean + rms_std * 2)
                    elif 'speed' in name or 'rotation' in name or 'animation' in name:
                        # Tempo-basiert
                        val = min_val + (max_val - min_val) * min(1.0, tempo / 180.0)
                    elif 'count' in name or 'particles' in name or 'bars' in name or 'rings' in name:
                        # Energie-basiert
                        val = min_val + (max_val - min_val) * rms_mean
                    elif 'smooth' in name or 'flow' in name or 'breathe' in name:
                        # Ruhig bei Speech
                        if mode == 'speech':
                            val = min_val + (max_val - min_val) * 0.3
                        else:
                            val = min_val + (max_val - min_val) * 0.7
                    else:
                        val = default
                    
                    # Auf Step runden
                    if isinstance(step, int):
                        val = round(val / step) * step
                        val = int(val)
                    else:
                        val = round(val / step) * step
                    
                    result[name] = max(min_val, min(max_val, val))
            else:
                result = current_params.copy()
            
            return result
        
        def _fallback_colors():
            mode = audio_features.get('mode', 'music')
            tempo = audio_features.get('tempo', 120)
            if mode == 'speech':
                return {"primary": "#667EEA", "secondary": "#764BA2", "background": "#1A1A2E"}
            elif tempo > 120:
                return {"primary": "#FF0055", "secondary": "#00CCFF", "background": "#0A0A0A"}
            else:
                return {"primary": "#4ECDC4", "secondary": "#96CEB4", "background": "#1A1A3E"}
        
        def _fallback_postprocess():
            mode = audio_features.get('mode', 'music')
            tempo = audio_features.get('tempo', 120)
            if mode == 'speech':
                return {"contrast": 1.05, "saturation": 0.8, "brightness": 0.0, "warmth": 0.1, "film_grain": 0.05}
            elif tempo > 120:
                return {"contrast": 1.2, "saturation": 1.3, "brightness": -0.03, "warmth": 0.0, "film_grain": 0.05}
            else:
                return {"contrast": 1.05, "saturation": 0.9, "brightness": 0.0, "warmth": 0.2, "film_grain": 0.1}
        
        def _fallback_quotes():
            mode = audio_features.get('mode', 'music')
            if mode == 'speech':
                return {
                    "font_size": 56, "box_color": "#1a1a2e", "font_color": "#FFFFFF",
                    "position": "bottom", "display_duration": 8.0, "auto_scale_font": True,
                    "slide_animation": "none", "slide_out_animation": "none", "scale_in": False,
                    "typewriter": False, "glow_pulse": False, "box_padding": 36,
                    "box_radius": 20, "box_margin_bottom": 100, "max_width_ratio": 0.7,
                    "fade_duration": 0.8, "line_spacing": 1.5, "max_font_size": 72,
                    "max_chars_per_line": 40,
                }
            else:
                return {
                    "font_size": 48, "box_color": "#0d0d1a", "font_color": "#FFFFFF",
                    "position": "bottom", "display_duration": 6.0, "auto_scale_font": True,
                    "slide_animation": "up", "slide_out_animation": "none", "scale_in": True,
                    "typewriter": False, "glow_pulse": True, "box_padding": 24,
                    "box_radius": 12, "box_margin_bottom": 80, "max_width_ratio": 0.8,
                    "fade_duration": 0.5, "line_spacing": 1.25, "max_font_size": 56,
                    "max_chars_per_line": 45,
                }
        
        default_result = {
            "params": _fallback_params(),
            "colors": _fallback_colors(),
            "postprocess": _fallback_postprocess(),
            "background": {"opacity": 0.3, "blur": 0.0, "vignette": 0.0},
            "quotes": _fallback_quotes(),
        }
        
        try:
            # Param-Spezifikationen fuer den Prompt aufbereiten
            param_info = ""
            if param_specs:
                for name, (default, min_val, max_val, step) in param_specs.items():
                    param_info += f"  - {name}: aktuell={default}, min={min_val}, max={max_val}, step={step}\n"
            else:
                param_info = "  (keine Spezifikationen verfuegbar)\n"
            
            # Erweiterte Audio-Features
            rms_std = audio_features.get('rms_std', 0.0)
            onset_std = audio_features.get('onset_std', 0.0)
            transient_mean = audio_features.get('transient_mean', 0.0)
            voice_clarity_mean = audio_features.get('voice_clarity_mean', 0.0)
            
            prompt = f"""
Du bist ein professioneller Motion-Graphics-Designer und Color-Grading-Experte.

AUDIO-ANALYSE:
- Dauer: {audio_features.get('duration', 0):.1f}s
- Tempo: {audio_features.get('tempo', 120):.0f} BPM
- Modus: {audio_features.get('mode', 'music')}
- RMS (Lautstaerke): mean={audio_features.get('rms_mean', 0.5):.2f}, std={rms_std:.2f}
- Onset (Beat-Staerke): mean={audio_features.get('onset_mean', 0.3):.2f}, std={onset_std:.2f}
- Transienten (Kick/Snare): {transient_mean:.2f}
- Voice-Clarity (Sprach-Anteil): {voice_clarity_mean:.2f}
- Spektrale Dominanz: {audio_features.get('spectral_mean', 0.5):.2f}

VISUALIZER: {visualizer_type}
AKTUELLE PARAMETER: {json.dumps(current_params, indent=2)}
AKTUELLE FARBEN: {json.dumps(colors, indent=2)}

PARAMETER-SPEZIFIKATIONEN (WICHTIG: Halte dich an min/max!):
{param_info}

GIB ALLE EINSTELLUNGEN ZURUECK als JSON-Objekt mit dieser Struktur:
{{
  "params": {{...}},
  "colors": {{"primary": "#...", "secondary": "#...", "background": "#..."}},
  "postprocess": {{"contrast": 1.0, "saturation": 1.0, "brightness": 0.0, "warmth": 0.0, "film_grain": 0.0}},
  "background": {{"opacity": 0.3, "blur": 0.0, "vignette": 0.0}},
  "quotes": {{
    "font_size": 52, "box_color": "#1a1a2e", "font_color": "#FFFFFF",
    "position": "bottom", "display_duration": 8.0, "auto_scale_font": true,
    "slide_animation": "none", "slide_out_animation": "none", "scale_in": false,
    "typewriter": false, "glow_pulse": false, "box_padding": 32,
    "box_radius": 16, "box_margin_bottom": 100, "max_width_ratio": 0.75,
    "fade_duration": 0.6, "line_spacing": 1.35, "max_font_size": 72,
    "max_chars_per_line": 40
  }}
}}

KONKRETE REGELN fuer Parameter:
- Intensitaet/Speed/Scale: RMS-Std > 0.15 bedeutet DYNAMISCHES Audio -> hoehere Werte
- Tempo > 130 BPM: schnellere Animationen, mehr Partikel, staerkere Effekte
- Modus == "speech": sanfte, langsame Werte, wenig Partikel, dezente Farben
- Modus == "music" + Tempo > 110: aggressiv, schnell, kontrastreich
- Modus == "music" + Tempo <= 110: fliessend, organisch, warm
- Voice-Clarity > 0.5: Podcast-Modus -> reduzierte Bewegung, lesbare Texte

KONKRETE REGELN fuer Farben:
- Musik/EDM: Klassisches Orange/Teal (#FF6B35 / #00CCFF) oder Neon (#FF00AA / #39FF14)
- Podcast: Gedämpft, professionell (#667EEA / #764BA2 / #1A1A2E)
- Chill/Ambient: Pastell-Tuerkis/Sand (#4ECDC4 / #96CEB4 / #1A1A3E)
- Aggressiv: Sattes Rot/Cyan (#FF0055 / #00CCFF)

KONKRETE REGELN fuer Post-Process:
- Speech/News: contrast 1.05, saturation 0.8, warmth 0.1, film_grain 0.05
- Musik/Energy: contrast 1.2, saturation 1.3, warmth 0.0, film_grain 0.05
- Chill: contrast 1.05, saturation 0.9, warmth 0.2, film_grain 0.1
- Cinematic: contrast 1.15, saturation 1.05, warmth 0.25, film_grain 0.15
- Vintage: contrast 0.9, saturation 0.75, warmth 0.4, film_grain 0.35

KONKRETE REGELN fuer Quotes:
- Podcast/News: grosse Schrift (52-64px), Position bottom, keine Animationen
- Musik/Energy: mittlere Schrift (40-48px), Position center, Slide-In up, Glow-Pulse
- Ruhig: sehr grosse Schrift (56-72px), Position center, nur Fade

WICHTIG: Jeder Parameter-Wert MUSS innerhalb seiner min/max-Grenzen liegen!
Runde Float-Werte auf den angegebenen step-Wert.

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
                "font_size": 52, "box_color": "#1a1a2e", "font_color": "#FFFFFF",
                "position": "bottom", "display_duration": 8.0, "auto_scale_font": True,
                "slide_animation": "none", "slide_out_animation": "none", "scale_in": False,
                "typewriter": False, "glow_pulse": False, "box_padding": 32,
                "box_radius": 16, "box_margin_bottom": 100, "max_width_ratio": 0.75,
                "fade_duration": 0.6, "line_spacing": 1.35, "max_font_size": 72,
                "max_chars_per_line": 40,
            }
            
            if not isinstance(optimized, dict):
                print("[Gemini] KI-Antwort war kein Dictionary, verwende Fallback")
                return default_result
            
            # === PARAM CLAMPING ===
            result_params = {}
            if optimized.get("params") and param_specs:
                for name, val in optimized["params"].items():
                    if name in param_specs:
                        default, min_val, max_val, step = param_specs[name]
                        # Clamping
                        val = max(min_val, min(max_val, val))
                        # Auf Step runden
                        if isinstance(step, int):
                            val = round(val / step) * step
                            val = int(val)
                        else:
                            val = round(val / step) * step
                        result_params[name] = val
                    else:
                        result_params[name] = val
            else:
                result_params = optimized.get("params", current_params) or current_params
            
            result = {
                "params": result_params,
                "colors": optimized.get("colors", colors),
                "postprocess": optimized.get("postprocess", {"contrast": 1.0, "saturation": 1.0, "brightness": 0.0,
                                                               "warmth": 0.0, "film_grain": 0.0}),
                "background": optimized.get("background", {"opacity": 0.3, "blur": 0.0, "vignette": 0.0}),
                "quotes": {**default_quotes, **(optimized.get("quotes", {}) or {})},
            }
            
            # Stelle sicher, dass alle erwarteten Keys existieren
            for k in ["contrast", "saturation", "brightness", "warmth", "film_grain"]:
                if k not in result["postprocess"]:
                    result["postprocess"][k] = default_result["postprocess"][k]
            for k in ["opacity", "blur", "vignette"]:
                if k not in result["background"]:
                    result["background"][k] = default_result["background"][k]
            
            return result
            
        except Exception as e:
            print(f"[Gemini] All-Settings-Optimierung fehlgeschlagen: {e}, verwende Fallback")
            return default_result

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
