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
import concurrent.futures
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None


# =============================================================================
# SEMANTIC PARAMETER DESCRIPTIONS
# =============================================================================
# Mapping von Parameter-Namen zu menschenlesbaren Beschreibungen.
# Wird im Prompt verwendet, damit Gemini die Bedeutung jedes Parameters
# versteht und keine unkontrollierten Werte raet.

SEMANTIC_PARAM_DESCRIPTIONS = {
    # Universal
    "viz_offset_x": "Horizontaler Offset des Visualizers. -1.0 = ganz links, 0.0 = Mitte, 1.0 = ganz rechts. Veraendere nur, wenn der User explizit eine Verschiebung will.",
    "viz_offset_y": "Vertikaler Offset des Visualizers. -1.0 = ganz unten, 0.0 = Mitte, 1.0 = ganz oben.",
    "viz_scale": "Skalierung des Visualizers. 0.5 = halbe Groesse, 1.0 = Original, 2.0 = doppelte Groesse. Bei kleiner Aufloesung (<720p) eher 0.8-1.2, bei 4K eher 1.0-1.5.",
    "offset_x": "Alias fuer viz_offset_x. Siehe dort.",
    "offset_y": "Alias fuer viz_offset_y. Siehe dort.",
    "scale": "Alias fuer viz_scale. Siehe dort.",

    # Pulsing Core
    "pulse_intensity": "Puls-Staerke des Zentrums. 0.1 = kaum sichtbarer Herzschlag, 0.5 = deutliches Pulsieren, 1.0 = extrem heftiges Aufblitzen. Hohe Werte bei starkem Beat (Onset > 0.4).",
    "glow_layers": "Anzahl der Glow-Schichten um den Kern. 1 = duenner Halo, 3 = weicher Glow, 5 = intensiver Lichtschein. Mehr Schichten = mehr GPU-Last.",
    "glow_radius": "Radius des Glows in Pixeln. 5 = eng, 20 = weicher Uebergang, 50 = riesiger Lichtschein. Bei hoher Aufloesung (1440p+) eher groesser.",
    "trail_length": "Laenge der Bewegungsspur. 5 = kurzer Schwung, 20 = langer Schweif, 50 = extrem lang gezogen. Bei schnellen Tempi (>130 BPM) kuerzer halten (10-15).",
    "trail_decay": "Abklinggeschwindigkeit der Spur. 0.1 = verschwindet sofort, 0.5 = moderate Nachleuchtzeit, 0.95 = sehr lange sichtbar. Bei hektischer Musik niedriger, bei ruhiger hoeher.",

    # Spectrum Bars
    "bar_count": "Anzahl der Frequenz-Balken. 20 = grob, 64 = fein aufgeloest, 128 = sehr detailliert. Mehr Balken = mehr CPU-Last. Bei Speech eher 20-32, bei Musik 48-64.",
    "smoothing": "Glaettung der Balken-Bewegung. 0.0 = direkte Reaktion (zackig), 0.3 = sanft, 0.8 = sehr traeg. Bei Sprache hoehere Werte (0.4-0.6), bei EDM niedriger (0.1-0.3).",
    "bar_width": "Breite jedes Balkens relativ zum Abstand. 0.5 = duenne Linien mit Luftzwickel, 0.8 = fast beruehrend, 1.0 = solid block. Aesthetische Praeferenz.",
    "bar_spacing": "Abstand zwischen Balken in Pixeln. 0 = kein Abstand, 1 = 1px Luecke, 5 = breiter Zwischenraum.",

    # Chroma Field
    "field_resolution": "Aufloesung des Chromafelds. 50 = grob/pixelig, 100 = mittel, 200 = sehr fein. Hoehere Werte = mehr GPU-Last. Bei 4K unbedingt >= 100.",
    "color_saturation": "Saettigung der Farben. 0.0 = Graustufen, 0.5 = gedaempft, 1.0 = knallig bunt. Bei Podcast 0.3-0.5, bei EDM 0.8-1.0.",

    # Particle Swarm
    "particle_count": "Anzahl der Partikel. 50 = sparsam, 150 = dicht, 500 = extrem voll. Bei langsamer Musik mehr Partikel (ruhigere Bewegung), bei schneller weniger (uebersichtlicher).",
    "explosion_threshold": "Schwelle fuer Partikel-Explosionen. 0.2 = bei leichten Beats, 0.5 = nur bei starken Beats, 0.8 = fast nie. Bei sehr dynamischer Musik niedriger, bei flachem Verlauf hoeher.",
    "fluidity": "Fluessigkeit der Partikel-Bewegung. 0.1 = steif/geometrisch, 0.5 = organisch, 1.0 = vollkommen chaotisch/fließend. Bei Chill/Ambient hoeher, bei Techno niedriger.",

    # Typographic
    "text_size": "Schriftgroesse in Pixeln. 24 = klein/untertitel-artig, 48 = lesbar, 72 = dominant/gross. Auf 1080p 36-48, auf 4K 56-72.",
    "animation_speed": "Geschwindigkeit der Text-Animation. 0.1 = sehr langsam, 0.5 = moderat, 1.0 = extrem schnell. Bei langsamen Songs < 0.4, bei schnellen > 0.6.",
    "typewriter_speed": "Geschwindigkeit des Typewriter-Effekts in Zeichen/Sekunde. 5 = langsame Morse-artige Darbietung, 15 = normal, 50 = unlesbar schnell.",

    # Neon Oscilloscope
    "line_thickness": "Liniendicke in Pixeln. 1 = duenn/fragil, 4 = markant, 10 = dick/massiv. Bei hoher Aufloesung (>1080p) dicker.",
    "num_points": "Aufloesung der Wellenform. 100 = grob, 200 = fein, 500 = sehr detailliert. Hoehere Werte = mehr GPU-Last.",

    # Sacred Mandala
    "rotation_speed": "Drehgeschwindigkeit. 0.001 = fast stehend, 0.005 = langsame Meditation, 0.02 = schnell hypnotisch. Bei Chill 0.002-0.005, bei Trance 0.01-0.02.",
    "num_petals": "Anzahl der Bluetenblaetter. 3 = minimalistisch, 8 = klassisch, 16 = komplex. Bei ruhiger Musik mehr Blaetter (ruhiger Eindruck), bei schneller weniger (uebersichtlicher).",
    "layer_count": "Anzahl der ueberlagerten Mandala-Schichten. 1 = einfach, 3 = tief/raeumlich, 6 = sehr komplex. Mehr Schichten = mehr GPU-Last.",

    # Liquid Blobs
    "blob_count": "Anzahl der Blobs. 3 = minimalistisch, 6 = ausgewogen, 12 = voll. Bei kleinem Screen 3-4, bei grossem 6-8.",

    # Neon Wave Circle
    "circle_count": "Anzahl der konzentrischen Kreise. 3 = reduziert, 5 = ausgewogen, 10 = dicht. Bei schnellem Tempo weniger Kreise (klarer), bei langsamem mehr.",
    "wave_amplitude": "Wellen-Hoehe. 0.5 = sanfte Huegel, 1.0 = normale Wellen, 2.0 = extreme Spitzen. RMS-mapped: leise=0.3-0.6, laut=1.0-1.5.",

    # Frequency Flower
    "num_petals": "Anzahl der Bluetenblaetter. 3 = minimalistisch, 8 = klassisch, 16 = komplex. Bei ruhiger Musik mehr, bei schneller weniger.",

    # Voice Flow
    "flow_speed": "Geschwindigkeit der Wellenbewegung. 0.1 = fast stehend, 0.5 = moderate Fluss, 1.0 = extrem hektisch. Bei Speech 0.2-0.4, bei Musik 0.5-0.8.",
    "wave_depth": "Tiefe/Amplitude der Wellen. 0.2 = flache Wellen, 0.6 = ausgepraegt, 1.0 = extreme Auslenkung. Bei leiser Stimme 0.4-0.6, bei lautem Schreien 0.8-1.0.",
    "breathe_intensity": "Atmungs-Effekt. 0.1 = kaum sichtbar, 0.35 = deutliche Ein- und Ausatmung, 0.8 = hyperventilierend. Bei Meditation 0.3-0.5, bei Action 0.1-0.2.",
    "line_count": "Anzahl der Wellenlinien. 3 = reduziert, 5 = ausgewogen, 10 = dicht. Mehr Linien bei grosser Aufloesung, weniger bei kleiner.",
    "glow_strength": "Leuchtstaerke der Linien. 0.2 = dezent, 0.5 = sichtbarer Neon-Effekt, 1.0 = extrem hell. Bei dunklem Hintergrund hoeher, bei hellem niedriger.",
    "line_width": "Liniendicke. 0.001 = Haarfein, 0.004 = markant, 0.01 = dick. Bei 4K unbedingt >= 0.003.",
    "trail_decay": "Nachleuchten der Linien. 0.5 = schnell verblassend, 0.75 = moderate Spur, 0.95 = sehr langsam. Bei schneller Rede niedriger, bei langsamer Monolog hoeher.",
    "brightness": "Gesamthelligkeit. 0.5 = dunkel, 1.0 = normal, 1.5 = ueberhell. Bei Podcast 1.0-1.2, bei Musik 0.8-1.1.",
}


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
        # ThreadPool fuer non-blocking API-Calls (verhindert Render-Loop-Stalls)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="gemini_"
        )

    def shutdown(self):
        """Faehrt den internen ThreadPool sauber herunter."""
        self._executor.shutdown(wait=True)

    def transcribe_audio_async(self, audio_path: str) -> concurrent.futures.Future:
        """Asynchrone Transkription. Gibt ein Future zurueck.

        Der blockierende Netzwerk-Call laeuft in einem Hintergrund-Thread,
        damit der Render-Loop nicht auf API-Latenz wartet.
        """
        return self._executor.submit(self.transcribe_audio, audio_path)

    def extract_quotes_async(self, audio_path: str, audio_duration: float = None,
                              max_quotes: int = None) -> concurrent.futures.Future:
        """Asynchrone Zitat-Extraktion. Gibt ein Future zurueck."""
        return self._executor.submit(
            self.extract_quotes, audio_path, audio_duration, max_quotes
        )

    def optimize_all_settings_async(self, visualizer_type: str, current_params: dict,
                                     audio_features: dict, colors: dict,
                                     param_specs: dict = None,
                                     user_prompt: str = None) -> concurrent.futures.Future:
        """Asynchrone Parameter-Optimierung. Gibt ein Future zurueck."""
        return self._executor.submit(
            self.optimize_all_settings, visualizer_type, current_params,
            audio_features, colors, param_specs, user_prompt
        )

    def generate_background_prompt_async(self, audio_features: dict) -> concurrent.futures.Future:
        """Asynchrone Prompt-Generierung. Gibt ein Future zurueck."""
        return self._executor.submit(self.generate_background_prompt, audio_features)

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
        
        def _build_semantic_param_info(specs):
            """Baut den Parameter-Info-Block mit Semantik + Hard-Bounds fuer den Prompt."""
            if not specs:
                return "  (keine Spezifikationen verfuegbar)\n"
            lines = []
            for name, (default, min_val, max_val, step) in specs.items():
                desc = SEMANTIC_PARAM_DESCRIPTIONS.get(name, "")
                if desc:
                    lines.append(
                        f'  - "{name}":\n'
                        f'      Standardwert: {default}\n'
                        f'      Bereich: [{min_val} ... {max_val}]  (Schrittweite: {step})\n'
                        f'      Bedeutung: {desc}\n'
                    )
                else:
                    lines.append(
                        f'  - "{name}": Standard={default}, min={min_val}, max={max_val}, step={step}\n'
                    )
            return "".join(lines)

        try:
            param_info = _build_semantic_param_info(param_specs)

            # Erweiterte Audio-Features
            rms_std = audio_features.get('rms_std', 0.0)
            onset_std = audio_features.get('onset_std', 0.0)
            transient_mean = audio_features.get('transient_mean', 0.0)
            voice_clarity_mean = audio_features.get('voice_clarity_mean', 0.0)
            mode = audio_features.get('mode', 'music')
            tempo = audio_features.get('tempo', 120)
            rms_mean = audio_features.get('rms_mean', 0.5)
            onset_mean = audio_features.get('onset_mean', 0.3)

            prompt = f"""Du bist ein professioneller Motion-Graphics-Designer und Color-Grading-Experte.

Deine Aufgabe: Optimiere die Visualisierungs-Einstellungen fuer ein Audio-Video basierend auf einer detaillierten Audio-Analyse.

================================================================================
AUDIO-ANALYSE
================================================================================
- Dauer: {audio_features.get('duration', 0):.1f}s
- Tempo: {tempo:.0f} BPM
- Modus: {mode}
- RMS (Lautstaerke): mean={rms_mean:.2f}, std={rms_std:.2f}
- Onset (Beat-Staerke): mean={onset_mean:.2f}, std={onset_std:.2f}
- Transienten (Kick/Snare-Praesenz): {transient_mean:.2f}
- Voice-Clarity (Sprach-Anteil): {voice_clarity_mean:.2f}
- Spektrale Dominanz: {audio_features.get('spectral_mean', 0.5):.2f}

================================================================================
VISUALIZER: {visualizer_type}
================================================================================
AKTUELLE PARAMETER (diese sind dein Ausgangspunkt):
{json.dumps(current_params, indent=2)}

AKTUELLE FARBEN:
{json.dumps(colors, indent=2)}

================================================================================
PARAMETER-SPEZIFIKATIONEN
================================================================================
Jeder Parameter hat einen STANDARDWERT, einen erlaubten BEREICH und eine BEDEUTUNG.
Du MUSST die neuen Werte INNERHALB der min/max-Grenzen liefern.
Werte ausserhalb des Bereichs fuehren zu Fehlern.

{param_info}

================================================================================
RELATIVE ANPASSUNG (WICHTIG)
================================================================================
Passe die Werte RELATIV zu den aktuellen Standardwerten an. Nicht von Null raten.

- LEICHTE Anpassung:  +/- 10-20% vom Standardwert  (bei geringen Veraenderungen)
- MODERATE Anpassung: +/- 30-50% vom Standardwert  (bei deutlichen Unterschieden)
- STARKE Anpassung:   +/- 60-100% vom Standardwert (bei extremen Audio-Eigenschaften)

Beispiel: Standardwert=0.5, Bereich=[0.0, 1.0], Schritt=0.05
  - Leicht (+20%)  -> 0.60
  - Moderat (+50%) -> 0.75
  - Stark (+100%)  -> 1.00

Wenn ein Parameter keinen Sinn fuer den aktuellen Modus macht (z.B. "particle_count" bei "speech"),
passe ihn nur LEICHT an oder belasse ihn beim Standardwert.

================================================================================
REGELN NACH AUDIO-TYP
================================================================================
Parameter-Regeln:
- Intensitaet/Speed/Scale: RMS-Std > 0.15 = DYNAMISCHES Audio -> leicht bis moderat hoeher
- Tempo > 130 BPM: schnellere Animationen, evtl. mehr Partikel, staerkere Effekte
- Modus == "speech": sanfte, langsame Werte, wenig Partikel, dezente Farben
- Modus == "music" + Tempo > 110: moderat aggressiv, kontrastreich
- Modus == "music" + Tempo <= 110: fliessend, organisch, warm
- Voice-Clarity > 0.5: Podcast-Modus -> reduzierte Bewegung, lesbare Texte

Farb-Regeln:
- Musik/EDM: Klassisches Orange/Teal (#FF6B35 / #00CCFF) oder Neon (#FF00AA / #39FF14)
- Podcast: Gedämpft, professionell (#667EEA / #764BA2 / #1A1A2E)
- Chill/Ambient: Pastell-Tuerkis/Sand (#4ECDC4 / #96CEB4 / #1A1A3E)
- Aggressiv: Sattes Rot/Cyan (#FF0055 / #00CCFF)

Post-Process-Regeln:
- Speech/News:    contrast 1.05, saturation 0.8,  warmth 0.1, film_grain 0.05
- Musik/Energy:   contrast 1.2,  saturation 1.3,  warmth 0.0, film_grain 0.05
- Chill:          contrast 1.05, saturation 0.9,  warmth 0.2, film_grain 0.1
- Cinematic:      contrast 1.15, saturation 1.05, warmth 0.25, film_grain 0.15
- Vintage:        contrast 0.9,  saturation 0.75, warmth 0.4, film_grain 0.35

Quote-Regeln:
- Podcast/News: grosse Schrift (52-64px), Position bottom, keine Animationen
- Musik/Energy: mittlere Schrift (40-48px), Position center, Slide-In up, Glow-Pulse
- Ruhig:        sehr grosse Schrift (56-72px), Position center, nur Fade

================================================================================
AUSGABEFORMAT (STRIKTES JSON)
================================================================================
Gib NUR ein JSON-Objekt zurueck. Keine Erklaerungen, kein Markdown-Code-Block.

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

WICHTIGE HINWEISE:
1. Jeder Wert in "params" MUSS innerhalb des jeweiligen min/max-Bereichs liegen.
2. Float-Werte MUSSEN auf die angegebene step-Schrittweite gerundet werden.
3. Bei "colors" verwende gueltige Hex-Codes (#RRGGBB).
4. Der User-Prompt hat Prioritaet ueber alle automatischen Regeln.
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
