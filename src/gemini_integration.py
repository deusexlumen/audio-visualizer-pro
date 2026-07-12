"""
Gemini Integration für Audio Visualizer Pro.

Nutzt ein konfigurierbares Gemini-Modell (Standard: Flash-Lite) für:
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

import numpy as np

from .app_logging import get_logger
from .app_settings import load_settings
from .ai_costs import get_cost_ledger
from .quote_cache import (
    save_upload_id, load_upload_id, save_transcript, load_transcript,
    save_json_result, load_json_result,
)
from .types import Quote

logger = get_logger(__name__)

# Version der Prompt-/Schema-Logik. Bei Prompt-Aenderungen erhoehen, damit
# alte gecachte KI-Ergebnisse verworfen werden.
PROMPT_VERSION = 1

try:
    from google import genai
except ImportError:
    genai = None

try:
    from google.genai import errors as genai_errors
except ImportError:
    genai_errors = None


# =============================================================================
# SEMANTIC PARAMETER DESCRIPTIONS
# =============================================================================
# Mapping von Parameter-Namen zu menschenlesbaren Beschreibungen.
# Wird im Prompt verwendet, damit Gemini die Bedeutung jedes Parameters
# versteht und keine unkontrollierten Werte raet.

# =============================================================================
# PARAMETER CATEGORIES FOR DETERMINISTIC FALLBACK
# =============================================================================
# Kategorisierung von Parametern fuer den Fallback-Algorithmus. Statt auf
# string-basierte Heuristiken zu setzen, ordnen wir jeden bekannten Parameter
# einer semantischen Kategorie zu und berechnen seinen Wert aus den
# Audio-Features.

PARAM_CATEGORIES = {
    # Intensitaet / Staerke / Helligkeit
    "intensity": {
        "pulse_intensity", "core_intensity", "bass_intensity", "glow_strength",
        "swell_intensity", "wave_intensity", "beat_flash", "strobe_intensity",
        "sparkle_intensity", "accent_intensity", "glow_size", "line_brightness",
        "spotlight_strength", "ray_strength", "noise_intensity", "vignette_intensity",
        "vignette_strength", "scanline_intensity", "brightness_cap", "bg_brightness",
    },
    # Geschwindigkeit / Animation / Zeit
    "speed": {
        "flow_speed", "rotation_speed", "animation_speed", "breathe_speed",
        "shockwave_speed", "trail_time_offset", "trail_decay", "glow_layers",
    },
    # Anzahl / Dichte
    "count": {
        "bar_count", "particle_count", "ring_count", "line_count", "vu_segments",
        "num_petals", "layer_count", "blob_count", "circle_count", "glow_radius",
        "trail_length", "num_points", "wave_complexity", "wave_frequency",
    },
    # Groesse / Abstand / Skalierung
    "size": {
        "bar_width", "bar_spacing", "ring_spacing", "ring_width", "base_radius",
        "core_base_radius", "ring_base_radius", "line_width", "temple_scale",
        "wave_amplitude", "bar_height", "height_scale", "base_height", "height_boost",
        "line_spacing", "particle_spread", "wave_count", "color_spread",
        "field_resolution", "connection_dist", "particle_size", "ring_spacing",
    },
    # Farbe / Farbverschiebung
    "color": {
        "color_shift", "color_saturation", "gold_tint", "color_mode",
    },
    # Reaktivitaet / Glättung / Fluss
    "reactivity": {
        "smoothing", "fluidity", "response_speed", "dynamics_response",
        "breathe_intensity", "explosion_threshold", "strobe_threshold",
        "contrast_gamma", "noise_amount", "grain_amount",
    },
    # Position / universelle Transform
    "transform": {
        "viz_offset_x", "viz_offset_y", "viz_scale", "offset_x", "offset_y", "scale",
    },
    # Spezielle String-/Hex-Parameter (werden beibehalten oder gemaess Modus gesetzt)
    "special": {
        "background_color", "text_size",
    },
}


def _get_param_category(name: str) -> str:
    """Gibt die Kategorie eines Parameters zurueck."""
    for category, members in PARAM_CATEGORIES.items():
        if name in members:
            return category
    return "other"


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
    "base_radius": "Basis-Radius des Kerns. 0.02 = winzig, 0.1 = normal, 0.3 = sehr gross.",
    "ring_count": "Anzahl konzentrischer Ringe. 1 = minimalistisch, 3 = ausgewogen, 8 = komplex.",
    "ring_spacing": "Abstand zwischen Ringen. 0.02 = dicht, 0.08 = normal, 0.15 = weit.",
    "ring_width": "Dicke jedes Rings. 0.005 = duenn, 0.015 = normal, 0.05 = dick.",
    "bg_brightness": "Helligkeit des Hintergrunds. 0.0 = schwarz, 0.15 = subtil, 0.5 = hell.",

    # Spectrum Bars
    "bar_count": "Anzahl der Frequenz-Balken. 20 = grob, 64 = fein aufgeloest, 128 = sehr detailliert. Mehr Balken = mehr CPU-Last. Bei Speech eher 20-32, bei Musik 48-64.",
    "smoothing": "Glaettung der Balken-Bewegung. 0.0 = direkte Reaktion (zackig), 0.3 = sanft, 0.8 = sehr traeg. Bei Sprache hoehere Werte (0.4-0.6), bei EDM niedriger (0.1-0.3).",
    "bar_width": "Breite jedes Balkens relativ zum Abstand. 0.5 = duenne Linien mit Luftzwickel, 0.8 = fast beruehrend, 1.0 = solid block. Aesthetische Praeferenz.",
    "bar_spacing": "Abstand zwischen Balken in Pixeln. 0 = kein Abstand, 1 = 1px Luecke, 5 = breiter Zwischenraum.",
    "height_scale": "Hoehen-Skalierung der Balken. 0.2 = sehr klein, 1.0 = normal, 3.0 = riesig. An RMS anpassen.",
    "base_height": "Mindesthoehe der Balken. 0.0 = keine, 0.1 = kleiner Sockel, 0.5 = grosser Sockel.",
    "height_boost": "Zusaetzlicher Hoehen-Boost durch Energie. 0.0 = aus, 1.0 = voll.",
    "wave_count": "Anzahl der Wellen-Modulation ueber die Balken. 0.0 = keine, 2.0 = sehr wellig.",
    "color_spread": "Farbverschiebung zwischen benachbarten Balken. 0.0 = gleichfarbig, 0.1 = regenbogen-artig.",
    "color_shift": "Globaler Hue-Offset. 0.0 = Standard, 1.0 = voller Farbkreis.",

    # Chroma Field
    "field_resolution": "Aufloesung des Chromafelds. 50 = grob/pixelig, 100 = mittel, 200 = sehr fein. Hoehere Werte = mehr GPU-Last. Bei 4K unbedingt >= 100.",
    "color_saturation": "Saettigung der Farben. 0.0 = Graustufen, 0.5 = gedaempft, 1.0 = knallig bunt. Bei Podcast 0.3-0.5, bei EDM 0.8-1.0.",
    "connection_dist": "Maximale Distanz fuer Partikel-Verbindungen. 50 = eng, 150 = normal, 300 = weit.",
    "particle_size": "Groesse der Partikel im Feld. 2 = klein, 8 = normal, 20 = gross.",

    # Particle Swarm
    "particle_count": "Anzahl der Partikel. 50 = sparsam, 150 = dicht, 500 = extrem voll. Bei langsamer Musik mehr Partikel (ruhigere Bewegung), bei schneller weniger (uebersichtlicher).",
    "explosion_threshold": "Schwelle fuer Partikel-Explosionen. 0.2 = bei leichten Beats, 0.5 = nur bei starken Beats, 0.8 = fast nie. Bei sehr dynamischer Musik niedriger, bei flachem Verlauf hoeher.",
    "fluidity": "Fluessigkeit der Partikel-Bewegung. 0.1 = steif/geometrisch, 0.5 = organisch, 1.0 = vollkommen chaotisch/fließend. Bei Chill/Ambient hoeher, bei Techno niedriger.",
    "glow_size": "Groesse des Partikel-Glows. 1 = punktuell, 3 = weich, 10 = riesig.",

    # Typographic
    "text_size": "Schriftgroesse in Pixeln. 24 = klein/untertitel-artig, 48 = lesbar, 72 = dominant/gross. Auf 1080p 36-48, auf 4K 56-72.",
    "animation_speed": "Geschwindigkeit der Text-Animation. 0.1 = sehr langsam, 0.5 = moderat, 1.0 = extrem schnell. Bei langsamen Songs < 0.4, bei schnellen > 0.6.",

    # Neon Oscilloscope
    "line_thickness": "Liniendicke in Pixeln. 1 = duenn/fragil, 4 = markant, 10 = dick/massiv. Bei hoher Aufloesung (>1080p) dicker.",
    "num_points": "Aufloesung der Wellenform. 100 = grob, 200 = fein, 500 = sehr detailliert. Hoehere Werte = mehr GPU-Last.",
    "glow_radius": "Weite des Glows um die Linie. 4 = schmal, 16 = neon-artig, 30 = sehr weich.",

    # Sacred Mandala
    "rotation_speed": "Drehgeschwindigkeit. 0.001 = fast stehend, 0.005 = langsame Meditation, 0.02 = schnell hypnotisch. Bei Chill 0.002-0.005, bei Trance 0.01-0.02.",
    "num_petals": "Anzahl der Bluetenblaetter. 3 = minimalistisch, 8 = klassisch, 16 = komplex. Bei ruhiger Musik mehr Blaetter (ruhiger Eindruck), bei schneller weniger (uebersichtlicher).",
    "layer_count": "Anzahl der ueberlagerten Mandala-Schichten. 1 = einfach, 3 = tief/raeumlich, 6 = sehr komplex. Mehr Schichten = mehr GPU-Last.",

    # Liquid Blobs
    "blob_count": "Anzahl der Blobs. 3 = minimalistisch, 6 = ausgewogen, 12 = voll. Bei kleinem Screen 3-4, bei grossem 6-8.",
    "fluidity": "Fluessigkeit der Blob-Bewegung. 0.1 = steif, 0.5 = organisch, 1.0 = sehr fließend.",

    # Neon Wave Circle
    "circle_count": "Anzahl der konzentrischen Kreise. 3 = reduziert, 5 = ausgewogen, 10 = dicht. Bei schnellem Tempo weniger Kreise (klarer), bei langsamem mehr.",
    "wave_amplitude": "Wellen-Hoehe. 0.5 = sanfte Huegel, 1.0 = normale Wellen, 2.0 = extreme Spitzen. RMS-mapped: leise=0.3-0.6, laut=1.0-1.5.",

    # Frequency Flower
    "num_petals": "Anzahl der Bluetenblaetter. 3 = minimalistisch, 8 = klassisch, 16 = komplex. Bei ruhiger Musik mehr, bei schneller weniger.",

    # Voice Flow
    "flow_speed": "Geschwindigkeit der Wellenbewegung. 0.1 = fast stehend, 0.5 = moderate Fluss, 1.0 = extrem hektisch. Bei Speech 0.2-0.4, bei Musik 0.5-0.8.",
    "wave_depth": "Tiefe/Amplitude der Wellen. 0.2 = flache Wellen, 0.6 = ausgepraegt, 1.0 = extreme Auslenkung. Bei leiser Stimme 0.4-0.6, bei lautem Schreien 0.8-1.0.",
    "breathe_intensity": "Atmungs-Effekt. 0.1 = kaum sichtbar, 0.35 = deutliche Ein- und Ausatmung, 0.8 = hyperventilierend. Bei Meditation 0.3-0.5, bei Action 0.1-0.2.",
    "breathe_speed": "Atmungs-Tempo. 0.1 = sehr langsam, 0.6 = normal, 2.0 = schnell.",
    "line_count": "Anzahl der Wellenlinien. 3 = reduziert, 5 = ausgewogen, 10 = dicht. Mehr Linien bei grosser Aufloesung, weniger bei kleiner.",
    "line_spacing": "Vertikaler Abstand der Wellenlinien. 0.2 = eng, 0.6 = normal, 1.5 = weit.",
    "glow_strength": "Leuchtstaerke der Linien. 0.2 = dezent, 0.5 = sichtbarer Neon-Effekt, 1.0 = extrem hell. Bei dunklem Hintergrund hoeher, bei hellem niedriger.",
    "glow_size": "Weite des Glows um die Wellenlinien. 0.01 = schmal, 0.08 = normal, 0.3 = sehr weich.",
    "line_width": "Liniendicke. 0.001 = Haarfein, 0.004 = markant, 0.01 = dick. Bei 4K unbedingt >= 0.003.",
    "trail_decay": "Nachleuchten der Linien. 0.5 = schnell verblassend, 0.75 = moderate Spur, 0.95 = sehr langsam. Bei schneller Rede niedriger, bei langsamer Monolog hoeher.",
    "trail_time_offset": "Zeitlicher Abstand der Trail-Echos. 0.01 = fast gleichzeitig, 0.1 = deutlich versetzt.",
    "scanline_intensity": "Staerke des Scanline-Effekts. 0.0 = aus, 0.05 = subtil, 0.3 = stark.",
    "vignette_intensity": "Staerke der Vignette. 0.0 = aus, 0.6 = normal, 1.5 = sehr dunkle Raender.",
    "brightness": "Gesamthelligkeit. 0.5 = dunkel, 1.0 = normal, 1.5 = ueberhell. Bei Podcast 1.0-1.2, bei Musik 0.8-1.1.",

    # Speech Focus
    "vu_segments": "Anzahl der VU-Meter-Segmente. 4 = grob, 12 = normal, 24 = fein.",
    "response_speed": "Reaktionsgeschwindigkeit auf Sprache. 0.2 = sehr traeg, 0.8 = schnell, 1.5 = hyper-reaktiv.",
    "wave_amp": "Amplitude der Wellenform. 0.0 = flach, 0.025 = subtil, 0.08 = stark.",
    "line_brightness": "Helligkeit der Wellenform. 0.05 = dezent, 0.18 = normal, 0.5 = hell.",
    "accent_intensity": "Staerke der Akzent-Farbe bei Sprache. 0.0 = aus, 0.55 = normal, 1.0 = sehr stark.",
    "grain_amount": "Film-Grain-Menge. 0.0 = sauber, 0.01 = subtil, 0.05 = deutlich.",
    "brightness_cap": "Maximale Helligkeit. 0.1 = sehr dunkel, 0.4 = normal, 0.8 = hell.",
    "background_color": "Hintergrundfarbe als Hex-String. Im Speech-Modus sehr dunkel halten (#060607).",

    # Bass Temple
    "strobe_threshold": "Schwelle fuer Stroboskop-Blitze. 0.0 = immer, 0.55 = normale Beats, 1.0 = nie.",
    "shockwave_speed": "Geschwindigkeit der Shockwave-Ringe. 0.5 = langsam, 2.5 = normal, 6.0 = schnell.",
    "temple_scale": "Skalierung der Tempel-Form. 0.5 = klein, 1.0 = normal, 2.0 = gross.",
    "contrast_gamma": "Gamma-Kontrast. 0.5 = sehr hell, 0.88 = normal, 1.5 = sehr dunkel.",
    "noise_intensity": "Staerke des Hintergrund-Rauschens. 0.0 = aus, 0.2 = subtil, 1.0 = stark.",

    # Lumina Core
    "chromatic_aberration": "Staerke der chromatischen Aberration bei Beats. 0.0 = aus, 0.003 = subtil, 0.02 = stark.",
    "core_base_radius": "Basis-Radius des Kerns. 0.05 = klein, 0.15 = normal, 0.4 = gross.",
    "ring_base_radius": "Basis-Radius des ersten Rings. 0.1 = nah, 0.25 = normal, 0.5 = weit.",
    "noise_scale": "Skalierung des Noise-Details. 0.5 = grob, 2.0 = normal, 5.0 = sehr fein.",
    "noise_amount": "Staerke der Noise-Displacement. 0.0 = glatt, 0.03 = subtil, 0.1 = stark.",
    "specular_power": "Glanzlicht-Schaerfe. 4 = matt, 32 = normal, 128 = sehr spiegelnd.",

    # Spectrum Genesis
    "bar_height": "Maximale Balkenhoehe. 0.1 = niedrig, 0.35 = normal, 0.7 = sehr hoch.",
    "wave_frequency": "Frequenz der Wellenform. 1 = sehr langsam, 10 = normal, 40 = sehr schnell.",
    "wave_complexity": "Anzahl der Wellen-Komponenten. 1 = einfach, 3 = normal, 6 = komplex.",

    # Orchestral Swell
    "gold_tint": "Gold-Anteil in der Farbgebung. 0.0 = neutral, 0.5 = warm, 1.0 = sehr golden.",
    "dynamics_response": "Reaktion auf Dynamikwechsel. 0.5 = gedaempft, 1.2 = normal, 2.5 = sehr stark.",
    "particle_spread": "Ausbreitung der Partikel. 0.0 = zentriert, 1.0 = normal, 4.0 = sehr weit.",
}


# =============================================================================
# RESPONSE SCHEMA FOR DETERMINISTIC JSON OUTPUT
# =============================================================================
# Wird an Gemini uebergeben, damit die Antwort exakt diesem Schema folgt.
# Reduziert Halluzinationen und erzwingt gueltige Wertebereiche.

OPTIMIZE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "params": {
            "type": "object",
            "description": "Optimierte Visualizer-Parameter. Werte muessen innerhalb der angegebenen min/max-Grenzen liegen.",
            "additionalProperties": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "string"},
                    {"type": "boolean"},
                ]
            }
        },
        "colors": {
            "type": "object",
            "properties": {
                "primary": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                "secondary": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                "background": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
            },
            "required": ["primary", "secondary", "background"],
            "additionalProperties": False,
        },
        "postprocess": {
            "type": "object",
            "properties": {
                "contrast": {"type": "number"},
                "saturation": {"type": "number"},
                "brightness": {"type": "number"},
                "warmth": {"type": "number"},
                "film_grain": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "background": {
            "type": "object",
            "properties": {
                "opacity": {"type": "number"},
                "blur": {"type": "number"},
                "vignette": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "quotes": {
            "type": "object",
            "description": "Quote-Overlay-Einstellungen.",
            "additionalProperties": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "string"},
                    {"type": "boolean"},
                    {"type": "array", "items": {"type": "integer"}},
                ]
            }
        },
    },
    "required": ["params", "colors", "postprocess", "background", "quotes"],
    "additionalProperties": False,
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


class GeminiIntegration:
    """
    Wrapper für die Gemini-API (Modell konfigurierbar via settings.json/env).
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

        # Modell konfigurierbar (env/settings.json), nie hartcodiert.
        # Die tatsaechliche Validierung passiert lazy beim ersten Call, damit
        # __init__ keinen Netzwerk-Zugriff macht.
        self._settings = load_settings()
        self._configured_model = self._settings.gemini_model
        self._active_model: Optional[str] = None
        self._model_validated = False

        # ThreadPool fuer non-blocking API-Calls (verhindert Render-Loop-Stalls)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="gemini_"
        )

    @property
    def model(self) -> str:
        """Das aktuell aktive Modell (nach Lazy-Validierung, sonst konfiguriert)."""
        return self._active_model or self._configured_model

    def _ensure_model(self) -> str:
        """Validiert die konfigurierte Modell-ID einmalig und waehlt bei
        Bedarf ein passendes Ersatzmodell.

        Faellt eine veraltete/ungueltige ID auf, wird per models.list() das
        neueste passende Modell gewaehlt (Praeferenz aus settings.json).
        Wirft nie — im Zweifel bleibt die konfigurierte ID bestehen.
        """
        if self._model_validated:
            return self.model

        self._model_validated = True
        candidate = self._configured_model
        try:
            self.client.models.get(model=candidate)
            self._active_model = candidate
            return candidate
        except Exception as e:
            logger.warning(
                f"[Gemini] Modell '{candidate}' nicht verfuegbar ({e}). "
                f"Suche Ersatz..."
            )

        replacement = self._pick_fallback_model()
        if replacement:
            logger.warning(f"[Gemini] Nutze Ersatzmodell '{replacement}'.")
            self._active_model = replacement
        else:
            # Kein Ersatz gefunden — konfigurierte ID beibehalten, Call darf
            # dann mit einer klaren API-Fehlermeldung scheitern.
            self._active_model = candidate
        return self.model

    def _pick_fallback_model(self) -> Optional[str]:
        """Waehlt aus models.list() das neueste Modell nach Praeferenzliste."""
        try:
            models = list(self.client.models.list())
        except Exception as e:
            logger.warning(f"[Gemini] models.list() fehlgeschlagen: {e}")
            return None

        # Nur Modelle, die generateContent unterstuetzen
        def supports_generate(m) -> bool:
            actions = getattr(m, "supported_actions", None) or \
                getattr(m, "supported_generation_methods", None)
            if not actions:
                return True  # unbekannt -> nicht ausschliessen
            return any("generateContent" in a or "generate_content" in a for a in actions)

        names = [
            getattr(m, "name", "").replace("models/", "")
            for m in models if supports_generate(m)
        ]
        names = [n for n in names if n]

        for pref in self._settings.model_preference:
            matches = [n for n in names if pref in n]
            if matches:
                # 'latest'-Alias bevorzugen, sonst laengsten (meist neuesten) Namen
                latest = [n for n in matches if "latest" in n]
                return latest[0] if latest else sorted(matches)[-1]
        return names[0] if names else None

    # -------------------------------------------------------------------------
    # Retry-Wrapper fuer alle Gemini API-Calls
    # -------------------------------------------------------------------------

    # HTTP-Status-Codes, bei denen sich ein Retry lohnt (transiente Fehler)
    _RETRYABLE_CODES = {408, 409, 429, 500, 502, 503, 504}
    # Codes, die auf ein Konto-/Auth-Problem hindeuten (kein Retry sinnvoll)
    _AUTH_CODES = {401, 403}

    def _error_code(self, error: Exception) -> Optional[int]:
        """Extrahiert den HTTP-Status aus einem genai-APIError, falls vorhanden."""
        if genai_errors is not None and isinstance(error, genai_errors.APIError):
            code = getattr(error, "code", None)
            if isinstance(code, int):
                return code
        return None

    def _is_retryable_error(self, error: Exception) -> bool:
        """Prueft, ob ein Fehler retry-bar ist.

        Bevorzugt den typisierten HTTP-Status des SDKs; faellt nur bei
        Transport-Fehlern ohne Status auf String-Heuristik zurueck.
        """
        code = self._error_code(error)
        if code is not None:
            return code in self._RETRYABLE_CODES

        # Kein API-Status -> Transport-/Verbindungsfehler heuristisch behandeln
        error_str = str(error).lower()
        transport_indicators = [
            "timeout", "timed out", "deadline exceeded",
            "connection", "reset", "refused", "temporarily",
            "unavailable", "broken pipe",
        ]
        return any(ind in error_str for ind in transport_indicators)

    def _retry_after_seconds(self, error: Exception) -> Optional[float]:
        """Liest einen 'Retry-After'-Hinweis aus der Fehlerantwort, falls vorhanden."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    def _call_gemini_with_retry(self, call_fn, max_retries: int = 5,
                                 base_delay: float = 2.0, track_cost: bool = True):
        """
        Fuehrt einen Gemini API-Call mit Exponential Backoff aus.

        Validiert vor dem ersten Call die Modell-ID und erfasst — sofern das
        Ergebnis usage_metadata traegt — die Kosten im Session-Ledger.

        Args:
            call_fn: Callable, die den API-Call durchfuehrt (keine Argumente).
            max_retries: Maximale Anzahl Versuche (inkl. erster Versuch).
            base_delay: Basis-Wartezeit in Sekunden (verdoppelt sich pro Retry).
            track_cost: Ob die Token-Kosten des Ergebnisses erfasst werden sollen.

        Returns:
            Das Ergebnis von call_fn().

        Raises:
            RuntimeError: Wenn alle Versuche fehlschlagen.
        """
        # Modell-ID einmalig validieren (kann self.model auf ein Ersatzmodell setzen)
        self._ensure_model()

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = call_fn()
                if track_cost:
                    self._track_cost(result)
                return result
            except Exception as e:
                last_error = e
                code = self._error_code(e)

                # Auth-/Kontingent-Fehler: kein Retry, klare Meldung
                if code in self._AUTH_CODES:
                    raise RuntimeError(
                        "Gemini-Zugriff verweigert (Auth/Berechtigung). "
                        "Bitte GEMINI_API_KEY pruefen."
                    ) from e

                if not self._is_retryable_error(e):
                    raise

                if attempt < max_retries:
                    wait_time = self._retry_after_seconds(e) or base_delay * (2 ** (attempt - 1))
                    kind = "Kontingent (429)" if code == 429 else f"Fehler{f' ({code})' if code else ''}"
                    logger.warning(
                        f"[Gemini] Retry {attempt}/{max_retries} nach {kind}: {e}. "
                        f"Warte {wait_time:.0f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"[Gemini] Alle {max_retries} Versuche fehlgeschlagen: {e}")

        raise RuntimeError(
            f"Gemini API nach {max_retries} Versuchen nicht erreichbar. "
            f"Letzter Fehler: {last_error}"
        )

    def _track_cost(self, response) -> None:
        """Erfasst Token-Verbrauch eines Antwort-Objekts im Kosten-Ledger."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        try:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = (
                getattr(usage, "candidates_token_count", 0)
                or getattr(usage, "total_token_count", 0) - prompt_tokens
                or 0
            )
            get_cost_ledger().record(self.model, int(prompt_tokens), int(max(0, output_tokens)))
        except Exception as e:
            logger.debug(f"[Gemini] Kosten konnten nicht erfasst werden: {e}")

    @staticmethod
    def _load_default_config() -> dict:
        """Laedt die Default-Config als Fallback bei API/Parsing-Fehlern."""
        default_path = Path(__file__).parent.parent / "config" / "default.json"
        try:
            with open(default_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return {
                "params": cfg.get("visual", {}).get("params", {}),
                "colors": cfg.get("visual", {}).get("colors", {}),
                "postprocess": cfg.get("postprocess", {}),
                "background": {"opacity": 0.3, "blur": 0.0, "vignette": 0.0},
                "quotes": {},
            }
        except Exception as e:
            logger.warning(f"[Gemini] Konnte default.json nicht laden: {e}")
            return {}

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
                              max_quotes: int = None,
                              use_cache: bool = True) -> concurrent.futures.Future:
        """Asynchrone Zitat-Extraktion. Gibt ein Future zurueck."""
        return self._executor.submit(
            self.extract_quotes, audio_path, audio_duration, max_quotes,
            None, use_cache
        )

    def optimize_all_settings_async(self, visualizer_type: str, current_params: dict,
                                     audio_features: dict, colors: dict,
                                     param_specs: dict = None,
                                     user_prompt: str = None,
                                     recommendation: dict = None) -> concurrent.futures.Future:
        """Asynchrone Parameter-Optimierung. Gibt ein Future zurueck."""
        return self._executor.submit(
            self.optimize_all_settings, visualizer_type, current_params,
            audio_features, colors, param_specs, user_prompt, recommendation
        )

    def generate_background_prompt_async(self, audio_features: dict) -> concurrent.futures.Future:
        """Asynchrone Prompt-Generierung. Gibt ein Future zurueck."""
        return self._executor.submit(self.generate_background_prompt, audio_features)

    def _upload_audio_with_retry(self, audio_path: str, max_retries: int = 3,
                                   progress_callback=None):
        """
        Laedt Audio zu Gemini hoch mit Retry-Logik, Komprimierung und Warten auf ACTIVE.
        Nutzt gecachte Upload-IDs, um wiederholte Uploads zu vermeiden.
        
        Args:
            audio_path: Pfad zur Audio-Datei
            max_retries: Maximale Anzahl Upload-Versuche
            progress_callback: Optional callback(status_msg) fuer Fortschrittsupdates
            
        Returns:
            Das hochgeladene File-Objekt (Status ACTIVE)
        """
        audio_path = Path(audio_path)
        original_size = audio_path.stat().st_size / (1024 * 1024)  # MB
        
        # === Cache-Check: Vorhandene Upload-ID wiederverwenden ===
        cached_id = load_upload_id(str(audio_path))
        if cached_id:
            if progress_callback:
                progress_callback("Cache-Check...")
            try:
                cached_file = self.client.files.get(name=cached_id)
                state_name = getattr(cached_file.state, 'name', str(cached_file.state))
                if state_name == "ACTIVE":
                    if progress_callback:
                        progress_callback("Gecachte Upload-ID verwendet")
                    logger.info(f"[Gemini] Verwende gecachte Upload-ID: {cached_id}")
                    return cached_file
                else:
                    logger.info(f"[Gemini] Gecachte Upload-ID nicht mehr ACTIVE ({state_name}), neu hochladen...")
            except Exception as e:
                logger.warning(f"[Gemini] Gecachte Upload-ID ungueltig: {e}")
        
        # Wenn Datei > 5MB, vorher komprimieren
        upload_path = str(audio_path)
        temp_compressed = None
        
        if original_size > 5:
            if progress_callback:
                progress_callback("Audio komprimieren...")
            temp_compressed = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_compressed.close()
            if _compress_audio_for_upload(str(audio_path), temp_compressed.name):
                compressed_size = os.path.getsize(temp_compressed.name) / (1024 * 1024)
                if progress_callback:
                    progress_callback(f"Audio komprimiert: {compressed_size:.1f}MB")
                logger.info(f"[Gemini] Audio komprimiert: {original_size:.1f}MB -> {compressed_size:.1f}MB")
                upload_path = temp_compressed.name
            else:
                if progress_callback:
                    progress_callback("Komprimierung fehlgeschlagen, verwende Original")
                logger.warning(f"[Gemini] Komprimierung fehlgeschlagen, verwende Original ({original_size:.1f}MB)")
        
        if progress_callback:
            progress_callback("Zu Gemini hochladen...")
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                uploaded_file = self.client.files.upload(file=upload_path)
                
                # WICHTIG: Auf ACTIVE warten! Sonst bricht Gemini die Verbindung ab.
                max_wait = 60  # Max 60 Sekunden warten
                waited = 0
                if progress_callback:
                    progress_callback("Warte auf Verarbeitung...")
                while getattr(uploaded_file.state, 'name', str(uploaded_file.state)) == "PROCESSING" and waited < max_wait:
                    time.sleep(2)
                    waited += 2
                    uploaded_file = self.client.files.get(name=uploaded_file.name)
                    state_name = getattr(uploaded_file.state, 'name', str(uploaded_file.state))
                    if progress_callback:
                        progress_callback(f"Verarbeitung... ({waited}s)")
                    logger.info(f"[Gemini] Datei-Status: {state_name} ({waited}s)")
                
                state_name = getattr(uploaded_file.state, 'name', str(uploaded_file.state))
                if state_name != "ACTIVE":
                    raise RuntimeError(f"Datei nicht ACTIVE nach Upload: {state_name}")
                
                # Upload-ID cachen fuer zukuenftige Verwendung
                save_upload_id(str(audio_path), uploaded_file.name)
                
                if progress_callback:
                    progress_callback("Upload fertig")
                
                # Cleanup temp file
                if temp_compressed and os.path.exists(temp_compressed.name):
                    os.unlink(temp_compressed.name)
                return uploaded_file
            except Exception as e:
                last_error = e
                logger.warning(f"[Gemini] Upload Versuch {attempt}/{max_retries} fehlgeschlagen: {e}")
                if progress_callback:
                    progress_callback(f"Upload Versuch {attempt}/{max_retries} fehlgeschlagen")
                if attempt < max_retries:
                    import random
                    base_wait = 2 * (2 ** (attempt - 1))  # Exponential: 2s, 4s, 8s
                    wait = base_wait + random.uniform(0, 1.0)  # + Jitter bis 1s
                    logger.info(f"[Gemini] Warte {wait:.1f}s vor naechstem Versuch...")
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

            # Transkript-Cache prüfen
            cached = load_transcript(str(audio_path))
            if cached:
                logger.info("[Gemini] Gecachtes Transkript verwendet.")
                return cached

            uploaded_file = self._upload_audio_with_retry(str(audio_path))

            prompt = (
                "Erstelle ein genaues Transkript dieses Audios. "
                "Gib nur den gesprochenen Text aus, keine zusätzlichen Kommentare."
            )

            response = self._call_gemini_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt, uploaded_file]
                )
            )

            transcript = response.text.strip()
            # Transkript cachen
            save_transcript(str(audio_path), transcript)
            return transcript
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Unerwarteter Fehler bei der Transkription: {e}") from e

    def extract_quotes(self, audio_path: str, audio_duration: float = None,
                        max_quotes: int = None, progress_callback=None,
                        use_cache: bool = True) -> List[Quote]:
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
            progress_callback: Optional callback(status_msg) fuer Fortschrittsupdates

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

            # Result-Cache pruefen (spart Upload + API-Call bei identischer Eingabe)
            cache_sig = f"{self.model}|{PROMPT_VERSION}|{max_quotes}"
            if use_cache:
                cached = load_json_result(str(audio_path), "quotes", cache_sig)
                if cached is not None:
                    logger.info("[Gemini] Gecachte Zitate verwendet.")
                    if progress_callback:
                        progress_callback(f"{len(cached)} Zitate (Cache)")
                    return [
                        Quote(
                            text=q.get("text", ""),
                            start_time=float(q.get("start_time", 0.0)),
                            end_time=float(q.get("end_time", 0.0)),
                            confidence=float(q.get("confidence", 0.5)),
                        )
                        for q in cached
                    ]

            if progress_callback:
                progress_callback("Audio wird vorbereitet...")

            try:
                uploaded_file = self._upload_audio_with_retry(
                    str(audio_path),
                    progress_callback=progress_callback
                )
            except Exception as e:
                raise RuntimeError(f"Audio-Upload zu Gemini fehlgeschlagen: {e}") from e

            if progress_callback:
                progress_callback("KI verarbeitet Audio...")

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

            response = self._call_gemini_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt, uploaded_file],
                    config={
                        "response_mime_type": "application/json",
                    }
                )
            )

            if progress_callback:
                progress_callback("Zitate werden gefiltert...")

            quotes_data = self._parse_json_response(response.text)
            if not isinstance(quotes_data, list):
                logger.warning(f"[Gemini] Zitat-Antwort war kein Array (Typ: {type(quotes_data).__name__}), verwende leere Liste")
                quotes_data = []

            quotes = []
            for q in quotes_data:
                text = str(q.get("text", "")).strip()
                # Mindestlaenge: 3 Woerter (CJK-aware: bei CJK-Zeichen >= 6 Zeichen)
                word_count = len(text.split())
                char_count = len(text)
                if word_count < 3 and char_count < 6:
                    continue
                try:
                    start_t = float(q.get("start_time", 0.0))
                    end_t = float(q.get("end_time", 0.0))
                    conf = float(q.get("confidence", 0.5))
                except (ValueError, TypeError):
                    logger.warning(f"[Gemini] Ungültige Zeitstempel fuer Zitat '{text[:30]}...', überspringe.")
                    continue
                # Endzeit darf nicht vor Startzeit liegen
                if end_t < start_t:
                    end_t = start_t + 0.5
                quotes.append(Quote(
                    text=text,
                    start_time=start_t,
                    end_time=end_t,
                    confidence=conf
                ))

            # --- ADAPTIVE CONFIDENCE-FILTERUNG ---
            base_threshold = 0.6
            filtered = [q for q in quotes if q.confidence >= base_threshold]

            # Zu wenige Zitate -> Threshold senken
            if len(filtered) < 2 and len(quotes) > len(filtered):
                if progress_callback:
                    progress_callback(f"Nur {len(filtered)} Zitate bei 0.6, senke auf 0.4...")
                filtered = [q for q in quotes if q.confidence >= 0.4]

            # Zu viele Zitate bei kurzem Audio -> Threshold erhöhen
            if len(filtered) > 15 and audio_duration is not None and audio_duration < 600:
                if progress_callback:
                    progress_callback(f"{len(filtered)} Zitate, erhöhe auf 0.7...")
                filtered = [q for q in quotes if q.confidence >= 0.7]

            quotes = filtered

            # Nach Confidence sortieren (beste zuerst)
            quotes = sorted(quotes, key=lambda x: x.confidence, reverse=True)

            # Auf max_quotes begrenzen
            quotes = quotes[:max_quotes]

            # Nach Startzeit sortieren fuer finale Ausgabe
            quotes = sorted(quotes, key=lambda x: x.start_time)

            if progress_callback:
                progress_callback(f"{len(quotes)} Zitate extrahiert")

            # Ergebnis cachen (weitere Klicks ohne Parameteraenderung sind gratis)
            if use_cache:
                save_json_result(str(audio_path), "quotes", cache_sig, [
                    {
                        "text": q.text,
                        "start_time": q.start_time,
                        "end_time": q.end_time,
                        "confidence": q.confidence,
                    }
                    for q in quotes
                ])

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
            AKTUELLE PARAMETER: {json.dumps(GeminiIntegration._sanitize_for_json(current_params), indent=2)}
            
            GIB DIE OPTIMALEN PARAMETER ZURUECK als JSON-Objekt.
            
            Regeln:
            - Aggressives Musik (hohes Tempo, hoher RMS) -> Hohe Intensitaet, schnelles Easing, viele Partikel
            - Ruhiger Podcast (niedriges Tempo, niedriger RMS) -> Sanfte Werte, langsames Easing, weniger Partikel
            - Hybrid -> Ausgewogene mittlere Werte
            - Halte dich an die Wertebereiche der aktuellen Parameter
            - Antworte NUR mit JSON, keine Erklaerungen
            """
            
            response = self._call_gemini_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt],
                    config={
                        "response_mime_type": "application/json",
                    }
                )
            )

            optimized = self._parse_json_response(response.text)

            # Validiere und filtere nur bekannte Parameter
            if isinstance(optimized, dict):
                return {k: v for k, v in optimized.items() if k in current_params}
            logger.warning(f"[Gemini] Parameter-Antwort ungueltig (Typ: {type(optimized).__name__}), verwende aktuelle Parameter")
            return current_params

        except Exception as e:
            logger.warning(f"[Gemini] Parameter-Optimierung fehlgeschlagen: {e}")
            return current_params

    @staticmethod
    def _sanitize_for_json(obj):
        """Rekursive Konvertierung von numpy-Typen zu nativen Python-Typen."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: GeminiIntegration._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [GeminiIntegration._sanitize_for_json(v) for v in obj]
        return obj

    @staticmethod
    def _is_valid_hex(color: str) -> bool:
        """Prueft ob ein String ein gueltiger #RRGGBB Hex-Code ist."""
        if not isinstance(color, str):
            return False
        color = color.strip()
        return len(color) == 7 and color.startswith('#') and all(
            c in '0123456789abcdefABCDEF' for c in color[1:]
        )

    def _validate_optimized_result(self, optimized: dict, current_params: dict,
                                   colors: dict, param_specs: dict) -> dict:
        """Validiert und korrigiert das KI-Ergebnis (Clamp, Hex-Check, Defaults)."""
        import re

        default_quotes = {
            "font_size": 52, "box_color": "#1a1a2e", "font_color": "#FFFFFF",
            "position": "bottom", "display_duration": 8.0, "auto_scale_font": True,
            "text_shadow_enabled": True, "box_gradient": True, "accent_line": True,
            "accent_line_color": "#FFC864", "box_padding": 32,
            "box_radius": 16, "box_margin_bottom": 100, "max_width_ratio": 0.75,
            "fade_duration": 0.6, "line_spacing": 1.35, "max_font_size": 72,
            "max_chars_per_line": 40,
        }

        # --- Params validieren ---
        result_params = {}
        raw_params = optimized.get("params") or {}
        for name, val in raw_params.items():
            if name in (param_specs or {}):
                default, min_val, max_val, step = param_specs[name]
                # String-Parameter ohne numerische Bounds
                if min_val is None or max_val is None or step is None:
                    if isinstance(val, str):
                        allowed = {
                            "color_mode": {"chroma", "fixed", "monochrome", "warm", "cool"},
                        }
                        if name in allowed and val in allowed[name]:
                            result_params[name] = val
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = default
                val = max(min_val, min(max_val, val))
                if isinstance(step, int):
                    val = round(val / step) * step
                    val = int(val)
                else:
                    val = round(val / step) * step
                result_params[name] = val
            elif name in current_params:
                # Bekannte aktuelle Parameter, die nicht in specs sind (z.B. String-Parameter)
                if isinstance(val, str):
                    allowed = {
                        "color_mode": {"chroma", "fixed", "monochrome", "warm", "cool"},
                    }
                    if name in allowed and val in allowed[name]:
                        result_params[name] = val
                elif isinstance(val, (int, float)):
                    result_params[name] = val

        # --- Farben validieren ---
        result_colors = {}
        for key, default in [("primary", colors.get("primary", "#FF0055")),
                             ("secondary", colors.get("secondary", "#00CCFF")),
                             ("background", colors.get("background", "#0A0A0A"))]:
            val = optimized.get("colors", {}).get(key)
            result_colors[key] = val if self._is_valid_hex(val) else default

        # --- Postprocess validieren ---
        pp_defaults = {"contrast": 1.0, "saturation": 1.0, "brightness": 0.0,
                       "warmth": 0.0, "film_grain": 0.0}
        result_pp = {}
        raw_pp = optimized.get("postprocess") or {}
        for key, default in pp_defaults.items():
            val = raw_pp.get(key, default)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = default
            if key in ("contrast", "saturation"):
                val = max(0.3, min(2.5, val))
            elif key == "brightness":
                val = max(-0.5, min(0.5, val))
            elif key in ("warmth", "film_grain"):
                val = max(0.0, min(1.0, val))
            result_pp[key] = val

        # --- Hintergrund validieren ---
        bg_defaults = {"opacity": 0.3, "blur": 0.0, "vignette": 0.0}
        result_bg = {}
        raw_bg = optimized.get("background") or {}
        for key, default in bg_defaults.items():
            val = raw_bg.get(key, default)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = default
            result_bg[key] = max(0.0, min(1.0, val))

        # --- Quotes validieren ---
        raw_quotes = optimized.get("quotes") or {}
        result_quotes = {**default_quotes}
        for key, default in default_quotes.items():
            val = raw_quotes.get(key)
            if val is None:
                continue
            if key in ("font_size", "box_padding", "box_radius", "box_margin_bottom",
                       "max_chars_per_line", "max_font_size"):
                try:
                    result_quotes[key] = int(val)
                except (TypeError, ValueError):
                    pass
            elif key in ("display_duration", "fade_duration", "line_spacing", "max_width_ratio"):
                try:
                    result_quotes[key] = max(0.0, float(val))
                except (TypeError, ValueError):
                    pass
            elif key in ("auto_scale_font", "text_shadow_enabled", "box_gradient", "accent_line"):
                result_quotes[key] = bool(val)
            elif key in ("box_color", "font_color", "accent_line_color"):
                if self._is_valid_hex(val):
                    result_quotes[key] = val
            elif key == "position" and val in {"bottom", "center", "top"}:
                result_quotes[key] = val

        return {
            "params": result_params,
            "colors": result_colors,
            "postprocess": result_pp,
            "background": result_bg,
            "quotes": result_quotes,
        }

    def optimize_all_settings(self, visualizer_type: str, current_params: dict,
                              audio_features: dict, colors: dict,
                              param_specs: dict = None,
                              user_prompt: str = None,
                              recommendation: dict = None) -> dict:
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
        # Fallback: kategorienbasierte deterministische Parameter-Berechnung
        def _fallback_params():
            """Berechnet Parameter deterministisch aus Audio-Features.

            Verwendet PARAM_CATEGORIES statt string-basierter Heuristiken, um
            jeden Parameter anhand seiner semantischen Bedeutung zuzuordnen.
            """
            tempo = audio_features.get('tempo', 120)
            mode = audio_features.get('mode', 'music')
            rms_mean = np.clip(audio_features.get('rms_mean', 0.5), 0.0, 1.0)
            rms_std = np.clip(audio_features.get('rms_std', 0.1), 0.0, 1.0)
            onset_mean = np.clip(audio_features.get('onset_mean', 0.3), 0.0, 1.0)
            dynamic_range = np.clip(audio_features.get('dynamic_range', 0.3), 0.0, 1.0)
            brightness = np.clip(audio_features.get('brightness', 0.5), 0.0, 1.0)
            voice_clarity = np.clip(audio_features.get('voice_clarity_mean', 0.0), 0.0, 1.0)

            # Normalisierte Steuergrössen
            speed_factor = np.clip(tempo / 180.0, 0.0, 1.0)
            energy_factor = rms_mean
            dynamics_factor = min(1.0, rms_std * 2 + dynamic_range * 0.5)
            rhythm_factor = min(1.0, onset_mean * 3)
            speech_factor = 1.0 if mode == 'speech' else 0.0

            result = {}
            if not param_specs:
                return current_params.copy()

            for name, (default, min_val, max_val, step) in param_specs.items():
                # String-/Pseudo-Parameter ohne numerische Bounds beibehalten
                if min_val is None or max_val is None or step is None:
                    if isinstance(default, str):
                        result[name] = default
                    continue

                category = _get_param_category(name)

                if category == "intensity":
                    # Dynamik + Energie steuern Intensitaet
                    val = min_val + (max_val - min_val) * (
                        0.3 * energy_factor + 0.4 * dynamics_factor + 0.3 * rhythm_factor
                    )
                elif category == "speed":
                    # Tempo + Rhythmik steuern Geschwindigkeit
                    val = min_val + (max_val - min_val) * (
                        0.5 * speed_factor + 0.3 * rhythm_factor + 0.2 * energy_factor
                    )
                    # Speech-Modus: langsamer
                    if mode == 'speech':
                        val = min_val + (val - min_val) * 0.5
                elif category == "count":
                    # Energie + Groesse des Screens/Details
                    val = min_val + (max_val - min_val) * (
                        0.4 * energy_factor + 0.3 * dynamics_factor + 0.3 * brightness
                    )
                    # Speech: weniger Elemente
                    if mode == 'speech':
                        val = min_val + (val - min_val) * 0.6
                elif category == "size":
                    # Ausgewogen nach Energie und Geschwindigkeit
                    val = min_val + (max_val - min_val) * (
                        0.4 * energy_factor + 0.3 * (1.0 - speed_factor) + 0.3 * brightness
                    )
                elif category == "color":
                    # Farbparameter leicht an Energie und Stimmung anpassen
                    val = min_val + (max_val - min_val) * (
                        0.5 + 0.3 * energy_factor - 0.2 * speech_factor
                    )
                elif category == "reactivity":
                    # Speech braucht weiche Reaktion; Musik direktere
                    if mode == 'speech':
                        val = min_val + (max_val - min_val) * (0.3 + 0.2 * voice_clarity)
                    else:
                        val = min_val + (max_val - min_val) * (0.5 + 0.4 * dynamics_factor)
                elif category == "transform":
                    # Transformations-Parameter nur minimal anpassen
                    val = default
                elif category == "special":
                    # Spezielle Parameter (z.B. Hex-Farben) beibehalten
                    val = default
                else:
                    # Unbekannte Parameter: moderat energieabhaengig
                    val = min_val + (max_val - min_val) * (
                        0.4 + 0.3 * energy_factor + 0.3 * (1.0 - speech_factor)
                    )

                # Auf Step runden
                if isinstance(step, int):
                    val = round(val / step) * step
                    val = int(val)
                else:
                    val = round(val / step) * step

                result[name] = max(min_val, min(max_val, val))

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
                    "text_shadow_enabled": True, "box_gradient": True, "accent_line": True,
                    "accent_line_color": "#FFC864", "box_padding": 36,
                    "box_radius": 20, "box_margin_bottom": 100, "max_width_ratio": 0.7,
                    "fade_duration": 0.8, "line_spacing": 1.5, "max_font_size": 72,
                    "max_chars_per_line": 40,
                }
            else:
                return {
                    "font_size": 48, "box_color": "#0d0d1a", "font_color": "#FFFFFF",
                    "position": "bottom", "display_duration": 6.0, "auto_scale_font": True,
                    "text_shadow_enabled": True, "box_gradient": True, "accent_line": True,
                    "accent_line_color": "#FFC864", "box_padding": 24,
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
            brightness = audio_features.get('brightness', 0.5)
            noisiness = audio_features.get('noisiness', 0.1)
            mode = audio_features.get('mode', 'music')
            tempo = audio_features.get('tempo', 120)
            rms_mean = audio_features.get('rms_mean', 0.5)
            onset_mean = audio_features.get('onset_mean', 0.3)

            rec_text = ""
            if recommendation:
                rec_text = f"""
================================================================================
SMARTMATCHER-EMPFEHLUNG
================================================================================
- Visualizer: {recommendation.get('visualizer', visualizer_type)}
- Konfidenz: {recommendation.get('confidence', 0.0):.0%}
- Begruendung: {recommendation.get('reason', 'Keine')}
- Vorgeschlagene Farben: Primary={recommendation.get('colors', {}).get('primary', '-')}, Secondary={recommendation.get('colors', {}).get('secondary', '-')}, Background={recommendation.get('colors', {}).get('background', '-')}
"""

            system_instruction = (
                "Du bist ein Motion-Graphics-Experte. Optimiere Visualizer-Parameter, "
                "Farben, Post-Process und Quote-Einstellungen fuer ein Audio-Video. "
                "Antworte ausschliesslich mit einem JSON-Objekt. Halte dich an min/max-Grenzen."
            )

            prompt = f"""AUDIO: {audio_features.get('duration', 0):.1f}s, {tempo:.0f} BPM, Modus={mode}
RMS mean={rms_mean:.2f}/std={rms_std:.2f}, Onset mean={onset_mean:.2f}/std={onset_std:.2f}, Transient={transient_mean:.2f}, Voice={voice_clarity_mean:.2f}, Brightness={brightness:.2f}, Noise={noisiness:.2f}
{rec_text}
VISUALIZER: {visualizer_type}
AKTUELLE PARAMETER: {json.dumps(GeminiIntegration._sanitize_for_json(current_params), indent=2)}
AKTUELLE FARBEN: {json.dumps(GeminiIntegration._sanitize_for_json(colors), indent=2)}

PARAMETER-SPEZIFIKATIONEN (Standard, [min..max], Schritt, Bedeutung):
{param_info}

REGELN:
- Passe Werte relativ zum Standardwert an (leicht ±10-20%, moderat ±30-50%, stark ±60-100%).
- Speech: sanfte, langsame Werte, dezente Farben, grosse lesbare Schrift.
- Music + Tempo > 110: aggressiver, kontrastreich, mehr Partikel/Balken.
- Music + Tempo <= 110: fliessend, organisch, warm.
- Post-Process: Speech (contrast 1.05, saturation 0.8, warmth 0.1, grain 0.05), Energy (1.2/1.3/0/0.05), Chill (1.05/0.9/0.2/0.1).
- Farben: Podcast (#667EEA/#764BA2/#1A1A2E), Energy (#FF0055/#00CCFF/#0A0A0A), Chill (#4ECDC4/#96CEB4/#1A1A3E).
- Quotes: Podcast (font_size 52-64, bottom), Musik (40-48, center).

Gib NUR ein JSON-Objekt zurueck. Keine Erklaerungen, kein Markdown.
"""

            if user_prompt:
                prompt += f"\nUser-Wunsch (Prioritaet): {user_prompt}\n"

            response = self._call_gemini_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt],
                    config={
                        "system_instruction": system_instruction,
                        "response_mime_type": "application/json",
                        "response_schema": OPTIMIZE_RESPONSE_SCHEMA,
                        "temperature": 0.2,
                    }
                )
            )

            optimized = self._parse_json_response(response.text)

            if not isinstance(optimized, dict):
                logger.warning(f"[Gemini] KI-Antwort war kein Dictionary (Typ: {type(optimized).__name__}), verwende Fallback")
                cfg_fallback = self._load_default_config()
                if cfg_fallback:
                    return {
                        "params": cfg_fallback.get("params", default_result["params"]),
                        "colors": cfg_fallback.get("colors", default_result["colors"]),
                        "postprocess": {**default_result["postprocess"], **cfg_fallback.get("postprocess", {})},
                        "background": default_result["background"],
                        "quotes": default_result["quotes"],
                    }
                return default_result

            return self._validate_optimized_result(optimized, current_params, colors, param_specs)
            
        except Exception as e:
            logger.warning(f"[Gemini] All-Settings-Optimierung fehlgeschlagen: {e}, verwende Fallback")
            # Versuche default.json zu laden und mit internem Fallback zu mergen
            cfg_fallback = self._load_default_config()
            if cfg_fallback:
                return {
                    "params": cfg_fallback.get("params", default_result["params"]),
                    "colors": cfg_fallback.get("colors", default_result["colors"]),
                    "postprocess": {**default_result["postprocess"], **cfg_fallback.get("postprocess", {})},
                    "background": default_result["background"],
                    "quotes": default_result["quotes"],
                }
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
            
            response = self._call_gemini_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt]
                )
            )

            return response.text.strip().strip('"').strip("'")

        except Exception as e:
            logger.warning(f"[Gemini] Bild-Prompt-Generierung fehlgeschlagen: {e}")
            return "abstract ambient background with soft gradients and atmospheric lighting, cinematic color grading, minimal composition, 8k quality"

    @staticmethod
    def _parse_json_response(text: str):
        """Hilfsmethode: Extrahiert JSON aus der API-Antwort.

        Bei komplettem Parsing-Fehler wird None zurueckgegeben (nicht []),
        damit der Aufrufer zwischen "leeres Array" und "ungueltige Antwort"
        unterscheiden kann.
        """
        if not text or not text.strip():
            logger.warning("[Gemini] API-Antwort war leer")
            return None

        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback 1: JSON aus Markdown-Code-Bloecken extrahieren
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError, ValueError):
            pass

        # Fallback 2: Suche nach dem ersten { und letzten }
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

        # Fallback 3: Suche nach dem ersten [ und letzten ]
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

        logger.warning(f"[Gemini] JSON-Parsing fehlgeschlagen. Antwort (erste 200 Zeichen): {text[:200]!r}")
        return None
