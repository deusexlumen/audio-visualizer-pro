"""
AI Matcher – Smarte Visualizer-Empfehlung basierend auf Audio-Features.

Nutzt die existierende Audio-Analyse, um automatisch den passendsten
Visualizer, Farbpalette und Parameter zu wählen.
Kein externes KI-Modell nötig – alles regelbasiert auf Features.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from .types import AudioFeatures, VisualConfig


class AIRecommendation:
    """Ergebnis einer KI-Empfehlung."""

    def __init__(
        self,
        visualizer: str,
        reason: str,
        confidence: float,
        colors: Dict[str, str],
        params: Dict,
        top_candidates: Optional[List[Tuple[str, float]]] = None,
    ):
        self.visualizer = visualizer
        self.reason = reason
        self.confidence = confidence  # 0.0 - 1.0
        self.colors = colors
        self.params = params
        # Top-3-Kandidaten mit Suitability-Score (Name, Score)
        self.top_candidates = top_candidates or []

    def to_visual_config(self, resolution: Tuple[int, int] = (1920, 1080), fps: int = 60) -> VisualConfig:
        """Wandelt die Empfehlung in eine vollständige VisualConfig um."""
        return VisualConfig(
            type=self.visualizer,
            params=self.params,
            colors=self.colors,
            resolution=resolution,
            fps=fps,
        )


class SmartMatcher:
    """
    Analysiert AudioFeatures und empfiehlt den besten Visualizer.

    Die Logik basiert auf kontinuierlichen Suitability-Scores:
    - Mode (speech/music/hybrid) → grundlegende Kategorie
    - RMS & Onset-Dichte → Energie-Level
    - Tempo → Geschwindigkeit
    - Key → Farbharmonie
    - Zusätzlich: beat_frames, tempogram, mfcc, spectral_rolloff
    """

    # Mapping: Note → Grundfarbton (Hex)
    KEY_COLORS = {
        'C': '#FF6B6B',   # Rot (energisch)
        'C#': '#FF8E53',  # Orange-Rot
        'D': '#FFA726',   # Orange (warm)
        'D#': '#FFCA28',  # Gelb-Orange
        'E': '#FFEE58',   # Gelb (hell)
        'F': '#66BB6A',   # Grün (natur)
        'F#': '#26A69A',  # Türkis
        'G': '#42A5F5',   # Blau (stabil)
        'G#': '#5C6BC0',  # Indigo
        'A': '#AB47BC',   # Lila (emotional)
        'A#': '#EC407A',  # Pink
        'B': '#EF5350',   # Rot-Rosa
    }

    # Visualizer-Kategorien für die Score-Berechnung
    SPEECH_VISUALS = {'typographic', 'voice_flow', 'speech_focus', 'aurora_voice'}
    MUSIC_VISUALS = {
        'pulsing_core', 'spectrum_bars', 'chroma_field', 'particle_swarm',
        'neon_oscilloscope', 'sacred_mandala', 'liquid_blobs',
        'neon_wave_circle', 'frequency_flower',
        'lumina_core', 'voice_flow', 'spectrum_genesis',
        'speech_focus', 'bass_temple', 'orchestral_swell', 'nebula_drift',
    }
    HYBRID_VISUALS = {
        'neon_wave_circle', 'pulsing_core', 'liquid_blobs', 'frequency_flower',
        'spectrum_genesis', 'nebula_drift',
    }

    # Parameter-Profile pro Visualizer für schnelle Default-Vorschläge
    VISUAL_DEFAULTS = {
        'pulsing_core': {'pulse_intensity': 1.0, 'ring_count': 3, 'glow_radius': 1.0, 'bg_brightness': 0.05},
        'spectrum_bars': {'bar_count': 64, 'height_scale': 1.2, 'spacing': 0.25, 'color_shift': 0.0},
        'chroma_field': {'field_resolution': 100, 'connection_dist': 100, 'particle_size': 8},
        'particle_swarm': {'particle_count': 150, 'explosion_threshold': 0.6, 'glow_size': 3, 'trail_length': 5},
        'typographic': {'animation_speed': 0.2, 'bar_width': 4, 'bar_spacing': 2},
        'neon_oscilloscope': {'line_thickness': 3, 'trail_length': 12, 'num_points': 200, 'glow_radius': 16},
        'sacred_mandala': {'rotation_speed': 0.005, 'num_petals': 8, 'layer_count': 3},
        'liquid_blobs': {'blob_count': 6, 'fluidity': 0.5},
        'neon_wave_circle': {'wave_amplitude': 0.8, 'circle_count': 3},
        'frequency_flower': {'num_petals': 8, 'layer_count': 3},
        'speech_focus': {'line_thickness': 2.5, 'vu_segments': 16, 'response_speed': 1.0, 'accent_intensity': 0.55},
        'voice_flow': {'flow_speed': 0.4, 'wave_depth': 0.5, 'breathe_intensity': 0.4, 'line_count': 5},
        'bass_temple': {'bass_intensity': 1.2, 'strobe_threshold': 0.55, 'shockwave_speed': 2.5},
        'lumina_core': {'core_intensity': 1.2, 'ring_count': 4, 'noise_scale': 2.0, 'glow_strength': 0.8},
        'spectrum_genesis': {'bar_count': 64, 'wave_intensity': 1.0, 'glow_radius': 1.0, 'beat_flash': 0.5},
        'orchestral_swell': {'swell_intensity': 1.0, 'particle_count': 64, 'dynamics_response': 1.2},
        'aurora_voice': {'band_count': 4, 'flow_speed': 0.15, 'voice_response': 0.7, 'glow_strength': 0.8},
        'nebula_drift': {'nebula_scale': 2.2, 'nebula_density': 0.8, 'beat_pulse': 0.6, 'particle_count': 40},
    }

    # Visualizer-Beschreibungen für die Reason-Texte
    VISUAL_DESCRIPTIONS = {
        'pulsing_core': 'Ein pulsierender Kern, der sich sanft zur Musik bewegt',
        'spectrum_bars': 'Klassische Frequenz-Balken für energiegeladene Musik',
        'chroma_field': 'Farbharmonien basierend auf der Tonart',
        'particle_swarm': 'Partikel-Explosionen im Takt der Beats',
        'typographic': 'Minimalistische Typografie für ruhigen Content',
        'neon_oscilloscope': 'Retro-Oszilloskop-Look für elektronische Musik',
        'sacred_mandala': 'Meditative Mandala-Muster für entspannte Stimmung',
        'liquid_blobs': 'Organische, fließende Formen für sanfte Übergänge',
        'neon_wave_circle': 'Wellenförmige Kreise für dynamischen Mix',
        'frequency_flower': 'Blumen-artiges Frequenzmuster für melodische Stücke',
        'speech_focus': 'Diskrete Wellenform für Sprach-Content ohne Ablenkung',
        'voice_flow': 'Sanfte Stimm-Atmung für Podcasts und Gespräche',
        'bass_temple': 'Tiefe Bass-Resonanz für kraftvolle Musik',
        'lumina_core': 'Subtiler Leuchtkern für elegante Visuals',
        'spectrum_genesis': 'Fein aufgelöstes Spektrum für detailreiche Musik',
        'orchestral_swell': 'Orchestrale Dynamik für filmische Stimmungen',
        'aurora_voice': 'Ruhige Aurora-Bänder für lange Sprach-Inhalte',
        'nebula_drift': 'Treibende Nebelwolken mit Sternen für Ambient bis EDM',
    }

    def __init__(self):
        pass

    def _extract_features(self, features: AudioFeatures) -> Dict:
        """
        Berechnet aggregierte Merkmale aus den rohen Audio-Features.
        """
        rms_mean = float(np.mean(features.rms))
        rms_std = float(np.std(features.rms))
        onset_mean = float(np.mean(features.onset))
        onset_density = float(np.mean(features.onset > 0.3))  # Anteil "starker" Beats

        # Dynamik-Range: wie sehr schwankt die Lautstärke?
        dynamic_range = rms_std / (rms_mean + 0.001)  # +0.001 vermeidet Division durch Null

        # Spectral features
        brightness = float(np.mean(features.spectral_centroid))
        spectral_rolloff_mean = float(np.mean(features.spectral_rolloff))
        noisiness = float(np.mean(features.zero_crossing_rate))

        # Voice features (fuer bessere Podcast-Erkennung)
        voice_clarity = np.asarray(features.voice_clarity)
        voice_band = np.asarray(features.voice_band)
        voice_clarity_mean = float(voice_clarity.mean()) if voice_clarity.size else 0.0
        voice_band_mean = float(voice_band.mean()) if voice_band.size else 0.0

        # Rhythmik
        beat_frames = np.asarray(features.beat_frames)
        beat_count = int(beat_frames.size)
        beat_density = beat_count / max(1.0, features.duration)

        tempogram = np.asarray(features.tempogram)
        tempogram_mean = float(tempogram.mean()) if tempogram.size else 0.0
        tempogram_std = float(tempogram.std()) if tempogram.size else 0.0

        # Timbre (erste MFCC-Koeffizienten)
        mfcc = np.asarray(features.mfcc)
        if mfcc.size and mfcc.ndim >= 2:
            mfcc_mean = float(mfcc[:, :].mean())
            mfcc_std = float(mfcc[:, :].std())
        else:
            mfcc_mean = 0.0
            mfcc_std = 0.0

        return {
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'onset_mean': onset_mean,
            'onset_density': onset_density,
            'dynamic_range': dynamic_range,
            'brightness': brightness,
            'spectral_rolloff_mean': spectral_rolloff_mean,
            'noisiness': noisiness,
            'voice_clarity_mean': voice_clarity_mean,
            'voice_band_mean': voice_band_mean,
            'tempo': features.tempo,
            'mode': features.mode,
            'key': features.key,
            'beat_count': beat_count,
            'beat_density': beat_density,
            'tempogram_mean': tempogram_mean,
            'tempogram_std': tempogram_std,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'duration': float(features.duration),
        }

    def _compute_suitability_scores(self, f: Dict) -> Dict[str, float]:
        """
        Berechnet kontinuierliche Suitability-Scores für jeden Visualizer.

        Jeder Score liegt zwischen 0.0 und 1.0. Höher = besser passend.
        """
        mode = f['mode']
        tempo = f['tempo']
        rms_mean = f['rms_mean']
        onset_density = f['onset_density']
        dynamic_range = f['dynamic_range']
        brightness = f['brightness']
        spectral_rolloff_mean = f['spectral_rolloff_mean']
        voice_clarity_mean = f['voice_clarity_mean']
        voice_band_mean = f['voice_band_mean']
        beat_density = f['beat_density']
        tempogram_std = f['tempogram_std']
        mfcc_std = f['mfcc_std']

        # Basis-Modus-Scores
        speech_score = (
            voice_clarity_mean * 0.5
            + voice_band_mean * 0.5
            + (1.0 - min(1.0, onset_density * 5.0)) * 0.2
        )
        music_score = (
            onset_density * min(2.0, tempo / 100.0)
            + dynamic_range * 0.3
            + (1.0 - voice_clarity_mean) * 0.2
        )

        if mode == 'speech':
            speech_score += 0.15
        elif mode == 'music':
            music_score += 0.15

        # Fallback auf Analyzer-Modus, wenn keine Voice-Features vorhanden
        if voice_clarity_mean == 0 and voice_band_mean == 0:
            mode = f['mode']
        elif speech_score > 0.35 and music_score < 0.30:
            mode = 'speech'
        elif music_score > 0.25 and speech_score < 0.35:
            mode = 'music'
        else:
            mode = 'hybrid'

        # Normalisierte Hilfsgrößen
        energy = np.clip(rms_mean, 0.0, 1.0)
        speed = np.clip(tempo / 180.0, 0.0, 1.0)
        rhythm_strength = np.clip(onset_density * 5.0, 0.0, 1.0)
        beat_strength = np.clip(beat_density / 2.0, 0.0, 1.0)
        dynamics = np.clip(dynamic_range, 0.0, 1.0)
        high_freq = np.clip(spectral_rolloff_mean, 0.0, 1.0)
        timbre_richness = np.clip(mfcc_std / 0.3, 0.0, 1.0)
        tonal_clarity = np.clip(brightness, 0.0, 1.0)

        # Kategorie-Gewichtung je nach Modus
        if mode == 'speech':
            cat_weights = {name: 1.0 if name in self.SPEECH_VISUALS else 0.0 for name in self.VISUAL_DEFAULTS}
        elif mode == 'music':
            cat_weights = {name: 1.0 if name in self.MUSIC_VISUALS else 0.0 for name in self.VISUAL_DEFAULTS}
        else:
            cat_weights = {name: 1.0 if name in self.HYBRID_VISUALS else 0.0 for name in self.VISUAL_DEFAULTS}

        scores = {}

        # Speech-optimierte Visualizer
        scores['typographic'] = (
            0.40 * (1.0 - energy)
            + 0.30 * (1.0 - dynamics)
            + 0.20 * voice_clarity_mean
            + 0.10 * (1.0 - speed)
        )
        scores['voice_flow'] = (
            0.35 * voice_band_mean
            + 0.25 * voice_clarity_mean
            + 0.20 * (1.0 - rhythm_strength)
            + 0.20 * (1.0 - speed)
        )
        scores['speech_focus'] = (
            0.30 * dynamics
            + 0.25 * voice_clarity_mean
            + 0.25 * (1.0 - rhythm_strength)
            + 0.20 * (1.0 - energy)
        )
        # Aurora Voice: ruhige Sprach-Baender, ideal fuer lange, gleichmaessige Rede
        scores['aurora_voice'] = (
            0.35 * voice_band_mean
            + 0.30 * (1.0 - rhythm_strength)
            + 0.20 * (1.0 - speed)
            + 0.15 * (1.0 - dynamics)
        )

        # Musik-Visualizer
        scores['spectrum_bars'] = (
            0.30 * energy
            + 0.25 * rhythm_strength
            + 0.20 * speed
            + 0.15 * high_freq
            + 0.10 * timbre_richness
        )
        scores['neon_oscilloscope'] = (
            0.25 * speed
            + 0.25 * rhythm_strength
            + 0.20 * high_freq
            + 0.20 * tonal_clarity
            + 0.10 * energy
        )
        scores['particle_swarm'] = (
            0.30 * rhythm_strength
            + 0.25 * dynamics
            + 0.20 * speed
            + 0.15 * energy
            + 0.10 * beat_strength
        )
        scores['bass_temple'] = (
            0.30 * (1.0 - high_freq)
            + 0.25 * energy
            + 0.20 * rhythm_strength
            + 0.15 * dynamics
            + 0.10 * beat_strength
        )
        scores['lumina_core'] = (
            0.25 * energy
            + 0.25 * dynamics
            + 0.20 * tonal_clarity
            + 0.15 * rhythm_strength
            + 0.15 * timbre_richness
        )
        scores['spectrum_genesis'] = (
            0.30 * high_freq
            + 0.25 * timbre_richness
            + 0.20 * energy
            + 0.15 * rhythm_strength
            + 0.10 * tonal_clarity
        )
        scores['frequency_flower'] = (
            0.35 * tonal_clarity
            + 0.25 * timbre_richness
            + 0.20 * (1.0 - rhythm_strength)
            + 0.15 * energy
            + 0.05 * (1.0 - speed)
        )
        scores['sacred_mandala'] = (
            0.35 * (1.0 - speed)
            + 0.25 * (1.0 - energy)
            + 0.20 * (1.0 - rhythm_strength)
            + 0.10 * tonal_clarity
            + 0.10 * (1.0 - dynamics)
        )
        scores['chroma_field'] = (
            0.30 * timbre_richness
            + 0.25 * tonal_clarity
            + 0.20 * (1.0 - rhythm_strength)
            + 0.15 * energy
            + 0.10 * dynamics
        )
        scores['pulsing_core'] = (
            0.25 * energy
            + 0.25 * rhythm_strength
            + 0.20 * dynamics
            + 0.15 * beat_strength
            + 0.15 * (1.0 - speed)
        )
        scores['liquid_blobs'] = (
            0.30 * (1.0 - speed)
            + 0.25 * dynamics
            + 0.20 * energy
            + 0.15 * timbre_richness
            + 0.10 * (1.0 - rhythm_strength)
        )
        scores['neon_wave_circle'] = (
            0.30 * dynamics
            + 0.25 * energy
            + 0.20 * rhythm_strength
            + 0.15 * speed
            + 0.10 * beat_strength
        )
        scores['orchestral_swell'] = (
            0.30 * timbre_richness
            + 0.25 * dynamics
            + 0.20 * (1.0 - speed)
            + 0.15 * energy
            + 0.10 * tonal_clarity
        )
        # Nebula Drift: atmosphaerischer Nebel, gut fuer Ambient/EDM mit Beat
        scores['nebula_drift'] = (
            0.30 * energy
            + 0.20 * beat_strength
            + 0.20 * timbre_richness
            + 0.15 * dynamics
            + 0.15 * high_freq
        )

        # Kategorie-Gewichtung anwenden
        for name in scores:
            scores[name] = np.clip(scores[name] * cat_weights.get(name, 0.0), 0.0, 1.0)

        return scores

    def rank_visualizers(self, features: AudioFeatures) -> List[Tuple[str, float]]:
        """
        Gibt die Top-3 Visualizer mit ihren Suitability-Scores zurück.

        Returns:
            Liste von Tupeln (visualizer_name, score), absteigend sortiert.
        """
        f = self._extract_features(features)
        scores = self._compute_suitability_scores(f)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:3]

    def _get_color_from_key(self, key: Optional[str], is_minor: bool = False,
                            energy: float = 0.5, dynamic_range: float = 0.5) -> Tuple[str, str, str]:
        """
        Erzeugt eine stimmungs- und energieabhängige Farbpalette aus der Tonart.

        Kombiniert Note, Dur/Moll, RMS-Mittelwert und Dynamik-Range für
        Primary, Secondary und Background.

        Returns:
            Tuple von (primary, secondary, background) als Hex-Codes.
        """
        if not key:
            # Fallback: neutrale Podcast-Farben, leicht angepasst an Energie
            primary = '#667EEA'
            secondary = '#764BA2'
            bg = '#0F0F1A' if is_minor else '#1A1A2E'
            return primary, secondary, bg

        # Extrahiere die Note (erster Buchstabe, evtl. mit #)
        key_clean = key.split()[0]  # "C major" → "C"
        if len(key_clean) >= 2 and key_clean[1] == '#':
            note = key_clean[:2]
        else:
            note = key_clean[0]

        primary = self.KEY_COLORS.get(note, '#667EEA')

        # Secondary: Komplementaerfarbe zur Primary fuer harmonischen Kontrast
        primary_hsv = self._hex_to_hsv(primary)
        secondary_hue = (primary_hsv[0] + 0.5) % 1.0
        # Moll: etwas gedaempfter; hohe Energie: saettiger
        secondary_sat = min(1.0, primary_hsv[1] * (0.8 if is_minor else 0.95))
        secondary_val = min(1.0, primary_hsv[2] * (1.0 + 0.15 * energy))
        secondary = self._hsv_to_hex((secondary_hue, secondary_sat, secondary_val))

        # Background: je nach Modus und Energie dunkler/heller
        if is_minor:
            # Moll: tief, nachtblau
            bg_brightness = 0.04 + 0.06 * energy
        else:
            # Dur: etwas heller, wärmer
            bg_brightness = 0.06 + 0.08 * energy

        # Dynamik-Range leicht einfließen lassen (hohe Dynamik = dunkler für Kontrast)
        bg_brightness *= (1.0 - 0.2 * dynamic_range)
        bg_brightness = np.clip(bg_brightness, 0.02, 0.20)

        bg = self._hsv_to_hex((primary_hsv[0], 0.25, bg_brightness))

        # Primary selbst an Energie anpassen (leicht aufgehellt/satürter)
        primary_sat = min(1.0, primary_hsv[1] * (0.9 + 0.2 * energy))
        primary_val = min(1.0, primary_hsv[2] * (0.95 + 0.1 * energy))
        primary = self._hsv_to_hex((primary_hsv[0], primary_sat, primary_val))

        return primary, secondary, bg

    @staticmethod
    def _hex_to_hsv(hex_color: str) -> Tuple[float, float, float]:
        """Wandelt Hex-Farbe in HSV-Tupel (0.0-1.0) um."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0

        mx = max(r, g, b)
        mn = min(r, g, b)
        diff = mx - mn

        if diff == 0:
            h = 0.0
        elif mx == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360

        s = 0.0 if mx == 0 else diff / mx
        v = mx
        return (h / 360.0, s, v)

    @staticmethod
    def _hsv_to_hex(hsv: Tuple[float, float, float]) -> str:
        """Wandelt HSV-Tupel (0.0-1.0) in Hex-Farbe um."""
        h, s, v = hsv
        h = h % 1.0
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        i = i % 6
        rgb_map = [
            (v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q),
        ]
        r, g, b = rgb_map[i]
        return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'

    def _darken_color(self, hex_color: str, factor: float) -> str:
        """Dunkelt eine Hex-Farbe ab."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
        return f'#{r:02x}{g:02x}{b:02x}'

    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Hellt eine Hex-Farbe auf."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f'#{r:02x}{g:02x}{b:02x}'

    def _select_params(self, visualizer: str, f: Dict) -> Dict:
        """
        Wählt Parameter für den empfohlenen Visualizer basierend auf Features.
        """
        defaults = self.VISUAL_DEFAULTS.get(visualizer, {}).copy()
        energy = f['rms_mean']
        dynamics = f['dynamic_range']
        tempo = f['tempo']
        speed = np.clip(tempo / 180.0, 0.0, 1.0)

        if visualizer == 'typographic':
            defaults['animation_speed'] = 0.1 + 0.1 * speed
            defaults['bar_width'] = 4
            defaults['bar_spacing'] = 2
        elif visualizer == 'voice_flow':
            defaults['flow_speed'] = 0.25 + 0.25 * speed
            defaults['wave_depth'] = 0.4 + 0.3 * energy
            defaults['breathe_intensity'] = 0.3 + 0.2 * (1.0 - dynamics)
            defaults['line_count'] = 5
        elif visualizer == 'speech_focus':
            defaults['line_thickness'] = 2.0 + 1.5 * energy
            defaults['vu_segments'] = 12 + int(12 * energy)
            defaults['response_speed'] = 0.7 + 0.6 * dynamics
        elif visualizer == 'spectrum_bars':
            defaults['height_scale'] = 0.8 + 0.8 * energy
            defaults['bar_count'] = 48 + int(32 * energy)
            defaults['color_shift'] = 0.05 * speed
        elif visualizer == 'neon_oscilloscope':
            defaults['line_thickness'] = 2 + 2 * energy
            defaults['trail_length'] = 8 + int(8 * speed)
            defaults['num_points'] = 150 + int(150 * energy)
        elif visualizer == 'particle_swarm':
            defaults['particle_count'] = 80 + int(120 * energy)
            defaults['explosion_threshold'] = max(0.2, 0.7 - 0.3 * dynamics)
            defaults['glow_size'] = 2 + 2 * energy
        elif visualizer == 'sacred_mandala':
            defaults['rotation_speed'] = 0.002 + 0.01 * speed
            defaults['num_petals'] = 6 + int(6 * (1.0 - speed))
        elif visualizer == 'bass_temple':
            defaults['bass_intensity'] = 0.8 + 1.6 * energy
            defaults['strobe_threshold'] = max(0.3, 0.65 - 0.2 * dynamics)
            defaults['shockwave_speed'] = 1.5 + 3.0 * speed
        elif visualizer == 'lumina_core':
            defaults['core_intensity'] = 0.8 + 1.6 * energy
            defaults['glow_strength'] = 0.5 + 1.2 * energy
            defaults['pulse_intensity'] = 0.2 + 0.6 * dynamics
        elif visualizer == 'orchestral_swell':
            defaults['swell_intensity'] = 0.6 + 1.0 * dynamics
            defaults['dynamics_response'] = 0.8 + 1.2 * dynamics
            defaults['particle_count'] = 32 + int(64 * energy)

        return defaults

    def _build_reason(self, visualizer: str, f: Dict, top3: List[Tuple[str, float]]) -> str:
        """Erzeugt einen aussagekräftigen Reason-Text."""
        desc = self.VISUAL_DESCRIPTIONS.get(visualizer, 'Passender Visualizer')
        mode_text = {
            'speech': 'Sprach-Content',
            'music': 'Musik',
            'hybrid': 'Mix aus Sprache und Musik',
        }.get(f['mode'], f['mode'])

        runners_up = ', '.join(f"{name} ({score:.0%})" for name, score in top3[1:] if score > 0)
        reason = f"{mode_text} erkannt: {desc}."
        if runners_up:
            reason += f" Alternativen: {runners_up}."
        return reason

    def match(self, features: AudioFeatures) -> AIRecommendation:
        """
        Hauptmethode: Empfiehlt Visualizer + Config basierend auf Audio-Features.
        """
        f = self._extract_features(features)

        # Top-3-Kandidaten mit kontinuierlichen Scores
        top3 = self.rank_visualizers(features)
        visualizer = top3[0][0]
        top_score = top3[0][1]

        # Mode-Erkennung (konsistent mit den Scores)
        mode = f['mode']
        if f['voice_clarity_mean'] == 0 and f['voice_band_mean'] == 0:
            pass  # Vertraue Analyzer-Modus
        else:
            speech_score = (
                f['voice_clarity_mean'] * 0.5
                + f['voice_band_mean'] * 0.5
                + (1.0 - min(1.0, f['onset_density'] * 5.0)) * 0.2
            )
            music_score = (
                f['onset_density'] * min(2.0, f['tempo'] / 100.0)
                + f['dynamic_range'] * 0.3
                + (1.0 - f['voice_clarity_mean']) * 0.2
            )
            if mode == 'speech':
                speech_score += 0.15
            elif mode == 'music':
                music_score += 0.15
            if speech_score > 0.35 and music_score < 0.30:
                mode = 'speech'
            elif music_score > 0.25 and speech_score < 0.35:
                mode = 'music'
            else:
                mode = 'hybrid'

        # Key für Farben
        key_str = f['key'] or ''
        is_minor = 'minor' in key_str.lower() if key_str else False
        primary, secondary, bg = self._get_color_from_key(
            f['key'], is_minor, f['rms_mean'], f['dynamic_range']
        )

        params = self._select_params(visualizer, f)
        reason = self._build_reason(visualizer, f, top3)

        # Confidence aus Top-Score ableiten, aber sinnvoll begrenzen
        confidence = 0.55 + 0.35 * min(1.0, top_score / 0.6)
        confidence = round(np.clip(confidence, 0.55, 0.95), 2)

        # Farbpalette zusammenbauen
        colors = {
            'primary': primary,
            'secondary': secondary,
            'background': bg,
        }

        return AIRecommendation(
            visualizer=visualizer,
            reason=reason,
            confidence=confidence,
            colors=colors,
            params=params,
            top_candidates=top3,
        )
