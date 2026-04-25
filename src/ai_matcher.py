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
    ):
        self.visualizer = visualizer
        self.reason = reason
        self.confidence = confidence  # 0.0 - 1.0
        self.colors = colors
        self.params = params
    
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
    
    Die Logik basiert auf einfachen Heuristiken:
    - Mode (speech/music/hybrid) → grundlegende Kategorie
    - RMS & Onset-Dichte → Energie-Level
    - Tempo → Geschwindigkeit
    - Key → Farbharmonie
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
        noisiness = float(np.mean(features.zero_crossing_rate))
        
        return {
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'onset_mean': onset_mean,
            'onset_density': onset_density,
            'dynamic_range': dynamic_range,
            'brightness': brightness,
            'noisiness': noisiness,
            'tempo': features.tempo,
            'mode': features.mode,
            'key': features.key,
        }
    
    def _get_color_from_key(self, key: Optional[str], is_minor: bool = False) -> Tuple[str, str, str]:
        """
        Erzeugt eine Farbpalette aus der Tonart.
        
        Returns:
            Tuple von (primary, secondary, background) als Hex-Codes.
        """
        if not key:
            # Fallback: neutrale Podcast-Farben
            return '#667EEA', '#764BA2', '#1A1A2E'
        
        # Extrahiere die Note (erster Buchstabe, evtl. mit #)
        key_clean = key.split()[0]  # "C major" → "C"
        if len(key_clean) >= 2 and key_clean[1] == '#':
            note = key_clean[:2]
        else:
            note = key_clean[0]
        
        primary = self.KEY_COLORS.get(note, '#667EEA')
        
        # Secondary: etwas dunkler/heller je nach Modus
        if is_minor:
            # Moll = dunkler, gedämpfter
            secondary = self._darken_color(primary, 0.3)
            bg = '#0F0F1A'
        else:
            # Dur = heller, offener
            secondary = self._lighten_color(primary, 0.2)
            bg = '#1A1A2E'
        
        return primary, secondary, bg
    
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
    
    def match(self, features: AudioFeatures) -> AIRecommendation:
        """
        Hauptmethode: Empfiehlt Visualizer + Config basierend auf Audio-Features.
        """
        f = self._extract_features(features)
        
        mode = f['mode']
        tempo = f['tempo']
        rms_mean = f['rms_mean']
        onset_density = f['onset_density']
        dynamic_range = f['dynamic_range']
        
        # Key für Farben
        key_str = f['key'] or ''
        is_minor = 'minor' in key_str.lower() if key_str else False
        primary, secondary, bg = self._get_color_from_key(f['key'], is_minor)
        
        # --- ENTSCHEIDUNGSLOGIK ---
        
        if mode == 'speech':
            # Podcast / Sprache – Sub-Genre-Erkennung
            # WICHTIG: Für Speech NUR dezente, podcast-optimierte Visualizer nutzen.
            # Nie pulsing_core, particle_swarm, spectrum_bars etc. – die wirken im
            # Sprach-Modus wie "Kinderdisko" und lenken vom Content ab.
            if rms_mean < 0.25 and dynamic_range < 0.3:
                # News: Sehr gleichmäßig, monoton, ein Sprecher
                visualizer = 'typographic'
                reason = 'News-Format erkannt: Gleichmäßiger Sprecher, sachlicher Ton – klare Typografie passt bestens.'
                confidence = 0.90
                params = {'text_size': 56, 'animation_speed': 0.15, 'bar_width': 4, 'bar_spacing': 2}
            elif dynamic_range < 0.55:
                # Interview: Zwei Sprecher, moderate Dynamik, Pausen
                visualizer = 'voice_flow'
                reason = 'Interview-Format erkannt: Gesprächiger Wechsel, moderate Dynamik – sanfte Stimm-Atmung unterstreicht den Dialog.'
                confidence = 0.85
                params = {'flow_intensity': 0.5, 'smoothness': 0.8}
            elif dynamic_range > 0.85:
                # Story: Viel Dynamik, Soundeffekte, Einspieler
                visualizer = 'speech_focus'
                reason = 'Storytelling erkannt: Hohe Dynamik, atmosphärische Passagen – diskrete Wellenform fängt die Stimmung ein.'
                confidence = 0.88
                params = {'wave_scale': 0.6, 'reactive_intensity': 0.4}
            else:
                # Mixed/Allround Podcast – sicherster Default
                visualizer = 'voice_flow'
                reason = 'Allround-Sprach-Content – sanfte Stimm-Atmung gibt visuelles Feedback ohne Ablenkung.'
                confidence = 0.78
                params = {'flow_intensity': 0.5, 'smoothness': 0.8}
        
        elif mode == 'music':
            # Musik
            if tempo > 120 and onset_density > 0.15:
                if rms_mean > 0.5:
                    visualizer = 'spectrum_bars'
                    reason = 'Energische, laute Musik mit vielen Beats – klassische Spektrum-Balken zeigen die Power.'
                    confidence = 0.90
                    params = {'bar_count': 64, 'smoothing': 0.3}
                else:
                    visualizer = 'neon_oscilloscope'
                    reason = 'Schnelle, aber leise Musik – der Oszilloskop-Look betont die Rhythmus-Struktur.'
                    confidence = 0.85
                    params = {'line_thickness': 3, 'trail_length': 20}
            elif tempo > 100:
                visualizer = 'particle_swarm'
                reason = 'Moderate Musik mit Drive – Partikel-Explosionen passen zum Tempo.'
                confidence = 0.82
                params = {'particle_count': 150, 'explosion_threshold': 0.6}
            elif tempo < 80 and rms_mean < 0.3:
                visualizer = 'sacred_mandala'
                reason = 'Langsame, sanfte Musik – meditative Mandala-Match für entspannte Stimmung.'
                confidence = 0.88
                params = {'rotation_speed': 0.2, 'layer_count': 7}
            else:
                visualizer = 'chroma_field'
                reason = 'Melodische Musik mit klaren Harmonien – Farbfelder basierend auf der Tonart.'
                confidence = 0.80
                params = {'field_resolution': 32, 'color_blend': 0.6}
        
        else:  # hybrid
            if dynamic_range > 0.7:
                visualizer = 'neon_wave_circle'
                reason = 'Mix aus Sprache und Musik mit hoher Dynamik – wellenförmige Kreise fangen beides ein.'
                confidence = 0.78
                params = {'wave_amplitude': 0.8, 'circle_count': 3}
            else:
                visualizer = 'pulsing_core'
                reason = 'Ausgewogener Mix – der pulsierende Kern ist vielseitig genug für hybriden Content.'
                confidence = 0.72
                params = {'pulse_intensity': 0.5, 'glow_radius': 60}
        
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
        )
