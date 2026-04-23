"""
GPU-Visualisierer Registry.
Liste aller verfuegbaren GPU-beschleunigten Visualizer.
"""

# Classic Visualizer (bestehende 10)
from .spectrum_bars import SpectrumBarsGPU
from .pulsing_core import PulsingCoreGPU
from .particle_swarm import ParticleSwarmGPU
from .neon_oscilloscope import NeonOscilloscopeGPU
from .chroma_field import ChromaFieldGPU
from .typographic import TypographicGPU
from .sacred_mandala import SacredMandalaGPU
from .liquid_blobs import LiquidBlobsGPU
from .neon_wave_circle import NeonWaveCircleGPU
from .frequency_flower import FrequencyFlowerGPU

# Signature Pro Visualizer (neu in v2.0)
from .lumina_core import LuminaCoreGPU
from .voice_flow import VoiceFlowGPU
from .spectrum_genesis import SpectrumGenesisGPU
from .speech_focus import SpeechFocusGPU
from .bass_temple import BassTempleGPU
from .orchestral_swell import OrchestralSwellGPU

__all__ = [
    "SpectrumBarsGPU", "PulsingCoreGPU", "ParticleSwarmGPU",
    "NeonOscilloscopeGPU", "ChromaFieldGPU", "TypographicGPU",
    "SacredMandalaGPU", "LiquidBlobsGPU", "NeonWaveCircleGPU",
    "FrequencyFlowerGPU",
    # Signature Pro
    "LuminaCoreGPU", "VoiceFlowGPU", "SpectrumGenesisGPU", "SpeechFocusGPU",
    "BassTempleGPU", "OrchestralSwellGPU",
]

# Mapping fuer einfachen Zugriff per Name
VISUALIZER_MAP = {
    # Classic
    "spectrum_bars": SpectrumBarsGPU,
    "pulsing_core": PulsingCoreGPU,
    "particle_swarm": ParticleSwarmGPU,
    "neon_oscilloscope": NeonOscilloscopeGPU,
    "chroma_field": ChromaFieldGPU,
    "typographic": TypographicGPU,
    "sacred_mandala": SacredMandalaGPU,
    "liquid_blobs": LiquidBlobsGPU,
    "neon_wave_circle": NeonWaveCircleGPU,
    "frequency_flower": FrequencyFlowerGPU,
    # Signature Pro
    "lumina_core": LuminaCoreGPU,
    "voice_flow": VoiceFlowGPU,
    "spectrum_genesis": SpectrumGenesisGPU,
    "speech_focus": SpeechFocusGPU,
    "bass_temple": BassTempleGPU,
    "orchestral_swell": OrchestralSwellGPU,
}


def get_visualizer(name: str):
    """Gibt die Visualizer-Klasse fuer den angegebenen Namen zurueck."""
    if name not in VISUALIZER_MAP:
        raise ValueError(f"Unbekannter Visualizer: {name}. Verfuegbar: {list(VISUALIZER_MAP.keys())}")
    return VISUALIZER_MAP[name]


def list_visualizers():
    """Liste aller registrierten GPU-Visualizer-Namen."""
    return list(VISUALIZER_MAP.keys())
