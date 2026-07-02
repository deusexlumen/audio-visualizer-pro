"""
GPU-Visualisierer Registry.

Bietet eine Rueckwaerts-kompatible Registry mit:
- Manuellem Mapping der eingebauten Visualizer (bestehende 16)
- Automatischer Discovery weiterer Module im Paket via pkgutil/importlib
- Validierung von Visualizer-Klassen
- Refresh-Funktion zur Laufzeit
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

from .base import BaseGPUVisualizer

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


# Manuelle Registry fuer Rueckwaerts-Kompatibilitaet.
# Diese Eintraege haben Vorrang vor automatisch entdeckten Klassen.
_MANUAL_VISUALIZER_MAP = {
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

# Oeffentliches Mapping: wird beim Import und bei refresh_registry() befuellt.
VISUALIZER_MAP = {}


def _class_name_to_snake(name: str) -> str:
    """Wandelt einen Klassennamen wie LuminaCoreGPU in lumina_core um."""
    name = re.sub(r"(GPU|Visualizer)$", "", name)
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def _discover_visualizer_modules():
    """Entdeckt Module im Paket und liefert deren Import-Pfade."""
    package_path = Path(__file__).parent
    modules = []
    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
        if is_pkg or module_name.startswith("_") or module_name == "base":
            continue
        modules.append(f"{__package__}.{module_name}")
    return modules


def _discover_visualizers():
    """Durchsucht das Paket dynamisch nach BaseGPUVisualizer-Subklassen.

    Der Registry-Name entspricht dem Modulnamen (Dateiname ohne .py), damit
    benutzerdefinierte Visualizer vorhersagbar ueber ihren Dateinamen adressierbar sind.
    """
    discovered = {}
    for module_path in _discover_visualizer_modules():
        module_name = module_path.rsplit(".", 1)[-1]
        try:
            module = importlib.import_module(module_path)
        except Exception:
            # Defekte oder nicht ladbare Module werden uebersprungen.
            continue

        candidates = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is BaseGPUVisualizer:
                continue
            if issubclass(obj, BaseGPUVisualizer) and obj.__module__ == module.__name__:
                candidates.append(obj)

        if not candidates:
            continue

        # Wenn mehrere Klassen im Modul liegen, bevorzuge diejenige, deren
        # Klassenname snake_case zum Modulnamen passt.
        chosen = candidates[0]
        for candidate in candidates:
            if _class_name_to_snake(candidate.__name__) == module_name:
                chosen = candidate
                break

        if module_name not in discovered:
            discovered[module_name] = chosen
    return discovered


def refresh_registry():
    """Baut die gemischte Registry aus manuellen und dynamisch entdeckten Eintraegen neu auf."""
    VISUALIZER_MAP.clear()
    VISUALIZER_MAP.update(_MANUAL_VISUALIZER_MAP)
    discovered = _discover_visualizers()
    for name, cls in discovered.items():
        if name not in VISUALIZER_MAP:
            VISUALIZER_MAP[name] = cls


def validate_visualizer_class(cls) -> list[str]:
    """Validiert eine Visualizer-Klasse auf das erwartete Interface.

    Prueft:
    - Klasse erbt von BaseGPUVisualizer
    - render() ist vorhanden und aufrufbar
    - PARAMS hat fuer jeden Eintrag das Format (default, min, max, step)
      mit numerischen Werten

    Args:
        cls: Zu validierende Klasse.

    Returns:
        Liste von Fehlerstrings. Leere Liste bedeutet valide.
    """
    errors: list[str] = []

    if not inspect.isclass(cls):
        errors.append("Keine Klasse uebergeben")
        return errors

    if not issubclass(cls, BaseGPUVisualizer):
        errors.append("Muss von BaseGPUVisualizer erben")

    render_method = getattr(cls, "render", None)
    if render_method is None:
        errors.append("Methode render() fehlt")
    elif not callable(render_method):
        errors.append("render() ist nicht aufrufbar")
    elif getattr(render_method, "__isabstractmethod__", False):
        errors.append("render() ist abstrakt und muss in der Subklasse implementiert werden")

    params = getattr(cls, "PARAMS", {})
    if not isinstance(params, dict):
        errors.append("PARAMS muss ein Dictionary sein")
    else:
        for param_name, spec in params.items():
            if not isinstance(spec, (list, tuple)) or len(spec) != 4:
                errors.append(
                    f"PARAMS['{param_name}'] muss (default, min, max, step) sein"
                )
                continue
            for idx, label in enumerate(["default", "min", "max", "step"]):
                try:
                    float(spec[idx])
                except (TypeError, ValueError):
                    errors.append(
                        f"PARAMS['{param_name}'].{label} muss numerisch sein"
                    )

    return errors


def get_visualizer(name: str):
    """Gibt die Visualizer-Klasse fuer den angegebenen Namen zurueck."""
    if name not in VISUALIZER_MAP:
        raise ValueError(f"Unbekannter Visualizer: {name}. Verfuegbar: {list(VISUALIZER_MAP.keys())}")
    return VISUALIZER_MAP[name]


def list_visualizers():
    """Liste aller registrierten GPU-Visualizer-Namen."""
    return list(VISUALIZER_MAP.keys())


# Registry initial befuellen
refresh_registry()
