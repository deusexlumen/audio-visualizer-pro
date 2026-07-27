"""Modus-Profile (Spec §5): Whitelists, Parameter-Korridore, FX-Budgets.

Whitelist-Keys werden beim Laden gegen VISUALIZER_MAP geprüft —
Fail-fast statt stillem Laufzeitfehler.
"""

from pydantic import BaseModel, model_validator


class StudioProfile(BaseModel):
    """Regelwerk eines Modus (MUSIC oder PODCAST)."""

    name: str
    version: int
    mode: str  # "music" | "podcast"
    visualizer_whitelist: list[str]
    param_corridors: dict[str, tuple[float, float]]
    postfx_budget: dict[str, float]
    vitality_corridor: tuple[float, float]
    subject_strength: float = 0.8
    desaturate_colors: bool = False

    @model_validator(mode="after")
    def _whitelist_gegen_registry(self):
        from ..gpu_visualizers import VISUALIZER_MAP
        for key in self.visualizer_whitelist:
            if key not in VISUALIZER_MAP:
                raise ValueError(
                    f"Profil '{self.name}': unbekannter Visualizer '{key}'"
                )
        return self


BUILTIN_PROFILES: dict[str, StudioProfile] = {
    "music_default": StudioProfile(
        name="music_default", version=1, mode="music",
        visualizer_whitelist=[
            "spectrum_bars", "lumina_core", "bass_temple", "particle_swarm",
            "chroma_field", "neon_oscilloscope", "spectrum_genesis",
            "orchestral_swell", "sacred_mandala", "liquid_blobs",
            "neon_wave_circle", "frequency_flower", "pulsing_core",
            "typographic",
        ],
        param_corridors={"intensity": (0.2, 3.0), "speed": (0.2, 5.0)},
        postfx_budget={"bloom_intensity": 1.0, "film_grain": 0.5},
        vitality_corridor=(0.02, 1.0),
    ),
    "podcast_default": StudioProfile(
        name="podcast_default", version=1, mode="podcast",
        visualizer_whitelist=[
            "voice_flow", "speech_focus", "neon_wave_circle",
            "pulsing_core", "aurora_voice", "nebula_drift",
        ],
        param_corridors={"intensity": (0.1, 1.0), "speed": (0.1, 1.0)},
        postfx_budget={"bloom_intensity": 0.4, "film_grain": 0.1},
        vitality_corridor=(0.0, 0.09),
        desaturate_colors=True,
    ),
}


def load_profile(name: str) -> StudioProfile:
    """Lädt ein Built-in-Profil; unbekannte Namen sind ein Fehler."""
    if name not in BUILTIN_PROFILES:
        raise KeyError(f"Unbekanntes Studio-Profil: '{name}'")
    return BUILTIN_PROFILES[name]
