"""
schemas.py - Validierung für Konfigurationsdateien

Pydantic-Schemas für die Validierung von Config-JSONs.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Tuple, Literal


class ColorConfig(BaseModel):
    """Farb-Konfiguration."""
    primary: str = Field(default="#FF0055", regex=r"^#[0-9A-Fa-f]{6}$")
    secondary: str = Field(default="#00CCFF", regex=r"^#[0-9A-Fa-f]{6}$")
    background: str = Field(default="#0A0A0A", regex=r"^#[0-9A-Fa-f]{6}$")


class VisualParams(BaseModel):
    """Visualizer-spezifische Parameter."""
    particle_intensity: float = Field(default=1.0, ge=0.0, le=10.0)
    shake_on_beat: bool = False
    bar_count: int = Field(default=40, ge=10, le=100)
    bar_style: Literal["solid", "gradient", "glow"] = "gradient"
    show_waveform: bool = True
    show_progress: bool = True
    show_time: bool = True
    particle_count: int = Field(default=100, ge=10, le=500)
    connection_distance: int = Field(default=100, ge=50, le=200)
    chroma_influence: float = Field(default=1.0, ge=0.0, le=2.0)
    physics_enabled: bool = True
    gravity_center: bool = True
    explosion_intensity: float = Field(default=1.0, ge=0.5, le=3.0)


class VisualConfigSchema(BaseModel):
    """Schema für Visual-Konfiguration."""
    type: Literal[
        "pulsing_core",
        "spectrum_bars",
        "chroma_field",
        "particle_swarm",
        "typographic"
    ]
    resolution: List[int] = Field(default=[1920, 1080], min_items=2, max_items=2)
    fps: int = Field(default=60, ge=24, le=120)
    colors: ColorConfig = Field(default_factory=ColorConfig)
    params: VisualParams = Field(default_factory=VisualParams)
    
    @validator('resolution')
    def validate_resolution(cls, v):
        if v[0] < 320 or v[1] < 240:
            raise ValueError("Auflösung zu klein (min 320x240)")
        if v[0] > 3840 or v[1] > 2160:
            raise ValueError("Auflösung zu groß (max 4K)")
        return v


class PostProcessConfig(BaseModel):
    """Schema für Post-Processing-Konfiguration."""
    contrast: float = Field(default=1.0, ge=0.5, le=2.0)
    saturation: float = Field(default=1.0, ge=0.0, le=2.0)
    brightness: float = Field(default=1.0, ge=0.5, le=2.0)
    grain: float = Field(default=0.0, ge=0.0, le=1.0)
    vignette: float = Field(default=0.0, ge=0.0, le=1.0)
    chromatic_aberration: float = Field(default=0.0, ge=0.0, le=5.0)
    lut: Optional[str] = None


class ProjectConfigSchema(BaseModel):
    """Vollständiges Schema für Projekt-Konfiguration."""
    audio_file: str
    output_file: str
    visual: VisualConfigSchema
    postprocess: PostProcessConfig = Field(default_factory=PostProcessConfig)
    
    @validator('audio_file')
    def validate_audio_file(cls, v):
        valid_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Audio-Datei muss eine der Endungen haben: {valid_extensions}")
        return v
    
    @validator('output_file')
    def validate_output_file(cls, v):
        if not v.lower().endswith('.mp4'):
            raise ValueError("Output-Datei muss .mp4 Endung haben")
        return v


def validate_config(config_dict: dict) -> ProjectConfigSchema:
    """
    Validiert eine Konfigurations-Dictionary.
    
    Args:
        config_dict: Dictionary mit Konfiguration
    
    Returns:
        Validiertes ProjectConfigSchema
    
    Raises:
        ValidationError: Bei ungültiger Konfiguration
    """
    return ProjectConfigSchema(**config_dict)


def load_and_validate_config(config_path: str) -> ProjectConfigSchema:
    """
    Lädt und validiert eine Konfigurationsdatei.
    
    Args:
        config_path: Pfad zur JSON-Config-Datei
    
    Returns:
        Validiertes ProjectConfigSchema
    """
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return validate_config(config_dict)
