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
    # Universelle Effekt-Parameter
    line_width: float = Field(default=0.003, ge=0.001, le=0.02)
    trail_length: int = Field(default=0, ge=0, le=12)
    trail_decay: float = Field(default=0.7, ge=0.1, le=0.95)
    brightness: float = Field(default=1.0, ge=0.5, le=2.0)
    # Farb-Parameter
    color_mode: Literal["chroma", "fixed", "monochrome", "warm", "cool"] = "chroma"
    base_hue: float = Field(default=0.55, ge=0.0, le=1.0)
    color_saturation: float = Field(default=0.7, ge=0.0, le=1.0)
    # Klassische Parameter (rueckwaertskompatibel)
    particle_intensity: float = Field(default=1.0, ge=0.0, le=10.0)
    shake_on_beat: bool = False
    bar_count: int = Field(default=40, ge=10, le=128)
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


class QuoteOverlayConfigSchema(BaseModel):
    """Schema für Quote-Overlay-Konfiguration."""
    enabled: bool = True
    font_size: int = Field(default=52, ge=16, le=96)
    font_color: str = Field(default="#FFFFFF", regex=r"^#[0-9A-Fa-f]{6}$")
    box_color: str = Field(default="#1A1A2E", regex=r"^#[0-9A-Fa-f]{6}$")
    box_alpha: int = Field(default=200, ge=0, le=255)
    fade_duration: float = Field(default=0.6, ge=0.1, le=2.0)
    max_chars_per_line: int = Field(default=40, ge=20, le=80)
    line_spacing: int = Field(default=10, ge=0, le=30)
    display_duration: float = Field(default=8.0, ge=2.0, le=20.0)
    position: Literal["bottom", "center", "top"] = "bottom"
    slide_animation: Literal["none", "up", "down", "left", "right"] = "none"
    scale_in: bool = False
    glow_pulse: bool = False
    compensation_blur: float = Field(default=12.0, ge=0.0, le=30.0)
    latency_offset: float = Field(default=0.0, ge=-2.0, le=2.0)


class VisualConfigSchema(BaseModel):
    """Schema für Visual-Konfiguration."""
    type: Literal[
        # Classic
        "pulsing_core", "spectrum_bars", "chroma_field",
        "particle_swarm", "typographic", "neon_oscilloscope",
        "sacred_mandala", "liquid_blobs", "neon_wave_circle",
        "frequency_flower",
        # Signature Pro
        "lumina_core", "voice_flow", "spectrum_genesis",
        "speech_focus", "bass_temple", "orchestral_swell",
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
    warmth: float = Field(default=0.0, ge=-1.0, le=1.0)
    grain: float = Field(default=0.0, ge=0.0, le=1.0)
    film_grain: float = Field(default=0.0, ge=0.0, le=1.0)
    vignette: float = Field(default=0.0, ge=0.0, le=1.0)
    chromatic_aberration: float = Field(default=0.0, ge=0.0, le=5.0)
    lut: Optional[str] = None


class ProjectConfigSchema(BaseModel):
    """Vollständiges Schema für Projekt-Konfiguration."""
    audio_file: str
    output_file: str
    visual: VisualConfigSchema
    postprocess: PostProcessConfig = Field(default_factory=PostProcessConfig)
    quotes: Optional[List[dict]] = None
    quote_overlay: QuoteOverlayConfigSchema = Field(default_factory=QuoteOverlayConfigSchema)
    background_image: Optional[str] = None
    background_blur: float = Field(default=0.0, ge=0.0, le=20.0)
    background_vignette: float = Field(default=0.0, ge=0.0, le=1.0)
    background_opacity: float = Field(default=0.3, ge=0.0, le=1.0)
    
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
