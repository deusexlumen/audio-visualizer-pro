"""
schemas.py - Validierung fuer Konfigurationsdateien

Pydantic v2 Schemas fuer die Validierung von Config-JSONs.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

# Statische Fallback-Liste der eingebauten Visualizer. Wird nur genutzt, wenn
# die Laufzeit-Registry nicht ladbar ist (z.B. Validierung ohne OpenGL/Import).
_BUILTIN_VISUALIZER_NAMES = {
    "pulsing_core", "spectrum_bars", "chroma_field", "particle_swarm",
    "typographic", "neon_oscilloscope", "sacred_mandala", "liquid_blobs",
    "neon_wave_circle", "frequency_flower", "lumina_core", "voice_flow",
    "spectrum_genesis", "speech_focus", "bass_temple", "orchestral_swell",
    "aurora_voice", "nebula_drift", "glass_prism",
}


def _known_visualizer_names() -> set:
    """Liefert die gueltigen Visualizer-Namen (Registry, sonst Fallback-Liste).

    Der Import der Registry kann OpenGL/Module ziehen; schlaegt er fehl, faellt
    die Validierung auf die statische Liste zurueck statt hart zu scheitern.
    """
    try:
        from src.gpu_visualizers import list_visualizers
        names = set(list_visualizers())
        return names | _BUILTIN_VISUALIZER_NAMES
    except Exception:
        return set(_BUILTIN_VISUALIZER_NAMES)


class ColorConfig(BaseModel):
    """Farb-Konfiguration als Hex-Strings."""
    primary: str = Field(default="#FF0055", pattern=r"^#[0-9A-Fa-f]{6}$")
    secondary: str = Field(default="#00CCFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    background: str = Field(default="#0A0A0A", pattern=r"^#[0-9A-Fa-f]{6}$")


class RGBAColor(list):
    """Hilfsklasse fuer RGBA-Farben als Listen (Runtime-Typ)."""
    pass


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    """Wandelt einen 6-stelligen Hex-String in ein RGBA-Tupel um."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


class QuoteSchema(BaseModel):
    """Schema fuer ein einzelnes Zitat in der Config."""
    text: str = Field(..., min_length=1)
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., ge=0.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_end_after_start(self):
        if self.end_time < self.start_time:
            self.end_time = self.start_time + 0.5
        return self


class QuoteOverlayConfigSchema(BaseModel):
    """Schema fuer Quote-Overlay-Konfiguration.

    Runtime verwendet RGBA-Tupel; in JSON koennen Farben als Hex-String
    oder als [r,g,b,a]-Liste angegeben werden.
    """
    enabled: bool = True
    font_size: int = Field(default=52, ge=16, le=96)
    font_color: Union[str, List[int]] = Field(default="#FFFFFF")
    box_color: Union[str, List[int]] = Field(default="#1A1A2E")
    box_alpha: int = Field(default=200, ge=0, le=255)
    fade_duration: float = Field(default=0.6, ge=0.0, le=5.0)
    max_chars_per_line: int = Field(default=40, ge=10, le=120)
    line_spacing: int = Field(default=10, ge=0, le=50)
    display_duration: float = Field(default=8.0, ge=1.0, le=60.0)
    position: Literal["bottom", "center", "top"] = "bottom"
    text_shadow_enabled: bool = True
    text_shadow_color: Union[str, List[int]] = Field(default="#000000")
    text_shadow_offset: List[int] = Field(default=[2, 2])
    text_shadow_blur: float = Field(default=2.0, ge=0.0, le=20.0)
    box_gradient: bool = True
    accent_line: bool = True
    accent_line_color: Union[str, List[int]] = Field(default="#FFC864")
    accent_line_height: int = Field(default=3, ge=1, le=20)
    spatial_compensation: bool = True
    compensation_blur: float = Field(default=12.0, ge=0.0, le=50.0)
    compensation_darken: float = Field(default=0.55, ge=0.0, le=1.0)
    latency_offset: float = Field(default=0.0, ge=-5.0, le=5.0)
    offset_x: int = 0
    offset_y: int = 0
    scale: float = Field(default=1.0, ge=0.1, le=5.0)
    text_align: Literal["left", "center", "right"] = "center"
    auto_scale_font: bool = True
    min_font_size: int = Field(default=16, ge=8, le=96)
    max_font_size: int = Field(default=72, ge=16, le=200)
    box_padding: int = Field(default=32, ge=0, le=200)
    box_radius: int = Field(default=16, ge=0, le=100)
    box_margin_bottom: int = Field(default=100, ge=0, le=500)
    max_width_ratio: float = Field(default=0.75, ge=0.1, le=1.0)
    font_path: Optional[str] = None
    buffer_lookahead: float = Field(default=2.0, ge=0.0, le=10.0)

    @field_validator("font_color", "box_color", "text_shadow_color", "accent_line_color", mode="before")
    @classmethod
    def normalize_color(cls, v: Any) -> Tuple[int, int, int, int]:
        if isinstance(v, str):
            if v.startswith("#") and len(v) == 7:
                return _hex_to_rgba(v)
            raise ValueError(f"Ungueltige Farbe: {v}")
        if isinstance(v, (list, tuple)):
            if len(v) not in (3, 4):
                raise ValueError("RGBA-Farbe muss 3 oder 4 Werte haben")
            for c in v:
                if not isinstance(c, int) or not 0 <= c <= 255:
                    raise ValueError("RGBA-Werte muessen Integers zwischen 0 und 255 sein")
            if len(v) == 3:
                return tuple(v) + (255,)
            return tuple(v)
        raise ValueError(f"Farbe muss Hex-String oder RGBA-Liste sein, erhalten: {v}")


class VisualParams(BaseModel):
    """Visualizer-spezifische Parameter.

    Offenes Dict-Modell, da jeder GPU-Visualizer eigene PARAMS hat.
    """
    model_config = {"extra": "allow"}

    # Universelle Effekt-Parameter (Defaults fuer Rueckwaertskompatibilitaet)
    line_width: float = Field(default=0.003, ge=0.0001, le=0.1)
    trail_length: int = Field(default=0, ge=0, le=60)
    trail_decay: float = Field(default=0.7, ge=0.0, le=1.0)
    brightness: float = Field(default=1.0, ge=0.0, le=3.0)
    color_mode: Literal["chroma", "fixed", "monochrome", "warm", "cool"] = "chroma"
    base_hue: float = Field(default=0.55, ge=0.0, le=1.0)
    color_saturation: float = Field(default=0.7, ge=0.0, le=1.0)


class VisualConfigSchema(BaseModel):
    """Schema fuer Visual-Konfiguration."""
    # Kein festes Literal mehr: neue Visualizer und Studio-Rezepte sollen ohne
    # Schema-Aenderung nutzbar sein. Der Typ wird gegen die Laufzeit-Registry
    # geprueft (siehe Validator), mit Fallback auf eine statische Liste, falls
    # die Registry (z.B. ohne OpenGL) nicht ladbar ist.
    type: str = Field(default="lumina_core")
    resolution: List[int] = Field(default=[1920, 1080], min_length=2, max_length=2)
    fps: int = Field(default=60, ge=1, le=240)
    colors: ColorConfig = Field(default_factory=ColorConfig)
    params: VisualParams = Field(default_factory=VisualParams)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        known = _known_visualizer_names()
        if known and v not in known:
            raise ValueError(
                f"Unbekannter Visualizer '{v}'. Verfuegbar: {', '.join(sorted(known))}"
            )
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("Resolution muss [width, height] sein")
        if v[0] < 320 or v[1] < 240:
            raise ValueError("Aufloesung zu klein (min 320x240)")
        if v[0] > 7680 or v[1] > 4320:
            raise ValueError("Aufloesung zu gross (max 8K)")
        return v


class PostProcessConfig(BaseModel):
    """Schema fuer Post-Processing.

    Hinweis: brightness im GPU-Shader ist ein Offset (-0.5..0.5),
    nicht ein Multiplikator. Schema angepasst.
    """
    exposure: float = Field(default=1.0, ge=0.1, le=4.0)
    contrast: float = Field(default=1.0, ge=0.0, le=3.0)
    saturation: float = Field(default=1.0, ge=0.0, le=3.0)
    brightness: float = Field(default=0.0, ge=-0.5, le=0.5)
    warmth: float = Field(default=0.0, ge=-1.0, le=1.0)
    grain: float = Field(default=0.0, ge=0.0, le=1.0)
    film_grain: float = Field(default=0.0, ge=0.0, le=1.0)
    vignette: float = Field(default=0.0, ge=0.0, le=1.0)
    chromatic_aberration: float = Field(default=0.0, ge=0.0, le=5.0)
    bloom_intensity: float = Field(default=0.6, ge=0.0, le=2.0)
    bloom_threshold: float = Field(default=1.0, ge=0.0, le=3.0)
    bloom_radius: float = Field(default=1.0, ge=0.5, le=2.0)
    lut: Optional[str] = None
    lut_strength: float = Field(default=1.0, ge=0.0, le=1.0)


class BackgroundConfig(BaseModel):
    """Verschachtelte Background-Konfiguration (optional, fuer alte Configs)."""
    image: Optional[str] = None
    blur: float = Field(default=0.0, ge=0.0, le=20.0)
    vignette: float = Field(default=0.0, ge=0.0, le=1.0)
    opacity: float = Field(default=0.3, ge=0.0, le=1.0)


class SceneSchema(BaseModel):
    """Eine Szene der Timeline: ein Visualizer fuer einen Zeitabschnitt."""
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    visualizer: str
    params: Dict[str, Any] = Field(default_factory=dict)
    transition: Literal["cut", "crossfade"] = "crossfade"
    transition_duration: float = Field(default=0.6, ge=0.0, le=5.0)
    label: Optional[str] = None

    @model_validator(mode="after")
    def check_end_after_start(self):
        if self.end <= self.start:
            raise ValueError("Szenen-Ende muss nach dem Start liegen")
        return self

    @field_validator("visualizer")
    @classmethod
    def validate_visualizer(cls, v: str) -> str:
        known = _known_visualizer_names()
        if known and v not in known:
            raise ValueError(f"Unbekannter Visualizer '{v}' in Szene")
        return v


class TimelineSchema(BaseModel):
    """Geordnete, lueckenlose Folge von Szenen ueber die Track-Dauer."""
    scenes: List[SceneSchema] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_sorted_non_overlapping(self):
        scenes = self.scenes
        for a, b in zip(scenes, scenes[1:]):
            if b.start < a.end - 1e-3:
                raise ValueError("Szenen duerfen sich nicht ueberlappen")
            if b.start < a.start:
                raise ValueError("Szenen muessen nach Startzeit sortiert sein")
        return self


class LayerTransformSchema(BaseModel):
    """Transformation einer Studio-Ebene (Verschiebung, Skalierung, Rotation)."""
    offset_x: float = Field(default=0.0, ge=-1.0, le=1.0)
    offset_y: float = Field(default=0.0, ge=-1.0, le=1.0)
    scale: float = Field(default=1.0, ge=0.1, le=4.0)
    rotation_speed: float = Field(default=0.0, ge=-3.0, le=3.0)


class LayerMappingSchema(BaseModel):
    """Audio-Mapping: verknuepft eine Audio-Groesse mit einem Baustein-Parameter."""
    target: str                       # Parametername des Bausteins
    source: str                       # Audio-Uniform (u_energy, ...)
    gain: float = Field(default=0.3, ge=-4.0, le=4.0)
    offset: float = Field(default=0.0, ge=-4.0, le=4.0)
    smooth: float = Field(default=0.2, ge=0.0, le=0.95)


class RecipeLayerSchema(BaseModel):
    """Eine Ebene eines Studio-Rezepts: ein Baustein mit Params und Mappings."""
    block: str
    blend: Literal["add", "screen", "max"] = "add"
    transform: LayerTransformSchema = Field(default_factory=LayerTransformSchema)
    params: Dict[str, float] = Field(default_factory=dict)
    mappings: List[LayerMappingSchema] = Field(default_factory=list)

    @field_validator("block")
    @classmethod
    def validate_block(cls, v: str) -> str:
        try:
            from src.gpu_visualizers.blocks import BLOCK_LIBRARY
            if v not in BLOCK_LIBRARY:
                raise ValueError(
                    f"Unbekannter Baustein '{v}'. Verfuegbar: {', '.join(BLOCK_LIBRARY)}"
                )
        except ImportError:
            pass
        return v


class RecipeSchema(BaseModel):
    """Deklaratives Rezept fuer einen datengetriebenen Studio-Visualizer."""
    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str = ""
    description: str = ""
    mode_hint: Literal["music", "speech", "hybrid"] = "music"
    layers: List[RecipeLayerSchema] = Field(default_factory=list)
    color: ColorConfig = Field(default_factory=ColorConfig)
    version: int = 1

    @model_validator(mode="after")
    def default_display_name(self):
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()
        return self


class ProjectConfigSchema(BaseModel):
    """Vollstaendiges Schema fuer Projekt-Konfiguration."""
    audio_file: str
    output_file: str
    visual: VisualConfigSchema
    postprocess: PostProcessConfig = Field(default_factory=PostProcessConfig)
    quotes: Optional[List[QuoteSchema]] = None
    quote_overlay: QuoteOverlayConfigSchema = Field(default_factory=QuoteOverlayConfigSchema)
    background: Optional[BackgroundConfig] = None
    background_image: Optional[str] = None
    background_blur: float = Field(default=0.0, ge=0.0, le=20.0)
    background_vignette: float = Field(default=0.0, ge=0.0, le=1.0)
    background_opacity: float = Field(default=0.3, ge=0.0, le=1.0)
    background_color: str = Field(default="#0A0A0A", pattern=r"^#[0-9A-Fa-f]{6}$")

    intro_video: Optional[str] = None
    intro_fade_duration: float = Field(default=1.0, ge=0.1, le=2.0)

    # Optionale Szenen-Timeline. Ist sie gesetzt, ueberschreibt sie beim
    # Rendern den einzelnen visual.type (Visualizer-Wechsel ueber die Zeit).
    timeline: Optional[TimelineSchema] = None

    @model_validator(mode="after")
    def flatten_background(self):
        """Verschachtelte Background-Config in flache Felder uebernehmen."""
        if self.background is not None:
            if self.background_image is None and self.background.image is not None:
                self.background_image = self.background.image
            self.background_blur = self.background.blur
            self.background_vignette = self.background.vignette
            self.background_opacity = self.background.opacity
        return self

    @field_validator("audio_file")
    @classmethod
    def validate_audio_file(cls, v: str) -> str:
        valid_extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".opus"]
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Audio-Datei muss eine der Endungen haben: {valid_extensions}")
        return v

    @field_validator("output_file")
    @classmethod
    def validate_output_file(cls, v: str) -> str:
        if not v.lower().endswith(".mp4"):
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
        ValidationError: Bei ungueltiger Konfiguration
    """
    return ProjectConfigSchema(**config_dict)


def load_and_validate_config(config_path: str) -> ProjectConfigSchema:
    """
    Laedt und validiert eine Konfigurationsdatei.

    Args:
        config_path: Pfad zur JSON-Config-Datei

    Returns:
        Validiertes ProjectConfigSchema
    """
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    return validate_config(config_dict)
