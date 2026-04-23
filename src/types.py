"""
Type Definitions für das Audio Visualizer Pro System.
Pydantic Models für alle Konfigurationen und Audio-Features.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List, Dict, Optional, Tuple
import numpy as np


class Quote(BaseModel):
    """Schema fuer ein Key-Zitat mit Zeitstempel."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


class AudioFeatures(BaseModel):
    """Schema für alle Audio-Features. Einheitlich für alle Renderer."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    duration: float
    sample_rate: int
    fps: int = 60
    
    frame_count: int = 0
    
    def model_post_init(self, __context):
        """Berechnet frame_count automatisch, wenn nicht gesetzt."""
        if self.frame_count == 0:
            self.frame_count = int(self.duration * self.fps)
    
    # Zeitliche Features (Frame-basiert)
    rms: np.ndarray = Field(..., description="Loudness 0.0-1.0")
    onset: np.ndarray = Field(..., description="Beat detection 0.0-1.0")
    spectral_centroid: np.ndarray = Field(..., description="Brightness")
    spectral_rolloff: np.ndarray = Field(..., description="Bandwidth")
    zero_crossing_rate: np.ndarray = Field(..., description="Noisiness vs Tonal")
    
    # Neue Features für Pro-Visualizer
    transient: np.ndarray = Field(default_factory=lambda: np.array([]), description="Kick/Snare transients 0.0-1.0")
    voice_clarity: np.ndarray = Field(default_factory=lambda: np.array([]), description="Voice presence 80Hz-3kHz 0.0-1.0")
    
    # Tonaale Features
    chroma: np.ndarray = Field(..., description="Shape: (12, frames) - C,C#,D...")
    mfcc: np.ndarray = Field(..., description="Timbre fingerprint")
    tempogram: np.ndarray = Field(..., description="Rhythmic structure")
    
    # Metadaten
    tempo: float
    key: Optional[str] = None
    mode: Literal["music", "speech", "hybrid"]
    
    # Beat-Sync (fuer Audio-Sync Features)
    beat_frames: np.ndarray = Field(default_factory=lambda: np.array([]), description="Frame-Indizes der erkannten Beats")


class VisualConfig(BaseModel):
    """Jeder Visualizer hat diese Konfiguration."""
    type: str
    params: Dict = Field(default_factory=dict)
    colors: Dict[str, str] = Field(default_factory=dict)
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 60


class ProjectConfig(BaseModel):
    """Gesamtkonfiguration einer Render-Job."""
    audio_file: str
    output_file: str
    visual: VisualConfig
    postprocess: Dict = Field(default_factory=dict)
    quotes: Optional[List[Quote]] = None
    
    background_image: Optional[str] = None
    background_blur: float = 0.0
    background_vignette: float = 0.0
    background_opacity: float = 0.3
    
    turbo_mode: bool = False
    frame_skip: int = 1
