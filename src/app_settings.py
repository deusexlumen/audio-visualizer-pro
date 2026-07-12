"""
App-Einstellungen fuer Audio Visualizer Pro.

Laedt und validiert config/settings.json (Modell-IDs, Preistabellen).
Werte lassen sich per Umgebungsvariable ueberschreiben, ohne die Datei
anzufassen — wichtig, weil Modell-IDs und Preise sich aendern und nie
im Code hartcodiert sein sollen.
"""

import json
import os
from typing import Dict, List

from pydantic import BaseModel, Field

from .app_logging import get_logger
from .paths import resource_path

logger = get_logger(__name__)

_SETTINGS_PATH = resource_path("config", "settings.json")


class TokenPrice(BaseModel):
    input: float = 0.15
    output: float = 0.60


class AppSettings(BaseModel):
    """Validierte App-Einstellungen mit sinnvollen Defaults."""

    gemini_model: str = "gemini-flash-lite-latest"
    imagen_model: str = "imagen-4.0-generate-001"
    # Bevorzugte Modell-Familien fuer den Fallback, geordnet nach Wunsch
    model_preference: List[str] = Field(default_factory=lambda: ["flash-lite", "flash"])
    pricing_usd_per_million_tokens: Dict[str, TokenPrice] = Field(
        default_factory=lambda: {"default": TokenPrice()}
    )
    pricing_usd_per_image: Dict[str, float] = Field(
        default_factory=lambda: {"default": 0.04}
    )

    def price_for_model(self, model: str) -> TokenPrice:
        """Liefert die Token-Preise fuer ein Modell (Fallback: 'default')."""
        return self.pricing_usd_per_million_tokens.get(
            model, self.pricing_usd_per_million_tokens.get("default", TokenPrice())
        )

    def price_per_image(self, model: str) -> float:
        """Liefert den Preis pro generiertem Bild (Fallback: 'default')."""
        return self.pricing_usd_per_image.get(
            model, self.pricing_usd_per_image.get("default", 0.04)
        )


_cached: AppSettings | None = None


def load_settings(force_reload: bool = False) -> AppSettings:
    """Laedt die App-Einstellungen (einmalig gecacht).

    Umgebungsvariablen ueberschreiben die Datei:
    - GEMINI_MODEL  -> gemini_model
    - IMAGEN_MODEL  -> imagen_model
    """
    global _cached
    if _cached is not None and not force_reload:
        return _cached

    data: dict = {}
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.info("[Settings] config/settings.json nicht gefunden, nutze Defaults.")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[Settings] settings.json unlesbar ({e}), nutze Defaults.")

    try:
        settings = AppSettings(**data)
    except Exception as e:  # pydantic ValidationError o.ae.
        logger.warning(f"[Settings] settings.json ungueltig ({e}), nutze Defaults.")
        settings = AppSettings()

    env_model = os.environ.get("GEMINI_MODEL")
    if env_model:
        settings.gemini_model = env_model.strip()
    env_imagen = os.environ.get("IMAGEN_MODEL")
    if env_imagen:
        settings.imagen_model = env_imagen.strip()

    _cached = settings
    return settings
