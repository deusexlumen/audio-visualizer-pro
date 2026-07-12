"""
Kosten-Tracking fuer KI-Aufrufe (Gemini/Imagen).

Fuehrt ein Session-Ledger ueber Token-Verbrauch und geschaetzte Kosten.
Preise stammen aus config/settings.json (nie hartcodiert) und koennen sich
aendern, ohne den Code anzufassen.
"""

import threading
from dataclasses import dataclass, field

from .app_logging import get_logger
from .app_settings import load_settings

logger = get_logger(__name__)


@dataclass
class CostLedger:
    """Sammelt Token-Verbrauch und Kosten der aktuellen Sitzung (thread-sicher)."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    image_count: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def record(self, model: str, prompt_tokens: int, output_tokens: int) -> float:
        """Erfasst einen Token-basierten Call und liefert dessen Kosten in USD."""
        settings = load_settings()
        price = settings.price_for_model(model)
        call_cost = (
            prompt_tokens / 1_000_000 * price.input
            + output_tokens / 1_000_000 * price.output
        )
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.output_tokens += output_tokens
            self.cost_usd += call_cost
            self.calls += 1
        logger.info(
            f"[Kosten] {model}: {prompt_tokens}+{output_tokens} Tokens "
            f"= ~{call_cost:.4f} $ (Sitzung: {self.cost_usd:.4f} $)"
        )
        return call_cost

    def record_image(self, model: str, count: int = 1) -> float:
        """Erfasst generierte Bilder und liefert deren Kosten in USD."""
        settings = load_settings()
        call_cost = settings.price_per_image(model) * count
        with self._lock:
            self.image_count += count
            self.cost_usd += call_cost
            self.calls += 1
        logger.info(f"[Kosten] {model}: {count} Bild(er) = ~{call_cost:.4f} $")
        return call_cost

    def summary(self) -> str:
        """Kurze deutsche Zusammenfassung fuer die GUI."""
        with self._lock:
            if self.calls == 0:
                return "Sitzung: noch keine KI-Kosten"
            teile = [f"~{self.cost_usd:.2f} $"]
            if self.total_tokens:
                teile.append(f"{self.total_tokens:,} Tokens".replace(",", "."))
            if self.image_count:
                teile.append(f"{self.image_count} Bild(er)")
            return "Sitzung: " + " · ".join(teile)

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.output_tokens = 0
            self.image_count = 0
            self.cost_usd = 0.0
            self.calls = 0


_ledger: CostLedger | None = None


def get_cost_ledger() -> CostLedger:
    """Liefert das globale Session-Ledger (Singleton)."""
    global _ledger
    if _ledger is None:
        _ledger = CostLedger()
    return _ledger
