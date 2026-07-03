"""
Zentrales Logging fuer Audio Visualizer Pro.

Schreibt Diagnose-Meldungen sowohl auf die Konsole als auch rotierend
nach logs/app.log, damit Fehlerberichte auch ohne Konsole (GUI-Start)
nachvollziehbar sind.

Verwendung:
    from src.app_logging import get_logger
    logger = get_logger(__name__)
    logger.info("Meldung")
"""

import logging
import logging.handlers
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Initialisiert das Logging einmalig (Konsole + rotierendes Logfile)."""
    global _configured
    if _configured:
        return

    root = logging.getLogger("avp")
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)
    except OSError:
        # Kein Schreibzugriff (z.B. schreibgeschuetztes Verzeichnis) —
        # Konsolen-Logging reicht dann aus.
        pass

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Liefert einen Logger unterhalb des App-Namensraums.

    Stellt sicher, dass das Logging konfiguriert ist, auch wenn
    setup_logging() noch nicht explizit aufgerufen wurde.
    """
    setup_logging()
    short = name.removeprefix("src.").removeprefix("src")
    return logging.getLogger(f"avp.{short}" if short else "avp")
