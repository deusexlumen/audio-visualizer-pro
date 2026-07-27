"""Provenance-Sidecar (Spec §12).

Jede Engine-Entscheidung wird reproduzierbar protokolliert:
<output>.studio.json mit schema_version studio-decision/2.1.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "studio-decision/2.1"

REQUIRED_SECTIONS = (
    "input", "mode", "profile", "thresholds", "mask",
    "sampling", "solver", "verify", "renderer",
)


def build_sidecar(sections: dict) -> dict:
    """Baut das Sidecar-Dict; fehlende Pflichtblöcke sind ein Fehler."""
    missing = [s for s in REQUIRED_SECTIONS if s not in sections]
    if missing:
        raise ValueError(f"Sidecar-Pflichtblöcke fehlen: {', '.join(missing)}")
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    sidecar.update(sections)
    return sidecar


def write_sidecar(output_path: str, sidecar: dict) -> str:
    """Schreibt <output>.studio.json neben die Output-Datei."""
    out = Path(output_path)
    sidecar_path = out.with_suffix("").with_suffix(".studio.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str))
    return str(sidecar_path)
