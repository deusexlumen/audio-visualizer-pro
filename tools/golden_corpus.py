"""Golden-Set-Korpus: Manifest laden und validieren (Spec studio-spec/2.1 §3.5).

Das Manifest (Schema ``golden-corpus/1``) beschreibt die Referenz-Audios
für das Golden Set: id, Pfad (relativ zum Manifest), Modus und eine
menschliche Begründung (description/source), warum die Datei im Korpus ist.
"""

import json
from pathlib import Path

VALID_MODES = {"music", "podcast", "hybrid"}
REQUIRED_FIELDS = ("path", "description", "source")


class CorpusError(ValueError):
    """Ungültiges Korpus-Manifest."""


def load_corpus(path):
    """Lädt corpus.json und validiert das Schema.

    Rückgabe: Liste der Audio-Dicts; ``path`` ist dabei absolut aufgelöst
    (relativ zum Manifest-Verzeichnis).
    """
    manifest = Path(path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("version") != "golden-corpus/1":
        raise CorpusError(f"Unbekannte Version: {data.get('version')!r}")
    audios = data.get("audios")
    if not isinstance(audios, list) or not audios:
        raise CorpusError("Manifest braucht nicht-leere Liste 'audios'.")
    seen: set[str] = set()
    out: list[dict] = []
    for entry in audios:
        eid = entry.get("id")
        if not eid or not isinstance(eid, str):
            raise CorpusError(f"Audio ohne gueltige id: {entry!r}")
        if eid in seen:
            raise CorpusError(f"Doppelte Audio-id: {eid!r}")
        seen.add(eid)
        mode = entry.get("mode")
        if mode not in VALID_MODES:
            raise CorpusError(
                f"{eid}: ungueltiger mode {mode!r} (erlaubt: {sorted(VALID_MODES)})")
        for field in REQUIRED_FIELDS:
            if not entry.get(field):
                raise CorpusError(f"{eid}: Feld {field!r} fehlt oder leer.")
        resolved = (manifest.parent / entry["path"]).resolve()
        out.append({**entry, "path": str(resolved)})
    return out


def missing_audio_files(audios):
    """Liste der ids, deren Datei nicht existiert."""
    return [a["id"] for a in audios if not Path(a["path"]).is_file()]
