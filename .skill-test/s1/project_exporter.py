"""Projekt-Export: buendelt Projekt-JSON + referenzierte Assets als ZIP.

GUI-unabhaengiger Kern. Erkennt fehlende Assets vor dem Export und kann
optional eine Manifest-Datei mit SHA256-Pruefsummen erzeugen.

ZIP-Layout:
    <projektname>/
        project.json          (Original-Projektdatei, unveraendert)
        audio/...             (audio_path)
        backgrounds/...       (background_path)
        intro/...             (intro_path)
        configs/...           (*.json-Referenzen ausser der Projektdatei)
        assets/...            (sonstige *_path-Referenzen)
        manifest.json         (optional, SHA256 + Groessen)
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

EXPORT_VERSION = 1

# Bekannte Pfad-Felder im Projekt-JSON (siehe src/gui/state.py AppState.to_dict)
# und ihre Ziel-Rolle im ZIP.
KNOWN_ASSET_KEYS = {
    "audio_path": "audio",
    "background_path": "backgrounds",
    "intro_path": "intro",
}

_CHUNK = 1024 * 1024  # 1 MiB Lesepuffer fuer SHA256


@dataclass
class AssetEntry:
    """Eine Datei im Export-Plan."""

    key: str            # Feldname im Projekt-JSON ("project" fuer die Projektdatei)
    role: str           # Zielordner im ZIP ("", "audio", "backgrounds", ...)
    source: Path        # Absoluter Quellpfad
    arcname: str        # Pfad innerhalb des ZIP
    exists: bool


@dataclass
class ExportPlan:
    """Vollstaendiger Plan vor dem Export."""

    project_path: Path
    root_name: str
    entries: list[AssetEntry] = field(default_factory=list)

    @property
    def missing(self) -> list[AssetEntry]:
        return [e for e in self.entries if not e.exists]


class MissingAssetsError(Exception):
    """Wird geworfen, wenn Assets fehlen und allow_missing=False ist."""

    def __init__(self, missing: list[AssetEntry]):
        self.missing = missing
        lines = "\n".join(f"  - [{e.key}] {e.source}" for e in missing)
        super().__init__(f"Fehlende Dateien im Projekt:\n{lines}")


def sha256_file(path: Path) -> str:
    """SHA256-Hexdigest einer Datei (chunked, speicherschonend)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _role_for_key(key: str, value: str) -> str:
    """Bestimmt die Ziel-Rolle (Ordner) fuer ein Pfad-Feld."""
    if key in KNOWN_ASSET_KEYS:
        return KNOWN_ASSET_KEYS[key]
    if value.lower().endswith(".json"):
        return "configs"
    return "assets"


def find_asset_references(project: dict) -> dict[str, str]:
    """Findet alle Pfad-Referenzen im Projekt-JSON.

    Bekannte Felder plus generisch alle Strings, deren Key auf '_path'
    endet. Nur Top-Level-Felder (Deckungsgleich mit AppState.to_dict).
    """
    refs: dict[str, str] = {}
    for key, value in project.items():
        if not isinstance(value, str) or not value.strip():
            continue
        if key in KNOWN_ASSET_KEYS or key.endswith("_path"):
            refs[key] = value
    return refs


def build_export_plan(project_path: str | Path) -> ExportPlan:
    """Liest das Projekt-JSON und erstellt den Export-Plan.

    Pfade im Projekt werden relativ zur Projektdatei aufgeloest, falls
    sie nicht absolut sind.
    """
    project_path = Path(project_path).resolve()
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    root_name = project_path.stem
    plan = ExportPlan(project_path=project_path, root_name=root_name)

    # Die Projektdatei selbst liegt immer dabei
    plan.entries.append(AssetEntry(
        key="project", role="", source=project_path,
        arcname=f"{root_name}/project.json", exists=project_path.exists(),
    ))

    used_names: set[str] = {"project.json"}
    for key, raw in find_asset_references(project).items():
        src = Path(raw)
        if not src.is_absolute():
            src = (project_path.parent / src).resolve()
        role = _role_for_key(key, raw)
        name = src.name or "unnamed"
        # Kollisions-Schutz: gleiche Dateinamen im selben Ordner
        candidate = name
        n = 2
        while f"{role}/{candidate}" in used_names:
            candidate = f"{src.stem}_{n}{src.suffix}"
            n += 1
        used_names.add(f"{role}/{candidate}")
        arcname = f"{root_name}/{role}/{candidate}" if role else f"{root_name}/{candidate}"
        plan.entries.append(AssetEntry(
            key=key, role=role, source=src, arcname=arcname, exists=src.is_file(),
        ))
    return plan


def export_project(
    project_path: str | Path,
    zip_path: str | Path,
    include_manifest: bool = True,
    allow_missing: bool = False,
) -> dict:
    """Exportiert das Projekt als ZIP.

    Wirft MissingAssetsError, wenn Dateien fehlen und allow_missing=False.
    Gibt das Manifest-Dict zurueck (auch wenn include_manifest=False —
    dann wird es nur nicht ins ZIP geschrieben).
    """
    plan = build_export_plan(project_path)
    if plan.missing and not allow_missing:
        raise MissingAssetsError(plan.missing)

    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_files = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in plan.entries:
            if not entry.exists:
                continue
            zf.write(entry.source, entry.arcname)
            manifest_files.append({
                "path": entry.arcname,
                "key": entry.key,
                "role": entry.role or "project",
                "size_bytes": entry.source.stat().st_size,
                "sha256": sha256_file(entry.source),
            })

        manifest = {
            "version": EXPORT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "project": plan.project_path.name,
            "file_count": len(manifest_files),
            "missing": [
                {"key": e.key, "expected_path": str(e.source)} for e in plan.missing
            ],
            "files": manifest_files,
        }
        if include_manifest:
            zf.writestr(
                f"{plan.root_name}/manifest.json",
                json.dumps(manifest, indent=2, ensure_ascii=False),
            )
    return manifest


def verify_export(zip_path: str | Path) -> list[str]:
    """Prueft ein Export-ZIP gegen sein manifest.json.

    Rueckgabe: Liste von Problemen (leer = alles ok).
    Wirft ValueError, wenn kein Manifest im ZIP liegt.
    """
    problems: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        manifest_name = next(
            (n for n in zf.namelist() if n.endswith("/manifest.json")), None
        )
        if manifest_name is None:
            raise ValueError("Kein manifest.json im Export-Archiv gefunden.")
        manifest = json.loads(zf.read(manifest_name).decode("utf-8"))

        names = set(zf.namelist())
        for entry in manifest.get("files", []):
            arc = entry["path"]
            if arc not in names:
                problems.append(f"Fehlt im Archiv: {arc}")
                continue
            digest = hashlib.sha256(zf.read(arc)).hexdigest()
            if digest != entry["sha256"]:
                problems.append(f"Pruefsummen-Fehler: {arc}")
            size = zf.getinfo(arc).file_size
            if size != entry["size_bytes"]:
                problems.append(f"Groessen-Fehler: {arc}")
    return problems
