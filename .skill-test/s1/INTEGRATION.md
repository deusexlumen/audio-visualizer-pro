# Projekt-Export — Integration

## Dateien

- `project_exporter.py` — GUI-unabhaengiger Kern (Plan, ZIP, SHA256-Manifest, Verifikation)
- `export_integration.py` — PyQt6-Anbindung (Button im Studio-Tab + Menue-Eintrag)
- `test_project_exporter.py` — 9 Tests, alle gruen
- `podcast_export.zip` — Demo-Export des echten `projects/podcast.json`

## Verdrahtung (sobald Repo-Schreibzugriff erlaubt ist)

1. `project_exporter.py` nach `src/project_exporter.py` verschieben.
2. In `src/gui/main_window.py` am Ende von `MainWindow.__init__` eine Zeile ergaenzen:

```python
from src.project_exporter import ...  # bzw. Export-Logik direkt hier
```

   Alternativ die Standalone-Variante nutzen — `export_integration.py`
   haengt Button und Menueeintrag zur Laufzeit an, ohne Repo-Aenderung:

```python
# in gui.py nach dem Erzeugen des MainWindow:
sys.path.insert(0, ".skill-test/s1")
from export_integration import install_project_export
install_project_export(window)
```

3. Tests nach `tests/test_project_exporter.py` verschieben und den
   `sys.path.insert` im Testfile entfernen.

## Verhalten

- Export basiert auf dem gespeicherten Projekt-JSON (`AppState.to_dict`);
  ungespeicherte Aenderungen loesen einen Speichern-Dialog aus.
- Pfad-Felder: `audio_path`, `background_path`, `intro_path` + generisch alle
  `*_path`-Strings; `*.json`-Referenzen landen unter `configs/`.
- Fehlende Dateien werden vor dem Export gemeldet (Dialog mit Detail-Liste,
  optional "Trotzdem exportieren"); sonst Abbruch via `MissingAssetsError`.
- Manifest (optional, Default an): `manifest.json` im ZIP mit SHA256 +
  Groesse je Datei; `verify_export(zip)` prueft das Archiv.
