"""GUI-Anbindung fuer den Projekt-Export (PyQt6).

Haengt zur Laufzeit einen "Projekt exportieren"-Knopf in den Studio-Tab
und einen Menue-Eintrag ins Datei-Menue des MainWindow — ohne bestehende
Repo-Dateien zu veraendern.

Verwendung (einmalig nach dem Aufbau des MainWindow aufrufen):

    from export_integration import install_project_export
    install_project_export(main_window)

Ablauf: Projekt muss gespeichert sein (wird ggf. nachgeholt) -> Plan
erstellen -> fehlende Assets melden -> Manifest-Option -> ZIP schreiben.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QLabel, QMessageBox,
    QPushButton, QVBoxLayout,
)
from PyQt6.QtGui import QAction

from project_exporter import (
    MissingAssetsError, build_export_plan, export_project,
)


class _ExportOptionsDialog(QDialog):
    """Kleiner Dialog: Manifest-Option vor dem Export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Projekt exportieren")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Das Projekt wird als ZIP buendelt (Projekt-JSON, Audio,\n"
            "Hintergrundbilder, Configs)."
        ))
        self.manifest_check = QCheckBox("Manifest mit SHA256-Pruefsummen erzeugen")
        self.manifest_check.setChecked(True)
        layout.addWidget(self.manifest_check)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _run_export(window) -> None:
    """Kompletter Export-Flow fuer ein MainWindow."""
    # 1) Projekt muss auf Disk existieren (Export basiert auf dem JSON)
    if getattr(window, "_dirty", False) or not getattr(window, "_project_path", None):
        ret = QMessageBox.question(
            window, "Projekt exportieren",
            "Das Projekt muss vor dem Export gespeichert werden. Jetzt speichern?",
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        window._save_project()
        if not getattr(window, "_project_path", None):
            return  # Nutzer hat Speichern-Dialog abgebrochen

    project_path = Path(window._project_path)

    # 2) Plan + fehlende Assets melden
    try:
        plan = build_export_plan(project_path)
    except Exception as e:
        QMessageBox.critical(window, "Export fehlgeschlagen", str(e))
        return

    allow_missing = False
    if plan.missing:
        details = "\n".join(f"[{e.key}] {e.source}" for e in plan.missing)
        box = QMessageBox(window)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Fehlende Dateien")
        box.setText(f"{len(plan.missing)} referenzierte Datei(en) fehlen:")
        box.setDetailedText(details)
        box.setStandardButtons(
            QMessageBox.StandardButton.Ignore | QMessageBox.StandardButton.Cancel
        )
        box.button(QMessageBox.StandardButton.Ignore).setText("Trotzdem exportieren")
        if box.exec() != QMessageBox.StandardButton.Ignore:
            return
        allow_missing = True

    # 3) Optionen (Manifest) + Ziel-Datei
    dlg = _ExportOptionsDialog(window)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    start = str(project_path.with_suffix(".zip"))
    zip_path, _ = QFileDialog.getSaveFileName(
        window, "Projekt exportieren", start, "ZIP-Archiv (*.zip)"
    )
    if not zip_path:
        return
    if not zip_path.lower().endswith(".zip"):
        zip_path += ".zip"

    # 4) Export
    try:
        manifest = export_project(
            project_path, zip_path,
            include_manifest=dlg.manifest_check.isChecked(),
            allow_missing=allow_missing,
        )
    except MissingAssetsError as e:  # Safety-Net, sollte oben abgefangen sein
        QMessageBox.critical(window, "Export fehlgeschlagen", str(e))
        return
    except Exception as e:
        QMessageBox.critical(window, "Export fehlgeschlagen", str(e))
        return

    msg = f"Export nach {zip_path}\n{manifest['file_count']} Datei(en) gebuendelt."
    if manifest["missing"]:
        msg += f"\n{len(manifest['missing'])} Datei(en) fehlten und wurden uebersprungen."
    QMessageBox.information(window, "Export abgeschlossen", msg)
    if hasattr(window, "_set_status"):
        window._set_status(f"Projekt exportiert: {Path(zip_path).name}", "ok")


def install_project_export(window) -> QPushButton:
    """Installiert Export-Knopf (Studio-Tab) und Menue-Eintrag.

    Gibt den erzeugten Button zurueck (nuetzlich fuer Tests).
    """
    # Menue-Eintrag im Datei-Menue, direkt hinter "Projekt speichern unter…"
    menubar = window.menuBar()
    for action in menubar.actions():
        if action.menu() and action.text().replace("&", "").lower().startswith("datei"):
            act = QAction("Projekt exportieren…", window)
            act.triggered.connect(lambda checked=False: _run_export(window))
            action.menu().addAction(act)
            break

    # Button unten im Studio-Tab
    btn = QPushButton("Projekt exportieren…")
    btn.setToolTip("Projekt mit allen Assets als ZIP exportieren")
    btn.clicked.connect(lambda checked=False: _run_export(window))
    studio = getattr(window, "studio_panel", None)
    if studio is not None and studio.layout() is not None:
        studio.layout().addWidget(btn)
    return btn
