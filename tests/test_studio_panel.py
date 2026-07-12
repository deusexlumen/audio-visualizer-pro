"""Tests fuer das Visualizer-Studio-Panel (ohne GPU)."""

import json

from src.gui.state import AppState
from src.gui.studio_panel import StudioPanel


def _panel(qtbot):
    state = AppState()
    panel = StudioPanel(state, gemini=None)
    qtbot.addWidget(panel)
    return panel


def test_ebene_hinzufuegen_und_entfernen(qtbot):
    panel = _panel(qtbot)
    panel.block_combo.setCurrentIndex(0)  # erster Baustein
    panel._on_add_layer()
    assert len(panel._layers) == 1
    assert panel.layer_list.count() == 1
    panel._on_remove_layer()
    assert len(panel._layers) == 0


def test_ebene_verschieben(qtbot):
    panel = _panel(qtbot)
    panel.block_combo.setCurrentIndex(0)
    panel._on_add_layer()
    panel.block_combo.setCurrentIndex(1)
    panel._on_add_layer()
    first, second = panel._layers[0]["block"], panel._layers[1]["block"]
    panel._current_layer = 1
    panel._move_layer(-1)
    assert panel._layers[0]["block"] == second
    assert panel._layers[1]["block"] == first


def test_speichern_schreibt_rezept(qtbot, tmp_path, monkeypatch):
    import src.gpu_visualizers as reg
    monkeypatch.setattr(reg, "recipe_dirs", lambda: [tmp_path])

    panel = _panel(qtbot)
    panel.block_combo.setCurrentIndex(0)
    panel._on_add_layer()
    panel.name_input.setText("mein_studio_test")

    # QMessageBox unterdruecken
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)

    saved = []
    panel.recipe_saved.connect(saved.append)
    panel._on_save()

    path = tmp_path / "mein_studio_test.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["name"] == "mein_studio_test"
    assert len(data["layers"]) == 1
    assert saved == ["mein_studio_test"]


def test_speichern_ohne_name_bricht_ab(qtbot, monkeypatch):
    panel = _panel(qtbot)
    panel.block_combo.setCurrentIndex(0)
    panel._on_add_layer()
    from PyQt6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(1))
    panel.name_input.setText("")
    panel._on_save()
    assert warned  # Warnung ausgeloest, nichts gespeichert
