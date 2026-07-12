"""Visualizer-Studio: Bausteine zu einem eigenen Visualizer kombinieren.

Der Nutzer stellt Ebenen aus Bausteinen zusammen, stellt Parameter per Regler
ein, sieht eine Vorschau und speichert das Ergebnis als vollwertigen Visualizer
(JSON-Rezept). Kein Code noetig.
"""

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QDoubleSpinBox, QGroupBox, QGridLayout, QLineEdit,
    QMessageBox, QScrollArea, QInputDialog,
)

from src.app_logging import get_logger
from src.gpu_visualizers.blocks import BLOCK_LIBRARY, BLEND_MODES, AUDIO_SOURCES

logger = get_logger(__name__)


class StudioPanel(QWidget):
    """Editor fuer rezeptbasierte Visualizer mit Live-Vorschau."""

    recipe_saved = pyqtSignal(str)  # Name des gespeicherten Visualizers

    def __init__(self, state, gemini=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.gemini = gemini
        # Arbeits-Rezept: Liste von Ebenen-Dicts
        self._layers = []
        self._current_layer = -1

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._render_preview)

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QHBoxLayout(self)

        # --- Linke Spalte: Vorschau + Ebenenliste ---
        left = QVBoxLayout()

        self.preview_label = QLabel("Vorschau")
        self.preview_label.setMinimumSize(320, 180)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background:#111; border:1px solid #333;")
        left.addWidget(self.preview_label)

        # KI-Assistent
        ai_box = QGroupBox("KI-Assistent")
        ai_layout = QVBoxLayout(ai_box)
        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText("Beschreibe deinen Visualizer…")
        ai_layout.addWidget(self.ai_input)
        self.btn_ai = QPushButton("Rezept vorschlagen")
        self.btn_ai.clicked.connect(self._on_ai_suggest)
        ai_layout.addWidget(self.btn_ai)
        left.addWidget(ai_box)

        layer_box = QGroupBox("Ebenen")
        layer_layout = QVBoxLayout(layer_box)
        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self._on_layer_selected)
        layer_layout.addWidget(self.layer_list)

        add_row = QHBoxLayout()
        self.block_combo = QComboBox()
        for key, block in BLOCK_LIBRARY.items():
            self.block_combo.addItem(block["display_name"], key)
        add_row.addWidget(self.block_combo)
        btn_add = QPushButton("+ Ebene")
        btn_add.clicked.connect(self._on_add_layer)
        add_row.addWidget(btn_add)
        layer_layout.addLayout(add_row)

        ctrl_row = QHBoxLayout()
        for text, slot in [("Entfernen", self._on_remove_layer),
                           ("▲", lambda: self._move_layer(-1)),
                           ("▼", lambda: self._move_layer(1))]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            ctrl_row.addWidget(b)
        layer_layout.addLayout(ctrl_row)
        left.addWidget(layer_box)

        root.addLayout(left, 1)

        # --- Rechte Spalte: Parameter der gewaehlten Ebene ---
        right = QVBoxLayout()
        self.detail_box = QGroupBox("Ebenen-Einstellungen")
        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        self.detail_inner = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_inner)
        self.detail_scroll.setWidget(self.detail_inner)
        db = QVBoxLayout(self.detail_box)
        db.addWidget(self.detail_scroll)
        right.addWidget(self.detail_box)

        # Speichern
        save_row = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("visualizer_name (klein, mit _)")
        save_row.addWidget(self.name_input)
        btn_save = QPushButton("Als Visualizer speichern")
        btn_save.clicked.connect(self._on_save)
        save_row.addWidget(btn_save)
        right.addLayout(save_row)

        root.addLayout(right, 1)

    # ------------------------------------------------------------------
    # Ebenen-Verwaltung
    # ------------------------------------------------------------------

    def _refresh_layer_list(self):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for i, layer in enumerate(self._layers):
            block = BLOCK_LIBRARY.get(layer["block"], {})
            self.layer_list.addItem(f"{i + 1}. {block.get('display_name', layer['block'])} ({layer['blend']})")
        self.layer_list.blockSignals(False)
        if 0 <= self._current_layer < len(self._layers):
            self.layer_list.setCurrentRow(self._current_layer)

    def _on_add_layer(self):
        block_key = self.block_combo.currentData()
        block = BLOCK_LIBRARY[block_key]
        self._layers.append({
            "block": block_key,
            "blend": "add",
            "transform": {"offset_x": 0.0, "offset_y": 0.0, "scale": 1.0, "rotation_speed": 0.0},
            "params": {p: spec[0] for p, spec in block["params"].items()},
            "mappings": [],
        })
        self._current_layer = len(self._layers) - 1
        self._refresh_layer_list()
        self._rebuild_detail()
        self._schedule_preview()

    def _on_remove_layer(self):
        if 0 <= self._current_layer < len(self._layers):
            self._layers.pop(self._current_layer)
            self._current_layer = min(self._current_layer, len(self._layers) - 1)
            self._refresh_layer_list()
            self._rebuild_detail()
            self._schedule_preview()

    def _move_layer(self, delta):
        i = self._current_layer
        j = i + delta
        if 0 <= i < len(self._layers) and 0 <= j < len(self._layers):
            self._layers[i], self._layers[j] = self._layers[j], self._layers[i]
            self._current_layer = j
            self._refresh_layer_list()
            self._schedule_preview()

    def _on_layer_selected(self, row):
        self._current_layer = row
        self._rebuild_detail()

    # ------------------------------------------------------------------
    # Detail-Editor (Blend, Params, Mappings)
    # ------------------------------------------------------------------

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _rebuild_detail(self):
        self._clear_layout(self.detail_layout)
        if not (0 <= self._current_layer < len(self._layers)):
            self.detail_layout.addWidget(QLabel("Keine Ebene gewaehlt."))
            return
        layer = self._layers[self._current_layer]
        block = BLOCK_LIBRARY[layer["block"]]

        # Blend-Modus
        blend_row = QHBoxLayout()
        blend_row.addWidget(QLabel("Mischung"))
        blend_combo = QComboBox()
        blend_combo.addItems(list(BLEND_MODES.keys()))
        blend_combo.setCurrentText(layer["blend"])
        blend_combo.currentTextChanged.connect(self._on_blend_changed)
        blend_row.addWidget(blend_combo)
        bw = QWidget(); bw.setLayout(blend_row)
        self.detail_layout.addWidget(bw)

        # Parameter
        pbox = QGroupBox("Parameter")
        grid = QGridLayout(pbox)
        for r, (pname, spec) in enumerate(block["params"].items()):
            grid.addWidget(QLabel(pname), r, 0)
            spin = QDoubleSpinBox()
            spin.setRange(spec[1], spec[2])
            spin.setSingleStep(spec[3])
            spin.setDecimals(0 if spec[3] >= 1 else 2)
            spin.setValue(float(layer["params"].get(pname, spec[0])))
            spin.valueChanged.connect(lambda v, n=pname: self._on_param_changed(n, v))
            grid.addWidget(spin, r, 1)
        self.detail_layout.addWidget(pbox)

        # Mappings
        mbox = QGroupBox("Audio-Verknuepfungen")
        mlayout = QVBoxLayout(mbox)
        for mi, m in enumerate(layer["mappings"]):
            mlayout.addWidget(QLabel(
                f"{m['target']} ← {m['source']} × {m.get('gain', 0.3):.2f}"
            ))
        add_map = QPushButton("+ Verknuepfung")
        add_map.clicked.connect(self._on_add_mapping)
        mlayout.addWidget(add_map)
        self.detail_layout.addWidget(mbox)
        self.detail_layout.addStretch()

    def _on_blend_changed(self, value):
        if 0 <= self._current_layer < len(self._layers):
            self._layers[self._current_layer]["blend"] = value
            self._refresh_layer_list()
            self._schedule_preview()

    def _on_param_changed(self, name, value):
        if 0 <= self._current_layer < len(self._layers):
            self._layers[self._current_layer]["params"][name] = float(value)
            self._schedule_preview()

    def _on_add_mapping(self):
        if not (0 <= self._current_layer < len(self._layers)):
            return
        layer = self._layers[self._current_layer]
        block = BLOCK_LIBRARY[layer["block"]]
        target, ok = QInputDialog.getItem(
            self, "Ziel-Parameter", "Parameter:", list(block["params"].keys()), 0, False
        )
        if not ok:
            return
        source, ok = QInputDialog.getItem(
            self, "Audio-Quelle", "Quelle:", AUDIO_SOURCES, 0, False
        )
        if not ok:
            return
        layer["mappings"].append({"target": target, "source": source,
                                  "gain": 0.3, "offset": 0.0, "smooth": 0.2})
        self._rebuild_detail()
        self._schedule_preview()

    # ------------------------------------------------------------------
    # Vorschau
    # ------------------------------------------------------------------

    def _schedule_preview(self):
        self._preview_timer.start(300)

    def _current_recipe(self, name="_studio_vorschau"):
        return {
            "name": name,
            "display_name": name,
            "mode_hint": "music",
            "layers": self._layers,
            "color": {
                "primary": self.state.primary_color,
                "secondary": self.state.secondary_color,
                "background": self.state.background_color,
            },
            "version": 1,
        }

    def _synthetic_features(self):
        from src.types import AudioFeatures
        n = 60
        return AudioFeatures(
            duration=2.0, sample_rate=22050, fps=30, frame_count=n,
            rms=np.abs(np.sin(np.linspace(0, 6, n))).astype(np.float32),
            onset=np.abs(np.cos(np.linspace(0, 12, n))).astype(np.float32),
            spectral_centroid=np.linspace(0.3, 0.7, n).astype(np.float32),
            spectral_rolloff=np.linspace(0.4, 0.6, n).astype(np.float32),
            zero_crossing_rate=np.full(n, 0.2, dtype=np.float32),
            transient=np.abs(np.cos(np.linspace(0, 20, n))).astype(np.float32),
            voice_clarity=np.full(n, 0.5, dtype=np.float32),
            voice_band=np.full(n, 0.5, dtype=np.float32),
            chroma=np.abs(np.random.rand(12, n)).astype(np.float32),
            mfcc=np.zeros((13, n), dtype=np.float32),
            tempogram=np.zeros((384, n), dtype=np.float32),
            tempo=120.0, key="C", mode="music",
            beat_frames=np.arange(0, n, 15).astype(np.int64),
        )

    def _render_preview(self):
        if not self._layers:
            self.preview_label.setText("Fuege eine Ebene hinzu.")
            return
        try:
            from src.gpu_visualizers import VISUALIZER_MAP
            from src.gpu_visualizers.composite import make_recipe_visualizer_class
            from src.gpu_preview import render_gpu_preview

            recipe = self._current_recipe()
            VISUALIZER_MAP["_studio_vorschau"] = make_recipe_visualizer_class(recipe)

            features = self.state.features or self._synthetic_features()
            audio_path = self.state.audio_path or "studio"
            img = render_gpu_preview(
                audio_path=audio_path,
                visualizer_type="_studio_vorschau",
                width=384, height=216, fps=30,
                preview_time_percent=self.state.preview_time_percent,
                features=features,
                background_color=self.state.background_color,
            )
            if img is not None:
                qimg = QImage(
                    img.tobytes(), img.width, img.height,
                    img.width * 3, QImage.Format.Format_RGB888,
                )
                self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                    self.preview_label.width(), self.preview_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        except Exception as e:
            logger.warning(f"[Studio] Vorschau fehlgeschlagen: {e}")
            self.preview_label.setText(f"Vorschau-Fehler:\n{e}")

    # ------------------------------------------------------------------
    # Speichern & KI
    # ------------------------------------------------------------------

    def _on_save(self):
        from config.schemas import RecipeSchema
        from src.gpu_visualizers import refresh_registry, recipe_dirs

        name = self.name_input.text().strip().lower().replace(" ", "_")
        if not name:
            QMessageBox.warning(self, "Name fehlt", "Bitte einen Namen eingeben.")
            return
        recipe = self._current_recipe(name=name)
        try:
            validated = RecipeSchema(**recipe)
        except Exception as e:
            QMessageBox.critical(self, "Rezept ungueltig", f"Das Rezept ist ungueltig:\n{e}")
            return

        # In das (beschreibbare) Nutzer-Verzeichnis speichern
        user_dir = recipe_dirs()[-1]
        try:
            user_dir.mkdir(parents=True, exist_ok=True)
            path = user_dir / f"{name}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(validated.model_dump(), fh, ensure_ascii=False, indent=2)
        except OSError as e:
            QMessageBox.critical(self, "Speichern fehlgeschlagen", str(e))
            return

        refresh_registry()
        self.recipe_saved.emit(name)
        QMessageBox.information(
            self, "Gespeichert",
            f"Visualizer '{name}' gespeichert und in der Auswahl verfuegbar.",
        )

    def _on_ai_suggest(self):
        text = self.ai_input.text().strip()
        if not text:
            return
        if self.gemini is None:
            QMessageBox.information(self, "KI nicht verfuegbar",
                                    "Kein Gemini-API-Key gesetzt.")
            return
        self.btn_ai.setEnabled(False)
        self.btn_ai.setText("⏳ …")
        try:
            layers = self.gemini.suggest_recipe(text, list(BLOCK_LIBRARY.keys()))
            if layers:
                self._layers = layers
                self._current_layer = 0
                self._refresh_layer_list()
                self._rebuild_detail()
                self._schedule_preview()
            else:
                QMessageBox.information(self, "Kein Vorschlag",
                                        "Die KI lieferte kein gueltiges Rezept.")
        except Exception as e:
            QMessageBox.warning(self, "KI-Fehler", str(e))
        finally:
            self.btn_ai.setEnabled(True)
            self.btn_ai.setText("Rezept vorschlagen")
