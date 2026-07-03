"""Quotes-Panel für die PyQt6-GUI."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QComboBox, QSlider,
    QColorDialog, QGroupBox, QGridLayout, QInputDialog,
)

from src.types import Quote


class QuotesPanel(QWidget):
    def __init__(self, state, gemini=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.gemini = gemini

        self._setup_ui()
        self._connect_signals()
        self._refresh_list()
        self._update_button_states()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # --- Extraktion ---
        extract_box = QGroupBox("Zitate extrahieren")
        extract_layout = QVBoxLayout(extract_box)

        self.chk_enabled = QCheckBox("Zitate aktivieren")
        self.chk_enabled.setChecked(self.state.quotes_enabled)
        self.chk_enabled.stateChanged.connect(self._on_enabled_changed)
        extract_layout.addWidget(self.chk_enabled)

        btn_row = QHBoxLayout()
        self.btn_extract = QPushButton("Key-Zitate extrahieren")
        self.btn_extract.clicked.connect(self._on_extract)
        btn_row.addWidget(self.btn_extract)

        self.btn_demo = QPushButton("Demo-Zitate")
        self.btn_demo.clicked.connect(self._on_demo)
        btn_row.addWidget(self.btn_demo)
        extract_layout.addLayout(btn_row)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        extract_layout.addWidget(self.lbl_status)

        layout.addWidget(extract_box)

        # --- Liste ---
        list_box = QGroupBox("Zitat-Liste")
        list_layout = QVBoxLayout(list_box)
        self.list_quotes = QListWidget()
        self.list_quotes.setMaximumHeight(180)
        list_layout.addWidget(self.list_quotes)

        from src.gui.icons import get_icon

        list_btn_row = QHBoxLayout()
        self.btn_add = QPushButton(" Hinzufügen")
        self.btn_add.setIcon(get_icon("plus"))
        self.btn_add.clicked.connect(self._on_add)
        list_btn_row.addWidget(self.btn_add)

        self.btn_remove = QPushButton(" Entfernen")
        self.btn_remove.setIcon(get_icon("trash"))
        self.btn_remove.clicked.connect(self._on_remove)
        list_btn_row.addWidget(self.btn_remove)

        self.btn_edit = QPushButton(" Bearbeiten")
        self.btn_edit.setIcon(get_icon("edit"))
        self.btn_edit.clicked.connect(self._on_edit)
        list_btn_row.addWidget(self.btn_edit)
        list_layout.addLayout(list_btn_row)

        layout.addWidget(list_box)

        # --- Erscheinungsbild ---
        style_box = QGroupBox("Erscheinungsbild")
        style_layout = QGridLayout(style_box)

        style_layout.addWidget(QLabel("Position"), 0, 0)
        self.combo_position = QComboBox()
        self.combo_position.addItems(["bottom", "center", "top"])
        self.combo_position.setCurrentText(self.state.quote_config.position)
        self.combo_position.currentTextChanged.connect(self._on_style_changed)
        style_layout.addWidget(self.combo_position, 0, 1)

        style_layout.addWidget(QLabel("Schriftgröße"), 1, 0)
        self.slider_font_size = self._make_slider(16, 96, self.state.quote_config.font_size)
        self.slider_font_size.valueChanged.connect(self._on_style_changed)
        style_layout.addWidget(self.slider_font_size, 1, 1)

        style_layout.addWidget(QLabel("Fade-Dauer"), 2, 0)
        self.slider_fade = self._make_slider(1, 20, int(self.state.quote_config.fade_duration * 10))
        self.slider_fade.valueChanged.connect(self._on_style_changed)
        style_layout.addWidget(self.slider_fade, 2, 1)

        style_layout.addWidget(QLabel("Anzeigedauer"), 3, 0)
        self.slider_display = self._make_slider(20, 200, int(self.state.quote_config.display_duration * 10))
        self.slider_display.valueChanged.connect(self._on_style_changed)
        style_layout.addWidget(self.slider_display, 3, 1)

        self.btn_font_color = QPushButton("Textfarbe wählen")
        self.btn_font_color.clicked.connect(self._on_font_color)
        style_layout.addWidget(self.btn_font_color, 4, 0)

        self.btn_box_color = QPushButton("Box-Farbe wählen")
        self.btn_box_color.clicked.connect(self._on_box_color)
        style_layout.addWidget(self.btn_box_color, 4, 1)

        layout.addWidget(style_box)
        layout.addStretch()

    def _make_slider(self, min_val: int, max_val: int, default: int):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        return slider

    def _connect_signals(self):
        self.state.changed.connect(self._on_state_changed)

    def _on_state_changed(self, key: str):
        if key in {"features", "audio_path"}:
            self._update_button_states()
        if key == "quotes":
            self._refresh_list()

    def _update_button_states(self):
        has_audio = bool(self.state.audio_path)
        has_features = self.state.features is not None
        has_gemini = self.gemini is not None
        self.btn_extract.setEnabled(has_audio and has_features and has_gemini)
        if not has_gemini:
            self.lbl_status.setText("KI nicht verfügbar. Prüfe API-Key.")

    def _refresh_list(self):
        self.list_quotes.clear()
        for q in self.state.quotes:
            text = f"{q.text[:40]}{'...' if len(q.text) > 40 else ''} ({q.start_time:.1f}s - {q.end_time:.1f}s)"
            self.list_quotes.addItem(text)

    def _on_enabled_changed(self, state):
        self.state.quotes_enabled = bool(state)

    def _on_extract(self):
        if self.state.audio_path is None or self.state.features is None or self.gemini is None:
            return
        self.state.quotes_extracting = True
        self.btn_extract.setEnabled(False)
        self.btn_extract.setText("⏳ Extrahiere...")
        self.lbl_status.setText("Sende Anfrage an Gemini...")

    def on_extract_finished(self, quotes: list):
        self.state.quotes_extracting = False
        self.btn_extract.setEnabled(True)
        self.btn_extract.setText("Key-Zitate extrahieren")
        self.state.quotes = quotes
        self.lbl_status.setText(f"{len(quotes)} Zitate extrahiert.")

    def on_extract_error(self, msg: str):
        self.state.quotes_extracting = False
        self.btn_extract.setEnabled(True)
        self.btn_extract.setText("Key-Zitate extrahieren")
        self.lbl_status.setText(f"Fehler: {msg}")

    def _on_demo(self):
        duration = getattr(self.state.features, "duration", 10.0) or 10.0
        demo = [
            Quote(text="Das ist ein Beispielzitat.", start_time=1.0, end_time=4.0, confidence=0.9),
            Quote(text="Und hier noch ein zweites Highlight.", start_time=duration * 0.4, end_time=duration * 0.4 + 3.0, confidence=0.85),
        ]
        self.state.quotes = demo
        self.lbl_status.setText("Demo-Zitate hinzugefügt.")

    def _on_add(self):
        duration = getattr(self.state.features, "duration", 10.0) or 10.0
        text, ok = QInputDialog.getText(self, "Zitat hinzufügen", "Text:")
        if ok and text:
            new_quote = Quote(text=text, start_time=duration * 0.3, end_time=duration * 0.3 + 3.0, confidence=1.0)
            self.state.quotes = self.state.quotes + [new_quote]

    def _on_remove(self):
        row = self.list_quotes.currentRow()
        if row >= 0:
            quotes = list(self.state.quotes)
            quotes.pop(row)
            self.state.quotes = quotes

    def _on_edit(self):
        row = self.list_quotes.currentRow()
        if row < 0 or row >= len(self.state.quotes):
            return
        q = self.state.quotes[row]
        text, ok = QInputDialog.getText(self, "Zitat bearbeiten", "Text:", text=q.text)
        if ok and text:
            quotes = list(self.state.quotes)
            quotes[row] = Quote(text=text, start_time=q.start_time, end_time=q.end_time, confidence=q.confidence)
            self.state.quotes = quotes

    def _on_style_changed(self):
        qc = self.state.quote_config
        qc.position = self.combo_position.currentText()
        qc.font_size = self.slider_font_size.value()
        qc.fade_duration = self.slider_fade.value() / 10.0
        qc.display_duration = self.slider_display.value() / 10.0
        self.state.set("quote_config", qc)

    def _on_font_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.state.quote_config.font_color = (color.red(), color.green(), color.blue())
            self.state.set("quote_config", self.state.quote_config)

    def _on_box_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
            self.state.quote_config.box_color = (r, g, b, a)
            self.state.set("quote_config", self.state.quote_config)

    def get_extract_request(self) -> dict:
        return {
            "gemini": self.gemini,
            "audio_path": self.state.audio_path,
            "audio_duration": getattr(self.state.features, "duration", None),
            "max_quotes": None,
        }
