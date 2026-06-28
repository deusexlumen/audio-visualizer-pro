"""KI-Panel fuer die PyQt6-GUI."""

import colorsys

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox,
)

from src.ai_matcher import SmartMatcher
from src.gpu_visualizers import get_visualizer


class KIPanel(QWidget):
    optimize_requested = pyqtSignal()

    def __init__(self, state, gemini=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.gemini = gemini
        self._matcher = SmartMatcher()
        self._last_recommendation = None

        self._setup_ui()
        self._connect_signals()
        self._update_button_states()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # --- KI Komplett-Optimierung ---
        opt_box = QGroupBox("KI-Optimierung")
        opt_layout = QVBoxLayout(opt_box)

        self.lbl_recommendation = QLabel("Noch keine Optimierung durchgefuehrt.")
        self.lbl_recommendation.setWordWrap(True)
        opt_layout.addWidget(self.lbl_recommendation)

        opt_layout.addWidget(QLabel("Dein Wunsch (optional):"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("z.B. dunkler, mehr Kontrast, cyberpunk-Stil")
        opt_layout.addWidget(self.prompt_input)

        self.btn_optimize = QPushButton("Komplett optimieren")
        self.btn_optimize.setToolTip(
            "Waehlt automatisch den passenden Visualizer und optimiert Parameter, Farben und Post-Process."
        )
        self.btn_optimize.clicked.connect(self._on_optimize)
        opt_layout.addWidget(self.btn_optimize)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        opt_layout.addWidget(self.lbl_status)

        self.lbl_colors = QLabel("")
        self.lbl_colors.setWordWrap(True)
        opt_layout.addWidget(self.lbl_colors)

        layout.addWidget(opt_box)
        layout.addStretch()

    def _connect_signals(self):
        self.state.changed.connect(self._on_state_changed)

    def _on_state_changed(self, key: str):
        if key == "features":
            self._last_recommendation = None
            self.lbl_recommendation.setText("Noch keine Optimierung durchgefuehrt.")
            self.lbl_status.setText("")
            self._update_button_states()

    def _update_button_states(self):
        has_features = self.state.features is not None
        has_gemini = self.gemini is not None
        self.btn_optimize.setEnabled(has_features and has_gemini and not self.state.ki_optimizing)
        if not has_gemini:
            self.lbl_status.setText("KI nicht verfuegbar. Pruefe API-Key in .env (GEMINI_API_KEY).")

    def _on_optimize(self):
        """Startet die kombinierte SmartMatcher + Gemini Optimierung."""
        if self.state.features is None or self.gemini is None:
            return

        # 1. SmartMatcher: Visualizer + initiale Parameter + Farben
        try:
            rec = self._matcher.match(self.state.features)
            self._last_recommendation = rec
            self._apply_recommendation(rec)
            self.lbl_recommendation.setText(
                f"Empfohlener Visualizer: {rec.visualizer} ({rec.confidence:.0%})\n{rec.reason}"
            )
        except Exception as e:
            self.lbl_status.setText(f"SmartMatcher-Fehler: {e}")
            return

        # 2. Gemini-Optimierung anfordern (wird von MainWindow als Worker gestartet)
        self.state.ki_optimizing = True
        self.btn_optimize.setEnabled(False)
        self.btn_optimize.setText("⏳ KI denkt nach...")
        self.lbl_status.setText("Sende Anfrage an Gemini...")
        self.optimize_requested.emit()

    def _apply_recommendation(self, rec):
        """Uebernimmt SmartMatcher-Empfehlung in den Zustand."""
        try:
            viz_class = get_visualizer(rec.visualizer)
        except ValueError:
            viz_class = None
        if viz_class is None:
            self.lbl_status.setText(f"Fehler: Visualizer '{rec.visualizer}' nicht verfuegbar.")
            return
        self.state.visualizer_type = rec.visualizer
        self.state.viz_params = {**self.state.viz_params, **rec.params}
        self.state.ki_suggested_colors = rec.colors
        self._apply_colors_to_state(rec.colors)
        self.lbl_colors.setText(
            f"Primary: {rec.colors.get('primary', '-')}  "
            f"Secondary: {rec.colors.get('secondary', '-')}  "
            f"BG: {rec.colors.get('background', '-')}"
        )

    def _apply_colors_to_state(self, colors: dict):
        """Wendet eine Farbpalette auf den GUI-State an."""
        primary = colors.get("primary")
        if primary and primary.startswith('#') and len(primary) == 7:
            self.state.primary_color = primary.upper()
            try:
                r = int(primary[1:3], 16) / 255.0
                g = int(primary[3:5], 16) / 255.0
                b = int(primary[5:7], 16) / 255.0
                h, s, _ = colorsys.rgb_to_hsv(r, g, b)
                self.state.base_hue = h
                self.state.color_saturation = s
            except Exception:
                pass
        secondary = colors.get("secondary")
        if secondary and secondary.startswith('#') and len(secondary) == 7:
            self.state.secondary_color = secondary.upper()
        background = colors.get("background")
        if background and background.startswith('#') and len(background) == 7:
            self.state.background_color = background.upper()
        # Damit die vorgeschlagenen Farben tatsaechlich greifen, wechseln wir in den fixed-Modus
        self.state.color_mode = "fixed"

    def on_optimize_finished(self, result: dict):
        self.state.ki_optimizing = False
        self.btn_optimize.setEnabled(True)
        self.btn_optimize.setText("Komplett optimieren")
        self._apply_optimize_result(result)
        self.lbl_status.setText("Parameter optimiert!")

    def on_optimize_error(self, msg: str):
        self.state.ki_optimizing = False
        self.btn_optimize.setEnabled(True)
        self.btn_optimize.setText("Komplett optimieren")
        self.lbl_status.setText(f"KI-Fehler: {msg}")

    @staticmethod
    def _safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _apply_optimize_result(self, result: dict):
        if not isinstance(result, dict):
            return

        params = result.get("params", {})
        self.state.viz_params = {**self.state.viz_params, **params}

        pp = result.get("postprocess", {})
        val = self._safe_float(pp.get("contrast"))
        if val is not None:
            self.state.pp_contrast = val
        val = self._safe_float(pp.get("saturation"))
        if val is not None:
            self.state.pp_saturation = val
        val = self._safe_float(pp.get("brightness"))
        if val is not None:
            self.state.pp_brightness = val
        val = self._safe_float(pp.get("warmth"))
        if val is not None:
            self.state.pp_warmth = val
        val = self._safe_float(pp.get("film_grain"))
        if val is not None:
            self.state.pp_grain = val

        bg = result.get("background", {})
        val = self._safe_float(bg.get("blur"))
        if val is not None:
            self.state.bg_blur = val
        val = self._safe_float(bg.get("vignette"))
        if val is not None:
            self.state.bg_vignette = val
        val = self._safe_float(bg.get("opacity"))
        if val is not None:
            self.state.bg_opacity = val

        colors = result.get("colors", {})
        if colors:
            self.state.ki_suggested_colors = colors
            self._apply_colors_to_state(colors)
            self.lbl_colors.setText(
                f"Primary: {colors.get('primary', '-')}  "
                f"Secondary: {colors.get('secondary', '-')}  "
                f"BG: {colors.get('background', '-')}"
            )

    def get_optimize_request(self) -> dict:
        """Liefert die Daten, die der AIOptimizeWorker braucht."""
        try:
            viz_class = get_visualizer(self.state.visualizer_type)
        except ValueError:
            viz_class = None
        param_specs = {}
        if viz_class is not None:
            if hasattr(viz_class, "EFFECTS"):
                param_specs.update(viz_class.EFFECTS)
            if hasattr(viz_class, "COLOR_PARAMS"):
                # String-Parameter bekommen einen "Bereich" aus erlaubten Werten
                for k, v in viz_class.COLOR_PARAMS.items():
                    if isinstance(v, tuple) and len(v) == 4:
                        param_specs[k] = v
                    elif isinstance(v, str):
                        # Pseudo-Spec fuer color_mode etc. (keine numerische Clamp)
                        param_specs[k] = (v, None, None, None)
            if hasattr(viz_class, "PARAMS"):
                param_specs.update(viz_class.PARAMS)

        rec = None
        if self._last_recommendation is not None:
            rec = {
                "visualizer": self._last_recommendation.visualizer,
                "confidence": self._last_recommendation.confidence,
                "reason": self._last_recommendation.reason,
                "colors": self._last_recommendation.colors,
            }

        return {
            "gemini": self.gemini,
            "visualizer_type": self.state.visualizer_type,
            "current_params": self.state.get_params(),
            "audio_features": self._features_to_dict(self.state.features),
            "colors": self.state.ki_suggested_colors or {},
            "param_specs": param_specs,
            "user_prompt": self.prompt_input.text().strip() or None,
            "recommendation": rec,
        }

    @staticmethod
    def _features_to_dict(features) -> dict:
        import numpy as np

        def _mean(arr):
            arr = np.asarray(arr)
            return float(arr.mean()) if arr.size else 0.0

        def _std(arr):
            arr = np.asarray(arr)
            return float(arr.std()) if arr.size else 0.0

        return {
            "duration": float(getattr(features, "duration", 0)),
            "tempo": float(getattr(features, "tempo", 120)),
            "mode": str(getattr(features, "mode", "music")),
            "rms_mean": _mean(getattr(features, "rms", [])),
            "rms_std": _std(getattr(features, "rms", [])),
            "onset_mean": _mean(getattr(features, "onset", [])),
            "onset_std": _std(getattr(features, "onset", [])),
            "spectral_mean": _mean(getattr(features, "spectral_centroid", [])),
            "brightness": _mean(getattr(features, "spectral_centroid", [])),
            "noisiness": _mean(getattr(features, "zero_crossing_rate", [])),
            "transient_mean": _mean(getattr(features, "transient", [])),
            "voice_clarity_mean": _mean(getattr(features, "voice_clarity", [])),
        }
