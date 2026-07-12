"""KI-Panel fuer die PyQt6-GUI."""

import colorsys

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox,
)

from src.ai_matcher import SmartMatcher
from src.app_logging import get_logger
from src.gui.helpers import _features_to_dict
from src.gpu_visualizers import get_visualizer
from src.quote_overlay import QuoteOverlayConfig
from config.schemas import QuoteOverlayConfigSchema

logger = get_logger(__name__)


class KIPanel(QWidget):
    optimize_requested = pyqtSignal()

    def __init__(self, state, gemini=None, parent=None, gemini_error=None):
        super().__init__(parent)
        self.state = state
        self.gemini = gemini
        self.gemini_error = gemini_error
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

        self.lbl_cost = QLabel("")
        self.lbl_cost.setWordWrap(True)
        self.lbl_cost.setToolTip("Geschaetzte KI-Kosten dieser Sitzung.")
        opt_layout.addWidget(self.lbl_cost)

        layout.addWidget(opt_box)
        layout.addStretch()

        self._update_cost_label()

    def _update_cost_label(self):
        """Aktualisiert die Anzeige der geschaetzten KI-Kosten."""
        try:
            from src.ai_costs import get_cost_ledger
            self.lbl_cost.setText(get_cost_ledger().summary())
        except Exception:
            self.lbl_cost.setText("")

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
            reason = f"\nGrund: {self.gemini_error}" if self.gemini_error else ""
            self.lbl_status.setText(
                f"KI nicht verfuegbar. Pruefe API-Key in .env (GEMINI_API_KEY).{reason}"
            )

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
        self._update_cost_label()

    def on_optimize_error(self, msg: str, tb: str = ""):
        if tb:
            logger.error(f"[KI] Optimierungs-Fehler:\n{tb}")
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

        quotes = result.get("quotes")
        if quotes is not None:
            self._apply_quotes_to_state(quotes)

    def _apply_quotes_to_state(self, quotes_data: dict):
        """Wendet optimierte Quote-Einstellungen auf den State an.

        Validiert das Eingabe-Dictionary gegen das QuoteOverlayConfig-Schema
        und überträgt die Werte in die QuoteOverlayConfig-Dataclass.
        """
        if not isinstance(quotes_data, dict):
            return

        try:
            validated = QuoteOverlayConfigSchema(**quotes_data).model_dump()
        except Exception as e:
            logger.warning(f"[KIPanel] Quote-Config-Validierung fehlgeschlagen: {e}")
            return

        def _as_tuple(value):
            if isinstance(value, (list, tuple)):
                return tuple(value)
            return value

        current = self.state.quote_config
        config_kwargs = {
            "enabled": validated.get("enabled", current.enabled),
            "font_size": validated.get("font_size", current.font_size),
            "font_color": _as_tuple(validated.get("font_color", current.font_color)),
            "box_color": _as_tuple(validated.get("box_color", current.box_color)),
            "box_alpha": validated.get("box_alpha", current.box_alpha),
            "box_padding": validated.get("box_padding", current.box_padding),
            "box_radius": validated.get("box_radius", current.box_radius),
            "box_margin_bottom": validated.get("box_margin_bottom", current.box_margin_bottom),
            "max_width_ratio": validated.get("max_width_ratio", current.max_width_ratio),
            "fade_duration": validated.get("fade_duration", current.fade_duration),
            "shadow_color": _as_tuple(validated.get("text_shadow_color", current.shadow_color)),
            "shadow_offset": _as_tuple(validated.get("text_shadow_offset", current.shadow_offset)),
            "line_spacing": validated.get("line_spacing", current.line_spacing),
            "max_chars_per_line": validated.get("max_chars_per_line", current.max_chars_per_line),
            "display_duration": validated.get("display_duration", current.display_duration),
            "position": validated.get("position", current.position),
            "font_path": validated.get("font_path", current.font_path),
            "text_align": validated.get("text_align", current.text_align),
            "auto_scale_font": validated.get("auto_scale_font", current.auto_scale_font),
            "min_font_size": validated.get("min_font_size", current.min_font_size),
            "max_font_size": validated.get("max_font_size", current.max_font_size),
            "text_shadow_enabled": validated.get("text_shadow_enabled", current.text_shadow_enabled),
            "text_shadow_color": _as_tuple(validated.get("text_shadow_color", current.text_shadow_color)),
            "text_shadow_offset": _as_tuple(validated.get("text_shadow_offset", current.text_shadow_offset)),
            "text_shadow_blur": validated.get("text_shadow_blur", current.text_shadow_blur),
            "box_gradient": validated.get("box_gradient", current.box_gradient),
            "accent_line": validated.get("accent_line", current.accent_line),
            "accent_line_color": _as_tuple(validated.get("accent_line_color", current.accent_line_color)),
            "accent_line_height": validated.get("accent_line_height", current.accent_line_height),
            "spatial_compensation": validated.get("spatial_compensation", current.spatial_compensation),
            "compensation_blur": validated.get("compensation_blur", current.compensation_blur),
            "compensation_darken": validated.get("compensation_darken", current.compensation_darken),
            "latency_offset": validated.get("latency_offset", current.latency_offset),
            "buffer_lookahead": validated.get("buffer_lookahead", current.buffer_lookahead),
            "offset_x": validated.get("offset_x", current.offset_x),
            "offset_y": validated.get("offset_y", current.offset_y),
            "scale": validated.get("scale", current.scale),
        }
        self.state.quote_config = QuoteOverlayConfig(**config_kwargs)

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
            "audio_features": _features_to_dict(self.state.features),
            "colors": self.state.ki_suggested_colors or {},
            "param_specs": param_specs,
            "user_prompt": self.prompt_input.text().strip() or None,
            "recommendation": rec,
        }

