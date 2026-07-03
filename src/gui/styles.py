"""Dark Studio Theme fuer die Audio Visualizer Pro GUI."""


class Theme:
    # Basis-Farben
    BACKGROUND = (10, 10, 15)
    PANEL = (18, 19, 26)
    PANEL_ALT = (24, 26, 34)
    INPUT = (26, 28, 36)
    BORDER = (42, 45, 58)
    # Text
    TEXT_PRIMARY = (232, 233, 236)
    TEXT_SECONDARY = (139, 143, 153)
    TEXT_DISABLED = (90, 94, 107)
    # Akzent & Semantik
    ACCENT = (96, 176, 255)
    ACCENT_HOVER = (130, 195, 255)
    SELECTION = (44, 78, 115)
    SUCCESS = (80, 200, 120)
    ERROR = (255, 95, 95)
    WARNING = (240, 200, 90)

    # Abstands-Tokens (einheitliche Panel-Layouts)
    MARGIN = 8
    SPACING = 12

    @staticmethod
    def rgb(color: tuple[int, int, int]) -> str:
        return f"rgb({color[0]}, {color[1]}, {color[2]})"

    @staticmethod
    def rgba(color: tuple[int, int, int], alpha: float) -> str:
        return f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})"

    @staticmethod
    def hex(color: tuple[int, int, int]) -> str:
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def build_app_stylesheet() -> str:
    bg = Theme.hex(Theme.BACKGROUND)
    panel = Theme.rgb(Theme.PANEL)
    panel_alt = Theme.rgb(Theme.PANEL_ALT)
    inp = Theme.rgb(Theme.INPUT)
    border = Theme.rgb(Theme.BORDER)
    text_primary = Theme.rgb(Theme.TEXT_PRIMARY)
    text_secondary = Theme.rgb(Theme.TEXT_SECONDARY)
    text_disabled = Theme.rgb(Theme.TEXT_DISABLED)
    accent = Theme.rgb(Theme.ACCENT)
    accent_hover = Theme.rgb(Theme.ACCENT_HOVER)
    selection = Theme.rgb(Theme.SELECTION)

    return f"""
    QWidget {{
        background-color: {bg};
        color: {text_primary};
        font-family: "Inter", "Segoe UI", sans-serif;
        font-size: 13px;
        selection-background-color: {selection};
        selection-color: {text_primary};
    }}

    /* === Gruppen === */
    QGroupBox {{
        background-color: {panel};
        border: 1px solid {border};
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 8px;
        font-weight: 600;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {text_secondary};
    }}

    /* === Buttons === */
    QPushButton {{
        background-color: {inp};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 6px 14px;
        color: {text_primary};
    }}

    QPushButton:hover {{
        border-color: {accent};
        background-color: {panel_alt};
    }}

    QPushButton:pressed {{
        background-color: {Theme.rgba(Theme.ACCENT, 0.15)};
    }}

    QPushButton:focus {{
        border-color: {accent};
    }}

    QPushButton:disabled {{
        color: {text_disabled};
        background-color: {panel};
        border-color: {border};
    }}

    QPushButton#primary {{
        background-color: {accent};
        color: {bg};
        border: none;
        font-weight: 600;
    }}

    QPushButton#primary:hover {{
        background-color: {accent_hover};
    }}

    QPushButton#primary:disabled {{
        background-color: {border};
        color: {text_disabled};
    }}

    QPushButton#danger {{
        background-color: {Theme.rgba(Theme.ERROR, 0.15)};
        border: 1px solid {Theme.rgb(Theme.ERROR)};
        color: {Theme.rgb(Theme.ERROR)};
    }}

    /* === Slider === */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {border};
        border-radius: 2px;
    }}

    QSlider::handle:horizontal {{
        background: {accent};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}

    QSlider::handle:horizontal:hover {{
        background: {accent_hover};
    }}

    QSlider::handle:horizontal:disabled {{
        background: {text_disabled};
    }}

    QSlider::sub-page:horizontal {{
        background: {accent};
        border-radius: 2px;
    }}

    QSlider::sub-page:horizontal:disabled {{
        background: {text_disabled};
    }}

    /* === Eingabefelder === */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {inp};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 4px 8px;
    }}

    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {accent};
    }}

    QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {Theme.rgb(Theme.TEXT_SECONDARY)};
    }}

    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        color: {text_disabled};
        background-color: {panel};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {text_secondary};
        margin-right: 6px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {panel_alt};
        border: 1px solid {border};
        border-radius: 4px;
        selection-background-color: {selection};
        outline: none;
        padding: 4px;
    }}

    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background-color: {panel_alt};
        border: none;
        width: 16px;
    }}

    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-bottom: 4px solid {text_secondary};
    }}

    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-top: 4px solid {text_secondary};
    }}

    /* === Checkboxen === */
    QCheckBox {{
        spacing: 8px;
        background: transparent;
    }}

    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {border};
        border-radius: 3px;
        background-color: {inp};
    }}

    QCheckBox::indicator:hover {{
        border-color: {accent};
    }}

    QCheckBox::indicator:checked {{
        background-color: {accent};
        border-color: {accent};
    }}

    QCheckBox:disabled {{
        color: {text_disabled};
    }}

    /* === Tabs === */
    QTabWidget::pane {{
        border: 1px solid {border};
        border-radius: 6px;
        background-color: {panel};
        top: -1px;
    }}

    QTabBar::tab {{
        background-color: transparent;
        color: {text_secondary};
        padding: 7px 18px;
        border: 1px solid transparent;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        margin-right: 2px;
    }}

    QTabBar::tab:hover {{
        color: {text_primary};
    }}

    QTabBar::tab:selected {{
        background-color: {panel};
        color: {text_primary};
        border-color: {border};
        border-bottom-color: {panel};
        font-weight: 600;
    }}

    /* === Scrollbars === */
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background: {border};
        border-radius: 5px;
        min-height: 30px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {text_secondary};
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background: {border};
        border-radius: 5px;
        min-width: 30px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: {text_secondary};
    }}

    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0; height: 0;
    }}

    QScrollBar::add-page, QScrollBar::sub-page {{
        background: transparent;
    }}

    QScrollArea {{
        border: none;
        background: transparent;
    }}

    /* === Listen === */
    QListWidget {{
        background-color: {inp};
        border: 1px solid {border};
        border-radius: 4px;
        outline: none;
        padding: 2px;
    }}

    QListWidget::item {{
        padding: 5px 8px;
        border-radius: 3px;
    }}

    QListWidget::item:hover {{
        background-color: {panel_alt};
    }}

    QListWidget::item:selected {{
        background-color: {selection};
        color: {text_primary};
    }}

    /* === Menues === */
    QMenuBar {{
        background-color: {panel};
        border-bottom: 1px solid {border};
        padding: 2px;
    }}

    QMenuBar::item {{
        padding: 5px 10px;
        border-radius: 4px;
        background: transparent;
    }}

    QMenuBar::item:selected {{
        background-color: {panel_alt};
    }}

    QMenu {{
        background-color: {panel_alt};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 4px;
    }}

    QMenu::item {{
        padding: 6px 24px 6px 12px;
        border-radius: 4px;
    }}

    QMenu::item:selected {{
        background-color: {selection};
    }}

    QMenu::item:disabled {{
        color: {text_disabled};
    }}

    QMenu::separator {{
        height: 1px;
        background: {border};
        margin: 4px 8px;
    }}

    /* === Fortschritt === */
    QProgressBar {{
        background-color: {inp};
        border: 1px solid {border};
        border-radius: 4px;
        height: 8px;
        text-align: center;
        color: transparent;
    }}

    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {accent}, stop:1 {accent_hover});
        border-radius: 3px;
    }}

    /* === Splitter === */
    QSplitter::handle {{
        background-color: {bg};
        width: 4px;
    }}

    QSplitter::handle:hover {{
        background-color: {Theme.rgba(Theme.ACCENT, 0.4)};
    }}

    /* === Tooltips === */
    QToolTip {{
        background-color: {panel_alt};
        color: {text_primary};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 5px 8px;
    }}

    /* === Dialoge === */
    QMessageBox {{
        background-color: {panel};
    }}

    QMessageBox QLabel {{
        color: {text_primary};
    }}

    /* === Labels === */
    QLabel {{
        color: {text_secondary};
        background: transparent;
    }}

    QLabel#heading {{
        color: {text_primary};
        font-size: 16px;
        font-weight: 600;
    }}

    QLabel:disabled {{
        color: {text_disabled};
    }}

    QStatusBar {{
        background-color: {panel};
        color: {text_secondary};
    }}
    """
