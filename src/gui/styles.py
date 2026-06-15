"""Dark Studio Theme fuer die Audio Visualizer Pro GUI."""


class Theme:
    BACKGROUND = (10, 10, 15)
    PANEL = (18, 19, 26)
    INPUT = (26, 28, 36)
    BORDER = (42, 45, 58)
    TEXT_PRIMARY = (232, 233, 236)
    TEXT_SECONDARY = (139, 143, 153)
    ACCENT = (96, 176, 255)
    SUCCESS = (80, 200, 120)
    ERROR = (255, 95, 95)
    WARNING = (240, 200, 90)

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
    inp = Theme.rgb(Theme.INPUT)
    border = Theme.rgb(Theme.BORDER)
    text_primary = Theme.rgb(Theme.TEXT_PRIMARY)
    text_secondary = Theme.rgb(Theme.TEXT_SECONDARY)
    accent = Theme.rgb(Theme.ACCENT)

    return f"""
    QWidget {{
        background-color: {bg};
        color: {text_primary};
        font-family: "Segoe UI", "Inter", sans-serif;
        font-size: 13px;
    }}

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

    QPushButton {{
        background-color: {inp};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 6px 14px;
        color: {text_primary};
    }}

    QPushButton:hover {{
        border-color: {accent};
    }}

    QPushButton:pressed {{
        background-color: {Theme.rgba(Theme.ACCENT, 0.15)};
    }}

    QPushButton#primary {{
        background-color: {accent};
        color: {bg};
        border: none;
        font-weight: 600;
    }}

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

    QSlider::sub-page:horizontal {{
        background: {accent};
        border-radius: 2px;
    }}

    QLineEdit, QComboBox, QSpinBox {{
        background-color: {inp};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 4px 8px;
    }}

    QLabel {{
        color: {text_secondary};
    }}

    QLabel#heading {{
        color: {text_primary};
        font-size: 16px;
        font-weight: 600;
    }}

    QStatusBar {{
        background-color: {panel};
        color: {text_secondary};
    }}
    """
